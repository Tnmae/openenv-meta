"""
Inference Script — Brand-Safe Ad Review Environment
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables
"""

import json
import os
import re
import statistics
import textwrap
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")
ENV_URL = os.getenv("ENV_URL", "http://localhost:8000")

TEMPERATURE = 0.1
MAX_TOKENS = 600

# ---------------------------------------------------------------------------
# Valid values (must match ad_review_env/models.py)
# ---------------------------------------------------------------------------

VALID_DECISIONS = {"APPROVE", "REJECT", "ESCALATE"}
VALID_IAB = {
    "IAB_SAFE", "IAB_ADULT", "IAB_VIOLENCE", "IAB_HATE_SPEECH", "IAB_ILLEGAL",
    "IAB_MISINFORMATION", "IAB_PROFANITY", "IAB_DRUGS", "IAB_GAMBLING",
    "IAB_CONTROVERSIAL",
}
VALID_GARM = {
    "GARM_SAFE", "GARM_ADULT_EXPLICIT", "GARM_ARMS_AMMUNITION",
    "GARM_CRIME_HARMFUL", "GARM_DEATH_INJURY", "GARM_HATE_SPEECH",
    "GARM_OBSCENITY_PROFANITY", "GARM_ONLINE_PIRACY", "GARM_SPAM_HARMFUL",
    "GARM_TERRORISM",
}
VALID_RISK = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}

# ---------------------------------------------------------------------------
# LLM system prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""\
You are an expert content moderator reviewing user-generated content (UGC)
for brand-safe ad placement using IAB Content Taxonomy 3.0 and GARM Brand
Safety Floor standards.

For each content item, respond with ONLY a JSON object (no other text):
{
  "decision": "APPROVE" | "REJECT" | "ESCALATE",
  "iab_category": "<IAB category>",
  "garm_category": "<GARM category>",
  "risk_level": "LOW" | "MEDIUM" | "HIGH" | "CRITICAL",
  "reasoning": "<10-500 char explanation>",
  "confidence": <float 0.0-1.0>,
  "flagged_elements": ["<element1>", ...]
}

IAB categories: IAB_SAFE, IAB_ADULT, IAB_VIOLENCE, IAB_HATE_SPEECH,
  IAB_ILLEGAL, IAB_MISINFORMATION, IAB_PROFANITY, IAB_DRUGS,
  IAB_GAMBLING, IAB_CONTROVERSIAL

GARM categories: GARM_SAFE, GARM_ADULT_EXPLICIT, GARM_ARMS_AMMUNITION,
  GARM_CRIME_HARMFUL, GARM_DEATH_INJURY, GARM_HATE_SPEECH,
  GARM_OBSCENITY_PROFANITY, GARM_ONLINE_PIRACY, GARM_SPAM_HARMFUL,
  GARM_TERRORISM

Decision guidelines:
- APPROVE: Content is clearly brand-safe for ad placement
- REJECT: Content contains harmful, explicit, illegal, or unsafe material
- ESCALATE: Content is ambiguous or context-dependent, needs human review

Pay close attention to:
- Satire vs genuine harmful content
- Educational/advocacy context vs promotion of harmful behavior
- Coded language, dog-whistles, and euphemisms
- Platform norms (Instagram, TikTok, YouTube, Twitter)
- Dual-use content (e.g. harm reduction vs drug promotion)

Respond with ONLY the JSON object.""")

# ---------------------------------------------------------------------------
# Fallback action (used when LLM parsing fails)
# ---------------------------------------------------------------------------

FALLBACK_ACTION: Dict[str, Any] = {
    "decision": "ESCALATE",
    "iab_category": "IAB_CONTROVERSIAL",
    "garm_category": "GARM_SAFE",
    "risk_level": "MEDIUM",
    "reasoning": "Unable to parse LLM response. Escalating for human review as a safety precaution.",
    "confidence": 0.3,
    "flagged_elements": [],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_json(text: str) -> Optional[Dict]:
    """Extract a JSON object from LLM output, handling markdown code blocks."""
    # Try markdown code block first
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find a raw JSON object
    match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # Try the full text
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return None


def validate_action(raw: Dict) -> Dict[str, Any]:
    """Validate and sanitize an action dict, fixing common LLM mistakes."""
    action: Dict[str, Any] = {}

    decision = str(raw.get("decision", "ESCALATE")).upper().strip()
    action["decision"] = decision if decision in VALID_DECISIONS else "ESCALATE"

    iab = str(raw.get("iab_category", "IAB_CONTROVERSIAL")).upper().strip()
    action["iab_category"] = iab if iab in VALID_IAB else "IAB_CONTROVERSIAL"

    garm = str(raw.get("garm_category", "GARM_SAFE")).upper().strip()
    action["garm_category"] = garm if garm in VALID_GARM else "GARM_SAFE"

    risk = str(raw.get("risk_level", "MEDIUM")).upper().strip()
    action["risk_level"] = risk if risk in VALID_RISK else "MEDIUM"

    reasoning = str(raw.get("reasoning", ""))
    if len(reasoning) < 10:
        reasoning = "Content reviewed by LLM agent. Decision based on contextual analysis of the content."
    action["reasoning"] = reasoning[:500]

    try:
        conf = float(raw.get("confidence", 0.5))
        action["confidence"] = max(0.0, min(1.0, conf))
    except (TypeError, ValueError):
        action["confidence"] = 0.5

    flagged = raw.get("flagged_elements", [])
    if isinstance(flagged, list):
        action["flagged_elements"] = [str(e) for e in flagged[:5]]
    else:
        action["flagged_elements"] = []

    return action


def call_llm(
    client: OpenAI,
    content_text: str,
    content_type: str,
    platform: str,
) -> Dict[str, Any]:
    """Use the LLM to review a UGC content item and return an action dict."""
    user_msg = (
        f"Platform: {platform}\n"
        f"Content type: {content_type}\n"
        f"\nContent:\n{content_text}"
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        raw_text = response.choices[0].message.content.strip()
        parsed = extract_json(raw_text)

        if parsed is None:
            print("    ⚠ Failed to parse LLM JSON, using fallback")
            return FALLBACK_ACTION.copy()

        return validate_action(parsed)

    except Exception as e:
        print(f"    ⚠ LLM call failed: {e}, using fallback")
        return FALLBACK_ACTION.copy()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run_evaluation(client: OpenAI, env_url: str) -> Dict[str, Any]:
    """Run inference across all tasks and collect scores."""

    # 1. Verify environment is reachable
    try:
        health = requests.get(f"{env_url}/health", timeout=10)
        health.raise_for_status()
        print(f"✓ Environment healthy at {env_url}\n")
    except Exception as e:
        raise RuntimeError(f"Cannot reach environment at {env_url}: {e}") from e

    # 2. Fetch all 30 tasks (seed=42 for reproducibility)
    resp = requests.get(f"{env_url}/tasks", params={"n": 30, "seed": 42}, timeout=30)
    resp.raise_for_status()
    tasks = resp.json()["tasks"]
    print(f"Fetched {len(tasks)} tasks. Running inference...\n")

    scores_by_difficulty: Dict[str, List[float]] = {"easy": [], "medium": [], "hard": []}
    all_scores: List[float] = []
    results: List[Dict[str, Any]] = []

    for i, task in enumerate(tasks, 1):
        content_id = task["content_id"]
        difficulty = task["difficulty"]

        # LLM reviews the content
        action = call_llm(
            client, task["content_text"], task["content_type"], task["platform"],
        )

        # Grade via the environment's grader endpoint
        grader_payload = {"content_id": content_id, **action}
        grade_resp = requests.post(
            f"{env_url}/grader", json=grader_payload, timeout=30,
        )
        grade_resp.raise_for_status()
        result = grade_resp.json()

        score = result["total_reward"]
        all_scores.append(score)
        scores_by_difficulty[difficulty].append(score)

        status = "✓" if result["your_decision"] == result["gold_decision"] else "✗"
        print(
            f"  [{status}] {i:2d}/{len(tasks)}  {content_id:12s} | "
            f"{difficulty:6s} | pred={result['your_decision']:8s} "
            f"gold={result['gold_decision']:8s} | score={score:.3f}"
        )

        results.append({
            "content_id": content_id,
            "difficulty": difficulty,
            "decision": result["your_decision"],
            "gold_decision": result["gold_decision"],
            "score": score,
        })

    return {
        "results": results,
        "all_scores": all_scores,
        "scores_by_difficulty": scores_by_difficulty,
    }


def print_report(eval_data: Dict[str, Any]) -> None:
    """Print the final evaluation report."""
    all_scores = eval_data["all_scores"]
    scores_by_difficulty = eval_data["scores_by_difficulty"]
    results = eval_data["results"]

    correct = sum(1 for r in results if r["decision"] == r["gold_decision"])
    total = len(results)

    print(f"\n{'=' * 60}")
    print("Brand-Safe Ad Review — LLM Inference Results")
    print(f"{'=' * 60}")
    print(f"  Model:             {MODEL_NAME}")
    print(f"  Items evaluated:   {total}")
    print(f"  Correct decisions: {correct}/{total} ({100 * correct / total:.1f}%)")
    print(f"  Overall mean:      {statistics.mean(all_scores):.4f}")
    print(f"  Overall median:    {statistics.median(all_scores):.4f}")
    if len(all_scores) > 1:
        print(f"  Std deviation:     {statistics.stdev(all_scores):.4f}")

    print()
    for diff in ["easy", "medium", "hard"]:
        scores = scores_by_difficulty.get(diff, [])
        if scores:
            diff_correct = sum(
                1 for r in results
                if r["difficulty"] == diff and r["decision"] == r["gold_decision"]
            )
            print(
                f"  {diff:6s}:  mean={statistics.mean(scores):.4f}  "
                f"correct={diff_correct}/{len(scores)}"
            )

    print(f"\n{'=' * 60}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """Run LLM-based inference against the Brand-Safe Ad Review environment."""
    assert API_KEY, (
        "Missing API key. Set HF_TOKEN or API_KEY environment variable."
    )
    assert MODEL_NAME, (
        "Missing model name. Set MODEL_NAME environment variable."
    )

    print(f"\n{'=' * 60}")
    print("Brand-Safe Ad Review — LLM Inference")
    print(f"{'=' * 60}")
    print(f"  API base:  {API_BASE_URL}")
    print(f"  Model:     {MODEL_NAME}")
    print(f"  Env URL:   {ENV_URL}")
    print(f"{'=' * 60}\n")

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    eval_data = run_evaluation(client, ENV_URL)
    print_report(eval_data)

    return eval_data


if __name__ == "__main__":
    main()
