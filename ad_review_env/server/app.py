"""FastAPI application for the Brand-Safe Ad Review environment."""

import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel, Field

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError("openenv is required. Install with: uv sync") from e

try:
    from ..models import AdReviewAction, AdReviewObservation
    from ..data import CONTENT_ITEMS, CONTENT_INDEX
    from ..grader import grade
    from ..agent import smart_agent, evaluate_agent
    from .environment import AdReviewEnvironment
except ImportError:
    from models import AdReviewAction, AdReviewObservation
    from data import CONTENT_ITEMS, CONTENT_INDEX
    from grader import grade
    from agent import smart_agent, evaluate_agent
    from server.environment import AdReviewEnvironment

app = create_app(
    AdReviewEnvironment, AdReviewAction, AdReviewObservation,
    env_name="ad_review_env", max_concurrent_envs=4,
)


@app.get("/", tags=["Root"])
def root():
    return RedirectResponse(url="/web")


def _filter_by_difficulty(pool: List[Dict[str, Any]], difficulty: Optional[str]) -> List[Dict[str, Any]]:
    if not difficulty:
        return pool
    if difficulty not in ("easy", "medium", "hard"):
        raise HTTPException(status_code=400, detail="difficulty must be easy, medium, or hard")
    return [item for item in pool if item["difficulty"] == difficulty]


def _lookup_content(content_id: str) -> Dict[str, Any]:
    item = CONTENT_INDEX.get(content_id)
    if item is None:
        raise HTTPException(status_code=404, detail=f"content_id '{content_id}' not found")
    return item


def _strip_gold_labels(item: Dict[str, Any]) -> Dict[str, Any]:
    return {k: item[k] for k in ("content_id", "content_text", "content_type", "platform", "difficulty")}


_BASELINE_RULES = [
    (["kill", "murder", "rape", "terrorist", "bomb"], "REJECT", "IAB_VIOLENCE", "GARM_DEATH_INJURY", "CRITICAL", "MATURE", 0.85),
    (["child porn", "pedophil", "csam"], "REJECT", "IAB_ADULT", "GARM_ADULT_EXPLICIT", "CRITICAL", "ADULT", 0.95),
    (["proud boys", "great replacement", "white power", "supremacist", "deportthem", "slur"],
     "REJECT", "IAB_HATE_SPEECH", "GARM_HATE_SPEECH", "HIGH", "ADULT", 0.85),
    (["18+", "adult content", "explicit", "nude", "xxx", "🔞", "uncensored"],
     "REJECT", "IAB_ADULT", "GARM_ADULT_EXPLICIT", "CRITICAL", "ADULT", 0.88),
    (["f***", "fuck", "shit", "bitch"],
     "REJECT", "IAB_PROFANITY", "GARM_OBSCENITY_PROFANITY", "HIGH", "MATURE", 0.78),
    (["graphic violence", "knockouts", "brutal", "street fight"],
     "REJECT", "IAB_VIOLENCE", "GARM_DEATH_INJURY", "HIGH", "MATURE", 0.82),
    (["xanax", "no prescription", "pirat", "credit card number", "discreet shipping"],
     "REJECT", "IAB_ILLEGAL", "GARM_CRIME_HARMFUL", "HIGH", "ADULT", 0.85),
    (["cancer cure", "doctors don't want", "wake up sheeple", "one weird trick", "instead of chemo"],
     "REJECT", "IAB_MISINFORMATION", "GARM_SPAM_HARMFUL", "MEDIUM", "TEEN", 0.78),
    (["autism", "vaccine", "mmr"],
     "REJECT", "IAB_MISINFORMATION", "GARM_SPAM_HARMFUL", "HIGH", "TEEN", 0.75),
    (["join my team", "boss babe", "$10k/month", "earn from your phone"],
     "REJECT", "IAB_MISINFORMATION", "GARM_SPAM_HARMFUL", "HIGH", "TEEN", 0.80),
    (["psilocybin", "microdosing", "cbd"],
     "ESCALATE", "IAB_DRUGS", "GARM_SAFE", "MEDIUM", "TEEN", 0.60),
    (["gambling", "jackpot", "casino", "free spins", "slot"],
     "ESCALATE", "IAB_GAMBLING", "GARM_SAFE", "MEDIUM", "TEEN", 0.62),
    (["gun control", "abortion", "leaked", "crypto", "government lying", "politician", "democracy"],
     "ESCALATE", "IAB_CONTROVERSIAL", "GARM_SAFE", "MEDIUM", "TEEN", 0.58),
    (["israeli", "palestinian", "israel", "palestine"],
     "ESCALATE", "IAB_CONTROVERSIAL", "GARM_SAFE", "MEDIUM", "TEEN", 0.55),
]


def baseline_agent(text: str, content_type: str = "", platform: str = "") -> Dict[str, Any]:
    text_lower = text.lower()
    all_kws = []
    for kws in _BASELINE_RULES:
        all_kws.extend(kws[0])
    flagged = [kw for kw in all_kws if kw in text_lower][:5]

    for keywords, decision, iab, garm, risk, age, confidence in _BASELINE_RULES:
        if any(kw in text_lower for kw in keywords):
            reason_action = "rejected" if decision == "REJECT" else "escalated"
            reasoning = (
                f"Keyword baseline {reason_action} this {content_type or 'content'}. "
                f"Flagged elements: {', '.join(flagged[:3]) if flagged else 'none'}. "
                f"Category: {iab} / {garm}. Risk assessment: {risk}. "
                f"Platform: {platform or 'unknown'}. Age rating: {age}. "
                f"This content violates brand safety guidelines."
            )
            return {
                "decision": decision, "iab_category": iab, "garm_category": garm,
                "risk_level": risk, "age_rating": age,
                "reasoning": reasoning[:500],
                "confidence": confidence, "flagged_elements": flagged,
            }

    reasoning = (
        f"Keyword baseline approved this {content_type or 'content'}. "
        f"No harmful signals detected in keyword scan. "
        f"Platform: {platform or 'unknown'}. Content appears brand-safe. "
        f"No profanity, violence, hate speech, or misinformation detected."
    )
    return {
        "decision": "APPROVE", "iab_category": "IAB_SAFE", "garm_category": "GARM_SAFE",
        "risk_level": "LOW", "age_rating": "ALL_AGES",
        "reasoning": reasoning,
        "confidence": 0.7, "flagged_elements": [],
    }


@app.get("/tasks", tags=["Hackathon"])
def get_tasks(n: int = 5, difficulty: Optional[str] = None, seed: Optional[int] = None) -> Dict[str, Any]:
    if not 1 <= n <= 50:
        raise HTTPException(status_code=400, detail="n must be between 1 and 50")
    if seed is not None and not (0 <= seed <= 2**31 - 1):
        raise HTTPException(status_code=400, detail="seed must be between 0 and 2147483647")
    pool = _filter_by_difficulty(CONTENT_ITEMS, difficulty)
    sample = random.Random(seed).sample(pool, min(n, len(pool)))
    return {"tasks": [_strip_gold_labels(item) for item in sample], "count": len(sample), "total_available": len(pool)}


class GraderRequest(BaseModel):
    content_id: str = Field(...)
    decision: str = Field(...)
    iab_category: str = Field(...)
    garm_category: str = Field(...)
    risk_level: str = Field(...)
    age_rating: str = Field(default="TEEN")
    reasoning: str = Field(..., min_length=10, max_length=500)
    confidence: float = Field(..., ge=0.0, le=1.0)
    flagged_elements: List[str] = Field(default_factory=list)
    steps_taken: int = Field(default=1, ge=1, le=3)


@app.post("/grader", tags=["Hackathon"])
def grader_endpoint(request: GraderRequest) -> Dict[str, Any]:
    gold = _lookup_content(request.content_id)
    action_data = request.model_dump(exclude={"content_id", "steps_taken"})
    total_reward, scores, feedback = grade(action_data, gold, steps_taken=request.steps_taken)
    return {
        "content_id": request.content_id, "difficulty": gold["difficulty"],
        "your_decision": request.decision, "gold_decision": gold["gold_decision"],
        "scores": scores, "total_reward": total_reward, "feedback": feedback,
        "steps_taken": request.steps_taken,
    }


@app.get("/baseline", tags=["Hackathon"])
def baseline_demo(content_id: Optional[str] = None, seed: Optional[int] = None) -> Dict[str, Any]:
    item = _lookup_content(content_id) if content_id else random.Random(seed).choice(CONTENT_ITEMS)
    action = baseline_agent(item["content_text"])
    total_reward, scores, feedback = grade(action, item, steps_taken=1)
    return {
        "content_id": item["content_id"], "content_text": item["content_text"],
        "difficulty": item["difficulty"], "baseline_action": action,
        "scores": scores, "total_reward": total_reward, "feedback": feedback,
    }


_AGENTS = {"smart": smart_agent, "baseline": baseline_agent}


class AnalyzeRequest(BaseModel):
    content_text: str = Field(..., min_length=1, max_length=5000)
    content_type: str = Field(default="post", pattern=r"^(post|comment|caption|bio|reel|story|thread)$")
    platform: str = Field(default="social_media", pattern=r"^(instagram|tiktok|youtube|x|facebook|reddit|threads|linkedin|social_media)$")


@app.post("/analyze", tags=["Hackathon"])
def analyze_content(request: AnalyzeRequest) -> Dict[str, Any]:
    result = smart_agent(request.content_text, request.content_type, request.platform)
    return {"content_text": request.content_text, "analysis": result}


@app.get("/evaluate", tags=["Hackathon"])
def evaluate_endpoint(agent: str = "smart", difficulty: Optional[str] = None) -> Dict[str, Any]:
    if agent not in _AGENTS:
        raise HTTPException(status_code=400, detail=f"agent must be one of {list(_AGENTS.keys())}")
    pool = _filter_by_difficulty(CONTENT_ITEMS, difficulty)
    result = evaluate_agent(pool, grade, agent_fn=_AGENTS[agent])
    result["agent"] = agent
    return result


_DASHBOARD_PATH = Path(__file__).parent / "dashboard.html"


@app.get("/web", tags=["Dashboard"], response_class=HTMLResponse)
def web_dashboard():
    return _DASHBOARD_PATH.read_text(encoding="utf-8")


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
