# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Brand-Safe Ad Review Environment.

Auto-generated endpoints (via create_app):
    POST /reset       — start a new review episode
    POST /step        — submit a review decision
    GET  /state       — current episode state
    GET  /schema      — action/observation JSON schemas
    GET  /health      — health check
    WS   /ws          — WebSocket for persistent sessions

Hackathon endpoints (bolted on):
    GET  /tasks       — sample tasks for evaluation
    POST /grader      — standalone grader endpoint
    GET  /baseline    — baseline agent demonstration
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required. Install with: uv sync"
    ) from e

try:
    from ..models import AdReviewAction, AdReviewObservation
    from ..data import CONTENT_ITEMS, CONTENT_INDEX
    from ..grader import grade
    from .environment import AdReviewEnvironment
except ImportError:
    from models import AdReviewAction, AdReviewObservation
    from data import CONTENT_ITEMS, CONTENT_INDEX
    from grader import grade
    from server.environment import AdReviewEnvironment

import random
from typing import Any, Dict, List, Optional

from fastapi import HTTPException
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Core app — auto-generates /reset /step /state /schema /health /ws /docs
# ---------------------------------------------------------------------------
app = create_app(
    AdReviewEnvironment,
    AdReviewAction,
    AdReviewObservation,
    env_name="ad_review_env",
    max_concurrent_envs=4,
)


# ---------------------------------------------------------------------------
# Hackathon endpoint 1: GET /tasks
# Returns a sample of content items for evaluation
# ---------------------------------------------------------------------------
@app.get("/tasks", tags=["Hackathon"])
def get_tasks(
    n: int = 5,
    difficulty: Optional[str] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Return a sample of UGC content items for agent evaluation.

    Args:
        n: Number of tasks to return (1-30, default 5)
        difficulty: Filter by 'easy', 'medium', or 'hard' (optional)
        seed: Random seed for reproducible sampling (optional)
    """
    if not 1 <= n <= 30:
        raise HTTPException(status_code=400, detail="n must be between 1 and 30")

    pool = CONTENT_ITEMS
    if difficulty:
        if difficulty not in ("easy", "medium", "hard"):
            raise HTTPException(status_code=400, detail="difficulty must be easy, medium, or hard")
        pool = [item for item in pool if item["difficulty"] == difficulty]

    rng = random.Random(seed)
    sample = rng.sample(pool, min(n, len(pool)))

    # Strip gold labels from task presentation
    tasks = [
        {
            "content_id": item["content_id"],
            "content_text": item["content_text"],
            "content_type": item["content_type"],
            "platform": item["platform"],
            "difficulty": item["difficulty"],
        }
        for item in sample
    ]

    return {
        "tasks": tasks,
        "count": len(tasks),
        "total_available": len(pool),
    }


# ---------------------------------------------------------------------------
# Hackathon endpoint 2: POST /grader
# Standalone grader — accepts action + content_id, returns scored result
# ---------------------------------------------------------------------------
class GraderRequest(BaseModel):
    content_id: str = Field(..., description="Content ID from /tasks")
    decision: str = Field(..., description="APPROVE, REJECT, or ESCALATE")
    iab_category: str = Field(..., description="IAB Content Taxonomy category")
    garm_category: str = Field(..., description="GARM Brand Safety Floor category")
    risk_level: str = Field(..., description="LOW, MEDIUM, HIGH, or CRITICAL")
    reasoning: str = Field(..., description="Explanation of the decision")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score [0,1]")
    flagged_elements: List[str] = Field(default_factory=list)


@app.post("/grader", tags=["Hackathon"])
def grader_endpoint(request: GraderRequest) -> Dict[str, Any]:
    """
    Grade a review decision against gold labels.

    Submit your decision for a content_id and receive a detailed score breakdown.
    """
    gold = CONTENT_INDEX.get(request.content_id)
    if gold is None:
        raise HTTPException(
            status_code=404,
            detail=f"content_id '{request.content_id}' not found. Use GET /tasks to get valid IDs.",
        )

    action_data = {
        "decision": request.decision,
        "iab_category": request.iab_category,
        "garm_category": request.garm_category,
        "risk_level": request.risk_level,
        "reasoning": request.reasoning,
        "confidence": request.confidence,
        "flagged_elements": request.flagged_elements,
    }

    total_reward, scores, feedback = grade(action_data, gold)

    return {
        "content_id": request.content_id,
        "difficulty": gold["difficulty"],
        "your_decision": request.decision,
        "gold_decision": gold["gold_decision"],
        "scores": scores,
        "total_reward": total_reward,
        "feedback": feedback,
    }


# ---------------------------------------------------------------------------
# Hackathon endpoint 3: GET /baseline
# Demonstrates a simple rule-based baseline agent
# ---------------------------------------------------------------------------
@app.get("/baseline", tags=["Hackathon"])
def baseline_demo(content_id: Optional[str] = None, seed: Optional[int] = 42) -> Dict[str, Any]:
    """
    Run a simple keyword-based baseline agent on a content item.

    Demonstrates the environment API and provides a performance benchmark.
    If content_id is not provided, a random item is selected.
    """
    if content_id:
        item = CONTENT_INDEX.get(content_id)
        if item is None:
            raise HTTPException(status_code=404, detail=f"content_id '{content_id}' not found")
    else:
        rng = random.Random(seed)
        item = rng.choice(CONTENT_ITEMS)

    # Simple keyword-based baseline
    action = _baseline_agent(item["content_text"])

    action_data = {
        "decision": action["decision"],
        "iab_category": action["iab_category"],
        "garm_category": action["garm_category"],
        "risk_level": action["risk_level"],
        "reasoning": action["reasoning"],
        "confidence": action["confidence"],
        "flagged_elements": action["flagged_elements"],
    }

    total_reward, scores, feedback = grade(action_data, item)

    return {
        "content_id": item["content_id"],
        "content_text": item["content_text"],
        "difficulty": item["difficulty"],
        "baseline_action": action,
        "gold_decision": item["gold_decision"],
        "scores": scores,
        "total_reward": total_reward,
        "feedback": feedback,
        "note": "Keyword-based baseline. A well-tuned LLM agent should significantly outperform this.",
    }


def _baseline_agent(text: str) -> Dict[str, Any]:
    """
    Simple keyword-based baseline agent.
    Scans for red-flag keywords and makes a heuristic decision.
    """
    text_lower = text.lower()

    # Red-flag keyword sets
    critical_keywords = ["kill", "murder", "terrorist", "bomb", "rape", "child porn", "cp ", "slur"]
    reject_keywords = [
        "f***", "fuck", "shit", "bitch", "hate", "illegal", "pirat", "xanax",
        "no prescription", "18+", "adult content", "explicit", "graphic violence",
        "knockouts", "brutal", "proud boys", "cancer cure", "doctors don't want"
    ]
    escalate_keywords = [
        "psilocybin", "microdosing", "crypto", "gambling", "jackpot", "controversial",
        "leaked", "government lying", "wake up sheeple", "gun control", "abortion"
    ]

    flagged = []

    for kw in critical_keywords:
        if kw in text_lower:
            flagged.append(kw)

    for kw in reject_keywords:
        if kw in text_lower:
            flagged.append(kw)

    for kw in escalate_keywords:
        if kw in text_lower:
            flagged.append(kw)

    if any(kw in text_lower for kw in critical_keywords):
        decision = "REJECT"
        iab = "IAB_HATE_SPEECH"
        garm = "GARM_HATE_SPEECH"
        risk = "CRITICAL"
        confidence = 0.9
    elif any(kw in text_lower for kw in reject_keywords):
        decision = "REJECT"
        iab = "IAB_PROFANITY"
        garm = "GARM_OBSCENITY_PROFANITY"
        risk = "HIGH"
        confidence = 0.75
    elif any(kw in text_lower for kw in escalate_keywords):
        decision = "ESCALATE"
        iab = "IAB_CONTROVERSIAL"
        garm = "GARM_SAFE"
        risk = "MEDIUM"
        confidence = 0.6
    else:
        decision = "APPROVE"
        iab = "IAB_SAFE"
        garm = "GARM_SAFE"
        risk = "LOW"
        confidence = 0.8

    reasoning = (
        f"Keyword scan: {'flagged ' + str(flagged[:3]) if flagged else 'no red-flag keywords detected'}. "
        f"Decision based on heuristic rules."
    )

    return {
        "decision": decision,
        "iab_category": iab,
        "garm_category": garm,
        "risk_level": risk,
        "reasoning": reasoning,
        "confidence": confidence,
        "flagged_elements": flagged[:5],
    }


def main(host: str = "0.0.0.0", port: int = 8000):
    """Entry point for uv run or python -m."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
