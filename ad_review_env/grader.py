# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Deterministic grader for the Brand-Safe Ad Review environment.

Reward formula (max 1.0):
  0.4 × decision_score   — correct APPROVE/REJECT/ESCALATE
  0.3 × category_score   — correct IAB + GARM category
  0.2 × reasoning_score  — reasoning quality (length + flagged elements)
  0.1 × efficiency_score — confidence calibration × step-efficiency multiplier

Step-efficiency multiplier (multi-step episodes):
  1 step  → 1.0×  (decided immediately — full efficiency credit)
  2 steps → 0.7×  (requested context once — slight penalty)
  3 steps → 0.4×  (used all context — larger penalty)
"""

from typing import Any, Dict, Tuple


# Step-efficiency multipliers: fewer steps = higher efficiency
_STEP_EFFICIENCY = {1: 1.0, 2: 0.7, 3: 0.4}


def grade(
    action_data: Dict[str, Any],
    gold: Dict[str, Any],
    steps_taken: int = 1,
) -> Tuple[float, Dict[str, float], str]:
    """
    Grade an AdReviewAction against gold labels.

    Args:
        action_data: dict with keys: decision, iab_category, garm_category,
                     risk_level, reasoning, confidence, flagged_elements
        gold: gold-label dict from CONTENT_ITEMS
        steps_taken: number of steps the agent took (1 = immediate, 2+ = used context)

    Returns:
        (total_reward, component_scores, feedback_message)
    """
    decision = action_data.get("decision", "")
    iab_category = action_data.get("iab_category", "")
    garm_category = action_data.get("garm_category", "")
    reasoning = action_data.get("reasoning", "")
    confidence = float(action_data.get("confidence", 0.5))
    flagged_elements = action_data.get("flagged_elements", [])

    gold_decision = gold["gold_decision"]
    gold_iab = gold["gold_iab_category"]
    gold_garm = gold["gold_garm_category"]
    difficulty = gold["difficulty"]

    # -------------------------------------------------------------------------
    # 1. Decision score (0.0 – 1.0, weight 0.4)
    # -------------------------------------------------------------------------
    if decision == gold_decision:
        decision_score = 1.0
    elif _is_adjacent_decision(decision, gold_decision):
        # Partial credit: ESCALATE when gold is REJECT (or vice versa) is less wrong
        decision_score = 0.4
    else:
        decision_score = 0.0

    # -------------------------------------------------------------------------
    # 2. Category score (0.0 – 1.0, weight 0.3)
    #    0.5 for IAB correct + 0.5 for GARM correct
    # -------------------------------------------------------------------------
    iab_score = 1.0 if iab_category == gold_iab else 0.0
    garm_score = 1.0 if garm_category == gold_garm else 0.0
    category_score = 0.5 * iab_score + 0.5 * garm_score

    # -------------------------------------------------------------------------
    # 3. Reasoning score (0.0 – 1.0, weight 0.2)
    #    Based on reasoning length and presence of flagged elements
    # -------------------------------------------------------------------------
    reasoning_len = len(reasoning.strip())
    if reasoning_len >= 100:
        length_score = 1.0
    elif reasoning_len >= 50:
        length_score = 0.6
    elif reasoning_len >= 10:
        length_score = 0.3
    else:
        length_score = 0.0

    # Bonus for flagging specific elements when content is unsafe
    if gold_decision in ("REJECT", "ESCALATE") and len(flagged_elements) > 0:
        flagging_bonus = min(0.3, len(flagged_elements) * 0.1)
    else:
        flagging_bonus = 0.0

    reasoning_score = min(1.0, length_score + flagging_bonus)

    # -------------------------------------------------------------------------
    # 4. Efficiency / calibration score (0.0 – 1.0, weight 0.1)
    #    Reward well-calibrated confidence:
    #    - Correct + high confidence → full score
    #    - Correct + low confidence → partial score
    #    - Wrong + high confidence → penalized (overconfidence)
    #    - Wrong + low confidence → partial credit (epistemic humility)
    #    Then multiply by step-efficiency (fewer steps = better)
    # -------------------------------------------------------------------------
    correct = decision == gold_decision
    if correct:
        raw_efficiency = confidence  # reward certainty when right
    else:
        raw_efficiency = 1.0 - confidence  # reward uncertainty when wrong

    step_mult = _STEP_EFFICIENCY.get(min(steps_taken, 3), 0.4)
    efficiency_score = raw_efficiency * step_mult

    # -------------------------------------------------------------------------
    # Difficulty multiplier — hard tasks worth slightly more
    # -------------------------------------------------------------------------
    difficulty_multiplier = {"easy": 1.0, "medium": 1.05, "hard": 1.1}.get(difficulty, 1.0)

    # -------------------------------------------------------------------------
    # Weighted total
    # -------------------------------------------------------------------------
    raw_total = (
        0.4 * decision_score
        + 0.3 * category_score
        + 0.2 * reasoning_score
        + 0.1 * efficiency_score
    )
    total = min(1.0, raw_total * difficulty_multiplier)

    component_scores = {
        "decision": round(0.4 * decision_score, 4),
        "category": round(0.3 * category_score, 4),
        "reasoning": round(0.2 * reasoning_score, 4),
        "efficiency": round(0.1 * efficiency_score, 4),
        "total": round(total, 4),
    }

    feedback = _build_feedback(
        decision, gold_decision, iab_category, gold_iab,
        garm_category, gold_garm, component_scores, difficulty, steps_taken
    )

    return total, component_scores, feedback


def _is_adjacent_decision(predicted: str, gold: str) -> bool:
    """ESCALATE and REJECT are adjacent — both indicate unsafe content."""
    adjacent_pairs = {("ESCALATE", "REJECT"), ("REJECT", "ESCALATE")}
    return (predicted, gold) in adjacent_pairs


def _build_feedback(
    decision: str, gold_decision: str,
    iab: str, gold_iab: str,
    garm: str, gold_garm: str,
    scores: Dict[str, float],
    difficulty: str,
    steps_taken: int = 1,
) -> str:
    parts = []

    if decision == gold_decision:
        parts.append(f"✓ Decision '{decision}' is correct.")
    elif _is_adjacent_decision(decision, gold_decision):
        parts.append(f"~ Decision '{decision}' is adjacent to gold '{gold_decision}' (partial credit).")
    else:
        parts.append(f"✗ Decision '{decision}' is wrong. Gold label: '{gold_decision}'.")

    if iab != gold_iab:
        parts.append(f"✗ IAB category '{iab}' incorrect. Expected: '{gold_iab}'.")
    else:
        parts.append(f"✓ IAB category correct.")

    if garm != gold_garm:
        parts.append(f"✗ GARM category '{garm}' incorrect. Expected: '{gold_garm}'.")
    else:
        parts.append(f"✓ GARM category correct.")

    step_note = f" ({steps_taken} step{'s' if steps_taken != 1 else ''})"
    parts.append(
        f"Scores → decision:{scores['decision']:.2f} "
        f"category:{scores['category']:.2f} "
        f"reasoning:{scores['reasoning']:.2f} "
        f"efficiency:{scores['efficiency']:.2f} "
        f"| total:{scores['total']:.3f} [{difficulty}]{step_note}"
    )

    return " ".join(parts)
