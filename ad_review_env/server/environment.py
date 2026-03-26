# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Core environment logic for the Brand-Safe Ad Review environment.

Single-step episodes: step() always returns done=True.
"""

import random
from typing import Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import AdReviewAction, AdReviewObservation
    from ..data import CONTENT_ITEMS, CONTENT_INDEX
    from ..grader import grade
except ImportError:
    from models import AdReviewAction, AdReviewObservation
    from data import CONTENT_ITEMS, CONTENT_INDEX
    from grader import grade


class AdReviewEnvironment(Environment):
    """
    Brand-Safe Ad Review environment for UGC content moderation.

    Each episode presents one UGC content item. The agent must:
    1. Decide: APPROVE / REJECT / ESCALATE
    2. Classify: IAB Content Taxonomy + GARM Brand Safety Floor category
    3. Assess: risk level and confidence
    4. Explain: reasoning with specific flagged elements

    Episodes are single-step (done=True after every step).

    Reward formula (max 1.0):
        0.4 × decision accuracy
        0.3 × category accuracy (IAB + GARM)
        0.2 × reasoning quality
        0.1 × confidence calibration
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        super().__init__()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._current_item: Optional[dict] = None
        self._rng = random.Random()

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs,
    ) -> AdReviewObservation:
        """Reset: sample a new content item to review."""
        if seed is not None:
            self._rng.seed(seed)

        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )

        self._current_item = self._rng.choice(CONTENT_ITEMS)

        return AdReviewObservation(
            content_id=self._current_item["content_id"],
            content_text=self._current_item["content_text"],
            content_type=self._current_item["content_type"],
            platform=self._current_item["platform"],
            difficulty=self._current_item["difficulty"],
            done=False,
            reward=None,
        )

    def step(
        self,
        action: AdReviewAction,
        timeout_s: Optional[float] = None,
        **kwargs,
    ) -> AdReviewObservation:
        """
        Execute one review step.

        Always returns done=True (single-step episodes).
        """
        if self._current_item is None:
            # Auto-reset if step called without reset
            self.reset()

        self._state.step_count += 1

        action_data = {
            "decision": action.decision,
            "iab_category": action.iab_category,
            "garm_category": action.garm_category,
            "risk_level": action.risk_level,
            "reasoning": action.reasoning,
            "confidence": action.confidence,
            "flagged_elements": action.flagged_elements,
        }

        total_reward, scores, feedback = grade(action_data, self._current_item)

        obs = AdReviewObservation(
            content_id=self._current_item["content_id"],
            content_text=self._current_item["content_text"],
            content_type=self._current_item["content_type"],
            platform=self._current_item["platform"],
            difficulty=self._current_item["difficulty"],
            # Scoring
            score_decision=scores["decision"],
            score_category=scores["category"],
            score_reasoning=scores["reasoning"],
            score_efficiency=scores["efficiency"],
            total_score=scores["total"],
            # Feedback
            feedback=feedback,
            gold_decision=self._current_item["gold_decision"],
            gold_iab_category=self._current_item["gold_iab_category"],
            gold_garm_category=self._current_item["gold_garm_category"],
            # Always terminal
            done=True,
            reward=total_reward,
        )

        return obs

    @property
    def state(self) -> State:
        return self._state

    def get_metadata(self):
        from openenv.core.env_server.types import EnvironmentMetadata
        return EnvironmentMetadata(
            name="ad_review_env",
            description=(
                "Brand-Safe Ad Review environment for UGC content moderation. "
                "Agents review user-generated content and decide APPROVE/REJECT/ESCALATE "
                "using IAB Content Taxonomy 3.0 and GARM Brand Safety Floor standards."
            ),
            version="1.0.0",
            author="Meta OpenEnv Hackathon",
        )
