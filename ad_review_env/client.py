# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Brand-Safe Ad Review Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import AdReviewAction, AdReviewObservation


class AdReviewEnv(EnvClient[AdReviewAction, AdReviewObservation, State]):
    """
    Client for the Brand-Safe Ad Review Environment.

    Maintains a persistent WebSocket connection to the environment server.

    Example:
        >>> with AdReviewEnv(base_url="http://localhost:8000") as env:
        ...     result = env.reset()
        ...     print(result.observation.content_text)
        ...
        ...     action = AdReviewAction(
        ...         decision="APPROVE",
        ...         iab_category="IAB_SAFE",
        ...         garm_category="GARM_SAFE",
        ...         risk_level="LOW",
        ...         reasoning="Content is a standard lifestyle post with no harmful elements.",
        ...         confidence=0.95,
        ...     )
        ...     result = env.step(action)
        ...     print(f"Score: {result.observation.total_score:.3f}")
        ...     print(result.observation.feedback)
    """

    def _step_payload(self, action: AdReviewAction) -> Dict:
        return {
            "decision": action.decision,
            "iab_category": action.iab_category,
            "garm_category": action.garm_category,
            "risk_level": action.risk_level,
            "reasoning": action.reasoning,
            "confidence": action.confidence,
            "flagged_elements": action.flagged_elements,
            "metadata": action.metadata,
        }

    def _parse_result(self, payload: Dict) -> StepResult[AdReviewObservation]:
        obs_data = payload.get("observation", {})
        observation = AdReviewObservation(
            content_id=obs_data.get("content_id", ""),
            content_text=obs_data.get("content_text", ""),
            content_type=obs_data.get("content_type", ""),
            platform=obs_data.get("platform", ""),
            difficulty=obs_data.get("difficulty", ""),
            score_decision=obs_data.get("score_decision", 0.0),
            score_category=obs_data.get("score_category", 0.0),
            score_reasoning=obs_data.get("score_reasoning", 0.0),
            score_efficiency=obs_data.get("score_efficiency", 0.0),
            total_score=obs_data.get("total_score", 0.0),
            feedback=obs_data.get("feedback", ""),
            gold_decision=obs_data.get("gold_decision"),
            gold_iab_category=obs_data.get("gold_iab_category"),
            gold_garm_category=obs_data.get("gold_garm_category"),
            done=payload.get("done", True),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", True),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
