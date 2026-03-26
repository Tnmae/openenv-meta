# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Brand-Safe Ad Review Environment.

Mirrors exact ConfigDict, fields, and validators from openenv.core.env_server.types.
"""

from typing import Any, Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field, field_validator


# IAB Content Taxonomy 3.0 top-level categories (subset relevant to brand safety)
IAB_CATEGORIES = [
    "IAB_SAFE",           # Clearly brand-safe content
    "IAB_ADULT",          # Adult/explicit content
    "IAB_VIOLENCE",       # Graphic violence
    "IAB_HATE_SPEECH",    # Hate speech / discrimination
    "IAB_ILLEGAL",        # Illegal activity promotion
    "IAB_MISINFORMATION", # Misinformation / fake news
    "IAB_PROFANITY",      # Profanity / offensive language
    "IAB_DRUGS",          # Drug / substance promotion
    "IAB_GAMBLING",       # Gambling content
    "IAB_CONTROVERSIAL",  # Controversial / politically divisive
]

# GARM Brand Safety Floor categories
GARM_CATEGORIES = [
    "GARM_SAFE",
    "GARM_ADULT_EXPLICIT",
    "GARM_ARMS_AMMUNITION",
    "GARM_CRIME_HARMFUL",
    "GARM_DEATH_INJURY",
    "GARM_HATE_SPEECH",
    "GARM_OBSCENITY_PROFANITY",
    "GARM_ONLINE_PIRACY",
    "GARM_SPAM_HARMFUL",
    "GARM_TERRORISM",
]

VALID_DECISIONS = Literal["APPROVE", "REJECT", "ESCALATE"]
VALID_RISK_LEVELS = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]


class AdReviewAction(Action):
    """
    Action for the Brand-Safe Ad Review environment.

    The agent reviews a UGC content item and decides whether it is
    brand-safe for ad placement.
    """

    # model_config inherited from Action (extra="forbid", validate_assignment=True)

    decision: VALID_DECISIONS = Field(
        ...,
        description="Review decision: APPROVE (brand-safe), REJECT (unsafe), or ESCALATE (human review needed)",
    )
    iab_category: str = Field(
        ...,
        description=f"IAB Content Taxonomy category. One of: {IAB_CATEGORIES}",
    )
    garm_category: str = Field(
        ...,
        description=f"GARM Brand Safety Floor category. One of: {GARM_CATEGORIES}",
    )
    risk_level: VALID_RISK_LEVELS = Field(
        ...,
        description="Assessed risk level for brand safety: LOW, MEDIUM, HIGH, or CRITICAL",
    )
    reasoning: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description="Brief explanation of the review decision (10-500 chars)",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score for the decision [0.0, 1.0]",
    )
    flagged_elements: List[str] = Field(
        default_factory=list,
        description="Specific content elements that triggered the review decision",
    )

    @field_validator("iab_category")
    @classmethod
    def validate_iab_category(cls, v: str) -> str:
        if v not in IAB_CATEGORIES:
            raise ValueError(f"iab_category must be one of {IAB_CATEGORIES}, got '{v}'")
        return v

    @field_validator("garm_category")
    @classmethod
    def validate_garm_category(cls, v: str) -> str:
        if v not in GARM_CATEGORIES:
            raise ValueError(f"garm_category must be one of {GARM_CATEGORIES}, got '{v}'")
        return v


class AdReviewObservation(Observation):
    """
    Observation from the Brand-Safe Ad Review environment.

    Contains the UGC content item to review plus scoring feedback.
    """

    # model_config inherited from Observation (extra="forbid", validate_assignment=True)

    # Content to review
    content_id: str = Field(description="Unique identifier for the content item")
    content_text: str = Field(description="The UGC text content to review")
    content_type: str = Field(description="Type of content: post, comment, caption, bio")
    platform: str = Field(description="Platform context: instagram, tiktok, youtube, twitter")
    difficulty: str = Field(description="Task difficulty: easy, medium, hard")

    # Scoring feedback (populated after step)
    score_decision: float = Field(default=0.0, description="Score for decision correctness [0, 0.4]")
    score_category: float = Field(default=0.0, description="Score for category accuracy [0, 0.3]")
    score_reasoning: float = Field(default=0.0, description="Score for reasoning quality [0, 0.2]")
    score_efficiency: float = Field(default=0.0, description="Score for confidence calibration [0, 0.1]")
    total_score: float = Field(default=0.0, description="Total reward score [0.0, 1.0]")

    # Feedback
    feedback: str = Field(default="", description="Grader feedback on the decision")
    gold_decision: Optional[str] = Field(default=None, description="Gold-label decision (revealed after step)")
    gold_iab_category: Optional[str] = Field(default=None, description="Gold-label IAB category")
    gold_garm_category: Optional[str] = Field(default=None, description="Gold-label GARM category")
