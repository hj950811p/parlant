# Copyright 2025 Emcie Co Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Sequence

from parlant.core.common import DefaultBaseModel


@dataclass(frozen=True)
class EmotionAnalysis:
    """Structured emotion analysis result from semantic analysis."""

    intensity: int
    """Emotion intensity on a scale of 1-10."""

    pacing: Literal["fast", "slow"]
    """Recommended pacing for delivery based on emotional tone."""

    focus_keywords: Sequence[str]
    """Key words or phrases that capture the emotional tone."""

    recommended_scene_duration: int
    """Recommended scene duration in milliseconds."""


class EmotionAnalysisResponse(DefaultBaseModel):
    """Schema for emotion analysis response from the LLM."""

    intensity: int
    pacing: Literal["fast", "slow"]
    focus_keywords: list[str]
    recommended_scene_duration: int


class EmotionAnalyzer(ABC):
    """Interface for emotion analysis services."""

    @abstractmethod
    async def analyze(self, text: str) -> EmotionAnalysis:
        """Analyze the emotion in the given text.

        Args:
            text: The text to analyze for emotional content.

        Returns:
            EmotionAnalysis with structured sentiment attributes.

        Raises:
            Exception: If analysis fails and fallback is also unavailable.
        """
        ...
