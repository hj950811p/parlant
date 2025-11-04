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

import hashlib
from typing import Literal

from parlant.adapters.nlp.cerebras_service import Llama3_3_70B
from parlant.core.nlp.emotion import (
    EmotionAnalysis,
    EmotionAnalysisResponse,
    EmotionAnalyzer,
)
from parlant.core.loggers import Logger
from parlant.core.meter import Meter


# Fallback heuristic keywords for emotion detection
POSITIVE_KEYWORDS = {
    "happy",
    "joy",
    "love",
    "wonderful",
    "excellent",
    "great",
    "awesome",
    "fantastic",
    "pleased",
    "satisfied",
    "delighted",
}
NEGATIVE_KEYWORDS = {
    "angry",
    "frustrated",
    "disappointed",
    "terrible",
    "awful",
    "hate",
    "furious",
    "upset",
    "sad",
    "miserable",
    "unacceptable",
}
NEUTRAL_MARKERS = {"need", "want", "check", "help", "question", "information"}


class CerebrasEmotionAnalyzer(EmotionAnalyzer):
    """Emotion analyzer using Cerebras LLM with fallback heuristics."""

    def __init__(self, logger: Logger, meter: Meter) -> None:
        self._logger = logger
        self._meter = meter
        self._generator = Llama3_3_70B[EmotionAnalysisResponse](logger, meter)
        self._cache: dict[str, EmotionAnalysis] = {}

    async def analyze(self, text: str) -> EmotionAnalysis:
        """Analyze emotion in text using Cerebras, with fallback to heuristics.

        Args:
            text: The text to analyze.

        Returns:
            EmotionAnalysis with structured attributes.
        """
        text_hash = self._compute_hash(text)

        if text_hash in self._cache:
            self._logger.trace(f"Emotion analysis cache hit for hash {text_hash}")
            return self._cache[text_hash]

        try:
            result = await self._do_analyze(text)
            self._cache[text_hash] = result
            return result
        except Exception as e:
            self._logger.warning(
                f"Cerebras emotion analysis failed ({type(e).__name__}), using fallback heuristic: {e}"
            )
            result = self._analyze_fallback(text)
            self._cache[text_hash] = result
            return result

    async def _do_analyze(self, text: str) -> EmotionAnalysis:
        """Perform emotion analysis using Cerebras LLM.

        Args:
            text: The text to analyze.

        Returns:
            EmotionAnalysis from LLM response.
        """
        prompt = self._build_emotion_analysis_prompt(text)

        with self._logger.scope("Emotion Analysis via Cerebras"):
            try:
                result = await self._generator.generate(prompt)

                emotion_response = result.content
                analysis = EmotionAnalysis(
                    intensity=emotion_response.intensity,
                    pacing=emotion_response.pacing,
                    focus_keywords=emotion_response.focus_keywords,
                    recommended_scene_duration=emotion_response.recommended_scene_duration,
                )

                self._logger.trace(f"Emotion analysis result: {analysis}")
                return analysis
            except Exception as e:
                self._logger.error(f"Failed to analyze emotion with Cerebras: {e}")
                raise

    def _analyze_fallback(self, text: str) -> EmotionAnalysis:
        """Perform fallback heuristic-based emotion analysis.

        Args:
            text: The text to analyze.

        Returns:
            EmotionAnalysis from heuristic analysis.
        """
        text_lower = text.lower()

        positive_count = sum(1 for kw in POSITIVE_KEYWORDS if kw in text_lower)
        negative_count = sum(1 for kw in NEGATIVE_KEYWORDS if kw in text_lower)
        neutral_count = sum(1 for kw in NEUTRAL_MARKERS if kw in text_lower)

        intensity = self._compute_intensity(positive_count, negative_count, len(text_lower))
        pacing = self._compute_pacing(positive_count, negative_count, neutral_count)
        focus_keywords = self._extract_focus_keywords(text_lower, positive_count, negative_count)

        duration = self._compute_scene_duration(intensity, len(text))

        analysis = EmotionAnalysis(
            intensity=intensity,
            pacing=pacing,
            focus_keywords=focus_keywords,
            recommended_scene_duration=duration,
        )

        self._logger.trace(f"Fallback emotion analysis result: {analysis}")
        return analysis

    def _build_emotion_analysis_prompt(self, text: str) -> str:
        """Build a prompt for emotion analysis.

        Args:
            text: The text to analyze.

        Returns:
            Formatted prompt for LLM.
        """
        return f"""Analyze the emotional content and sentiment of the following text. 
Provide a structured analysis with:
1. intensity (1-10 scale where 1 is neutral and 10 is extremely emotional)
2. pacing (either "fast" for energetic/urgent emotions or "slow" for calm/contemplative)
3. focus_keywords (list of 1-3 words capturing the emotional tone)
4. recommended_scene_duration (recommended duration in milliseconds, typically 3000-6000)

Text to analyze: "{text}"

Return the response as valid JSON matching this schema:
{{
    "intensity": <integer 1-10>,
    "pacing": <"fast" or "slow">,
    "focus_keywords": [<keywords>],
    "recommended_scene_duration": <integer milliseconds>
}}"""

    @staticmethod
    def _compute_intensity(
        positive_count: int, negative_count: int, text_length: int
    ) -> int:
        """Compute intensity from keyword counts.

        Args:
            positive_count: Number of positive keywords found.
            negative_count: Number of negative keywords found.
            text_length: Length of the text.

        Returns:
            Intensity value between 1-10.
        """
        total_emotional_keywords = positive_count + negative_count
        base_intensity = max(1, min(10, 1 + (total_emotional_keywords * 2)))

        length_factor = min(1.5, text_length / 100)
        adjusted_intensity = int(base_intensity * length_factor)

        return max(1, min(10, adjusted_intensity))

    @staticmethod
    def _compute_pacing(positive_count: int, negative_count: int, neutral_count: int) -> Literal["fast", "slow"]:
        """Determine pacing based on emotion distribution.

        Args:
            positive_count: Number of positive keywords.
            negative_count: Number of negative keywords.
            neutral_count: Number of neutral keywords.

        Returns:
            "fast" for high emotional energy, "slow" for calm.
        """
        if positive_count + negative_count > neutral_count:
            return "fast"
        return "slow"

    @staticmethod
    def _extract_focus_keywords(text_lower: str, positive_count: int, negative_count: int) -> list[str]:
        """Extract focus keywords from text.

        Args:
            text_lower: Lowercase text.
            positive_count: Count of positive keywords.
            negative_count: Count of negative keywords.

        Returns:
            List of focus keywords.
        """
        keywords = []

        if positive_count > 0:
            for kw in POSITIVE_KEYWORDS:
                if kw in text_lower:
                    keywords.append(kw)
                    if len(keywords) >= 2:
                        break

        if negative_count > 0 and len(keywords) < 2:
            for kw in NEGATIVE_KEYWORDS:
                if kw in text_lower:
                    keywords.append(kw)
                    if len(keywords) >= 2:
                        break

        if not keywords:
            words = text_lower.split()
            keywords = [w.rstrip(".,!?;:") for w in words if len(w) > 4][:2]

        return keywords[:3]

    @staticmethod
    def _compute_scene_duration(intensity: int, text_length: int) -> int:
        """Compute recommended scene duration based on intensity and text length.

        Args:
            intensity: Emotion intensity 1-10.
            text_length: Length of text.

        Returns:
            Recommended duration in milliseconds.
        """
        base_duration = 3000
        intensity_factor = 1.0 + (intensity - 1) * 0.25
        length_factor = min(1.5, text_length / 100)

        duration = int(base_duration * intensity_factor * length_factor)
        return max(3000, min(6000, duration))

    @staticmethod
    def _compute_hash(text: str) -> str:
        """Compute a hash of the text for caching.

        Args:
            text: The text to hash.

        Returns:
            Hex hash string.
        """
        return hashlib.md5(text.encode()).hexdigest()
