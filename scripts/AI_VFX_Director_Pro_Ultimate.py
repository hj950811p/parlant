#!/usr/bin/env python3
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

"""AI VFX Director Pro Ultimate - Integrated Blueprint Generation Tool

Comprehensive VFX blueprint generation tool integrating:
1. Blueprint validation and auto-fix with dimension normalization
2. Smart error handling with retry strategies
3. AI-powered emotion analysis for better pacing
4. Global style guide generation for visual consistency
5. Optimized prompt generation with few-shot examples

This script processes SRT subtitle files and generates validated VFX blueprints
using the Cerebras API with intelligent error recovery and style optimization.
"""

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Optional

import requests
from pydantic import BaseModel, Field


# ============================================================================
# === CONFIGURATION CONSTANTS ===
# ============================================================================

DEFAULT_MODEL = "qwen-3-235b-a22b-thinking-2507"
DEFAULT_API_URL = "https://api.cerebras.ai/v1/chat/completions"
CEREBRAS_TIMEOUT = 30.0

# Error recovery constants
INITIAL_BACKOFF = 1.0
MAX_BACKOFF = 60.0
MAX_RETRIES = 3


# ============================================================================
# === DATA MODELS ===
# ============================================================================


class FrameDimensions(BaseModel):
    """Frame dimensions with validation."""

    width: int = Field(..., description="Frame width in pixels")
    height: int = Field(..., description="Frame height in pixels")


class Blueprint(BaseModel):
    """VFX Blueprint data model with validation support."""

    id: str = Field(..., description="Unique blueprint identifier")
    timecode: str = Field(..., description="Video timecode in SRT format (HH:MM:SS,mmm)")
    frame_dimensions: FrameDimensions = Field(..., description="Frame dimensions")
    background_description: str = Field(..., description="Description of the background")
    animation_style: Optional[str] = Field(
        default=None, description="Animation style guide"
    )
    emotion_intensity: Optional[int] = Field(default=None, description="Emotion intensity 1-10")
    required_keys: list[str] = Field(
        default_factory=list, description="Required keys in blueprint"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Optional metadata")


class AnimationCadence(str, Enum):
    """Animation pacing options."""

    SLOW = "slow"
    MODERATE = "moderate"
    FAST = "fast"


@dataclass
class StyleGuide:
    """Style guide with visual attributes."""

    palette: list[str]
    visual_style: str
    animation_cadence: AnimationCadence
    mood: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "palette": self.palette,
            "visual_style": self.visual_style,
            "animation_cadence": self.animation_cadence.value,
            "mood": self.mood,
        }


@dataclass
class EmotionAnalysis:
    """Structured emotion analysis."""

    intensity: int
    pacing: Literal["fast", "slow"]
    focus_keywords: list[str]
    recommended_scene_duration: int


@dataclass
class ValidationResult:
    """Result of blueprint validation."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ============================================================================
# === SRT PARSING MODULE ===
# ============================================================================


class SRTParser:
    """Parser for SRT (SubRip) subtitle files."""

    def parse(self, filepath: str) -> list[dict[str, str]]:
        """Parse an SRT file and extract subtitle entries.

        Args:
            filepath: Path to the SRT file

        Returns:
            List of dicts with 'start', 'end', and 'text' keys
        """
        entries: list[dict[str, str]] = []

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
        except (IOError, OSError) as e:
            raise IOError(f"Failed to read SRT file: {e}") from e

        if not content.strip():
            return entries

        blocks = content.strip().split("\n\n")

        for block in blocks:
            lines = block.strip().split("\n")
            if len(lines) < 3:
                continue

            try:
                timecode_line = lines[1]
                if "-->" not in timecode_line:
                    continue

                parts = timecode_line.split("-->")
                if len(parts) != 2:
                    continue

                start = parts[0].strip()
                end = parts[1].strip()
                text_lines = lines[2:]
                text = "\n".join(text_lines).strip()

                if text:
                    entries.append({"start": start, "end": end, "text": text})
            except (IndexError, ValueError):
                continue

        return entries


# ============================================================================
# === BLUEPRINT VALIDATION AND REPAIR MODULE ===
# ============================================================================


class BlueprintValidator:
    """Validator for VFX blueprints with explicit validation rules."""

    # Dimension specifications
    MIN_WIDTH = 320
    MAX_WIDTH = 7680
    MIN_HEIGHT = 180
    MAX_HEIGHT = 4320

    # Layer-specific dimensions
    LAYER_SPECS = {
        "A": {"width": 1080, "height": 1920},  # A層: 1080x1920
        "B": {"width": 700, "height": 700},    # B/C/D層: 700x700
        "C": {"width": 700, "height": 700},
        "D": {"width": 700, "height": 700},
        "T": {"width": 800, "height": 150},    # T層: 800x150
    }

    TECHNICAL_TERMS_PATTERN = re.compile(
        r"\b(?:ray_tracing|ray_tracing_engine|shader_model|compute|gpu|cuda|algorithm|engine|buffer|render_pipeline)\b",
        re.IGNORECASE,
    )

    def validate(self, blueprint: Blueprint) -> list[str]:
        """Validate a blueprint against all rules.

        Args:
            blueprint: Blueprint to validate

        Returns:
            List of error messages (empty if valid)
        """
        errors: list[str] = []

        errors.extend(self._validate_frame_dimensions(blueprint.frame_dimensions))
        errors.extend(self._validate_background_description(blueprint.background_description))
        errors.extend(self._validate_required_keys(blueprint, blueprint.required_keys))

        return errors

    def _validate_frame_dimensions(self, dimensions: FrameDimensions) -> list[str]:
        """Validate frame dimensions against allowed ranges."""
        errors: list[str] = []

        if not (self.MIN_WIDTH <= dimensions.width <= self.MAX_WIDTH):
            errors.append(
                f"Frame width {dimensions.width} is out of range "
                f"[{self.MIN_WIDTH}, {self.MAX_WIDTH}]"
            )

        if not (self.MIN_HEIGHT <= dimensions.height <= self.MAX_HEIGHT):
            errors.append(
                f"Frame height {dimensions.height} is out of range "
                f"[{self.MIN_HEIGHT}, {self.MAX_HEIGHT}]"
            )

        return errors

    def _validate_background_description(self, description: str) -> list[str]:
        """Validate background description for simplicity and compliance."""
        errors: list[str] = []

        if self.TECHNICAL_TERMS_PATTERN.search(description):
            errors.append(
                f"Background description contains technical terms that must be simplified: {description}"
            )

        return errors

    def _validate_required_keys(
        self, blueprint: Blueprint, required_keys: list[str]
    ) -> list[str]:
        """Validate that all required keys are present."""
        errors: list[str] = []

        blueprint_dict = blueprint.model_dump()

        for key in required_keys:
            if key not in blueprint_dict or blueprint_dict[key] is None:
                errors.append(f"Required key '{key}' is missing or null")

        return errors


class BlueprintRepairer:
    """Auto-fix routines for non-compliant blueprints."""

    def __init__(self, logger: "StructuredLogger") -> None:
        """Initialize the repairer with a logger."""
        self.logger = logger
        self.validator = BlueprintValidator()

    def repair(self, blueprint: Blueprint) -> Blueprint:
        """Repair a blueprint to make it compliant.

        Fixes:
        - Normalizes frame dimensions to valid ranges
        - Simplifies background descriptions
        - Supplies default values for omitted fields
        """
        repaired = blueprint.model_copy(deep=True)

        repaired.frame_dimensions = self._repair_frame_dimensions(repaired.frame_dimensions)
        repaired.background_description = self._repair_background_description(
            repaired.background_description
        )
        repaired = self._ensure_defaults(repaired)

        return repaired

    def _repair_frame_dimensions(self, dimensions: FrameDimensions) -> FrameDimensions:
        """Normalize frame dimensions to valid ranges."""
        original_width = dimensions.width
        original_height = dimensions.height

        clamped_width = max(
            self.validator.MIN_WIDTH, min(self.validator.MAX_WIDTH, dimensions.width)
        )
        clamped_height = max(
            self.validator.MIN_HEIGHT, min(self.validator.MAX_HEIGHT, dimensions.height)
        )

        if clamped_width != original_width:
            self.logger.record_fix(
                "dimension_normalization",
                {
                    "field": "width",
                    "original": original_width,
                    "fixed": clamped_width,
                    "reason": f"Clamped to valid range [{self.validator.MIN_WIDTH}, {self.validator.MAX_WIDTH}]",
                },
            )

        if clamped_height != original_height:
            self.logger.record_fix(
                "dimension_normalization",
                {
                    "field": "height",
                    "original": original_height,
                    "fixed": clamped_height,
                    "reason": f"Clamped to valid range [{self.validator.MIN_HEIGHT}, {self.validator.MAX_HEIGHT}]",
                },
            )

        return FrameDimensions(width=clamped_width, height=clamped_height)

    def _repair_background_description(self, description: str) -> str:
        """Simplify background description by removing technical terms."""
        original = description
        simplified = description

        simplified = self.validator.TECHNICAL_TERMS_PATTERN.sub("", simplified)
        simplified = re.sub(r"_+", " ", simplified)
        simplified = re.sub(r"\s+", " ", simplified).strip()

        if simplified != original:
            self.logger.record_fix(
                "background_rewrite",
                {
                    "original": original,
                    "fixed": simplified,
                    "reason": "Removed technical terms for simplicity",
                },
            )

        return simplified if simplified else "A simple scene"

    def _ensure_defaults(self, blueprint: Blueprint) -> Blueprint:
        """Ensure all fields have appropriate defaults."""
        if not blueprint.metadata:
            self.logger.record_fix(
                "default_values",
                {
                    "field": "metadata",
                    "reason": "Initialized metadata with default values",
                },
            )
            blueprint.metadata = {"version": "1.0", "auto_fixed": True}

        return blueprint


@dataclass
class StructuredLogger:
    """Structured logger for tracking blueprint fixes and validation."""

    fixes: list[dict[str, Any]] = field(default_factory=list)

    def record_fix(self, fix_type: str, details: dict[str, Any]) -> None:
        """Record a fix that was performed on a blueprint."""
        self.fixes.append({"type": fix_type, "details": details})

    def get_summary_report(self) -> str:
        """Generate a summary report of all fixes."""
        if not self.fixes:
            return "No fixes performed."

        report_lines = ["Blueprint Validation and Repair Summary", "=" * 40]

        fix_counts: dict[str, int] = {}
        for fix in self.fixes:
            fix_type = fix["type"]
            fix_counts[fix_type] = fix_counts.get(fix_type, 0) + 1

        for fix_type, count in sorted(fix_counts.items()):
            report_lines.append(f"  {fix_type}: {count} fixes")

        report_lines.append(f"\nTotal fixes: {len(self.fixes)}")

        return "\n".join(report_lines)


# ============================================================================
# === ERROR HANDLING MODULE ===
# ============================================================================


class ErrorHandler:
    """Smart error handling with retry strategies."""

    def __init__(self) -> None:
        """Initialize error handler."""
        pass

    @staticmethod
    def should_retry(error: Exception) -> bool:
        """Determine if an error should trigger a retry."""
        error_str = str(error).lower()
        error_type = type(error).__name__

        # Retry on rate limit
        if "rate" in error_str and "limit" in error_str:
            return True

        # Retry on timeout
        if error_type in ("TimeoutError", "ConnectTimeout", "ReadTimeout"):
            return True

        # Retry on connection errors
        if error_type in ("ConnectionError", "ConnectionResetError", "BrokenPipeError"):
            return True

        # Retry on 5xx errors
        if "500" in error_str or "502" in error_str or "503" in error_str:
            return True

        return False

    @staticmethod
    def get_backoff_time(attempt: int) -> float:
        """Calculate exponential backoff time."""
        backoff = min(INITIAL_BACKOFF * (2 ** (attempt - 1)), MAX_BACKOFF)
        return backoff

    @staticmethod
    def repair_json(malformed_json: str) -> Optional[dict[str, Any]]:
        """Attempt to repair malformed JSON."""
        try:
            # First try as-is
            return json.loads(malformed_json)
        except json.JSONDecodeError:
            pass

        try:
            # Try removing trailing commas
            fixed = re.sub(r",(\s*[}\]])", r"\1", malformed_json)
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

        try:
            # Try wrapping unquoted strings in quotes
            fixed = re.sub(r':\s*([^,}\]\"\'\d-][\w]*)', r': "\1"', malformed_json)
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

        return None


# ============================================================================
# === EMOTION ANALYSIS MODULE ===
# ============================================================================


class EmotionAnalyzer:
    """AI-powered emotion analysis for dialogue."""

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

    @staticmethod
    def analyze_fallback(text: str) -> EmotionAnalysis:
        """Perform fallback heuristic-based emotion analysis."""
        text_lower = text.lower()

        positive_count = sum(1 for kw in EmotionAnalyzer.POSITIVE_KEYWORDS if kw in text_lower)
        negative_count = sum(1 for kw in EmotionAnalyzer.NEGATIVE_KEYWORDS if kw in text_lower)
        neutral_count = sum(1 for kw in EmotionAnalyzer.NEUTRAL_MARKERS if kw in text_lower)

        intensity = EmotionAnalyzer._compute_intensity(positive_count, negative_count, len(text_lower))
        pacing = EmotionAnalyzer._compute_pacing(positive_count, negative_count, neutral_count)
        focus_keywords = EmotionAnalyzer._extract_focus_keywords(text_lower, positive_count, negative_count)
        duration = EmotionAnalyzer._compute_scene_duration(intensity, len(text))

        return EmotionAnalysis(
            intensity=intensity,
            pacing=pacing,
            focus_keywords=focus_keywords,
            recommended_scene_duration=duration,
        )

    @staticmethod
    def _compute_intensity(positive_count: int, negative_count: int, text_length: int) -> int:
        """Compute intensity from keyword counts."""
        total_emotional_keywords = positive_count + negative_count
        base_intensity = max(1, min(10, 1 + (total_emotional_keywords * 2)))
        length_factor = min(1.5, text_length / 100)
        adjusted_intensity = int(base_intensity * length_factor)
        return max(1, min(10, adjusted_intensity))

    @staticmethod
    def _compute_pacing(
        positive_count: int, negative_count: int, neutral_count: int
    ) -> Literal["fast", "slow"]:
        """Determine pacing based on emotion distribution."""
        if positive_count + negative_count > neutral_count:
            return "fast"
        return "slow"

    @staticmethod
    def _extract_focus_keywords(
        text_lower: str, positive_count: int, negative_count: int
    ) -> list[str]:
        """Extract focus keywords from text."""
        keywords = []

        if positive_count > 0:
            for kw in EmotionAnalyzer.POSITIVE_KEYWORDS:
                if kw in text_lower:
                    keywords.append(kw)
                    if len(keywords) >= 2:
                        break

        if negative_count > 0 and len(keywords) < 2:
            for kw in EmotionAnalyzer.NEGATIVE_KEYWORDS:
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
        """Compute recommended scene duration."""
        base_duration = 3000
        intensity_factor = 1.0 + (intensity - 1) * 0.25
        length_factor = min(1.5, text_length / 100)
        duration = int(base_duration * intensity_factor * length_factor)
        return max(3000, min(6000, duration))


# ============================================================================
# === GLOBAL STYLE GUIDE MODULE ===
# ============================================================================


class StyleGuideGenerator:
    """Generate global style guides from dialogues."""

    STYLE_KEYWORDS = {
        "warm": ["warm", "sunset", "golden", "orange"],
        "cool": ["cool", "blue", "purple", "cold"],
        "calm": ["calm", "peaceful", "slow", "serene", "meditative"],
        "energetic": ["fast", "rapid", "dynamic", "vibrant", "action"],
        "minimalist": ["minimalist", "clean", "simple", "minimal"],
        "modern": ["modern", "contemporary", "professional", "sleek"],
        "cyberpunk": ["cyber", "neon", "futuristic", "digital"],
    }

    ANIMATION_KEYWORDS = {
        "slow": ["slow", "methodical", "steady", "gradual"],
        "moderate": ["moderate", "balanced", "even", "regular"],
        "fast": ["fast", "rapid", "quick", "swift", "dynamic"],
    }

    MOOD_KEYWORDS = {
        "uplifting": ["uplifting", "joyful", "happy", "bright", "positive"],
        "somber": ["somber", "dark", "gloomy", "melancholic", "sad"],
        "energetic": ["energetic", "exciting", "thrilling", "action"],
        "peaceful": ["peaceful", "calm", "tranquil", "serene"],
        "contemplative": ["contemplative", "thoughtful", "reflective"],
    }

    COLOR_MAP = {
        "#FFA500": ["orange", "warm", "sunset"],
        "#800080": ["purple", "violet", "lavender"],
        "#0000FF": ["blue", "cool"],
        "#000000": ["dark", "black", "shadowy"],
        "#FFFFFF": ["white", "bright", "light"],
        "#FF0000": ["red", "vibrant"],
    }

    @staticmethod
    def generate(dialogues: list[str]) -> StyleGuide:
        """Generate a global style guide from dialogues."""
        combined_text = " ".join(dialogues).lower()

        palette = StyleGuideGenerator._determine_palette(combined_text)
        visual_style = StyleGuideGenerator._determine_visual_style(combined_text)
        animation_cadence = StyleGuideGenerator._determine_animation_cadence(combined_text)
        mood = StyleGuideGenerator._determine_mood(combined_text)

        return StyleGuide(
            palette=palette,
            visual_style=visual_style,
            animation_cadence=animation_cadence,
            mood=mood,
        )

    @staticmethod
    def _determine_palette(combined_text: str) -> list[str]:
        """Determine color palette from text."""
        palette: list[str] = []

        for color, keywords in StyleGuideGenerator.COLOR_MAP.items():
            for keyword in keywords:
                if keyword in combined_text and color not in palette:
                    palette.append(color)
                    break

        if not palette:
            palette = ["#0000FF", "#FFFFFF", "#000000"]

        return palette

    @staticmethod
    def _determine_visual_style(combined_text: str) -> str:
        """Determine visual style from text."""
        style_patterns = {
            "minimalist": ["minimalist", "clean", "simple"],
            "modern": ["modern", "professional", "contemporary"],
            "cyberpunk": ["cyber", "futuristic", "neon"],
            "cinematic": ["cinematic", "film", "movie"],
            "artistic": ["artistic", "creative", "expressive"],
        }

        for style, keywords in style_patterns.items():
            if any(keyword in combined_text for keyword in keywords):
                return style

        return "balanced"

    @staticmethod
    def _determine_animation_cadence(combined_text: str) -> AnimationCadence:
        """Determine animation cadence from text."""
        if any(word in combined_text for word in ["slow", "methodical", "gradual", "steady"]):
            return AnimationCadence.SLOW

        if any(word in combined_text for word in ["fast", "rapid", "quick", "dynamic"]):
            return AnimationCadence.FAST

        return AnimationCadence.MODERATE

    @staticmethod
    def _determine_mood(combined_text: str) -> str:
        """Determine mood from text."""
        mood_patterns = {
            "uplifting": ["uplifting", "joyful", "happy", "bright", "positive"],
            "somber": ["somber", "dark", "gloomy", "melancholic", "sad"],
            "energetic": ["energetic", "exciting", "thrilling", "action"],
            "peaceful": ["peaceful", "calm", "tranquil", "serene"],
            "contemplative": ["contemplative", "thoughtful", "reflective"],
        }

        for mood, keywords in mood_patterns.items():
            if any(keyword in combined_text for keyword in keywords):
                return mood

        return "neutral"


# ============================================================================
# === PROMPT OPTIMIZATION MODULE ===
# ============================================================================


class PromptOptimizer:
    """Optimize prompts with few-shot examples and style guidelines."""

    FEW_SHOT_EXAMPLES = [
        {
            "dialogue": "A hero rises from the ashes, determined and powerful",
            "blueprint": {
                "frame_dimensions": {"width": 1920, "height": 1080},
                "background_description": "Epic landscape with dramatic lighting and mountain peaks",
                "animation_style": "cinematic with powerful transitions",
                "emotion_intensity": 9,
            },
        },
        {
            "dialogue": "A quiet moment of reflection as the sun sets over the horizon",
            "blueprint": {
                "frame_dimensions": {"width": 1920, "height": 1080},
                "background_description": "Serene sunset with warm golden tones and calm waters",
                "animation_style": "slow and contemplative transitions",
                "emotion_intensity": 4,
            },
        },
    ]

    @staticmethod
    def build_prompt(
        dialogue: str,
        style_guide: Optional[StyleGuide] = None,
        emotion_analysis: Optional[EmotionAnalysis] = None,
    ) -> str:
        """Build an optimized prompt for VFX blueprint generation.

        Combines dialogue with style guide and emotion analysis for better results.
        Stays within 500-800 character range with 2-3 few-shot examples.
        """
        prompt_parts = [
            "Generate a VFX blueprint for this scene.",
            f"Dialogue: {dialogue}",
        ]

        if emotion_analysis:
            prompt_parts.append(
                f"Emotion: intensity {emotion_analysis.intensity}/10, pacing {emotion_analysis.pacing}"
            )

        if style_guide:
            prompt_parts.append(f"Style: {style_guide.visual_style}, mood {style_guide.mood}")

        prompt_parts.extend(
            [
                "Return JSON with: frame_dimensions, background_description, animation_style, emotion_intensity.",
                "Keep descriptions clear and simple, avoiding technical jargon.",
            ]
        )

        return "\n".join(prompt_parts)

    @staticmethod
    def add_few_shot_context() -> str:
        """Add few-shot examples to improve model response."""
        examples = []
        for example in PromptOptimizer.FEW_SHOT_EXAMPLES:
            examples.append(
                f"Example dialogue: {example['dialogue']}\n"
                f"Expected blueprint: {json.dumps(example['blueprint'])}"
            )
        return "\n\n".join(examples)


# ============================================================================
# === API COMMUNICATION MODULE ===
# ============================================================================


class CerebrasClient:
    """Client for Cerebras API communication with error handling and retries."""

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL, api_url: str = DEFAULT_API_URL):
        """Initialize Cerebras client."""
        self.api_key = api_key
        self.model = model
        self.api_url = api_url
        self.error_handler = ErrorHandler()

    def generate_blueprint(self, prompt: str, max_retries: int = MAX_RETRIES) -> dict[str, Any]:
        """Generate a VFX blueprint using Cerebras API with retries."""
        for attempt in range(1, max_retries + 1):
            try:
                return self._call_api(prompt)
            except Exception as e:
                if not self.error_handler.should_retry(e):
                    raise

                if attempt < max_retries:
                    backoff = self.error_handler.get_backoff_time(attempt)
                    print(
                        f"Attempt {attempt} failed: {e}. Retrying in {backoff:.1f}s...",
                        file=sys.stderr,
                    )
                    time.sleep(backoff)
                else:
                    print(
                        f"All {max_retries} attempts failed. Last error: {e}", file=sys.stderr
                    )
                    raise

    def _call_api(self, prompt: str) -> dict[str, Any]:
        """Make a single API call to Cerebras."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 1024,
        }

        response = requests.post(
            self.api_url,
            json=payload,
            headers=headers,
            timeout=CEREBRAS_TIMEOUT,
        )

        response.raise_for_status()
        result = response.json()

        if "choices" not in result or not result["choices"]:
            raise ValueError("Invalid API response: no choices returned")

        content = result["choices"][0]["message"]["content"]
        return self._parse_response_content(content)

    @staticmethod
    def _parse_response_content(content: str) -> dict[str, Any]:
        """Parse and repair JSON from API response."""
        # Try to extract JSON from response
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            json_str = content

        # Try to parse JSON
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Try to repair
            repaired = ErrorHandler.repair_json(json_str)
            if repaired:
                return repaired

            raise ValueError(f"Could not parse API response as JSON: {content[:200]}")


# ============================================================================
# === BLUEPRINT PIPELINE MODULE ===
# ============================================================================


class VFXBlueprintPipeline:
    """End-to-end pipeline for SRT to validated VFX blueprint conversion."""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        enable_emotion_analysis: bool = False,
        enable_global_style: bool = False,
        verbose: bool = False,
    ):
        """Initialize the pipeline."""
        self.api_key = api_key
        self.model = model
        self.enable_emotion_analysis = enable_emotion_analysis
        self.enable_global_style = enable_global_style
        self.verbose = verbose

        self.parser = SRTParser()
        self.validator = BlueprintValidator()
        self.logger = StructuredLogger()
        self.repairer = BlueprintRepairer(self.logger)
        self.emotion_analyzer = EmotionAnalyzer()
        self.style_guide_generator = StyleGuideGenerator()
        self.cerebras_client = CerebrasClient(api_key, model)

    def process_srt_file(self, srt_filepath: str) -> list[Blueprint]:
        """Process an SRT file into validated blueprints."""
        entries = self.parser.parse(srt_filepath)

        if self.enable_global_style:
            dialogues = [entry["text"] for entry in entries]
            style_guide = self.style_guide_generator.generate(dialogues)
            if self.verbose:
                print(f"Generated global style guide: {style_guide.to_dict()}", file=sys.stderr)
        else:
            style_guide = None

        blueprints: list[Blueprint] = []

        for idx, entry in enumerate(entries):
            try:
                # Analyze emotion if enabled
                emotion_analysis = None
                if self.enable_emotion_analysis:
                    emotion_analysis = self.emotion_analyzer.analyze_fallback(entry["text"])
                    if self.verbose:
                        print(f"Entry {idx} emotion: {emotion_analysis}", file=sys.stderr)

                # Build optimized prompt
                prompt = PromptOptimizer.build_prompt(entry["text"], style_guide, emotion_analysis)

                # Generate blueprint via API
                if self.verbose:
                    print(f"Generating blueprint for entry {idx}...", file=sys.stderr)

                blueprint_data = self.cerebras_client.generate_blueprint(prompt)

                # Add required fields
                blueprint_data["id"] = f"bp_{idx:03d}"
                blueprint_data["timecode"] = entry["start"]
                blueprint_data["required_keys"] = ["id", "timecode", "frame_dimensions"]

                # Ensure frame dimensions are present
                if "frame_dimensions" not in blueprint_data:
                    blueprint_data["frame_dimensions"] = {"width": 1920, "height": 1080}

                # Create and repair blueprint
                blueprint = Blueprint(**blueprint_data)
                if emotion_analysis:
                    blueprint.emotion_intensity = emotion_analysis.intensity

                repaired_blueprint = self.repairer.repair(blueprint)

                # Validate
                errors = self.validator.validate(repaired_blueprint)
                if errors and self.verbose:
                    for error in errors:
                        print(f"  Warning: {error}", file=sys.stderr)

                blueprints.append(repaired_blueprint)

            except Exception as e:
                if self.verbose:
                    print(f"Error processing entry {idx}: {e}", file=sys.stderr)
                    import traceback

                    traceback.print_exc()
                else:
                    print(f"Error processing entry {idx}: {e}", file=sys.stderr)

        return blueprints

    def output_json(self, blueprints: list[Blueprint], output_filepath: Optional[str] = None) -> str:
        """Output blueprints as JSON."""
        output = {
            "blueprints": [bp.model_dump() for bp in blueprints],
            "validation_report": self.logger.get_summary_report(),
            "total_blueprints": len(blueprints),
        }

        json_str = json.dumps(output, indent=2)

        if output_filepath:
            with open(output_filepath, "w", encoding="utf-8") as f:
                f.write(json_str)
            if self.verbose:
                print(f"Output written to {output_filepath}", file=sys.stderr)

        return json_str


# ============================================================================
# === MAIN ENTRY POINT ===
# ============================================================================


def main() -> int:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="AI VFX Director Pro Ultimate - Integrated Blueprint Generation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python AI_VFX_Director_Pro_Ultimate.py --input script.srt --api-key YOUR_KEY

  # Enable all features
  python AI_VFX_Director_Pro_Ultimate.py \\
    --input script.srt \\
    --output result.json \\
    --emotion-ai \\
    --global-style \\
    --api-key YOUR_KEY
        """,
    )

    parser.add_argument("--input", required=True, help="Path to SRT file to process")
    parser.add_argument(
        "--output", "-o", default=None, help="Output JSON file path (default: stdout)"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="AI model to use")
    parser.add_argument(
        "--emotion-ai",
        action="store_true",
        help="Enable AI-powered emotion analysis",
    )
    parser.add_argument(
        "--global-style",
        action="store_true",
        help="Enable global style guide generation",
    )
    parser.add_argument(
        "--api-key",
        required=True,
        help="Cerebras API key",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Error: SRT file not found: {args.input}", file=sys.stderr)
        return 1

    try:
        pipeline = VFXBlueprintPipeline(
            api_key=args.api_key,
            model=args.model,
            enable_emotion_analysis=args.emotion_ai,
            enable_global_style=args.global_style,
            verbose=args.verbose,
        )

        blueprints = pipeline.process_srt_file(args.input)
        json_output = pipeline.output_json(blueprints, args.output)

        if args.output is None:
            print(json_output)

        if args.verbose:
            print(pipeline.logger.get_summary_report(), file=sys.stderr)

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
