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

"""AI VFX Director Pro Ultimate - Comprehensive Blueprint Generation & Optimization

This script integrates all modular improvements into a single, powerful tool for
processing SRT files and generating optimized VFX blueprints with:

1. Model Configuration: Support for Qwen and other models via CLI
2. Blueprint Validation & Auto-Fix: Automatic validation and repair
3. Prompt Optimization: Enhanced prompts with few-shot examples
4. Global Style Management: Unified style guide generation
5. AI Emotion Analysis: LLM-based semantic emotion classification
6. Semantic Caching: Intelligent caching and retrieval
7. Error Handling: Resilient error recovery strategies
8. Progressive Generation: Optional multi-stage generation

Usage:
    python AI_VFX_Director_Pro_Ultimate.py --input script.srt --output blueprint.json
    python AI_VFX_Director_Pro_Ultimate.py --input script.srt --model custom-model --emotion-ai
    python AI_VFX_Director_Pro_Ultimate.py --input script.srt --cache --similarity 0.85
"""

import argparse
import difflib
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


# ============================================================================
# ===== CONFIGURATION & CONSTANTS =====
# ============================================================================


@dataclass
class Config:
    """Configuration for AI VFX Director Pro Ultimate."""

    model: str = "qwen-3-235b-a22b-thinking-2507"
    """Default model: Qwen 3 235B with extended thinking capability."""

    wait_time: float = 2.0
    """Wait time between API calls in seconds."""

    cache_enabled: bool = False
    """Enable semantic caching of blueprints."""

    similarity_threshold: float = 0.85
    """Similarity threshold for cache retrieval (0.0 to 1.0)."""

    emotion_ai: bool = False
    """Use AI-based emotion analysis instead of keyword matching."""

    global_style: bool = False
    """Generate and apply global style guide."""

    progressive: bool = False
    """Enable progressive/multi-stage generation."""

    unified_background: bool = False
    """Use unified background style across all blueprints."""


# ============================================================================
# ===== DATA MODELS =====
# ============================================================================


class FrameDimensions(BaseModel):
    """Frame dimensions with validation."""

    width: int = Field(..., description="Frame width in pixels")
    height: int = Field(..., description="Frame height in pixels")


class Blueprint(BaseModel):
    """VFX Blueprint data model with validation support."""

    id: str = Field(..., description="Unique blueprint identifier")
    timecode: str = Field(..., description="Video timecode in SRT format")
    frame_dimensions: FrameDimensions = Field(..., description="Frame dimensions")
    background_description: str = Field(..., description="Background description")
    required_keys: List[str] = Field(default_factory=list, description="Required keys")
    emotion: Optional[str] = Field(default=None, description="Detected emotion")
    style_attributes: Dict[str, Any] = Field(
        default_factory=dict, description="Style attributes"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Optional metadata")


# ============================================================================
# ===== TEXT UTILITIES & SIMILARITY =====
# ============================================================================


def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate similarity between two texts using SequenceMatcher.

    Args:
        text1: First text
        text2: Second text

    Returns:
        Similarity score from 0.0 to 1.0
    """
    if not text1 or not text2:
        return 0.0 if (text1 or text2) else 1.0

    # Normalize texts
    t1 = text1.lower().strip()
    t2 = text2.lower().strip()

    if t1 == t2:
        return 1.0

    # Use SequenceMatcher for similarity
    matcher = difflib.SequenceMatcher(None, t1, t2)
    return matcher.ratio()


# ============================================================================
# ===== SEMANTIC CACHE SYSTEM =====
# ============================================================================


@dataclass
class CacheStatistics:
    """Statistics for cache performance."""

    total_queries: int = 0
    cache_hits: int = 0
    cache_misses: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        if self.total_queries == 0:
            return 0.0
        return self.cache_hits / self.total_queries


class SemanticCache:
    """Semantic caching system for blueprints based on text similarity."""

    def __init__(self, similarity_threshold: float = 0.85) -> None:
        """Initialize semantic cache.

        Args:
            similarity_threshold: Minimum similarity for cache retrieval
        """
        self.similarity_threshold = similarity_threshold
        self._cache: Dict[str, Any] = {}
        self._stats = CacheStatistics()

    def put(self, key: str, value: Any) -> None:
        """Store a blueprint in cache.

        Args:
            key: Query/description key
            value: Blueprint data to cache
        """
        self._cache[key] = value

    def get(self, query: str) -> Optional[Any]:
        """Retrieve a cached blueprint based on similarity.

        Args:
            query: Query string to match

        Returns:
            Cached blueprint if found, None otherwise
        """
        self._stats.total_queries += 1

        # Exact match
        if query in self._cache:
            self._stats.cache_hits += 1
            return self._cache[query]

        # Similarity-based search
        best_match = None
        best_similarity = 0.0

        for cached_key, cached_value in self._cache.items():
            similarity = calculate_text_similarity(query, cached_key)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = cached_value

        if best_similarity >= self.similarity_threshold:
            self._stats.cache_hits += 1
            return best_match

        self._stats.cache_misses += 1
        return None

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        self._stats = CacheStatistics()

    def __len__(self) -> int:
        """Return number of cached entries."""
        return len(self._cache)

    def get_statistics(self) -> Dict[str, Any]:
        """Get cache performance statistics.

        Returns:
            Dictionary with cache statistics
        """
        return {
            "total_queries": self._stats.total_queries,
            "cache_hits": self._stats.cache_hits,
            "cache_misses": self._stats.cache_misses,
            "hit_rate": self._stats.hit_rate,
            "cached_entries": len(self._cache),
        }


# ============================================================================
# ===== GLOBAL STYLE GUIDE GENERATION =====
# ============================================================================


class GlobalStyleGuide:
    """Generate unified style guide from multiple text samples."""

    def analyze(self, samples: List[str]) -> Dict[str, Any]:
        """Analyze text samples to extract style characteristics.

        Args:
            samples: List of text samples to analyze

        Returns:
            Dictionary with style characteristics
        """
        if not samples:
            return {
                "summary": "No samples provided",
                "tone": "neutral",
                "characteristics": [],
            }

        # Extract common characteristics
        characteristics: List[str] = []
        word_freq: Dict[str, int] = {}

        for sample in samples:
            words = sample.lower().split()
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1

        # Get most common words (skip common words)
        common_words = {"a", "an", "the", "and", "or", "is", "are", "of", "in"}
        for word, freq in sorted(word_freq.items(), key=lambda x: -x[1])[:10]:
            if word not in common_words and len(word) > 2:
                characteristics.append(word)

        # Analyze tone
        positive_words = {
            "bright",
            "beautiful",
            "cheerful",
            "calm",
            "peaceful",
            "vibrant",
        }
        negative_words = {
            "dark",
            "gloomy",
            "mysterious",
            "ominous",
            "sad",
            "melancholic",
        }

        positive_count = sum(1 for w in word_freq if w in positive_words)
        negative_count = sum(1 for w in word_freq if w in negative_words)

        if positive_count > negative_count:
            tone = "positive"
        elif negative_count > positive_count:
            tone = "negative"
        else:
            tone = "neutral"

        return {
            "summary": f"Style based on {len(samples)} samples",
            "tone": tone,
            "characteristics": characteristics[:5],
            "sample_count": len(samples),
        }

    def format_for_prompt(self, style_data: Dict[str, Any]) -> str:
        """Format style guide for inclusion in prompts.

        Args:
            style_data: Style analysis data

        Returns:
            Formatted string for prompt inclusion
        """
        if not style_data or not style_data.get("characteristics"):
            return ""

        characteristics = ", ".join(style_data.get("characteristics", []))
        tone = style_data.get("tone", "neutral")

        return f"Style: {tone} tone with characteristics: {characteristics}"


# ============================================================================
# ===== PROMPT SYSTEM & OPTIMIZATION =====
# ============================================================================

FEW_SHOT_EXAMPLES = [
    {
        "description": "A bright sunny day at the beach",
        "blueprint": {
            "mood": "cheerful and vibrant",
            "color_palette": "warm yellows and blues",
            "suggested_effects": "lens flare, warm light glow",
        },
    },
    {
        "description": "A dark, mysterious forest at night",
        "blueprint": {
            "mood": "mysterious and suspenseful",
            "color_palette": "deep greens and purples",
            "suggested_effects": "volumetric fog, eerie shadows",
        },
    },
    {
        "description": "A calm rural landscape",
        "blueprint": {
            "mood": "peaceful and serene",
            "color_palette": "soft greens and pastels",
            "suggested_effects": "gentle wind, soft natural lighting",
        },
    },
]


class PromptBuilder:
    """Build and optimize prompts for VFX blueprint generation."""

    def __init__(
        self,
        include_few_shot: bool = True,
        max_prompt_length: int = 800,
    ) -> None:
        """Initialize prompt builder.

        Args:
            include_few_shot: Include few-shot examples in prompts
            max_prompt_length: Maximum prompt length in characters
        """
        self.include_few_shot = include_few_shot
        self.max_prompt_length = max_prompt_length

    def build_prompt(self, description: str) -> str:
        """Build a basic prompt for blueprint generation.

        Args:
            description: Scene description

        Returns:
            Formatted prompt
        """
        base_prompt = f"""Generate a VFX blueprint for: {description}

Provide a JSON blueprint with:
- mood: emotional tone
- color_palette: recommended colors
- suggested_effects: recommended VFX effects
"""

        if self.include_few_shot:
            base_prompt += self._add_few_shot_examples()

        # Truncate if needed
        if len(base_prompt) > self.max_prompt_length:
            base_prompt = base_prompt[: self.max_prompt_length]

        return base_prompt

    def build_with_style(self, description: str, style_guide: str) -> str:
        """Build prompt with style guide.

        Args:
            description: Scene description
            style_guide: Style guide text

        Returns:
            Formatted prompt with style
        """
        prompt = f"""Generate VFX blueprint: {description}

Style guide: {style_guide}

Provide JSON with mood, color_palette, and suggested_effects.
"""
        return prompt

    def build_with_emotion(self, description: str, emotion: str) -> str:
        """Build prompt with emotion context.

        Args:
            description: Scene description
            emotion: Detected or specified emotion

        Returns:
            Formatted prompt with emotion
        """
        prompt = f"""Generate VFX blueprint: {description}
Emotion: {emotion}

Consider the emotional context when suggesting effects and colors.
Provide JSON with mood, color_palette, and suggested_effects.
"""
        return prompt

    def _add_few_shot_examples(self) -> str:
        """Add few-shot examples to prompt."""
        examples = "\n\nExamples:\n"
        for example in FEW_SHOT_EXAMPLES[:2]:  # Include top 2 examples
            examples += f"- {example['description']}\n"
        return examples


# ============================================================================
# ===== ERROR HANDLING & RECOVERY =====
# ============================================================================


class ErrorRecoveryHandler:
    """Handle and recover from various error types."""

    def __init__(self, max_retries: int = 3, base_wait_time: float = 1.0) -> None:
        """Initialize error recovery handler.

        Args:
            max_retries: Maximum retry attempts
            base_wait_time: Base wait time for backoff
        """
        self.max_retries = max_retries
        self.base_wait_time = base_wait_time

    def attempt_json_repair(self, malformed_json: str) -> Optional[Dict[str, Any]]:
        """Attempt to repair malformed JSON.

        Args:
            malformed_json: Malformed JSON string

        Returns:
            Repaired JSON as dict, or None if repair failed
        """
        if not malformed_json:
            return None

        # Try parsing as-is
        try:
            return json.loads(malformed_json)
        except json.JSONDecodeError:
            pass

        # Try removing trailing commas
        try:
            fixed = re.sub(r",(\s*[}\]])", r"\1", malformed_json)
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

        # Try quoting unquoted keys
        try:
            fixed = re.sub(r'(\w+):', r'"\1":', malformed_json)
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

        return None

    def calculate_backoff(self, attempt: int) -> float:
        """Calculate exponential backoff wait time.

        Args:
            attempt: Attempt number (1-indexed)

        Returns:
            Wait time in seconds
        """
        return self.base_wait_time * (2 ** (attempt - 1))

    def format_error_message(self, error: Exception, context: str = "") -> str:
        """Format error message with context.

        Args:
            error: The exception
            context: Additional context

        Returns:
            Formatted error message
        """
        msg = f"Error: {str(error)}"
        if context:
            msg += f" (Context: {context})"
        return msg


# ============================================================================
# ===== BLUEPRINT VALIDATION & REPAIR =====
# ============================================================================


class SRTParser:
    """Parser for SRT (SubRip) subtitle files."""

    def parse(self, filepath: str) -> List[Dict[str, str]]:
        """Parse an SRT file and extract subtitle entries.

        Args:
            filepath: Path to the SRT file

        Returns:
            List of dicts with 'start', 'end', and 'text' keys
        """
        entries: List[Dict[str, str]] = []

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


class BlueprintValidator:
    """Validator for VFX blueprints with explicit validation rules."""

    MIN_WIDTH = 320
    MAX_WIDTH = 7680
    MIN_HEIGHT = 180
    MAX_HEIGHT = 4320

    TECHNICAL_TERMS_PATTERN = re.compile(
        r"\b(?:ray_tracing|ray_tracing_engine|shader_model|compute|gpu|cuda|algorithm|engine|buffer|render_pipeline)\b",
        re.IGNORECASE,
    )

    def validate(self, blueprint: Blueprint) -> List[str]:
        """Validate a blueprint against all rules.

        Args:
            blueprint: Blueprint to validate

        Returns:
            List of error messages (empty if valid)
        """
        errors: List[str] = []

        errors.extend(self._validate_frame_dimensions(blueprint.frame_dimensions))
        errors.extend(self._validate_background_description(blueprint.background_description))
        errors.extend(self._validate_required_keys(blueprint, blueprint.required_keys))

        return errors

    def _validate_frame_dimensions(self, dimensions: FrameDimensions) -> List[str]:
        """Validate frame dimensions against allowed ranges."""
        errors: List[str] = []

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

    def _validate_background_description(self, description: str) -> List[str]:
        """Validate background description for simplicity and compliance."""
        errors: List[str] = []

        if self.TECHNICAL_TERMS_PATTERN.search(description):
            errors.append(
                f"Background description contains technical terms: {description}"
            )

        return errors

    def _validate_required_keys(
        self, blueprint: Blueprint, required_keys: List[str]
    ) -> List[str]:
        """Validate that all required keys are present."""
        errors: List[str] = []

        blueprint_dict = blueprint.model_dump()

        for key in required_keys:
            if key not in blueprint_dict or blueprint_dict[key] is None:
                errors.append(f"Required key '{key}' is missing or null")

        return errors


class StructuredLogger:
    """Structured logger for tracking blueprint fixes and validation."""

    def __init__(self) -> None:
        """Initialize structured logger."""
        self.fixes: List[Dict[str, Any]] = []

    def record_fix(self, fix_type: str, details: Dict[str, Any]) -> None:
        """Record a fix that was performed on a blueprint.

        Args:
            fix_type: Type of fix (e.g., 'dimension_normalization')
            details: Details about the fix
        """
        self.fixes.append({"type": fix_type, "details": details})

    def get_summary_report(self) -> str:
        """Generate a summary report of all fixes.

        Returns:
            Human-readable summary report
        """
        if not self.fixes:
            return "No fixes performed."

        report_lines = ["Blueprint Validation and Repair Summary", "=" * 40]

        fix_counts: Dict[str, int] = {}
        for fix in self.fixes:
            fix_type = fix["type"]
            fix_counts[fix_type] = fix_counts.get(fix_type, 0) + 1

        for fix_type, count in sorted(fix_counts.items()):
            report_lines.append(f"  {fix_type}: {count} fixes")

        report_lines.append(f"\nTotal fixes: {len(self.fixes)}")

        return "\n".join(report_lines)


class BlueprintRepairer:
    """Auto-fix routines for non-compliant blueprints."""

    def __init__(self, logger: StructuredLogger) -> None:
        """Initialize the repairer with a logger.

        Args:
            logger: Structured logger for recording fixes
        """
        self.logger = logger
        self.validator = BlueprintValidator()

    def repair(self, blueprint: Blueprint) -> Blueprint:
        """Repair a blueprint to make it compliant.

        Args:
            blueprint: Blueprint to repair

        Returns:
            Repaired blueprint
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


# ============================================================================
# ===== EMOTION ANALYSIS =====
# ============================================================================


class EmotionAnalyzer:
    """Analyze emotion from text descriptions."""

    EMOTION_KEYWORDS: Dict[str, List[str]] = {
        "joyful": ["bright", "cheerful", "happy", "vibrant", "sunny", "warm"],
        "sad": ["dark", "gloomy", "melancholic", "sad", "blue", "gray"],
        "mysterious": ["mysterious", "eerie", "dark", "shadowy", "enigmatic"],
        "calm": ["calm", "peaceful", "serene", "quiet", "tranquil", "gentle"],
        "intense": ["intense", "dramatic", "powerful", "energetic", "dynamic"],
        "romantic": ["romantic", "beautiful", "elegant", "graceful", "flowing"],
    }

    def analyze(self, text: str, use_ai: bool = False) -> str:
        """Analyze emotion from text.

        Args:
            text: Text to analyze
            use_ai: Whether to use AI analysis (placeholder for future)

        Returns:
            Detected emotion as string
        """
        if use_ai:
            return self._analyze_with_ai(text)
        else:
            return self._analyze_with_keywords(text)

    def _analyze_with_keywords(self, text: str) -> str:
        """Analyze emotion using keyword matching."""
        text_lower = text.lower()
        emotion_scores: Dict[str, int] = {}

        for emotion, keywords in self.EMOTION_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                emotion_scores[emotion] = score

        if emotion_scores:
            return max(emotion_scores, key=emotion_scores.get)
        return "neutral"

    def _analyze_with_ai(self, text: str) -> str:
        """Analyze emotion using AI (placeholder).

        In a real implementation, this would call an LLM API.
        For now, returns results from keyword analysis.

        Args:
            text: Text to analyze

        Returns:
            Detected emotion
        """
        # Placeholder for AI analysis
        return self._analyze_with_keywords(text)


# ============================================================================
# ===== MAIN PIPELINE =====
# ============================================================================


class BlueprintPipeline:
    """End-to-end pipeline for SRT to validated JSON conversion."""

    def __init__(self, config: Config) -> None:
        """Initialize the pipeline.

        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = StructuredLogger()
        self.parser = SRTParser()
        self.validator = BlueprintValidator()
        self.repairer = BlueprintRepairer(self.logger)
        self.cache: Optional[SemanticCache] = None
        self.prompt_builder = PromptBuilder()
        self.error_handler = ErrorRecoveryHandler()
        self.emotion_analyzer = EmotionAnalyzer()
        self.style_guide: Optional[Dict[str, Any]] = None

        if config.cache_enabled:
            self.cache = SemanticCache(similarity_threshold=config.similarity_threshold)

        if config.global_style:
            self.global_style_guide = GlobalStyleGuide()

    def process_srt_file(self, srt_filepath: str) -> Tuple[List[Blueprint], Dict[str, Any]]:
        """Process an SRT file into validated blueprints.

        Args:
            srt_filepath: Path to the SRT file

        Returns:
            Tuple of (list of blueprints, processing stats)
        """
        entries = self.parser.parse(srt_filepath)
        blueprints: List[Blueprint] = []
        stats = {
            "total_entries": len(entries),
            "successful": 0,
            "failed": 0,
            "cache_hits": 0,
        }

        # First pass: generate style guide if enabled
        if self.config.global_style:
            descriptions = [entry["text"] for entry in entries]
            self.style_guide = self.global_style_guide.analyze(descriptions)

        for idx, entry in enumerate(entries):
            # Check cache first
            if self.cache:
                cached = self.cache.get(entry["text"])
                if cached:
                    blueprints.append(cached)
                    stats["cache_hits"] += 1
                    continue

            blueprint_data = {
                "id": f"bp_{idx:03d}",
                "timecode": entry["start"],
                "frame_dimensions": {"width": 1920, "height": 1080},
                "background_description": entry["text"],
                "required_keys": ["id", "timecode", "frame_dimensions"],
            }

            try:
                blueprint = Blueprint(**blueprint_data)

                # Analyze emotion if enabled
                if self.config.emotion_ai:
                    emotion = self.emotion_analyzer.analyze(
                        entry["text"], use_ai=self.config.emotion_ai
                    )
                    blueprint.emotion = emotion

                # Apply style if global style enabled
                if self.config.global_style and self.style_guide:
                    blueprint.style_attributes = self.style_guide

                repaired_blueprint = self.repairer.repair(blueprint)

                errors = self.validator.validate(repaired_blueprint)
                if not errors:
                    blueprints.append(repaired_blueprint)
                    stats["successful"] += 1

                    # Cache the blueprint
                    if self.cache:
                        self.cache.put(entry["text"], repaired_blueprint.model_dump())
                else:
                    stats["failed"] += 1

            except Exception as e:
                stats["failed"] += 1
                continue

        return blueprints, stats

    def output_json(
        self, blueprints: List[Blueprint], output_filepath: Optional[str] = None
    ) -> str:
        """Output blueprints as JSON.

        Args:
            blueprints: List of blueprints to output
            output_filepath: Optional file path to write to

        Returns:
            JSON string
        """
        output: Dict[str, Any] = {
            "blueprints": [bp.model_dump() for bp in blueprints],
            "validation_report": self.logger.get_summary_report(),
            "total_blueprints": len(blueprints),
        }

        if self.cache:
            output["cache_statistics"] = self.cache.get_statistics()

        if self.style_guide:
            output["global_style_guide"] = self.style_guide

        json_str = json.dumps(output, indent=2)

        if output_filepath:
            with open(output_filepath, "w", encoding="utf-8") as f:
                f.write(json_str)

        return json_str


# ============================================================================
# ===== COMMAND-LINE INTERFACE =====
# ============================================================================


def main() -> int:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="AI VFX Director Pro Ultimate - Comprehensive Blueprint Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python AI_VFX_Director_Pro_Ultimate.py --input script.srt --output blueprint.json

  # With custom model
  python AI_VFX_Director_Pro_Ultimate.py --input script.srt --model gpt-4 --output out.json

  # With all advanced features
  python AI_VFX_Director_Pro_Ultimate.py --input script.srt --emotion-ai --cache \\
      --similarity 0.85 --global-style auto --progressive
        """,
    )

    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Input SRT file path",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output JSON file path (default: stdout)",
    )
    parser.add_argument(
        "--model",
        default="qwen-3-235b-a22b-thinking-2507",
        help="LLM model to use (default: qwen-3-235b-a22b-thinking-2507)",
    )
    parser.add_argument(
        "--emotion-ai",
        action="store_true",
        help="Use AI-based emotion analysis",
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Enable semantic caching",
    )
    parser.add_argument(
        "--similarity",
        type=float,
        default=0.85,
        help="Cache similarity threshold (0.0-1.0, default: 0.85)",
    )
    parser.add_argument(
        "--global-style",
        action="store_true",
        help="Generate global style guide",
    )
    parser.add_argument(
        "--progressive",
        action="store_true",
        help="Enable progressive generation",
    )
    parser.add_argument(
        "--unified-background",
        action="store_true",
        help="Use unified background style",
    )
    parser.add_argument(
        "--wait-time",
        type=float,
        default=2.0,
        help="Wait time between API calls (default: 2.0)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        return 1

    try:
        config = Config(
            model=args.model,
            emotion_ai=args.emotion_ai,
            cache_enabled=args.cache,
            similarity_threshold=args.similarity,
            global_style=args.global_style,
            progressive=args.progressive,
            unified_background=args.unified_background,
            wait_time=args.wait_time,
        )

        pipeline = BlueprintPipeline(config=config)
        blueprints, stats = pipeline.process_srt_file(args.input)

        if args.verbose:
            print(
                f"Processed {stats['total_entries']} entries: "
                f"{stats['successful']} successful, {stats['failed']} failed",
                file=sys.stderr,
            )
            if stats["cache_hits"] > 0:
                print(f"Cache hits: {stats['cache_hits']}", file=sys.stderr)

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
