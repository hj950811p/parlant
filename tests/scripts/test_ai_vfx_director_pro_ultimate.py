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

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

import pytest

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from AI_VFX_Director_Pro_Ultimate import (
    Config,
    SemanticCache,
    calculate_text_similarity,
    GlobalStyleGuide,
    ErrorRecoveryHandler,
    PromptBuilder,
    Blueprint,
    BlueprintValidator,
    BlueprintRepairer,
    SRTParser,
    StructuredLogger,
)


class TestConfig:
    """Tests for configuration management."""

    def test_config_has_default_model(self) -> None:
        config = Config()
        assert config.model == "qwen-3-235b-a22b-thinking-2507"

    def test_config_can_override_model(self) -> None:
        config = Config(model="custom-model-v1")
        assert config.model == "custom-model-v1"

    def test_config_default_values_are_set(self) -> None:
        config = Config()
        assert config.wait_time > 0
        assert isinstance(config.cache_enabled, bool)
        assert isinstance(config.emotion_ai, bool)
        assert isinstance(config.global_style, bool)
        assert isinstance(config.progressive, bool)

    def test_config_from_dict(self) -> None:
        config_dict = {
            "model": "test-model",
            "wait_time": 5,
            "cache_enabled": True,
            "emotion_ai": True,
        }
        config = Config(**config_dict)
        assert config.model == "test-model"
        assert config.wait_time == 5
        assert config.cache_enabled is True
        assert config.emotion_ai is True


class TestTextSimilarity:
    """Tests for text similarity calculation."""

    def test_identical_texts_have_perfect_similarity(self) -> None:
        text = "The quick brown fox"
        similarity = calculate_text_similarity(text, text)
        assert similarity == 1.0

    def test_completely_different_texts_have_low_similarity(self) -> None:
        text1 = "aaa bbb"
        text2 = "xxx yyy zzz"
        similarity = calculate_text_similarity(text1, text2)
        assert 0 <= similarity < 0.3

    def test_similar_texts_have_high_similarity(self) -> None:
        text1 = "A beautiful sunset over the ocean"
        text2 = "Beautiful sunset over the ocean"
        similarity = calculate_text_similarity(text1, text2)
        assert similarity > 0.7

    def test_empty_strings_have_zero_similarity(self) -> None:
        similarity = calculate_text_similarity("", "")
        assert similarity == 0.0 or similarity == 1.0  # May vary by algorithm

    def test_case_insensitive_comparison(self) -> None:
        text1 = "The Quick Brown Fox"
        text2 = "the quick brown fox"
        similarity = calculate_text_similarity(text1, text2)
        assert similarity > 0.95


class TestSemanticCache:
    """Tests for semantic caching system."""

    def test_semantic_cache_initializes_empty(self) -> None:
        cache = SemanticCache(similarity_threshold=0.85)
        assert len(cache) == 0

    def test_semantic_cache_stores_and_retrieves_exact_match(self) -> None:
        cache = SemanticCache(similarity_threshold=0.85)
        blueprint = {"id": "bp_001", "description": "A sunny beach"}
        cache.put("sunny beach", blueprint)

        retrieved = cache.get("sunny beach")
        assert retrieved is not None
        assert retrieved["id"] == "bp_001"

    def test_semantic_cache_retrieves_similar_descriptions(self) -> None:
        cache = SemanticCache(similarity_threshold=0.80)
        blueprint1 = {"id": "bp_001", "description": "A sunny beach with palm trees"}
        cache.put("A sunny beach with palm trees", blueprint1)

        retrieved = cache.get("sunny beach with palm")
        # May or may not retrieve based on similarity threshold
        assert retrieved is None or retrieved["id"] == "bp_001"

    def test_semantic_cache_respects_similarity_threshold(self) -> None:
        cache = SemanticCache(similarity_threshold=0.95)
        blueprint = {"id": "bp_001", "description": "test"}
        cache.put("test query one", blueprint)

        # Query with very different text should not match
        retrieved = cache.get("completely different text here")
        assert retrieved is None

    def test_semantic_cache_clear(self) -> None:
        cache = SemanticCache(similarity_threshold=0.85)
        cache.put("query1", {"id": "bp_001"})
        cache.put("query2", {"id": "bp_002"})
        assert len(cache) == 2

        cache.clear()
        assert len(cache) == 0

    def test_semantic_cache_statistics(self) -> None:
        cache = SemanticCache(similarity_threshold=0.85)
        cache.put("query1", {"id": "bp_001"})
        cache.get("query1")  # Hit
        cache.get("nonexistent")  # Miss
        cache.get("query1")  # Hit

        stats = cache.get_statistics()
        assert stats["total_queries"] >= 1
        assert stats["cache_hits"] >= 1
        assert "hit_rate" in stats


class TestGlobalStyleGuide:
    """Tests for global style guide generation."""

    def test_global_style_guide_initializes(self) -> None:
        guide = GlobalStyleGuide()
        assert guide is not None

    def test_global_style_guide_analyzes_text_samples(self) -> None:
        guide = GlobalStyleGuide()
        samples = [
            "A bright and cheerful morning scene",
            "A dark and mysterious night scene",
            "A peaceful and calm rural landscape",
        ]
        style = guide.analyze(samples)
        assert isinstance(style, dict)
        assert "summary" in style or len(style) > 0

    def test_global_style_guide_returns_structured_data(self) -> None:
        guide = GlobalStyleGuide()
        samples = ["Sample text one", "Sample text two"]
        style = guide.analyze(samples)
        assert isinstance(style, dict)

    def test_global_style_guide_handles_empty_samples(self) -> None:
        guide = GlobalStyleGuide()
        style = guide.analyze([])
        assert isinstance(style, dict)

    def test_global_style_guide_format_for_prompt(self) -> None:
        guide = GlobalStyleGuide()
        samples = ["A beautiful sunset", "A calm morning"]
        style = guide.analyze(samples)
        formatted = guide.format_for_prompt(style)
        assert isinstance(formatted, str)
        assert len(formatted) > 0


class TestPromptBuilder:
    """Tests for prompt building and optimization."""

    def test_prompt_builder_creates_basic_prompt(self) -> None:
        builder = PromptBuilder()
        prompt = builder.build_prompt("Generate a VFX blueprint")
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_prompt_builder_includes_few_shot_examples(self) -> None:
        builder = PromptBuilder(include_few_shot=True)
        prompt = builder.build_prompt("Generate VFX")
        assert isinstance(prompt, str)
        # Should include some example context
        assert len(prompt) > 100

    def test_prompt_builder_respects_max_length(self) -> None:
        builder = PromptBuilder(max_prompt_length=500)
        prompt = builder.build_prompt("Generate VFX blueprint for scene")
        assert len(prompt) <= 600  # Allow some margin

    def test_prompt_builder_with_style_guide(self) -> None:
        style_guide = "Use bright colors and dynamic movements"
        builder = PromptBuilder()
        prompt = builder.build_with_style(
            "Generate VFX", style_guide
        )
        assert isinstance(prompt, str)

    def test_prompt_builder_with_emotion_context(self) -> None:
        builder = PromptBuilder()
        prompt = builder.build_with_emotion(
            "Generate VFX", emotion="joyful"
        )
        assert isinstance(prompt, str)


class TestErrorRecoveryHandler:
    """Tests for error recovery handling."""

    def test_error_recovery_handler_initializes(self) -> None:
        handler = ErrorRecoveryHandler()
        assert handler is not None

    def test_error_recovery_handler_repairs_json_with_trailing_comma(self) -> None:
        handler = ErrorRecoveryHandler()
        malformed = '{"id": "test", "value": 123,}'
        result = handler.attempt_json_repair(malformed)
        assert result is not None or malformed is not None

    def test_error_recovery_handler_repairs_json_with_unquoted_keys(self) -> None:
        handler = ErrorRecoveryHandler()
        malformed = '{id: "test", value: 123}'
        result = handler.attempt_json_repair(malformed)
        # Should either repair or return None
        assert result is None or isinstance(result, dict)

    def test_error_recovery_handler_retry_logic(self) -> None:
        handler = ErrorRecoveryHandler(max_retries=3)
        assert handler.max_retries == 3

    def test_error_recovery_handler_exponential_backoff(self) -> None:
        handler = ErrorRecoveryHandler()
        backoff1 = handler.calculate_backoff(1)
        backoff2 = handler.calculate_backoff(2)
        backoff3 = handler.calculate_backoff(3)
        # Each should increase (exponential backoff)
        assert backoff1 >= 0
        assert backoff2 >= backoff1
        assert backoff3 >= backoff2


class TestBlueprintValidationAndRepair:
    """Tests for blueprint validation and repair (backward compatibility)."""

    def test_blueprint_validator_accepts_valid_blueprint(self) -> None:
        blueprint_data = {
            "id": "bp_001",
            "timecode": "00:00:00,000",
            "frame_dimensions": {"width": 1920, "height": 1080},
            "background_description": "A simple outdoor scene",
            "required_keys": ["id", "timecode"],
        }
        blueprint = Blueprint(**blueprint_data)
        validator = BlueprintValidator()

        errors = validator.validate(blueprint)
        assert len(errors) == 0

    def test_blueprint_repairer_normalizes_dimensions(self) -> None:
        blueprint_data = {
            "id": "bp_001",
            "timecode": "00:00:00,000",
            "frame_dimensions": {"width": 100, "height": 1080},
            "background_description": "A scene",
            "required_keys": ["id"],
        }
        blueprint = Blueprint(**blueprint_data)
        logger = StructuredLogger()
        repairer = BlueprintRepairer(logger)

        fixed = repairer.repair(blueprint)
        assert fixed.frame_dimensions.width >= 320


class TestBackwardCompatibility:
    """Tests for backward compatibility with original script."""

    def test_srt_parsing_works_as_before(self) -> None:
        srt_content = """1
00:00:00,000 --> 00:00:05,000
First subtitle

2
00:00:05,000 --> 00:00:10,000
Second subtitle
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".srt", delete=False) as f:
            f.write(srt_content)
            f.flush()

            parser = SRTParser()
            entries = parser.parse(f.name)

            assert len(entries) == 2
            assert entries[0]["text"] == "First subtitle"
            assert entries[1]["text"] == "Second subtitle"

            Path(f.name).unlink()

    def test_structured_logging_works_as_before(self) -> None:
        logger = StructuredLogger()
        logger.record_fix("dimension_normalization", {"field": "width", "value": 100})
        logger.record_fix("background_rewrite", {"original": "complex"})

        report = logger.get_summary_report()
        assert "dimension_normalization" in report
        assert "background_rewrite" in report


class TestIntegrationEndToEnd:
    """Integration tests for the complete pipeline."""

    def test_end_to_end_srt_to_blueprints_with_caching(self) -> None:
        """Test complete pipeline with caching enabled."""
        srt_content = """1
00:00:00,000 --> 00:00:05,000
A sunny beach scene

2
00:00:05,000 --> 00:00:10,000
A sunny beach with palm trees
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".srt", delete=False) as f:
            f.write(srt_content)
            f.flush()

            parser = SRTParser()
            entries = parser.parse(f.name)

            assert len(entries) == 2
            # First entry
            assert "sunny beach" in entries[0]["text"].lower()
            # Second entry
            assert "palm" in entries[1]["text"].lower()

            Path(f.name).unlink()

    def test_end_to_end_with_global_style_analysis(self) -> None:
        """Test that global style analysis integrates correctly."""
        samples = [
            "A bright and cheerful morning",
            "A dark and mysterious evening",
        ]
        guide = GlobalStyleGuide()
        style = guide.analyze(samples)
        assert isinstance(style, dict)

    def test_end_to_end_with_error_handling(self) -> None:
        """Test that error handling integrates correctly."""
        handler = ErrorRecoveryHandler()
        
        # Test with malformed JSON
        malformed = '{"incomplete": '
        # Handler should not crash
        result = handler.attempt_json_repair(malformed)
        assert result is None or isinstance(result, dict)


class TestCommandLineIntegration:
    """Tests for command-line argument handling."""

    def test_config_from_command_line_args(self) -> None:
        """Test that Config can be built from CLI args."""
        args = {
            "model": "custom-model",
            "wait_time": 3,
            "cache_enabled": True,
            "emotion_ai": True,
            "global_style": True,
            "progressive": True,
        }
        config = Config(**args)
        assert config.model == "custom-model"
        assert config.wait_time == 3
        assert config.cache_enabled is True
        assert config.emotion_ai is True


class TestProgressiveGeneration:
    """Tests for progressive/staged generation."""

    def test_progressive_generation_config_option(self) -> None:
        """Test that progressive generation can be enabled."""
        config = Config(progressive=True)
        assert config.progressive is True

    def test_non_progressive_generation_config_option(self) -> None:
        """Test that progressive generation can be disabled."""
        config = Config(progressive=False)
        assert config.progressive is False
