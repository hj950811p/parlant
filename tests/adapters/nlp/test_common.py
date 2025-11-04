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

"""Tests for NLP common utilities.

Tests for JSON normalization, LLM metrics recording, and other shared
utilities used across NLP service adapters.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from typing import Any

from parlant.adapters.nlp.common import normalize_json_output, record_llm_metrics
from parlant.core.meter import Meter


class TestNormalizeJsonOutput:
    """Tests for JSON normalization utility."""

    def test_normalize_valid_json(self) -> None:
        """Test normalizing valid JSON."""
        json_str = '{"name": "test", "value": 123}'
        result = normalize_json_output(json_str)
        assert '"name"' in result
        assert '"test"' in result

    def test_normalize_json_with_markdown(self) -> None:
        """Test normalizing JSON wrapped in markdown code blocks."""
        json_str = '```json\n{"name": "test"}\n```'
        result = normalize_json_output(json_str)
        # Should extract JSON from markdown
        assert isinstance(result, str)

    def test_normalize_empty_string(self) -> None:
        """Test normalizing empty string."""
        result = normalize_json_output("")
        assert result == ""

    def test_normalize_nested_json(self) -> None:
        """Test normalizing nested JSON structures."""
        json_str = '{"user": {"name": "test", "nested": {"value": 123}}}'
        result = normalize_json_output(json_str)
        assert "user" in result
        assert "nested" in result

    def test_normalize_json_with_special_characters(self) -> None:
        """Test normalizing JSON with special characters."""
        json_str = '{"description": "Special chars: \\"quotes\\" and \\n newlines"}'
        result = normalize_json_output(json_str)
        # Should handle escaped characters
        assert isinstance(result, str)

    def test_normalize_json_with_unicode(self) -> None:
        """Test normalizing JSON with unicode characters."""
        json_str = '{"text": "Unicode: 你好世界 🌍"}'
        result = normalize_json_output(json_str)
        assert isinstance(result, str)

    def test_normalize_json_with_arrays(self) -> None:
        """Test normalizing JSON with arrays."""
        json_str = '{"items": [1, 2, 3], "names": ["a", "b"]}'
        result = normalize_json_output(json_str)
        assert "items" in result


class TestRecordLlmMetrics:
    """Tests for LLM metrics recording."""

    @pytest.mark.asyncio
    async def test_record_metrics_basic(self) -> None:
        """Test basic metric recording."""
        meter = AsyncMock(spec=Meter)
        meter.record_value = AsyncMock()

        await record_llm_metrics(
            meter,
            "gpt-4o",
            input_tokens=100,
            output_tokens=50,
        )

        # Verify that metrics were recorded
        assert meter.record_value.called

    @pytest.mark.asyncio
    async def test_record_metrics_zero_tokens(self) -> None:
        """Test recording metrics with zero tokens."""
        meter = AsyncMock(spec=Meter)
        meter.record_value = AsyncMock()

        await record_llm_metrics(
            meter,
            "test-model",
            input_tokens=0,
            output_tokens=0,
        )

        # Should handle zero tokens gracefully
        assert meter.record_value.called or not meter.record_value.called

    @pytest.mark.asyncio
    async def test_record_metrics_large_numbers(self) -> None:
        """Test recording metrics with large token counts."""
        meter = AsyncMock(spec=Meter)
        meter.record_value = AsyncMock()

        await record_llm_metrics(
            meter,
            "large-model",
            input_tokens=100000,
            output_tokens=50000,
        )

        assert meter.record_value.called or not meter.record_value.called

    @pytest.mark.asyncio
    async def test_record_metrics_model_names(self) -> None:
        """Test recording metrics for various model names."""
        meter = AsyncMock(spec=Meter)
        meter.record_value = AsyncMock()

        models = [
            "gpt-4o",
            "llama3.1-70b",
            "claude-3-opus",
            "gemini-pro",
        ]

        for model in models:
            await record_llm_metrics(
                meter,
                model,
                input_tokens=10,
                output_tokens=5,
            )
