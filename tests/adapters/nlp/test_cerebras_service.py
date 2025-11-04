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

"""Unit tests for Cerebras NLP service adapter.

This module provides comprehensive unit tests for the Cerebras service,
including tokenization, schema-based generation, error handling, and model
configuration. Tests use mocked API responses to avoid external dependencies.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, Mock
from pydantic import BaseModel, ValidationError
from typing import Any

from parlant.adapters.nlp.cerebras_service import (
    CerebrasSchematicGenerator,
    Llama3_3_8B,
    Llama3_3_70B,
    CerebrasService,
    LlamaEstimatingTokenizer,
)
from parlant.core.nlp.generation_info import GenerationInfo, UsageInfo
from parlant.core.loggers import Logger
from parlant.core.meter import Meter
from parlant.core.engines.alpha.prompt_builder import PromptBuilder


class SampleSchema(BaseModel):
    """Sample schema for testing."""
    name: str
    age: int
    description: str = "default"


@pytest.fixture
def mock_logger() -> Logger:
    """Create a mock logger for testing."""
    logger = MagicMock(spec=Logger)
    logger.scope = MagicMock()
    logger.scope.return_value.__enter__ = MagicMock(return_value=None)
    logger.scope.return_value.__exit__ = MagicMock(return_value=None)
    return logger


@pytest.fixture
def mock_meter() -> Meter:
    """Create a mock meter for testing."""
    return MagicMock(spec=Meter)


class TestLlamaEstimatingTokenizer:
    """Tests for LlamaEstimatingTokenizer."""

    @pytest.mark.asyncio
    async def test_tokenizer_initialization(self) -> None:
        """Test that tokenizer initializes correctly."""
        tokenizer = LlamaEstimatingTokenizer()
        assert tokenizer.encoding is not None

    @pytest.mark.asyncio
    async def test_estimate_token_count_empty_string(self) -> None:
        """Test token count estimation for empty string."""
        tokenizer = LlamaEstimatingTokenizer()
        count = await tokenizer.estimate_token_count("")
        assert count > 0  # Should still return some minimum count

    @pytest.mark.asyncio
    async def test_estimate_token_count_short_text(self) -> None:
        """Test token count estimation for short text."""
        tokenizer = LlamaEstimatingTokenizer()
        text = "Hello, world!"
        count = await tokenizer.estimate_token_count(text)
        assert count > 0
        assert isinstance(count, int)

    @pytest.mark.asyncio
    async def test_estimate_token_count_long_text(self) -> None:
        """Test token count estimation for longer text."""
        tokenizer = LlamaEstimatingTokenizer()
        text = "This is a longer piece of text. " * 100
        count = await tokenizer.estimate_token_count(text)
        assert count > 0
        assert isinstance(count, int)

    @pytest.mark.asyncio
    async def test_estimate_token_count_consistency(self) -> None:
        """Test that token estimation is consistent for same text."""
        tokenizer = LlamaEstimatingTokenizer()
        text = "Consistent text for testing"
        count1 = await tokenizer.estimate_token_count(text)
        count2 = await tokenizer.estimate_token_count(text)
        assert count1 == count2


class TestCerebrasSchematicGenerator:
    """Tests for CerebrasSchematicGenerator."""

    def test_initialization(self, mock_logger: Logger, mock_meter: Meter) -> None:
        """Test generator initialization."""
        generator = CerebrasSchematicGenerator(
            model_name="llama3.1-8b",
            logger=mock_logger,
            meter=mock_meter,
        )
        assert generator.model_name == "llama3.1-8b"
        assert generator._logger == mock_logger
        assert generator._meter == mock_meter

    def test_supported_hints(self, mock_logger: Logger, mock_meter: Meter) -> None:
        """Test that supported hints are defined."""
        generator = CerebrasSchematicGenerator(
            model_name="llama3.1-8b",
            logger=mock_logger,
            meter=mock_meter,
        )
        assert "temperature" in generator.supported_hints

    @pytest.mark.asyncio
    async def test_do_generate_with_string_prompt(
        self, mock_logger: Logger, mock_meter: Meter
    ) -> None:
        """Test generation with string prompt."""
        generator = CerebrasSchematicGenerator[SampleSchema](
            model_name="llama3.1-8b",
            logger=mock_logger,
            meter=mock_meter,
        )
        generator.schema = SampleSchema

        # Mock the client response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = '{"name": "John", "age": 30}'
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5

        generator._client = AsyncMock()
        generator._client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await generator._do_generate(
            "Generate a person",
            {"temperature": 0.5},
        )

        assert result.content.name == "John"
        assert result.content.age == 30
        assert result.info.usage.input_tokens == 10
        assert result.info.usage.output_tokens == 5

    @pytest.mark.asyncio
    async def test_do_generate_with_prompt_builder(
        self, mock_logger: Logger, mock_meter: Meter
    ) -> None:
        """Test generation with PromptBuilder."""
        generator = CerebrasSchematicGenerator[SampleSchema](
            model_name="llama3.1-8b",
            logger=mock_logger,
            meter=mock_meter,
        )
        generator.schema = SampleSchema

        prompt_builder = PromptBuilder()
        prompt_builder.add("Generate a person")

        # Mock the client response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = '{"name": "Jane", "age": 25}'
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5

        generator._client = AsyncMock()
        generator._client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await generator._do_generate(prompt_builder, {})

        assert result.content.name == "Jane"
        assert result.content.age == 25

    @pytest.mark.asyncio
    async def test_do_generate_handles_missing_content(
        self, mock_logger: Logger, mock_meter: Meter
    ) -> None:
        """Test that generator handles missing content in response."""
        generator = CerebrasSchematicGenerator[SampleSchema](
            model_name="llama3.1-8b",
            logger=mock_logger,
            meter=mock_meter,
        )
        generator.schema = SampleSchema

        # Mock response with no content
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = None

        generator._client = AsyncMock()
        generator._client.chat.completions.create = AsyncMock(return_value=mock_response)

        with pytest.raises(Exception):
            await generator._do_generate("Generate a person")

    @pytest.mark.asyncio
    async def test_do_generate_handles_invalid_json(
        self, mock_logger: Logger, mock_meter: Meter
    ) -> None:
        """Test that generator handles invalid JSON in response."""
        generator = CerebrasSchematicGenerator[SampleSchema](
            model_name="llama3.1-8b",
            logger=mock_logger,
            meter=mock_meter,
        )
        generator.schema = SampleSchema

        # Mock response with invalid JSON
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Invalid JSON"

        generator._client = AsyncMock()
        generator._client.chat.completions.create = AsyncMock(return_value=mock_response)

        with pytest.raises(Exception):
            await generator._do_generate("Generate a person")

    @pytest.mark.asyncio
    async def test_do_generate_handles_validation_error(
        self, mock_logger: Logger, mock_meter: Meter
    ) -> None:
        """Test that generator handles validation errors."""
        generator = CerebrasSchematicGenerator[SampleSchema](
            model_name="llama3.1-8b",
            logger=mock_logger,
            meter=mock_meter,
        )
        generator.schema = SampleSchema

        # Mock response with JSON that doesn't match schema
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = '{"invalid": "data"}'

        generator._client = AsyncMock()
        generator._client.chat.completions.create = AsyncMock(return_value=mock_response)

        with pytest.raises(Exception):
            await generator._do_generate("Generate a person")

    @pytest.mark.asyncio
    async def test_do_generate_passes_hints(
        self, mock_logger: Logger, mock_meter: Meter
    ) -> None:
        """Test that supported hints are passed to the API."""
        generator = CerebrasSchematicGenerator[SampleSchema](
            model_name="llama3.1-8b",
            logger=mock_logger,
            meter=mock_meter,
        )
        generator.schema = SampleSchema

        # Mock the client response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = '{"name": "John", "age": 30}'
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5

        generator._client = AsyncMock()
        generator._client.chat.completions.create = AsyncMock(return_value=mock_response)

        await generator._do_generate("Generate", {"temperature": 0.7})

        # Verify that create was called with temperature
        call_kwargs = generator._client.chat.completions.create.call_args[1]
        assert call_kwargs.get("temperature") == 0.7


class TestLlama3_3_8B:
    """Tests for Llama3.3 8B model."""

    def test_initialization(self, mock_logger: Logger, mock_meter: Meter) -> None:
        """Test model initialization."""
        model = Llama3_3_8B(mock_logger, mock_meter)
        assert model.model_name == "llama3.1-8b"

    def test_id_property(self, mock_logger: Logger, mock_meter: Meter) -> None:
        """Test that model ID is correctly set."""
        model = Llama3_3_8B(mock_logger, mock_meter)
        assert model.id == "llama3.1-8b"

    def test_max_tokens_property(self, mock_logger: Logger, mock_meter: Meter) -> None:
        """Test that max_tokens is correctly set."""
        model = Llama3_3_8B(mock_logger, mock_meter)
        assert model.max_tokens == 8192

    def test_tokenizer_property(self, mock_logger: Logger, mock_meter: Meter) -> None:
        """Test that tokenizer is available."""
        model = Llama3_3_8B(mock_logger, mock_meter)
        assert model.tokenizer is not None


class TestLlama3_3_70B:
    """Tests for Llama3.3 70B model."""

    def test_initialization(self, mock_logger: Logger, mock_meter: Meter) -> None:
        """Test model initialization."""
        model = Llama3_3_70B(mock_logger, mock_meter)
        assert model.model_name == "llama3.3-70b"

    def test_id_property(self, mock_logger: Logger, mock_meter: Meter) -> None:
        """Test that model ID is correctly set."""
        model = Llama3_3_70B(mock_logger, mock_meter)
        assert model.id == "llama3.3-70b"

    def test_max_tokens_property(self, mock_logger: Logger, mock_meter: Meter) -> None:
        """Test that max_tokens is correctly set."""
        model = Llama3_3_70B(mock_logger, mock_meter)
        assert model.max_tokens == 32 * 1024


class TestCerebrasService:
    """Tests for CerebrasService."""

    def test_initialization(self, mock_logger: Logger, mock_meter: Meter) -> None:
        """Test service initialization."""
        service = CerebrasService(mock_logger, mock_meter)
        assert service._logger == mock_logger
        assert service._meter == mock_meter

    @pytest.mark.asyncio
    async def test_get_schematic_generator(
        self, mock_logger: Logger, mock_meter: Meter
    ) -> None:
        """Test getting a schematic generator."""
        service = CerebrasService(mock_logger, mock_meter)
        generator = await service.get_schematic_generator(SampleSchema)
        assert isinstance(generator, CerebrasSchematicGenerator)

    @pytest.mark.asyncio
    async def test_get_embedder(self, mock_logger: Logger, mock_meter: Meter) -> None:
        """Test getting an embedder."""
        service = CerebrasService(mock_logger, mock_meter)
        embedder = await service.get_embedder()
        assert embedder is not None

    @pytest.mark.asyncio
    async def test_get_moderation_service(
        self, mock_logger: Logger, mock_meter: Meter
    ) -> None:
        """Test getting a moderation service."""
        service = CerebrasService(mock_logger, mock_meter)
        moderation = await service.get_moderation_service()
        assert moderation is not None

    def test_verify_environment_missing_key(self, monkeypatch: Any) -> None:
        """Test environment verification when API key is missing."""
        monkeypatch.delenv("CEREBRAS_API_KEY", raising=False)
        error_message = CerebrasService.verify_environment()
        assert error_message is not None
        assert "CEREBRAS_API_KEY" in error_message

    def test_verify_environment_with_key(self, monkeypatch: Any) -> None:
        """Test environment verification when API key is present."""
        monkeypatch.setenv("CEREBRAS_API_KEY", "test-key-12345")
        error_message = CerebrasService.verify_environment()
        assert error_message is None
