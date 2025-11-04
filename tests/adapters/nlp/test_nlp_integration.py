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

"""Integration tests for NLP service adapters.

These tests verify end-to-end functionality of NLP services with mocked
external API calls, ensuring proper integration between components. They
test complete pipelines including service initialization, generation,
error handling, and metrics recording.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, Mock
from pydantic import BaseModel
from typing import Any

from parlant.adapters.nlp.cerebras_service import (
    CerebrasService,
    Llama3_3_70B,
)
from parlant.core.nlp.generation_info import GenerationInfo
from parlant.core.loggers import Logger
from parlant.core.meter import Meter
from parlant.core.engines.alpha.prompt_builder import PromptBuilder


class UserProfile(BaseModel):
    """Sample user profile schema for integration testing."""
    username: str
    email: str
    age: int
    role: str


class SampleResponse(BaseModel):
    """Sample response schema for integration testing."""
    success: bool
    message: str
    data: dict[str, Any] = {}


@pytest.fixture
def mock_logger() -> Logger:
    """Create a mock logger for integration testing."""
    logger = MagicMock(spec=Logger)
    logger.scope = MagicMock()
    logger.scope.return_value.__enter__ = MagicMock(return_value=None)
    logger.scope.return_value.__exit__ = MagicMock(return_value=None)
    logger.info = MagicMock()
    logger.error = MagicMock()
    logger.debug = MagicMock()
    logger.trace = MagicMock()
    return logger


@pytest.fixture
def mock_meter() -> Meter:
    """Create a mock meter for integration testing."""
    meter = MagicMock(spec=Meter)
    meter.record_value = AsyncMock()
    return meter


class TestCerebrasIntegration:
    """Integration tests for Cerebras service."""

    @pytest.mark.asyncio
    async def test_end_to_end_generation_with_user_profile(
        self, mock_logger: Logger, mock_meter: Meter
    ) -> None:
        """Test end-to-end generation of user profile schema."""
        service = CerebrasService(mock_logger, mock_meter)
        generator = await service.get_schematic_generator(UserProfile)

        # Mock API response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = '''{
            "username": "john_doe",
            "email": "john@example.com",
            "age": 30,
            "role": "admin"
        }'''
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 50
        mock_response.usage.completion_tokens = 30

        generator._client = AsyncMock()
        generator._client.chat.completions.create = AsyncMock(return_value=mock_response)

        prompt = "Create a user profile"
        result = await generator._do_generate(prompt)

        assert result.content.username == "john_doe"
        assert result.content.email == "john@example.com"
        assert result.content.age == 30
        assert result.content.role == "admin"
        assert result.info.usage.input_tokens == 50
        assert result.info.usage.output_tokens == 30

    @pytest.mark.asyncio
    async def test_end_to_end_generation_with_prompt_builder(
        self, mock_logger: Logger, mock_meter: Meter
    ) -> None:
        """Test generation using PromptBuilder."""
        service = CerebrasService(mock_logger, mock_meter)
        generator = await service.get_schematic_generator(SampleResponse)

        # Mock API response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = '''{
            "success": true,
            "message": "Operation completed successfully",
            "data": {"result": "value"}
        }'''
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 40
        mock_response.usage.completion_tokens = 25

        generator._client = AsyncMock()
        generator._client.chat.completions.create = AsyncMock(return_value=mock_response)

        # Build prompt
        prompt_builder = PromptBuilder()
        prompt_builder.add("Generate a success response")

        result = await generator._do_generate(prompt_builder)

        assert result.content.success is True
        assert result.content.message == "Operation completed successfully"
        assert "result" in result.content.data

    @pytest.mark.asyncio
    async def test_error_handling_rate_limit(
        self, mock_logger: Logger, mock_meter: Meter
    ) -> None:
        """Test error handling for rate limit errors."""
        from cerebras.cloud.sdk import RateLimitError

        generator = Llama3_3_70B(mock_logger, mock_meter)

        # Mock rate limit error
        generator._client = AsyncMock()
        generator._client.chat.completions.create = AsyncMock(
            side_effect=RateLimitError("Rate limit exceeded")
        )

        with pytest.raises(Exception):
            await generator._do_generate("test prompt")

    @pytest.mark.asyncio
    async def test_error_handling_api_connection_error(
        self, mock_logger: Logger, mock_meter: Meter
    ) -> None:
        """Test error handling for connection errors."""
        from cerebras.cloud.sdk import APIConnectionError

        generator = Llama3_3_70B(mock_logger, mock_meter)

        # Mock connection error
        generator._client = AsyncMock()
        generator._client.chat.completions.create = AsyncMock(
            side_effect=APIConnectionError("Connection failed")
        )

        with pytest.raises(Exception):
            await generator._do_generate("test prompt")

    @pytest.mark.asyncio
    async def test_generation_with_custom_hints(
        self, mock_logger: Logger, mock_meter: Meter
    ) -> None:
        """Test generation with custom temperature hints."""
        service = CerebrasService(mock_logger, mock_meter)
        generator = await service.get_schematic_generator(SampleResponse)

        # Mock API response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = '''{
            "success": true,
            "message": "Generated with temperature"
        }'''
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 30
        mock_response.usage.completion_tokens = 20

        generator._client = AsyncMock()
        generator._client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await generator._do_generate(
            "Generate response",
            {"temperature": 0.8},
        )

        assert result.content.success is True

    @pytest.mark.asyncio
    async def test_generation_with_json_schema_validation(
        self, mock_logger: Logger, mock_meter: Meter
    ) -> None:
        """Test that generated content validates against schema."""
        service = CerebrasService(mock_logger, mock_meter)
        generator = await service.get_schematic_generator(UserProfile)

        # Mock API response with complete schema
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = '''{
            "username": "test_user",
            "email": "test@test.com",
            "age": 25,
            "role": "user"
        }'''
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 50
        mock_response.usage.completion_tokens = 30

        generator._client = AsyncMock()
        generator._client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await generator._do_generate("Generate user")

        # Verify all required fields are present
        assert hasattr(result.content, "username")
        assert hasattr(result.content, "email")
        assert hasattr(result.content, "age")
        assert hasattr(result.content, "role")

    @pytest.mark.asyncio
    async def test_service_initialization_and_components(
        self, mock_logger: Logger, mock_meter: Meter
    ) -> None:
        """Test that service properly initializes all components."""
        service = CerebrasService(mock_logger, mock_meter)

        # Get all components
        generator = await service.get_schematic_generator(SampleResponse)
        embedder = await service.get_embedder()
        moderation = await service.get_moderation_service()

        # Verify all components are properly initialized
        assert generator is not None
        assert embedder is not None
        assert moderation is not None

    @pytest.mark.asyncio
    async def test_multiple_sequential_generations(
        self, mock_logger: Logger, mock_meter: Meter
    ) -> None:
        """Test multiple sequential generation calls."""
        service = CerebrasService(mock_logger, mock_meter)
        generator = await service.get_schematic_generator(SampleResponse)

        # Mock API responses
        responses = [
            '''{
                "success": true,
                "message": "First generation"
            }''',
            '''{
                "success": true,
                "message": "Second generation"
            }''',
            '''{
                "success": true,
                "message": "Third generation"
            }''',
        ]

        generator._client = AsyncMock()

        for response_content in responses:
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message = Mock()
            mock_response.choices[0].message.content = response_content
            mock_response.usage = Mock()
            mock_response.usage.prompt_tokens = 30
            mock_response.usage.completion_tokens = 20

            generator._client.chat.completions.create = AsyncMock(
                return_value=mock_response
            )

            result = await generator._do_generate("Generate response")
            assert result.content.success is True
            assert result.content.message in [
                "First generation",
                "Second generation",
                "Third generation",
            ]
