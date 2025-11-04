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

"""Fixtures for Cerebras service testing.

This module provides reusable fixtures for testing Cerebras NLP service
adapters. These fixtures can be imported in test modules using:

    from tests.conftest_cerebras import mock_logger, mock_meter

Or by adding conftest.py imports in test directories.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock
from pydantic import BaseModel
from typing import Any

from parlant.core.loggers import Logger
from parlant.core.meter import Meter


# ============================================================================
# Logger and Meter Fixtures
# ============================================================================


@pytest.fixture
def mock_logger() -> Logger:
    """Create a mock logger for testing Cerebras services.

    This fixture provides a fully mocked Logger with scope context managers
    to allow testing without actual logging infrastructure.

    Returns:
        A MagicMock configured to behave like a Logger.

    Example:
        @pytest.mark.asyncio
        async def test_generation(mock_logger):
            generator = Llama3_3_8B(mock_logger, mock_meter)
            # Logger methods can be called without errors
            assert generator._logger is not None
    """
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
    """Create a mock meter for testing Cerebras services.

    This fixture provides a fully mocked Meter for recording metrics
    without requiring actual metrics infrastructure.

    Returns:
        A MagicMock configured to behave like a Meter with async methods.

    Example:
        @pytest.mark.asyncio
        async def test_metrics(mock_meter):
            await mock_meter.record_value("key", 100)
            assert mock_meter.record_value.called
    """
    meter = MagicMock(spec=Meter)
    meter.record_value = AsyncMock()
    return meter


# ============================================================================
# Schema Fixtures
# ============================================================================


class MockUserProfile(BaseModel):
    """Mock user profile schema for testing."""
    username: str
    email: str
    age: int
    role: str = "user"


class MockResponseSchema(BaseModel):
    """Mock response schema for testing."""
    success: bool
    message: str
    code: int = 200
    data: dict[str, Any] = {}


@pytest.fixture
def user_profile_schema() -> type[MockUserProfile]:
    """Provide a mock user profile schema for testing.

    Returns:
        The MockUserProfile pydantic model class.
    """
    return MockUserProfile


@pytest.fixture
def response_schema() -> type[MockResponseSchema]:
    """Provide a mock response schema for testing.

    Returns:
        The MockResponseSchema pydantic model class.
    """
    return MockResponseSchema


# ============================================================================
# API Response Fixtures
# ============================================================================


@pytest.fixture
def mock_api_response_success() -> Any:
    """Create a successful mock API response for generation.

    Returns:
        A mock response object with valid generation output.

    Example:
        @pytest.mark.asyncio
        async def test_generation(generator, mock_api_response_success):
            generator._client = AsyncMock()
            generator._client.chat.completions.create = AsyncMock(
                return_value=mock_api_response_success
            )
    """
    from unittest.mock import Mock

    response = Mock()
    response.choices = [Mock()]
    response.choices[0].message = Mock()
    response.choices[0].message.content = '''{
        "username": "test_user",
        "email": "test@example.com",
        "age": 30,
        "role": "admin"
    }'''
    response.usage = Mock()
    response.usage.prompt_tokens = 50
    response.usage.completion_tokens = 30
    return response


@pytest.fixture
def mock_api_response_empty_content() -> Any:
    """Create a mock API response with empty content.

    Used to test error handling when API returns no content.

    Returns:
        A mock response object with None content.
    """
    from unittest.mock import Mock

    response = Mock()
    response.choices = [Mock()]
    response.choices[0].message = Mock()
    response.choices[0].message.content = None
    return response


@pytest.fixture
def mock_api_response_invalid_json() -> Any:
    """Create a mock API response with invalid JSON content.

    Used to test JSON parsing error handling.

    Returns:
        A mock response object with malformed JSON.
    """
    from unittest.mock import Mock

    response = Mock()
    response.choices = [Mock()]
    response.choices[0].message = Mock()
    response.choices[0].message.content = "This is not valid JSON"
    response.usage = Mock()
    response.usage.prompt_tokens = 50
    response.usage.completion_tokens = 10
    return response


# ============================================================================
# Cerebras Error Fixtures
# ============================================================================


@pytest.fixture
def rate_limit_error():
    """Provide a RateLimitError for testing.

    Returns:
        A RateLimitError instance.

    Example:
        @pytest.mark.asyncio
        async def test_rate_limit(generator, rate_limit_error):
            generator._client = AsyncMock()
            generator._client.chat.completions.create = AsyncMock(
                side_effect=rate_limit_error
            )
            with pytest.raises(Exception):
                await generator._do_generate("prompt")
    """
    from cerebras.cloud.sdk import RateLimitError

    return RateLimitError("Rate limit exceeded")


@pytest.fixture
def connection_error():
    """Provide an APIConnectionError for testing.

    Returns:
        An APIConnectionError instance.
    """
    from cerebras.cloud.sdk import APIConnectionError

    return APIConnectionError("Connection failed")


@pytest.fixture
def timeout_error():
    """Provide an APITimeoutError for testing.

    Returns:
        An APITimeoutError instance.
    """
    from cerebras.cloud.sdk import APITimeoutError

    return APITimeoutError("Request timed out")


@pytest.fixture
def internal_server_error():
    """Provide an InternalServerError for testing.

    Returns:
        An InternalServerError instance.
    """
    from cerebras.cloud.sdk import InternalServerError

    return InternalServerError("Internal server error")
