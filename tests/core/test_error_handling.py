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

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, patch

from parlant.core.errors import (
    ErrorCategory,
    JSONParsingError,
    ValidationError,
    NetworkError,
    RateLimitError,
    InternalServerError,
    TimeoutError,
    ResourceExhaustedError,
    FailureContext,
    ParlantError,
)
from parlant.core.error_handling import (
    categorize_error,
    ErrorHandler,
    ErrorRecoveryResult,
    JSONErrorStrategy,
    ValidationErrorStrategy,
    RateLimitErrorStrategy,
    NetworkErrorStrategy,
    TimeoutErrorStrategy,
    DefaultErrorStrategy,
)
from parlant.core.loggers import Logger
from lagom import Container


@pytest.fixture
def container() -> Container:
    from tests.conftest import build_container
    return build_container()


@pytest.fixture
def logger(container: Container) -> Logger:
    return container[Logger]


class TestErrorCategorization:
    """Test error categorization logic."""

    async def test_that_json_decode_error_is_categorized_as_json_parsing(self) -> None:
        error = json.JSONDecodeError("test", "doc", 0)
        category = categorize_error(error)
        assert category == ErrorCategory.JSON_PARSING

    async def test_that_validation_error_is_categorized_as_validation(self) -> None:
        error = ValidationError("Validation failed")
        category = categorize_error(error)
        assert category == ErrorCategory.VALIDATION

    async def test_that_value_error_with_json_text_is_categorized_as_json_parsing(self) -> None:
        error = ValueError("Failed to parse JSON")
        category = categorize_error(error)
        assert category == ErrorCategory.JSON_PARSING

    async def test_that_type_error_is_categorized_as_validation(self) -> None:
        error = TypeError("Expected string, got int")
        category = categorize_error(error)
        assert category == ErrorCategory.VALIDATION

    async def test_that_connection_error_is_categorized_as_network(self) -> None:
        error = ConnectionError("Connection refused")
        category = categorize_error(error)
        assert category == ErrorCategory.NETWORK

    async def test_that_asyncio_timeout_error_is_categorized_as_timeout(self) -> None:
        error = asyncio.TimeoutError("Request timed out")
        category = categorize_error(error)
        assert category == ErrorCategory.TIMEOUT

    async def test_that_error_with_token_limit_text_is_categorized_as_resource_exhausted(self) -> None:
        error = Exception("Token limit exceeded")
        category = categorize_error(error)
        assert category == ErrorCategory.RESOURCE_EXHAUSTED

    async def test_that_error_with_rate_limit_text_is_categorized_as_rate_limit(self) -> None:
        error = Exception("Rate limit exceeded")
        category = categorize_error(error)
        assert category == ErrorCategory.RATE_LIMIT

    async def test_that_unknown_error_is_categorized_as_unknown(self) -> None:
        error = RuntimeError("Some random error")
        category = categorize_error(error)
        assert category == ErrorCategory.UNKNOWN


class TestJSONErrorStrategy:
    """Test JSON error recovery strategy."""

    async def test_that_valid_json_string_is_recovered(self, logger: Logger) -> None:
        strategy = JSONErrorStrategy(logger)
        error = json.JSONDecodeError("test", "doc", 0)
        context = {
            "malformed_input": '{"key": "value"}'
        }

        result = await strategy.apply(error, context)

        assert result.success is True
        assert result.recovered_data == {"key": "value"}
        assert "repair" in result.recovery_strategy_applied

    async def test_that_json_with_trailing_commas_is_repaired(self, logger: Logger) -> None:
        strategy = JSONErrorStrategy(logger)
        error = json.JSONDecodeError("test", "doc", 0)
        context = {
            "malformed_input": '{"key": "value",}'
        }

        result = await strategy.apply(error, context)

        assert result.success is True
        assert result.recovered_data == {"key": "value"}

    async def test_that_irreparable_json_returns_failure(self, logger: Logger) -> None:
        strategy = JSONErrorStrategy(logger)
        error = json.JSONDecodeError("test", "doc", 0)
        context = {
            "malformed_input": '{"key": value'  # Missing closing brace and quote
        }

        result = await strategy.apply(error, context)

        assert result.success is False
        assert result.recovered_data is None

    async def test_that_missing_malformed_input_returns_failure(self, logger: Logger) -> None:
        strategy = JSONErrorStrategy(logger)
        error = json.JSONDecodeError("test", "doc", 0)
        context = {}

        result = await strategy.apply(error, context)

        assert result.success is False


class TestValidationErrorStrategy:
    """Test validation error handling strategy."""

    async def test_that_validation_error_context_is_captured(self, logger: Logger) -> None:
        strategy = ValidationErrorStrategy(logger)
        error = ValidationError("Field validation failed")
        context = {
            "field_name": "email",
            "expected_type": "string",
            "received_value": "not-an-email",
        }

        result = await strategy.apply(error, context)

        assert result.success is False
        assert result.recovered_data is not None
        assert result.recovered_data["field_name"] == "email"
        assert result.recovered_data["expected_type"] == "string"
        assert result.recovery_strategy_applied == "validation_error_analysis"


class TestRateLimitErrorStrategy:
    """Test rate limit error handling strategy."""

    async def test_that_rate_limit_error_applies_exponential_backoff(self, logger: Logger) -> None:
        strategy = RateLimitErrorStrategy(logger, initial_wait_seconds=0.01, max_wait_seconds=1.0)
        error = RateLimitError("Rate limit exceeded")
        context = {
            "attempt_number": 1,
        }

        import time
        start_time = time.time()
        result = await strategy.apply(error, context)
        elapsed = time.time() - start_time

        assert result.success is True
        assert elapsed >= 0.01
        assert result.recovery_strategy_applied == "exponential_backoff"

    async def test_that_rate_limit_error_uses_retry_after_when_provided(self, logger: Logger) -> None:
        strategy = RateLimitErrorStrategy(logger, initial_wait_seconds=1.0)
        error = RateLimitError("Rate limit exceeded")
        context = {
            "attempt_number": 1,
            "retry_after_seconds": 0.05,
        }

        import time
        start_time = time.time()
        result = await strategy.apply(error, context)
        elapsed = time.time() - start_time

        assert result.success is True
        assert elapsed >= 0.05
        assert result.recovery_strategy_applied == "exponential_backoff"


class TestNetworkErrorStrategy:
    """Test network error handling strategy."""

    async def test_that_network_error_applies_backoff(self, logger: Logger) -> None:
        strategy = NetworkErrorStrategy(
            logger,
            initial_wait_seconds=0.01,
            max_wait_seconds=1.0,
            max_retries=3,
        )
        error = NetworkError("Connection refused")
        context = {
            "attempt_number": 1,
        }

        import time
        start_time = time.time()
        result = await strategy.apply(error, context)
        elapsed = time.time() - start_time

        assert result.success is True
        assert elapsed >= 0.01
        assert result.recovery_strategy_applied == "network_error_backoff"

    async def test_that_network_error_fails_after_max_retries(self, logger: Logger) -> None:
        strategy = NetworkErrorStrategy(
            logger,
            initial_wait_seconds=0.01,
            max_wait_seconds=1.0,
            max_retries=2,
        )
        error = NetworkError("Connection refused")
        context = {
            "attempt_number": 3,  # Exceeds max_retries
        }

        result = await strategy.apply(error, context)

        assert result.success is False


class TestTimeoutErrorStrategy:
    """Test timeout error handling strategy."""

    async def test_that_timeout_error_applies_backoff(self, logger: Logger) -> None:
        strategy = TimeoutErrorStrategy(logger, initial_wait_seconds=0.01, max_retries=2)
        error = TimeoutError("Request timed out", timeout_seconds=5.0)
        context = {
            "attempt_number": 1,
        }

        import time
        start_time = time.time()
        result = await strategy.apply(error, context)
        elapsed = time.time() - start_time

        assert result.success is True
        assert elapsed >= 0.01
        assert result.recovery_strategy_applied == "timeout_error_backoff"

    async def test_that_timeout_error_fails_after_max_retries(self, logger: Logger) -> None:
        strategy = TimeoutErrorStrategy(logger, initial_wait_seconds=0.01, max_retries=1)
        error = TimeoutError("Request timed out")
        context = {
            "attempt_number": 2,  # Exceeds max_retries
        }

        result = await strategy.apply(error, context)

        assert result.success is False


class TestDefaultErrorStrategy:
    """Test default error handling strategy."""

    async def test_that_default_strategy_returns_failure(self, logger: Logger) -> None:
        strategy = DefaultErrorStrategy(logger)
        error = RuntimeError("Some unknown error")
        context = {}

        result = await strategy.apply(error, context)

        assert result.success is False


class TestErrorHandler:
    """Test the orchestrated error handler."""

    async def test_that_error_handler_categorizes_and_applies_strategy(self, logger: Logger) -> None:
        handler = ErrorHandler(logger)
        error = json.JSONDecodeError("test", "doc", 0)

        result, failure_context = await handler.handle_error(
            error=error,
            stage_name="message_generation",
            attempt_number=1,
            additional_context={"malformed_input": '{"key": "value"}'},
        )

        assert failure_context.error_category == ErrorCategory.JSON_PARSING
        assert failure_context.stage_name == "message_generation"
        assert failure_context.attempt_number == 1

    async def test_that_error_handler_captures_failure_context(self, logger: Logger) -> None:
        handler = ErrorHandler(logger)
        error = ValidationError("Validation failed")

        result, failure_context = await handler.handle_error(
            error=error,
            stage_name="validation",
            attempt_number=2,
            additional_context={"field_name": "email", "expected_type": "string"},
        )

        assert failure_context.error_category == ErrorCategory.VALIDATION
        assert failure_context.attempt_number == 2
        assert "field_name" in failure_context.error_details

    async def test_that_error_handler_can_register_custom_strategy(self, logger: Logger) -> None:
        handler = ErrorHandler(logger)

        class CustomStrategy(object):  # type: ignore
            async def apply(self, error: BaseException, context: dict[str, object]) -> ErrorRecoveryResult:
                return ErrorRecoveryResult(
                    success=True,
                    recovery_strategy_applied="custom_strategy",
                )

        handler.register_strategy(ErrorCategory.UNKNOWN, CustomStrategy())

        error = RuntimeError("Unknown error")
        result, failure_context = await handler.handle_error(
            error=error,
            stage_name="test_stage",
            attempt_number=1,
        )

        assert failure_context.recovery_strategy_applied == "custom_strategy"


class TestFailureContext:
    """Test FailureContext data structure."""

    async def test_that_failure_context_can_be_serialized(self) -> None:
        from datetime import datetime, timezone

        context = FailureContext(
            error_category=ErrorCategory.JSON_PARSING,
            error_message="JSON parsing failed",
            stage_name="message_generation",
            attempt_number=1,
            recovery_strategy_applied="json_repair_trailing_comma",
            recovery_successful=True,
        )

        serialized = context.to_dict()

        assert serialized["error_category"] == "json_parsing"
        assert serialized["error_message"] == "JSON parsing failed"
        assert serialized["stage_name"] == "message_generation"
        assert serialized["attempt_number"] == 1
        assert serialized["recovery_strategy_applied"] == "json_repair_trailing_comma"
        assert serialized["recovery_successful"] is True
