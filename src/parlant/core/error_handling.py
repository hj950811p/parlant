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

from __future__ import annotations

import asyncio
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, TypeVar, Union

from parlant.core.errors import (
    ErrorCategory,
    JSONParsingError,
    ValidationError,
    NetworkError,
    RateLimitError,
    InternalServerError,
    TimeoutError,
    ResourceExhaustedError,
    ParlantError,
    FailureContext,
)
from parlant.core.loggers import Logger

T = TypeVar("T")


def categorize_error(exception: BaseException) -> ErrorCategory:
    """Categorizes an exception into an error category."""
    try:
        from openai import (
            APIConnectionError,
            APITimeoutError,
            RateLimitError as OpenAIRateLimitError,
        )
    except ImportError:
        APIConnectionError = None  # type: ignore
        APITimeoutError = None  # type: ignore
        OpenAIRateLimitError = None  # type: ignore

    exception_type = type(exception).__name__
    exception_message = str(exception)

    if isinstance(exception, json.JSONDecodeError):
        return ErrorCategory.JSON_PARSING
    elif isinstance(exception, (ValueError, TypeError)):
        if "json" in exception_message.lower():
            return ErrorCategory.JSON_PARSING
        return ErrorCategory.VALIDATION
    elif APIConnectionError is not None and isinstance(exception, APIConnectionError):
        return ErrorCategory.NETWORK
    elif isinstance(exception, (ConnectionError, OSError)):
        return ErrorCategory.NETWORK
    elif APITimeoutError is not None and isinstance(exception, APITimeoutError):
        return ErrorCategory.TIMEOUT
    elif isinstance(exception, asyncio.TimeoutError):
        return ErrorCategory.TIMEOUT
    elif OpenAIRateLimitError is not None and isinstance(exception, OpenAIRateLimitError):
        return ErrorCategory.RATE_LIMIT
    elif isinstance(exception, InternalServerError):
        return ErrorCategory.INTERNAL_SERVER
    elif exception_type == "InternalServerError":
        return ErrorCategory.INTERNAL_SERVER
    elif "token" in exception_message.lower() and "limit" in exception_message.lower():
        return ErrorCategory.RESOURCE_EXHAUSTED
    elif "rate" in exception_message.lower() and "limit" in exception_message.lower():
        return ErrorCategory.RATE_LIMIT

    return ErrorCategory.UNKNOWN


@dataclass
class ErrorRecoveryResult:
    """Result of attempting error recovery."""

    success: bool
    recovered_data: Optional[Any] = None
    recovery_strategy_applied: Optional[str] = None
    error_message: Optional[str] = None


class ErrorHandlingStrategy(ABC):
    """Abstract base class for error handling strategies."""

    @abstractmethod
    async def apply(
        self,
        error: BaseException,
        context: Mapping[str, Any],
    ) -> ErrorRecoveryResult:
        """
        Apply the error handling strategy.

        Args:
            error: The exception that occurred
            context: Additional context for error recovery

        Returns:
            ErrorRecoveryResult with recovery status and recovered data
        """
        pass


class JSONErrorStrategy(ErrorHandlingStrategy):
    """Strategy for handling JSON parsing errors."""

    def __init__(self, logger: Logger) -> None:
        self._logger = logger

    async def apply(
        self,
        error: BaseException,
        context: Mapping[str, Any],
    ) -> ErrorRecoveryResult:
        """
        Attempt to repair malformed JSON.

        This strategy tries several fixes:
        1. Remove common JSON formatting issues
        2. Wrap unquoted strings in quotes
        3. Remove trailing commas
        """
        malformed_input = context.get("malformed_input")
        if not malformed_input or not isinstance(malformed_input, str):
            return ErrorRecoveryResult(
                success=False,
                error_message="No malformed input provided for JSON repair",
            )

        repairs_attempted = []

        fixed_input = malformed_input
        try:
            fixed_input = json.loads(malformed_input)
            return ErrorRecoveryResult(
                success=True,
                recovered_data=fixed_input,
                recovery_strategy_applied="json_repair_success",
            )
        except json.JSONDecodeError:
            pass

        try:
            fixed_input = re.sub(r",(\s*[}\]])", r"\1", malformed_input)
            repairs_attempted.append("trailing_comma_removal")
            parsed = json.loads(fixed_input)
            return ErrorRecoveryResult(
                success=True,
                recovered_data=parsed,
                recovery_strategy_applied="json_repair_trailing_comma",
            )
        except json.JSONDecodeError:
            pass

        try:
            fixed_input = re.sub(
                r':\s*([^,}\]\"\'\d-][\w]*)',
                r': "\1"',
                malformed_input,
            )
            repairs_attempted.append("unquoted_string_wrapping")
            parsed = json.loads(fixed_input)
            return ErrorRecoveryResult(
                success=True,
                recovered_data=parsed,
                recovery_strategy_applied="json_repair_unquoted_strings",
            )
        except json.JSONDecodeError:
            pass

        return ErrorRecoveryResult(
            success=False,
            error_message=f"Could not repair JSON after attempts: {', '.join(repairs_attempted)}",
        )


class ValidationErrorStrategy(ErrorHandlingStrategy):
    """Strategy for handling validation errors."""

    def __init__(self, logger: Logger) -> None:
        self._logger = logger

    async def apply(
        self,
        error: BaseException,
        context: Mapping[str, Any],
    ) -> ErrorRecoveryResult:
        """
        Handle validation errors by capturing details for prompt adjustment.

        This strategy doesn't recover the data but provides information
        for adjusting prompts or configuration in the next retry.
        """
        field_name = context.get("field_name", "unknown")
        expected_type = context.get("expected_type", "unknown")

        recovery_details = {
            "field_name": field_name,
            "expected_type": expected_type,
            "error_message": str(error),
        }

        return ErrorRecoveryResult(
            success=False,
            recovered_data=recovery_details,
            recovery_strategy_applied="validation_error_analysis",
            error_message="Validation error details captured for retry adjustment",
        )


class RateLimitErrorStrategy(ErrorHandlingStrategy):
    """Strategy for handling rate limit errors with exponential backoff."""

    def __init__(
        self,
        logger: Logger,
        initial_wait_seconds: float = 1.0,
        max_wait_seconds: float = 60.0,
    ) -> None:
        self._logger = logger
        self._initial_wait_seconds = initial_wait_seconds
        self._max_wait_seconds = max_wait_seconds

    async def apply(
        self,
        error: BaseException,
        context: Mapping[str, Any],
    ) -> ErrorRecoveryResult:
        """
        Apply exponential backoff for rate limit errors.

        Uses context['attempt_number'] to calculate backoff duration.
        """
        attempt_number = context.get("attempt_number", 1)
        retry_after = context.get("retry_after_seconds")

        if retry_after is not None:
            wait_time = float(retry_after)
        else:
            wait_time = min(
                self._initial_wait_seconds * (2 ** (attempt_number - 1)),
                self._max_wait_seconds,
            )

        self._logger.info(
            f"Rate limit error detected. Applying exponential backoff: {wait_time}s"
        )
        await asyncio.sleep(wait_time)

        return ErrorRecoveryResult(
            success=True,
            recovery_strategy_applied="exponential_backoff",
        )


class NetworkErrorStrategy(ErrorHandlingStrategy):
    """Strategy for handling network errors."""

    def __init__(
        self,
        logger: Logger,
        initial_wait_seconds: float = 0.5,
        max_wait_seconds: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        self._logger = logger
        self._initial_wait_seconds = initial_wait_seconds
        self._max_wait_seconds = max_wait_seconds
        self._max_retries = max_retries

    async def apply(
        self,
        error: BaseException,
        context: Mapping[str, Any],
    ) -> ErrorRecoveryResult:
        """
        Handle network errors with limited retries and backoff.
        """
        attempt_number = context.get("attempt_number", 1)

        if attempt_number > self._max_retries:
            return ErrorRecoveryResult(
                success=False,
                error_message=f"Network error: maximum retries ({self._max_retries}) exceeded",
            )

        wait_time = min(
            self._initial_wait_seconds * (2 ** (attempt_number - 1)),
            self._max_wait_seconds,
        )

        self._logger.warning(
            f"Network error detected. Retrying with backoff: {wait_time}s (attempt {attempt_number}/{self._max_retries})"
        )
        await asyncio.sleep(wait_time)

        return ErrorRecoveryResult(
            success=True,
            recovery_strategy_applied="network_error_backoff",
        )


class TimeoutErrorStrategy(ErrorHandlingStrategy):
    """Strategy for handling timeout errors."""

    def __init__(
        self,
        logger: Logger,
        initial_wait_seconds: float = 1.0,
        max_retries: int = 2,
    ) -> None:
        self._logger = logger
        self._initial_wait_seconds = initial_wait_seconds
        self._max_retries = max_retries

    async def apply(
        self,
        error: BaseException,
        context: Mapping[str, Any],
    ) -> ErrorRecoveryResult:
        """
        Handle timeout errors with limited retries.
        """
        attempt_number = context.get("attempt_number", 1)

        if attempt_number > self._max_retries:
            return ErrorRecoveryResult(
                success=False,
                error_message=f"Timeout error: maximum retries ({self._max_retries}) exceeded",
            )

        wait_time = self._initial_wait_seconds * attempt_number

        self._logger.warning(
            f"Timeout error detected. Retrying with increased timeout window: {wait_time}s (attempt {attempt_number}/{self._max_retries})"
        )
        await asyncio.sleep(wait_time)

        return ErrorRecoveryResult(
            success=True,
            recovery_strategy_applied="timeout_error_backoff",
        )


class DefaultErrorStrategy(ErrorHandlingStrategy):
    """Default strategy for unknown error types."""

    def __init__(self, logger: Logger) -> None:
        self._logger = logger

    async def apply(
        self,
        error: BaseException,
        context: Mapping[str, Any],
    ) -> ErrorRecoveryResult:
        """
        Default error handling - no recovery attempted.
        """
        return ErrorRecoveryResult(
            success=False,
            error_message=f"Unknown error type: {type(error).__name__}",
        )


class ErrorHandler:
    """Orchestrates error handling and strategy application."""

    def __init__(self, logger: Logger) -> None:
        self._logger = logger
        self._strategies: dict[ErrorCategory, ErrorHandlingStrategy] = {
            ErrorCategory.JSON_PARSING: JSONErrorStrategy(logger),
            ErrorCategory.VALIDATION: ValidationErrorStrategy(logger),
            ErrorCategory.RATE_LIMIT: RateLimitErrorStrategy(logger),
            ErrorCategory.NETWORK: NetworkErrorStrategy(logger),
            ErrorCategory.TIMEOUT: TimeoutErrorStrategy(logger),
        }

    async def handle_error(
        self,
        error: BaseException,
        stage_name: str,
        attempt_number: int,
        additional_context: Optional[Mapping[str, Any]] = None,
    ) -> tuple[ErrorRecoveryResult, FailureContext]:
        """
        Handle an error and apply appropriate recovery strategy.

        Args:
            error: The exception that occurred
            stage_name: Name of the processing stage where error occurred
            attempt_number: Current attempt number
            additional_context: Additional context for error handling

        Returns:
            Tuple of (recovery result, failure context)
        """
        category = categorize_error(error)
        additional_context = additional_context or {}

        context: dict[str, Any] = {
            "attempt_number": attempt_number,
            **additional_context,
        }

        strategy = self._strategies.get(
            category,
            DefaultErrorStrategy(self._logger),
        )

        recovery_result = await strategy.apply(error, context)

        failure_context = FailureContext(
            error_category=category,
            error_message=str(error),
            stage_name=stage_name,
            attempt_number=attempt_number,
            error_details=dict(additional_context),
            recovery_strategy_applied=recovery_result.recovery_strategy_applied,
            recovery_successful=recovery_result.success,
        )

        return recovery_result, failure_context

    def register_strategy(
        self,
        category: ErrorCategory,
        strategy: ErrorHandlingStrategy,
    ) -> None:
        """Register a custom error handling strategy for a specific error category."""
        self._strategies[category] = strategy
