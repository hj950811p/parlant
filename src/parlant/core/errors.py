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

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional
from datetime import datetime, timezone


class ErrorCategory(Enum):
    """Categorizes different types of errors for targeted recovery strategies."""

    JSON_PARSING = "json_parsing"
    """JSON parsing or deserialization failed."""

    VALIDATION = "validation"
    """Schema or data validation failed."""

    NETWORK = "network"
    """Network-related error (connection, timeout)."""

    RATE_LIMIT = "rate_limit"
    """Rate limit exceeded."""

    INTERNAL_SERVER = "internal_server"
    """Internal server error (5xx)."""

    TIMEOUT = "timeout"
    """Request timeout."""

    RESOURCE_EXHAUSTED = "resource_exhausted"
    """Resource exhausted (e.g., token limit, memory)."""

    UNKNOWN = "unknown"
    """Unknown error category."""


class ParlantError(Exception):
    """Base exception class for Parlant errors with categorization."""

    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        original_exception: Optional[BaseException] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.category = category
        self.original_exception = original_exception
        self.context = dict(context) if context else {}


class JSONParsingError(ParlantError):
    """Raised when JSON parsing or deserialization fails."""

    def __init__(
        self,
        message: str,
        malformed_input: Optional[str] = None,
        original_exception: Optional[BaseException] = None,
    ) -> None:
        context: dict[str, Any] = {}
        if malformed_input:
            context["malformed_input"] = malformed_input[:500]
        super().__init__(
            message=message,
            category=ErrorCategory.JSON_PARSING,
            original_exception=original_exception,
            context=context,
        )
        self.malformed_input = malformed_input


class ValidationError(ParlantError):
    """Raised when schema or data validation fails."""

    def __init__(
        self,
        message: str,
        field_name: Optional[str] = None,
        expected_type: Optional[str] = None,
        received_value: Optional[str] = None,
        original_exception: Optional[BaseException] = None,
    ) -> None:
        context: dict[str, Any] = {}
        if field_name:
            context["field_name"] = field_name
        if expected_type:
            context["expected_type"] = expected_type
        if received_value:
            context["received_value"] = received_value[:200]
        super().__init__(
            message=message,
            category=ErrorCategory.VALIDATION,
            original_exception=original_exception,
            context=context,
        )
        self.field_name = field_name
        self.expected_type = expected_type
        self.received_value = received_value


class NetworkError(ParlantError):
    """Raised when a network error occurs."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        original_exception: Optional[BaseException] = None,
    ) -> None:
        context: dict[str, Any] = {}
        if error_code:
            context["error_code"] = error_code
        super().__init__(
            message=message,
            category=ErrorCategory.NETWORK,
            original_exception=original_exception,
            context=context,
        )
        self.error_code = error_code


class RateLimitError(ParlantError):
    """Raised when rate limit is exceeded."""

    def __init__(
        self,
        message: str,
        retry_after_seconds: Optional[int] = None,
        original_exception: Optional[BaseException] = None,
    ) -> None:
        context: dict[str, Any] = {}
        if retry_after_seconds:
            context["retry_after_seconds"] = retry_after_seconds
        super().__init__(
            message=message,
            category=ErrorCategory.RATE_LIMIT,
            original_exception=original_exception,
            context=context,
        )
        self.retry_after_seconds = retry_after_seconds


class InternalServerError(ParlantError):
    """Raised when an internal server error occurs."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        original_exception: Optional[BaseException] = None,
    ) -> None:
        context: dict[str, Any] = {}
        if status_code:
            context["status_code"] = status_code
        super().__init__(
            message=message,
            category=ErrorCategory.INTERNAL_SERVER,
            original_exception=original_exception,
            context=context,
        )
        self.status_code = status_code


class TimeoutError(ParlantError):
    """Raised when a request times out."""

    def __init__(
        self,
        message: str,
        timeout_seconds: Optional[float] = None,
        original_exception: Optional[BaseException] = None,
    ) -> None:
        context: dict[str, Any] = {}
        if timeout_seconds:
            context["timeout_seconds"] = timeout_seconds
        super().__init__(
            message=message,
            category=ErrorCategory.TIMEOUT,
            original_exception=original_exception,
            context=context,
        )
        self.timeout_seconds = timeout_seconds


class ResourceExhaustedError(ParlantError):
    """Raised when a resource is exhausted (e.g., token limit)."""

    def __init__(
        self,
        message: str,
        resource_type: Optional[str] = None,
        resource_limit: Optional[int] = None,
        original_exception: Optional[BaseException] = None,
    ) -> None:
        context: dict[str, Any] = {}
        if resource_type:
            context["resource_type"] = resource_type
        if resource_limit:
            context["resource_limit"] = resource_limit
        super().__init__(
            message=message,
            category=ErrorCategory.RESOURCE_EXHAUSTED,
            original_exception=original_exception,
            context=context,
        )
        self.resource_type = resource_type
        self.resource_limit = resource_limit


@dataclass
class FailureContext:
    """Captures context information about a failure for diagnostics."""

    error_category: ErrorCategory
    error_message: str
    stage_name: str
    attempt_number: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    error_details: dict[str, Any] = field(default_factory=dict)
    recovery_strategy_applied: Optional[str] = None
    recovery_successful: bool = False
    additional_context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Mapping[str, Any]:
        """Convert failure context to dictionary for storage."""
        return {
            "error_category": self.error_category.value,
            "error_message": self.error_message,
            "stage_name": self.stage_name,
            "attempt_number": self.attempt_number,
            "timestamp": self.timestamp.isoformat(),
            "error_details": self.error_details,
            "recovery_strategy_applied": self.recovery_strategy_applied,
            "recovery_successful": self.recovery_successful,
            "additional_context": self.additional_context,
        }
