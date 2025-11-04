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

from datetime import datetime, timezone

import pytest

from parlant.core.errors import ErrorCategory, FailureContext
from parlant.core.failure_analytics import (
    FailureAnalytics,
    ErrorStatistics,
    StageFailureStatistics,
)


class TestFailureAnalytics:
    """Test failure analytics collection and reporting."""

    def test_that_failures_are_recorded(self) -> None:
        analytics = FailureAnalytics()
        failure = FailureContext(
            error_category=ErrorCategory.JSON_PARSING,
            error_message="JSON parsing failed",
            stage_name="message_generation",
            attempt_number=1,
            recovery_strategy_applied="json_repair_trailing_comma",
            recovery_successful=True,
        )

        analytics.record_failure(failure)

        assert analytics.total_failures == 1
        assert len(analytics.failure_records) == 1

    def test_that_error_statistics_are_tracked(self) -> None:
        analytics = FailureAnalytics()
        failure1 = FailureContext(
            error_category=ErrorCategory.JSON_PARSING,
            error_message="JSON parsing failed",
            stage_name="message_generation",
            attempt_number=1,
            recovery_successful=True,
        )
        failure2 = FailureContext(
            error_category=ErrorCategory.JSON_PARSING,
            error_message="JSON parsing failed",
            stage_name="message_generation",
            attempt_number=2,
            recovery_successful=False,
        )

        analytics.record_failure(failure1)
        analytics.record_failure(failure2)

        stats = analytics.get_error_statistics(ErrorCategory.JSON_PARSING)
        assert stats.count == 2
        assert stats.successful_recoveries == 1
        assert stats.failed_recoveries == 1

    def test_that_stage_statistics_are_tracked(self) -> None:
        analytics = FailureAnalytics()
        failure1 = FailureContext(
            error_category=ErrorCategory.JSON_PARSING,
            error_message="JSON parsing failed",
            stage_name="message_generation",
            attempt_number=1,
            recovery_successful=True,
        )
        failure2 = FailureContext(
            error_category=ErrorCategory.VALIDATION,
            error_message="Validation failed",
            stage_name="message_generation",
            attempt_number=2,
            recovery_successful=False,
        )

        analytics.record_failure(failure1)
        analytics.record_failure(failure2)

        stage_stats = analytics.get_stage_statistics("message_generation")
        assert stage_stats.total_failures == 2
        assert stage_stats.successful_recoveries == 1
        assert stage_stats.failed_recoveries == 1
        assert len(stage_stats.failures_by_category) == 2

    def test_that_strategies_are_tracked(self) -> None:
        analytics = FailureAnalytics()
        failure1 = FailureContext(
            error_category=ErrorCategory.JSON_PARSING,
            error_message="JSON parsing failed",
            stage_name="message_generation",
            attempt_number=1,
            recovery_strategy_applied="json_repair_trailing_comma",
            recovery_successful=True,
        )
        failure2 = FailureContext(
            error_category=ErrorCategory.JSON_PARSING,
            error_message="JSON parsing failed",
            stage_name="message_generation",
            attempt_number=2,
            recovery_strategy_applied="json_repair_unquoted_strings",
            recovery_successful=True,
        )

        analytics.record_failure(failure1)
        analytics.record_failure(failure2)

        stats = analytics.get_error_statistics(ErrorCategory.JSON_PARSING)
        assert stats.strategies_applied["json_repair_trailing_comma"] == 1
        assert stats.strategies_applied["json_repair_unquoted_strings"] == 1

    def test_that_average_retry_attempts_are_calculated(self) -> None:
        analytics = FailureAnalytics()
        failure1 = FailureContext(
            error_category=ErrorCategory.RATE_LIMIT,
            error_message="Rate limit exceeded",
            stage_name="generation",
            attempt_number=1,
            recovery_successful=False,
        )
        failure2 = FailureContext(
            error_category=ErrorCategory.RATE_LIMIT,
            error_message="Rate limit exceeded",
            stage_name="generation",
            attempt_number=3,
            recovery_successful=True,
        )

        analytics.record_failure(failure1)
        analytics.record_failure(failure2)

        stats = analytics.get_error_statistics(ErrorCategory.RATE_LIMIT)
        assert stats.average_retry_attempts == 2.0

    def test_that_most_common_error_is_identified(self) -> None:
        analytics = FailureAnalytics()
        failure1 = FailureContext(
            error_category=ErrorCategory.JSON_PARSING,
            error_message="JSON parsing failed",
            stage_name="generation",
            attempt_number=1,
            recovery_successful=False,
        )
        failure2 = FailureContext(
            error_category=ErrorCategory.JSON_PARSING,
            error_message="JSON parsing failed",
            stage_name="generation",
            attempt_number=2,
            recovery_successful=False,
        )
        failure3 = FailureContext(
            error_category=ErrorCategory.VALIDATION,
            error_message="Validation failed",
            stage_name="generation",
            attempt_number=3,
            recovery_successful=False,
        )

        analytics.record_failure(failure1)
        analytics.record_failure(failure2)
        analytics.record_failure(failure3)

        stage_stats = analytics.get_stage_statistics("generation")
        assert stage_stats.most_common_error == ErrorCategory.JSON_PARSING

    def test_that_failure_records_can_be_filtered_by_stage(self) -> None:
        analytics = FailureAnalytics()
        failure1 = FailureContext(
            error_category=ErrorCategory.JSON_PARSING,
            error_message="JSON parsing failed",
            stage_name="message_generation",
            attempt_number=1,
            recovery_successful=False,
        )
        failure2 = FailureContext(
            error_category=ErrorCategory.VALIDATION,
            error_message="Validation failed",
            stage_name="guideline_matching",
            attempt_number=2,
            recovery_successful=False,
        )

        analytics.record_failure(failure1)
        analytics.record_failure(failure2)

        msg_gen_failures = analytics.get_failure_records(stage_name="message_generation")
        assert len(msg_gen_failures) == 1
        assert msg_gen_failures[0].stage_name == "message_generation"

    def test_that_failure_records_can_be_filtered_by_category(self) -> None:
        analytics = FailureAnalytics()
        failure1 = FailureContext(
            error_category=ErrorCategory.JSON_PARSING,
            error_message="JSON parsing failed",
            stage_name="message_generation",
            attempt_number=1,
            recovery_successful=False,
        )
        failure2 = FailureContext(
            error_category=ErrorCategory.VALIDATION,
            error_message="Validation failed",
            stage_name="message_generation",
            attempt_number=2,
            recovery_successful=False,
        )

        analytics.record_failure(failure1)
        analytics.record_failure(failure2)

        json_failures = analytics.get_failure_records(category=ErrorCategory.JSON_PARSING)
        assert len(json_failures) == 1
        assert json_failures[0].error_category == ErrorCategory.JSON_PARSING

    def test_that_comprehensive_report_is_generated(self) -> None:
        analytics = FailureAnalytics()
        failure1 = FailureContext(
            error_category=ErrorCategory.JSON_PARSING,
            error_message="JSON parsing failed",
            stage_name="message_generation",
            attempt_number=1,
            recovery_successful=True,
        )
        failure2 = FailureContext(
            error_category=ErrorCategory.RATE_LIMIT,
            error_message="Rate limit exceeded",
            stage_name="generation",
            attempt_number=2,
            recovery_successful=False,
        )

        analytics.record_failure(failure1)
        analytics.record_failure(failure2)

        report = analytics.generate_report()

        assert "summary" in report
        assert report["summary"]["total_failures"] == 2
        assert report["summary"]["recovery_rate"] == 0.5
        assert report["summary"]["error_categories_encountered"] == 2
        assert report["summary"]["stages_with_failures"] == 2
        assert "errors" in report
        assert "stages" in report
        assert "report_generated_at" in report

    def test_that_recovery_rate_is_calculated(self) -> None:
        analytics = FailureAnalytics()
        failure1 = FailureContext(
            error_category=ErrorCategory.JSON_PARSING,
            error_message="JSON parsing failed",
            stage_name="generation",
            attempt_number=1,
            recovery_successful=True,
        )
        failure2 = FailureContext(
            error_category=ErrorCategory.JSON_PARSING,
            error_message="JSON parsing failed",
            stage_name="generation",
            attempt_number=2,
            recovery_successful=True,
        )
        failure3 = FailureContext(
            error_category=ErrorCategory.VALIDATION,
            error_message="Validation failed",
            stage_name="generation",
            attempt_number=3,
            recovery_successful=False,
        )

        analytics.record_failure(failure1)
        analytics.record_failure(failure2)
        analytics.record_failure(failure3)

        report = analytics.generate_report()
        recovery_rate = report["summary"]["recovery_rate"]

        assert recovery_rate == 2 / 3

    def test_that_analytics_can_be_cleared(self) -> None:
        analytics = FailureAnalytics()
        failure = FailureContext(
            error_category=ErrorCategory.JSON_PARSING,
            error_message="JSON parsing failed",
            stage_name="generation",
            attempt_number=1,
            recovery_successful=False,
        )

        analytics.record_failure(failure)
        assert analytics.total_failures == 1

        analytics.clear()
        assert analytics.total_failures == 0
        assert len(analytics.failure_records) == 0


class TestErrorStatistics:
    """Test ErrorStatistics data structure."""

    def test_that_error_statistics_can_be_serialized(self) -> None:
        stats = ErrorStatistics(
            category=ErrorCategory.JSON_PARSING,
            count=5,
            successful_recoveries=3,
            failed_recoveries=2,
        )

        serialized = stats.to_dict()

        assert serialized["category"] == "json_parsing"
        assert serialized["count"] == 5
        assert serialized["successful_recoveries"] == 3
        assert serialized["failed_recoveries"] == 2


class TestStageFailureStatistics:
    """Test StageFailureStatistics data structure."""

    def test_that_stage_statistics_can_be_serialized(self) -> None:
        stats = StageFailureStatistics(
            stage_name="message_generation",
            total_failures=3,
            successful_recoveries=1,
            failed_recoveries=2,
            most_common_error=ErrorCategory.JSON_PARSING,
        )

        stats.failures_by_category[ErrorCategory.JSON_PARSING] = 2
        stats.failures_by_category[ErrorCategory.VALIDATION] = 1

        serialized = stats.to_dict()

        assert serialized["stage_name"] == "message_generation"
        assert serialized["total_failures"] == 3
        assert serialized["successful_recoveries"] == 1
        assert serialized["failed_recoveries"] == 2
        assert serialized["most_common_error"] == "json_parsing"
