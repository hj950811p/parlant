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

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Optional, Sequence

from parlant.core.errors import ErrorCategory, FailureContext


@dataclass
class ErrorStatistics:
    """Statistics for a specific error category."""

    category: ErrorCategory
    count: int = 0
    successful_recoveries: int = 0
    failed_recoveries: int = 0
    strategies_applied: dict[str, int] = field(default_factory=dict)
    average_retry_attempts: float = 0.0
    last_occurrence: Optional[datetime] = None

    def to_dict(self) -> Mapping[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "category": self.category.value,
            "count": self.count,
            "successful_recoveries": self.successful_recoveries,
            "failed_recoveries": self.failed_recoveries,
            "strategies_applied": self.strategies_applied,
            "average_retry_attempts": self.average_retry_attempts,
            "last_occurrence": self.last_occurrence.isoformat() if self.last_occurrence else None,
        }


@dataclass
class StageFailureStatistics:
    """Statistics for failures at a specific processing stage."""

    stage_name: str
    total_failures: int = 0
    failures_by_category: dict[ErrorCategory, int] = field(default_factory=dict)
    successful_recoveries: int = 0
    failed_recoveries: int = 0
    average_retry_attempts: float = 0.0
    most_common_error: Optional[ErrorCategory] = None

    def to_dict(self) -> Mapping[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "stage_name": self.stage_name,
            "total_failures": self.total_failures,
            "failures_by_category": {
                k.value: v for k, v in self.failures_by_category.items()
            },
            "successful_recoveries": self.successful_recoveries,
            "failed_recoveries": self.failed_recoveries,
            "average_retry_attempts": self.average_retry_attempts,
            "most_common_error": self.most_common_error.value if self.most_common_error else None,
        }


class FailureAnalytics:
    """Collects and analyzes failure patterns for diagnostics and feedback."""

    def __init__(self) -> None:
        self._failure_records: list[FailureContext] = []
        self._error_statistics: dict[ErrorCategory, ErrorStatistics] = {}
        self._stage_statistics: dict[str, StageFailureStatistics] = {}
        self._total_failures: int = 0

    def record_failure(self, failure: FailureContext) -> None:
        """Record a failure event."""
        self._failure_records.append(failure)
        self._total_failures += 1

        self._update_error_statistics(failure)
        self._update_stage_statistics(failure)

    def _update_error_statistics(self, failure: FailureContext) -> None:
        """Update error category statistics."""
        category = failure.error_category

        if category not in self._error_statistics:
            self._error_statistics[category] = ErrorStatistics(category=category)

        stats = self._error_statistics[category]
        stats.count += 1
        stats.last_occurrence = failure.timestamp

        if failure.recovery_successful:
            stats.successful_recoveries += 1
        else:
            stats.failed_recoveries += 1

        if failure.recovery_strategy_applied:
            strategy = failure.recovery_strategy_applied
            stats.strategies_applied[strategy] = stats.strategies_applied.get(strategy, 0) + 1

        if stats.count > 0:
            all_attempts = sum(f.attempt_number for f in self._failure_records if f.error_category == category)
            stats.average_retry_attempts = all_attempts / stats.count

    def _update_stage_statistics(self, failure: FailureContext) -> None:
        """Update stage-specific statistics."""
        stage_name = failure.stage_name

        if stage_name not in self._stage_statistics:
            self._stage_statistics[stage_name] = StageFailureStatistics(stage_name=stage_name)

        stats = self._stage_statistics[stage_name]
        stats.total_failures += 1

        category = failure.error_category
        stats.failures_by_category[category] = stats.failures_by_category.get(category, 0) + 1

        if failure.recovery_successful:
            stats.successful_recoveries += 1
        else:
            stats.failed_recoveries += 1

        if stats.total_failures > 0:
            all_attempts = sum(
                f.attempt_number for f in self._failure_records if f.stage_name == stage_name
            )
            stats.average_retry_attempts = all_attempts / stats.total_failures

        most_common = max(
            stats.failures_by_category.items(),
            key=lambda x: x[1],
            default=(None, 0),
        )
        stats.most_common_error = most_common[0]

    def get_error_statistics(
        self,
        category: Optional[ErrorCategory] = None,
    ) -> dict[ErrorCategory, ErrorStatistics] | ErrorStatistics:
        """Get error statistics, optionally filtered by category."""
        if category:
            return self._error_statistics.get(
                category,
                ErrorStatistics(category=category),
            )
        return self._error_statistics

    def get_stage_statistics(
        self,
        stage_name: Optional[str] = None,
    ) -> dict[str, StageFailureStatistics] | StageFailureStatistics:
        """Get stage statistics, optionally filtered by stage name."""
        if stage_name:
            return self._stage_statistics.get(
                stage_name,
                StageFailureStatistics(stage_name=stage_name),
            )
        return self._stage_statistics

    def get_failure_records(
        self,
        stage_name: Optional[str] = None,
        category: Optional[ErrorCategory] = None,
    ) -> Sequence[FailureContext]:
        """Get failure records, optionally filtered."""
        records = self._failure_records

        if stage_name:
            records = [r for r in records if r.stage_name == stage_name]

        if category:
            records = [r for r in records if r.error_category == category]

        return records

    def generate_report(self) -> Mapping[str, Any]:
        """Generate a comprehensive failure analysis report."""
        error_stats_by_category = {
            k.value: v.to_dict()
            for k, v in self._error_statistics.items()
        }

        stage_stats_dict = {
            k: v.to_dict()
            for k, v in self._stage_statistics.items()
        }

        recovery_rate = (
            sum(s.successful_recoveries for s in self._error_statistics.values())
            / self._total_failures
            if self._total_failures > 0
            else 0.0
        )

        return {
            "summary": {
                "total_failures": self._total_failures,
                "recovery_rate": recovery_rate,
                "error_categories_encountered": len(self._error_statistics),
                "stages_with_failures": len(self._stage_statistics),
            },
            "errors": error_stats_by_category,
            "stages": stage_stats_dict,
            "report_generated_at": datetime.now(timezone.utc).isoformat(),
        }

    def clear(self) -> None:
        """Clear all recorded failures and statistics."""
        self._failure_records = []
        self._error_statistics = {}
        self._stage_statistics = {}
        self._total_failures = 0

    @property
    def total_failures(self) -> int:
        """Get total number of failures recorded."""
        return self._total_failures

    @property
    def failure_records(self) -> Sequence[FailureContext]:
        """Get all failure records."""
        return list(self._failure_records)
