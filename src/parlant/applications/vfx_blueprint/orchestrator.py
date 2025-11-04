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

from typing import Any, Optional

from parlant.applications.vfx_blueprint.auto_fix import BlueprintAutoFixer
from parlant.applications.vfx_blueprint.models import Blueprint, ValidationReport
from parlant.applications.vfx_blueprint.reporter import BlueprintReporter
from parlant.applications.vfx_blueprint.validation import BlueprintValidator


class BlueprintOrchestrator:
    """Orchestrates blueprint validation and auto-fix operations"""

    def __init__(self) -> None:
        """Initialize orchestrator with validation and auto-fix components"""
        self._validator = BlueprintValidator()
        self._auto_fixer = BlueprintAutoFixer()
        self._reporter = BlueprintReporter()

    def validate_and_fix(
        self,
        blueprint: Blueprint,
        auto_fix: bool = True,
    ) -> tuple[Optional[Blueprint], ValidationReport]:
        """
        Validate a blueprint and optionally auto-fix it.

        Args:
            blueprint: Blueprint to validate
            auto_fix: Whether to attempt auto-fixing of recoverable errors

        Returns:
            Tuple of (fixed_blueprint or None if unrecoverable errors, report)

        Raises:
            No exceptions - returns errors in report instead
        """
        # Validate blueprint
        validation_report = self._validator.validate(blueprint)

        # If no errors, return the original blueprint
        if validation_report.is_valid:
            return blueprint, validation_report

        # If there are unrecoverable errors and auto-fix is disabled, return None
        if validation_report.has_unrecoverable_errors and not auto_fix:
            return None, validation_report

        # If there are unrecoverable errors even with auto-fix enabled, return None
        if validation_report.has_unrecoverable_errors:
            return None, validation_report

        # If all errors are recoverable and auto-fix is enabled, attempt fix
        if auto_fix and len(validation_report.detected_errors) > 0:
            fixed_blueprint, fixed_report = self._auto_fixer.auto_fix(
                blueprint, validation_report
            )

            # If still has unrecoverable errors after fix, return None
            if fixed_report.has_unrecoverable_errors:
                return None, fixed_report

            return fixed_blueprint, fixed_report

        return blueprint, validation_report

    def generate_report_dict(self, report: ValidationReport) -> dict[str, Any]:
        """Generate dictionary representation of report"""
        return self._reporter.generate_report_dict(report)

    def generate_report_json(self, report: ValidationReport) -> str:
        """Generate JSON representation of report"""
        return self._reporter.generate_report_json(report)

    def generate_report_text(self, report: ValidationReport) -> str:
        """Generate human-readable text representation of report"""
        return self._reporter.generate_report_text(report)
