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

import json
from datetime import datetime
from typing import Any

from parlant.applications.vfx_blueprint.models import ValidationReport


class BlueprintReporter:
    """Generates detailed reports for blueprint validation and auto-fix operations"""

    def generate_report_dict(self, report: ValidationReport) -> dict[str, Any]:
        """
        Generate a dictionary representation of the validation report.

        Args:
            report: ValidationReport to convert

        Returns:
            Dictionary with all report details
        """
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "summary": report.summary,
            "is_valid": report.is_valid,
            "has_unrecoverable_errors": report.has_unrecoverable_errors,
            "recovery_possible": report.recovery_possible,
            "detected_errors": {
                "count": len(report.detected_errors),
                "items": [err.to_dict() for err in report.detected_errors],
            },
            "applied_fixes": {
                "count": len(report.applied_fixes),
                "items": [fix.to_dict() for fix in report.applied_fixes],
            },
            "unresolved_errors": {
                "count": len(report.unresolved_errors),
                "items": [err.to_dict() for err in report.unresolved_errors],
            },
        }

    def generate_report_json(self, report: ValidationReport) -> str:
        """
        Generate a JSON string representation of the report.

        Args:
            report: ValidationReport to convert

        Returns:
            JSON string with report details
        """
        report_dict = self.generate_report_dict(report)
        return json.dumps(report_dict, indent=2)

    def generate_report_text(self, report: ValidationReport) -> str:
        """
        Generate a human-readable text report.

        Args:
            report: ValidationReport to convert

        Returns:
            Formatted text report
        """
        lines = [
            "=" * 80,
            "BLUEPRINT VALIDATION REPORT",
            "=" * 80,
            "",
            f"Status: {'VALID' if report.is_valid else 'INVALID'}",
            f"Summary: {report.summary}",
            "",
        ]

        if report.detected_errors:
            lines.append(f"Detected Errors ({len(report.detected_errors)}):")
            lines.append("-" * 80)
            for i, err in enumerate(report.detected_errors, 1):
                lines.append(
                    f"  {i}. [{err.code}] {err.message}"
                    + (f" (Layer: {err.layer_id})" if err.layer_id else "")
                )
                lines.append(f"     Recoverable: {err.recoverable}")
            lines.append("")

        if report.applied_fixes:
            lines.append(f"Applied Fixes ({len(report.applied_fixes)}):")
            lines.append("-" * 80)
            for i, fix in enumerate(report.applied_fixes, 1):
                lines.append(
                    f"  {i}. [{fix.code}] {fix.message}"
                    + (f" (Layer: {fix.layer_id})" if fix.layer_id else "")
                )
                if fix.details:
                    for key, value in fix.details.items():
                        lines.append(f"     {key}: {value}")
            lines.append("")

        if report.unresolved_errors:
            lines.append(f"Unresolved Errors ({len(report.unresolved_errors)}):")
            lines.append("-" * 80)
            for i, err in enumerate(report.unresolved_errors, 1):
                lines.append(
                    f"  {i}. [{err.code}] {err.message}"
                    + (f" (Layer: {err.layer_id})" if err.layer_id else "")
                )
                lines.append(f"     Recoverable: {err.recoverable}")
            lines.append("")

        lines.append("=" * 80)

        return "\n".join(lines)

    def log_report(
        self, report: ValidationReport, logger: Any, level: str = "info"
    ) -> None:
        """
        Log the report using provided logger.

        Args:
            report: ValidationReport to log
            logger: Logger instance with log methods
            level: Log level ('info', 'warning', 'error')
        """
        text_report = self.generate_report_text(report)

        log_method = getattr(logger, level, logger.info)
        log_method(text_report)
