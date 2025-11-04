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

import pytest

from parlant.applications.vfx_blueprint.models import AppliedFix, ValidationError, ValidationReport
from parlant.applications.vfx_blueprint.reporter import BlueprintReporter


@pytest.fixture
def reporter() -> BlueprintReporter:
    """Provide a BlueprintReporter instance"""
    return BlueprintReporter()


@pytest.fixture
def sample_report_with_errors() -> ValidationReport:
    """Provide a sample report with errors and fixes"""
    return ValidationReport(
        detected_errors=[
            ValidationError("CODE1", "Error message 1", layer_id="A"),
            ValidationError("CODE2", "Error message 2", layer_id="B"),
        ],
        applied_fixes=[
            AppliedFix("FIX1", "Applied fix 1", layer_id="A"),
            AppliedFix("FIX2", "Applied fix 2", layer_id="B"),
        ],
        unresolved_errors=[
            ValidationError("CODE2", "Error message 2", layer_id="B"),
        ],
        is_valid=False,
        summary="Test summary",
    )


class TestBlueprintReporterDictGeneration:
    """Tests for dictionary report generation"""

    def test_that_report_dict_contains_summary(
        self, reporter: BlueprintReporter, sample_report_with_errors: ValidationReport
    ) -> None:
        """Verify report dict includes summary"""
        report_dict = reporter.generate_report_dict(sample_report_with_errors)

        assert "summary" in report_dict
        assert report_dict["summary"] == "Test summary"

    def test_that_report_dict_includes_validity_status(
        self, reporter: BlueprintReporter, sample_report_with_errors: ValidationReport
    ) -> None:
        """Verify report dict includes is_valid status"""
        report_dict = reporter.generate_report_dict(sample_report_with_errors)

        assert "is_valid" in report_dict
        assert report_dict["is_valid"] is False

    def test_that_report_dict_includes_detected_errors(
        self, reporter: BlueprintReporter, sample_report_with_errors: ValidationReport
    ) -> None:
        """Verify report dict includes all detected errors"""
        report_dict = reporter.generate_report_dict(sample_report_with_errors)

        assert "detected_errors" in report_dict
        assert report_dict["detected_errors"]["count"] == 2
        assert len(report_dict["detected_errors"]["items"]) == 2

    def test_that_report_dict_includes_applied_fixes(
        self, reporter: BlueprintReporter, sample_report_with_errors: ValidationReport
    ) -> None:
        """Verify report dict includes all applied fixes"""
        report_dict = reporter.generate_report_dict(sample_report_with_errors)

        assert "applied_fixes" in report_dict
        assert report_dict["applied_fixes"]["count"] == 2
        assert len(report_dict["applied_fixes"]["items"]) == 2

    def test_that_report_dict_includes_unresolved_errors(
        self, reporter: BlueprintReporter, sample_report_with_errors: ValidationReport
    ) -> None:
        """Verify report dict includes unresolved errors"""
        report_dict = reporter.generate_report_dict(sample_report_with_errors)

        assert "unresolved_errors" in report_dict
        assert report_dict["unresolved_errors"]["count"] == 1

    def test_that_report_dict_includes_recovery_status(
        self, reporter: BlueprintReporter, sample_report_with_errors: ValidationReport
    ) -> None:
        """Verify report dict includes recovery status"""
        report_dict = reporter.generate_report_dict(sample_report_with_errors)

        assert "has_unrecoverable_errors" in report_dict
        assert "recovery_possible" in report_dict

    def test_that_report_dict_includes_timestamp(
        self, reporter: BlueprintReporter, sample_report_with_errors: ValidationReport
    ) -> None:
        """Verify report dict includes ISO timestamp"""
        report_dict = reporter.generate_report_dict(sample_report_with_errors)

        assert "timestamp" in report_dict
        # Should be ISO format
        assert "T" in report_dict["timestamp"]


class TestBlueprintReporterJsonGeneration:
    """Tests for JSON report generation"""

    def test_that_json_report_is_valid_json(
        self, reporter: BlueprintReporter, sample_report_with_errors: ValidationReport
    ) -> None:
        """Verify generated JSON string is valid JSON"""
        json_str = reporter.generate_report_json(sample_report_with_errors)

        # Should not raise
        report_dict = json.loads(json_str)
        assert report_dict is not None

    def test_that_json_report_contains_all_sections(
        self, reporter: BlueprintReporter, sample_report_with_errors: ValidationReport
    ) -> None:
        """Verify JSON report contains all required sections"""
        json_str = reporter.generate_report_json(sample_report_with_errors)
        report_dict = json.loads(json_str)

        assert "summary" in report_dict
        assert "is_valid" in report_dict
        assert "detected_errors" in report_dict
        assert "applied_fixes" in report_dict


class TestBlueprintReporterTextGeneration:
    """Tests for text report generation"""

    def test_that_text_report_contains_header(
        self, reporter: BlueprintReporter, sample_report_with_errors: ValidationReport
    ) -> None:
        """Verify text report contains header"""
        text_report = reporter.generate_report_text(sample_report_with_errors)

        assert "BLUEPRINT VALIDATION REPORT" in text_report

    def test_that_text_report_contains_status(
        self, reporter: BlueprintReporter, sample_report_with_errors: ValidationReport
    ) -> None:
        """Verify text report includes status"""
        text_report = reporter.generate_report_text(sample_report_with_errors)

        assert "Status:" in text_report
        assert "INVALID" in text_report

    def test_that_text_report_lists_detected_errors(
        self, reporter: BlueprintReporter, sample_report_with_errors: ValidationReport
    ) -> None:
        """Verify text report includes detected errors section"""
        text_report = reporter.generate_report_text(sample_report_with_errors)

        assert "Detected Errors" in text_report
        assert "CODE1" in text_report
        assert "CODE2" in text_report

    def test_that_text_report_lists_applied_fixes(
        self, reporter: BlueprintReporter, sample_report_with_errors: ValidationReport
    ) -> None:
        """Verify text report includes applied fixes section"""
        text_report = reporter.generate_report_text(sample_report_with_errors)

        assert "Applied Fixes" in text_report
        assert "FIX1" in text_report
        assert "FIX2" in text_report

    def test_that_text_report_lists_unresolved_errors(
        self, reporter: BlueprintReporter, sample_report_with_errors: ValidationReport
    ) -> None:
        """Verify text report includes unresolved errors section"""
        text_report = reporter.generate_report_text(sample_report_with_errors)

        assert "Unresolved Errors" in text_report

    def test_that_valid_report_shows_valid_status(
        self, reporter: BlueprintReporter
    ) -> None:
        """Verify valid report shows VALID status"""
        report = ValidationReport(is_valid=True)
        text_report = reporter.generate_report_text(report)

        assert "VALID" in text_report

    def test_that_text_report_shows_layer_ids_when_present(
        self, reporter: BlueprintReporter, sample_report_with_errors: ValidationReport
    ) -> None:
        """Verify text report includes layer IDs in error/fix listings"""
        text_report = reporter.generate_report_text(sample_report_with_errors)

        assert "Layer: A" in text_report
        assert "Layer: B" in text_report


class TestBlueprintReporterLogging:
    """Tests for logging functionality"""

    def test_that_report_can_be_logged(
        self, reporter: BlueprintReporter, sample_report_with_errors: ValidationReport
    ) -> None:
        """Verify report can be logged with a mock logger"""
        logged_messages: list[str] = []

        class MockLogger:
            def info(self, msg: str) -> None:
                logged_messages.append(msg)

            def warning(self, msg: str) -> None:
                logged_messages.append(msg)

            def error(self, msg: str) -> None:
                logged_messages.append(msg)

        logger = MockLogger()
        reporter.log_report(sample_report_with_errors, logger, "info")

        assert len(logged_messages) > 0
        assert "BLUEPRINT VALIDATION REPORT" in logged_messages[0]

    def test_that_report_can_be_logged_as_warning(
        self, reporter: BlueprintReporter, sample_report_with_errors: ValidationReport
    ) -> None:
        """Verify report can be logged at warning level"""
        logged_messages: list[str] = []

        class MockLogger:
            def warning(self, msg: str) -> None:
                logged_messages.append(msg)

        logger = MockLogger()
        reporter.log_report(sample_report_with_errors, logger, "warning")

        assert len(logged_messages) > 0

    def test_that_report_can_be_logged_as_error(
        self, reporter: BlueprintReporter, sample_report_with_errors: ValidationReport
    ) -> None:
        """Verify report can be logged at error level"""
        logged_messages: list[str] = []

        class MockLogger:
            def error(self, msg: str) -> None:
                logged_messages.append(msg)

        logger = MockLogger()
        reporter.log_report(sample_report_with_errors, logger, "error")

        assert len(logged_messages) > 0
