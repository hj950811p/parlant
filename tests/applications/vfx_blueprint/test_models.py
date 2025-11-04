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

import pytest

from parlant.applications.vfx_blueprint.models import (
    AppliedFix,
    Blueprint,
    LayerContent,
    ValidationError,
    ValidationReport,
)


class TestLayerContent:
    """Tests for LayerContent model"""

    def test_that_layer_content_stores_all_required_fields(self) -> None:
        """Verify LayerContent stores layer_id, dimensions, and prompt"""
        layer = LayerContent(
            layer_id="A",
            width=1080,
            height=1920,
            prompt="Test prompt",
        )

        assert layer.layer_id == "A"
        assert layer.width == 1080
        assert layer.height == 1920
        assert layer.prompt == "Test prompt"

    def test_that_layer_content_has_optional_metadata(self) -> None:
        """Verify LayerContent supports optional metadata"""
        layer = LayerContent(
            layer_id="B",
            width=700,
            height=700,
            prompt="Overlay prompt",
            metadata={"key": "value"},
        )

        assert layer.metadata == {"key": "value"}


class TestBlueprint:
    """Tests for Blueprint model"""

    def test_that_blueprint_contains_all_layers(self) -> None:
        """Verify Blueprint stores layers dictionary"""
        layers = {
            "A": LayerContent("A", 1080, 1920, "Background"),
            "B": LayerContent("B", 700, 700, "Overlay 1"),
        }
        blueprint = Blueprint(layers=layers)

        assert "A" in blueprint.layers
        assert "B" in blueprint.layers
        assert len(blueprint.layers) == 2

    def test_that_blueprint_supports_dict_like_access(self) -> None:
        """Verify Blueprint allows dict-like access to layers"""
        layers = {
            "A": LayerContent("A", 1080, 1920, "Background"),
        }
        blueprint = Blueprint(layers=layers)

        assert blueprint["A"].layer_id == "A"

    def test_that_blueprint_supports_dict_get_method(self) -> None:
        """Verify Blueprint.get() method works with defaults"""
        layers = {
            "A": LayerContent("A", 1080, 1920, "Background"),
        }
        blueprint = Blueprint(layers=layers)

        assert blueprint.get("A") is not None
        assert blueprint.get("X", "default") == "default"

    def test_that_blueprint_supports_in_operator(self) -> None:
        """Verify Blueprint supports 'in' operator"""
        layers = {
            "A": LayerContent("A", 1080, 1920, "Background"),
        }
        blueprint = Blueprint(layers=layers)

        assert "A" in blueprint
        assert "X" not in blueprint


class TestValidationError:
    """Tests for ValidationError model"""

    def test_that_validation_error_contains_code_and_message(self) -> None:
        """Verify ValidationError stores error code and message"""
        error = ValidationError(
            code="TEST_CODE",
            message="Test error message",
        )

        assert error.code == "TEST_CODE"
        assert error.message == "Test error message"

    def test_that_validation_error_can_be_marked_unrecoverable(self) -> None:
        """Verify ValidationError can mark errors as unrecoverable"""
        error = ValidationError(
            code="CRITICAL_ERROR",
            message="Cannot fix this",
            recoverable=False,
        )

        assert error.recoverable is False

    def test_that_validation_error_converts_to_dict(self) -> None:
        """Verify ValidationError can be converted to dictionary"""
        error = ValidationError(
            code="TEST_CODE",
            message="Test message",
            layer_id="A",
            recoverable=True,
        )

        error_dict = error.to_dict()

        assert error_dict["code"] == "TEST_CODE"
        assert error_dict["message"] == "Test message"
        assert error_dict["layer_id"] == "A"
        assert error_dict["recoverable"] is True


class TestAppliedFix:
    """Tests for AppliedFix model"""

    def test_that_applied_fix_contains_code_and_message(self) -> None:
        """Verify AppliedFix stores fix code and message"""
        fix = AppliedFix(
            code="FIXED_CODE",
            message="Applied fix",
        )

        assert fix.code == "FIXED_CODE"
        assert fix.message == "Applied fix"

    def test_that_applied_fix_stores_details(self) -> None:
        """Verify AppliedFix can store fix details"""
        fix = AppliedFix(
            code="DIMENSION_FIX",
            message="Fixed dimensions",
            details={"old": "640x480", "new": "700x700"},
        )

        assert fix.details["old"] == "640x480"
        assert fix.details["new"] == "700x700"

    def test_that_applied_fix_converts_to_dict(self) -> None:
        """Verify AppliedFix can be converted to dictionary"""
        fix = AppliedFix(
            code="FIX_CODE",
            message="Applied fix",
            layer_id="B",
            details={"key": "value"},
        )

        fix_dict = fix.to_dict()

        assert fix_dict["code"] == "FIX_CODE"
        assert fix_dict["message"] == "Applied fix"
        assert fix_dict["layer_id"] == "B"


class TestValidationReport:
    """Tests for ValidationReport model"""

    def test_that_validation_report_is_valid_when_no_errors(self) -> None:
        """Verify ValidationReport is valid when no errors present"""
        report = ValidationReport(
            detected_errors=[],
            is_valid=True,
        )

        assert report.is_valid is True
        assert len(report.detected_errors) == 0

    def test_that_validation_report_detects_unrecoverable_errors(self) -> None:
        """Verify ValidationReport identifies unrecoverable errors"""
        unrecoverable_error = ValidationError(
            code="CRITICAL",
            message="Cannot fix",
            recoverable=False,
        )
        report = ValidationReport(
            unresolved_errors=[unrecoverable_error],
            is_valid=False,
        )

        assert report.has_unrecoverable_errors is True
        assert report.recovery_possible is False

    def test_that_validation_report_allows_recovery_with_only_recoverable_errors(
        self,
    ) -> None:
        """Verify ValidationReport allows recovery when all errors are recoverable"""
        recoverable_error = ValidationError(
            code="FIXABLE",
            message="Can fix",
            recoverable=True,
        )
        report = ValidationReport(
            unresolved_errors=[recoverable_error],
            is_valid=False,
        )

        assert report.has_unrecoverable_errors is False
        assert report.recovery_possible is True

    def test_that_validation_report_converts_to_dict(self) -> None:
        """Verify ValidationReport can be converted to dictionary"""
        error = ValidationError("CODE", "Message")
        fix = AppliedFix("FIX_CODE", "Applied")

        report = ValidationReport(
            detected_errors=[error],
            applied_fixes=[fix],
            is_valid=False,
            summary="Test summary",
        )

        report_dict = report.to_dict()

        assert report_dict["is_valid"] is False
        assert report_dict["summary"] == "Test summary"
        assert "detected_errors" in report_dict
        assert "applied_fixes" in report_dict
        assert "unresolved_errors" in report_dict
