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

from parlant.applications.vfx_blueprint.constants import ValidationCode
from parlant.applications.vfx_blueprint.models import Blueprint, LayerContent
from parlant.applications.vfx_blueprint.validation import BlueprintValidator


@pytest.fixture
def validator() -> BlueprintValidator:
    """Provide a BlueprintValidator instance"""
    return BlueprintValidator()


@pytest.fixture
def valid_blueprint() -> Blueprint:
    """Provide a valid blueprint with all required layers"""
    return Blueprint(
        layers={
            "A": LayerContent("A", 1080, 1920, "Natural background scene"),
            "B": LayerContent("B", 700, 700, "Overlay element 1"),
            "C": LayerContent("C", 700, 700, "Overlay element 2"),
            "D": LayerContent("D", 700, 700, "Overlay element 3"),
            "T": LayerContent("T", 800, 150, "Title text overlay"),
        }
    )


class TestBlueprintValidatorRequiredLayers:
    """Tests for required layer validation"""

    def test_that_valid_blueprint_passes_validation(
        self, validator: BlueprintValidator, valid_blueprint: Blueprint
    ) -> None:
        """Verify a complete blueprint with all layers passes validation"""
        report = validator.validate(valid_blueprint)

        assert report.is_valid is True
        assert len(report.detected_errors) == 0

    def test_that_missing_required_layer_a_is_detected(
        self, validator: BlueprintValidator
    ) -> None:
        """Verify missing layer A is detected"""
        blueprint = Blueprint(
            layers={
                "B": LayerContent("B", 700, 700, "Overlay"),
                "C": LayerContent("C", 700, 700, "Overlay"),
                "D": LayerContent("D", 700, 700, "Overlay"),
                "T": LayerContent("T", 800, 150, "Title"),
            }
        )

        report = validator.validate(blueprint)

        assert report.is_valid is False
        assert any(err.code == ValidationCode.MISSING_REQUIRED_LAYER for err in report.detected_errors)
        assert any(err.layer_id == "A" for err in report.detected_errors)

    def test_that_missing_multiple_required_layers_are_detected(
        self, validator: BlueprintValidator
    ) -> None:
        """Verify multiple missing layers are all detected"""
        blueprint = Blueprint(
            layers={
                "A": LayerContent("A", 1080, 1920, "Background"),
            }
        )

        report = validator.validate(blueprint)

        assert report.is_valid is False
        missing_layer_errors = [
            err for err in report.detected_errors
            if err.code == ValidationCode.MISSING_REQUIRED_LAYER
        ]
        assert len(missing_layer_errors) == 4  # B, C, D, T


class TestBlueprintValidatorDimensions:
    """Tests for dimension validation"""

    def test_that_incorrect_layer_a_dimensions_are_detected(
        self, validator: BlueprintValidator
    ) -> None:
        """Verify incorrect layer A dimensions are detected"""
        blueprint = Blueprint(
            layers={
                "A": LayerContent("A", 640, 480, "Background"),
                "B": LayerContent("B", 700, 700, "Overlay"),
                "C": LayerContent("C", 700, 700, "Overlay"),
                "D": LayerContent("D", 700, 700, "Overlay"),
                "T": LayerContent("T", 800, 150, "Title"),
            }
        )

        report = validator.validate(blueprint)

        assert report.is_valid is False
        dimension_errors = [
            err for err in report.detected_errors
            if err.code == ValidationCode.INVALID_LAYER_DIMENSIONS
        ]
        assert len(dimension_errors) > 0
        assert dimension_errors[0].layer_id == "A"

    def test_that_incorrect_overlay_dimensions_are_detected(
        self, validator: BlueprintValidator
    ) -> None:
        """Verify incorrect overlay (B/C/D) dimensions are detected"""
        blueprint = Blueprint(
            layers={
                "A": LayerContent("A", 1080, 1920, "Background"),
                "B": LayerContent("B", 800, 600, "Overlay"),
                "C": LayerContent("C", 700, 700, "Overlay"),
                "D": LayerContent("D", 700, 700, "Overlay"),
                "T": LayerContent("T", 800, 150, "Title"),
            }
        )

        report = validator.validate(blueprint)

        assert report.is_valid is False
        dimension_errors = [
            err for err in report.detected_errors
            if err.code == ValidationCode.INVALID_LAYER_DIMENSIONS
        ]
        assert len(dimension_errors) > 0
        assert dimension_errors[0].layer_id == "B"

    def test_that_all_dimension_errors_are_marked_recoverable(
        self, validator: BlueprintValidator
    ) -> None:
        """Verify dimension errors are marked as recoverable"""
        blueprint = Blueprint(
            layers={
                "A": LayerContent("A", 640, 480, "Background"),
                "B": LayerContent("B", 700, 700, "Overlay"),
                "C": LayerContent("C", 700, 700, "Overlay"),
                "D": LayerContent("D", 700, 700, "Overlay"),
                "T": LayerContent("T", 800, 150, "Title"),
            }
        )

        report = validator.validate(blueprint)

        dimension_errors = [
            err for err in report.detected_errors
            if err.code == ValidationCode.INVALID_LAYER_DIMENSIONS
        ]
        assert all(err.recoverable for err in dimension_errors)


class TestBlueprintValidatorGreenScreenConflicts:
    """Tests for green-screen conflict detection"""

    def test_that_green_screen_in_overlay_layer_is_detected(
        self, validator: BlueprintValidator, valid_blueprint: Blueprint
    ) -> None:
        """Verify green screen keywords in overlay layers are detected"""
        valid_blueprint.layers["B"].prompt = "Green screen background"

        report = validator.validate(valid_blueprint)

        assert report.is_valid is False
        green_screen_errors = [
            err for err in report.detected_errors
            if err.code == ValidationCode.GREEN_SCREEN_CONFLICT
        ]
        assert len(green_screen_errors) > 0

    def test_that_green_screen_in_background_layer_is_not_flagged(
        self, validator: BlueprintValidator, valid_blueprint: Blueprint
    ) -> None:
        """Verify green screen keywords in background layer A are allowed"""
        valid_blueprint.layers["A"].prompt = "Green screen backdrop"

        report = validator.validate(valid_blueprint)

        green_screen_errors = [
            err for err in report.detected_errors
            if err.code == ValidationCode.GREEN_SCREEN_CONFLICT
        ]
        assert len(green_screen_errors) == 0

    def test_that_chroma_key_keyword_is_detected(
        self, validator: BlueprintValidator, valid_blueprint: Blueprint
    ) -> None:
        """Verify chroma key keyword is detected"""
        valid_blueprint.layers["B"].prompt = "Use chroma key for selection"

        report = validator.validate(valid_blueprint)

        green_screen_errors = [
            err for err in report.detected_errors
            if err.code == ValidationCode.GREEN_SCREEN_CONFLICT
        ]
        assert len(green_screen_errors) > 0


class TestBlueprintValidatorBracketWeights:
    """Tests for bracket weight detection"""

    def test_that_bracket_weights_in_prompts_are_detected(
        self, validator: BlueprintValidator, valid_blueprint: Blueprint
    ) -> None:
        """Verify bracket weights at start of prompt are detected"""
        valid_blueprint.layers["A"].prompt = "[0.8] Background scene"

        report = validator.validate(valid_blueprint)

        assert report.is_valid is False
        bracket_errors = [
            err for err in report.detected_errors
            if err.code == ValidationCode.BRACKET_WEIGHT_DETECTED
        ]
        assert len(bracket_errors) > 0

    def test_that_bracket_weights_with_decimals_are_detected(
        self, validator: BlueprintValidator, valid_blueprint: Blueprint
    ) -> None:
        """Verify bracket weights with decimal numbers are detected"""
        valid_blueprint.layers["B"].prompt = "[0.5] Overlay content"

        report = validator.validate(valid_blueprint)

        bracket_errors = [
            err for err in report.detected_errors
            if err.code == ValidationCode.BRACKET_WEIGHT_DETECTED
        ]
        assert len(bracket_errors) > 0

    def test_that_normal_brackets_in_prompts_are_not_flagged(
        self, validator: BlueprintValidator, valid_blueprint: Blueprint
    ) -> None:
        """Verify brackets in middle of prompt are not flagged"""
        valid_blueprint.layers["A"].prompt = "Background with [text] in middle"

        report = validator.validate(valid_blueprint)

        bracket_errors = [
            err for err in report.detected_errors
            if err.code == ValidationCode.BRACKET_WEIGHT_DETECTED
        ]
        assert len(bracket_errors) == 0


class TestBlueprintValidatorEmptyPrompts:
    """Tests for empty prompt detection"""

    def test_that_empty_prompt_in_background_is_detected(
        self, validator: BlueprintValidator, valid_blueprint: Blueprint
    ) -> None:
        """Verify empty prompt in background layer A is detected"""
        valid_blueprint.layers["A"].prompt = ""

        report = validator.validate(valid_blueprint)

        assert report.is_valid is False
        empty_errors = [
            err for err in report.detected_errors
            if err.code == ValidationCode.EMPTY_PROMPT
        ]
        assert len(empty_errors) > 0

    def test_that_whitespace_only_prompt_is_treated_as_empty(
        self, validator: BlueprintValidator, valid_blueprint: Blueprint
    ) -> None:
        """Verify whitespace-only prompt is treated as empty"""
        valid_blueprint.layers["A"].prompt = "   "

        report = validator.validate(valid_blueprint)

        empty_errors = [
            err for err in report.detected_errors
            if err.code == ValidationCode.EMPTY_PROMPT
        ]
        assert len(empty_errors) > 0


class TestBlueprintValidatorSummary:
    """Tests for validation summary generation"""

    def test_that_valid_blueprint_generates_success_summary(
        self, validator: BlueprintValidator, valid_blueprint: Blueprint
    ) -> None:
        """Verify valid blueprint generates success summary"""
        report = validator.validate(valid_blueprint)

        assert "passed" in report.summary.lower()
        assert "no issues" in report.summary.lower()

    def test_that_invalid_blueprint_generates_failure_summary(
        self, validator: BlueprintValidator
    ) -> None:
        """Verify invalid blueprint generates failure summary"""
        blueprint = Blueprint(layers={})

        report = validator.validate(blueprint)

        assert "failed" in report.summary.lower()
        assert "error" in report.summary.lower()
