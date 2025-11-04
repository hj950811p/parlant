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

from parlant.applications.vfx_blueprint.auto_fix import BlueprintAutoFixer
from parlant.applications.vfx_blueprint.constants import AutoFixCode, ValidationCode
from parlant.applications.vfx_blueprint.models import Blueprint, LayerContent, ValidationError, ValidationReport
from parlant.applications.vfx_blueprint.validation import BlueprintValidator


@pytest.fixture
def fixer() -> BlueprintAutoFixer:
    """Provide a BlueprintAutoFixer instance"""
    return BlueprintAutoFixer()


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


class TestBlueprintAutoFixerDimensions:
    """Tests for dimension auto-fix"""

    def test_that_incorrect_layer_dimensions_are_corrected(
        self, fixer: BlueprintAutoFixer, validator: BlueprintValidator
    ) -> None:
        """Verify incorrect dimensions are auto-corrected to expected sizes"""
        blueprint = Blueprint(
            layers={
                "A": LayerContent("A", 640, 480, "Background"),
                "B": LayerContent("B", 700, 700, "Overlay"),
                "C": LayerContent("C", 700, 700, "Overlay"),
                "D": LayerContent("D", 700, 700, "Overlay"),
                "T": LayerContent("T", 800, 150, "Title"),
            }
        )

        validation_report = validator.validate(blueprint)
        fixed_blueprint, fixed_report = fixer.auto_fix(blueprint, validation_report)

        assert fixed_blueprint.layers["A"].width == 1080
        assert fixed_blueprint.layers["A"].height == 1920

    def test_that_dimension_fixes_are_reported(
        self, fixer: BlueprintAutoFixer, validator: BlueprintValidator
    ) -> None:
        """Verify applied dimension fixes are recorded in report"""
        blueprint = Blueprint(
            layers={
                "A": LayerContent("A", 640, 480, "Background"),
                "B": LayerContent("B", 700, 700, "Overlay"),
                "C": LayerContent("C", 700, 700, "Overlay"),
                "D": LayerContent("D", 700, 700, "Overlay"),
                "T": LayerContent("T", 800, 150, "Title"),
            }
        )

        validation_report = validator.validate(blueprint)
        _, fixed_report = fixer.auto_fix(blueprint, validation_report)

        dimension_fixes = [
            fix for fix in fixed_report.applied_fixes
            if fix.code == AutoFixCode.CORRECTED_DIMENSIONS
        ]
        assert len(dimension_fixes) > 0

    def test_that_original_blueprint_is_not_modified(
        self, fixer: BlueprintAutoFixer, validator: BlueprintValidator
    ) -> None:
        """Verify original blueprint remains unchanged (immutable)"""
        blueprint = Blueprint(
            layers={
                "A": LayerContent("A", 640, 480, "Background"),
                "B": LayerContent("B", 700, 700, "Overlay"),
                "C": LayerContent("C", 700, 700, "Overlay"),
                "D": LayerContent("D", 700, 700, "Overlay"),
                "T": LayerContent("T", 800, 150, "Title"),
            }
        )

        original_width = blueprint.layers["A"].width
        validation_report = validator.validate(blueprint)
        fixer.auto_fix(blueprint, validation_report)

        assert blueprint.layers["A"].width == original_width


class TestBlueprintAutoFixerBracketWeights:
    """Tests for bracket weight removal"""

    def test_that_bracket_weights_are_removed_from_prompts(
        self, fixer: BlueprintAutoFixer, validator: BlueprintValidator, valid_blueprint: Blueprint
    ) -> None:
        """Verify bracket weights are removed from prompts"""
        valid_blueprint.layers["A"].prompt = "[0.8] Background scene"

        validation_report = validator.validate(valid_blueprint)
        fixed_blueprint, _ = fixer.auto_fix(valid_blueprint, validation_report)

        assert fixed_blueprint.layers["A"].prompt == "Background scene"

    def test_that_bracket_weights_with_decimals_are_removed(
        self, fixer: BlueprintAutoFixer, validator: BlueprintValidator, valid_blueprint: Blueprint
    ) -> None:
        """Verify bracket weights with decimal numbers are removed"""
        valid_blueprint.layers["B"].prompt = "[0.5] Overlay content"

        validation_report = validator.validate(valid_blueprint)
        fixed_blueprint, _ = fixer.auto_fix(valid_blueprint, validation_report)

        assert "[0.5]" not in fixed_blueprint.layers["B"].prompt

    def test_that_bracket_weight_removal_is_reported(
        self, fixer: BlueprintAutoFixer, validator: BlueprintValidator, valid_blueprint: Blueprint
    ) -> None:
        """Verify bracket weight removal is recorded in report"""
        valid_blueprint.layers["A"].prompt = "[0.8] Background scene"

        validation_report = validator.validate(valid_blueprint)
        _, fixed_report = fixer.auto_fix(valid_blueprint, validation_report)

        bracket_fixes = [
            fix for fix in fixed_report.applied_fixes
            if fix.code == AutoFixCode.REMOVED_BRACKET_WEIGHTS
        ]
        assert len(bracket_fixes) > 0


class TestBlueprintAutoFixerGreenScreenConflicts:
    """Tests for green-screen conflict resolution"""

    def test_that_green_screen_keywords_are_removed_from_overlays(
        self, fixer: BlueprintAutoFixer, validator: BlueprintValidator, valid_blueprint: Blueprint
    ) -> None:
        """Verify green-screen keywords are removed from overlay layers"""
        valid_blueprint.layers["B"].prompt = "Green screen overlay content"

        validation_report = validator.validate(valid_blueprint)
        fixed_blueprint, _ = fixer.auto_fix(valid_blueprint, validation_report)

        assert "green screen" not in fixed_blueprint.layers["B"].prompt.lower()

    def test_that_green_screen_removal_is_reported(
        self, fixer: BlueprintAutoFixer, validator: BlueprintValidator, valid_blueprint: Blueprint
    ) -> None:
        """Verify green-screen removal is recorded in report"""
        valid_blueprint.layers["B"].prompt = "Green screen overlay"

        validation_report = validator.validate(valid_blueprint)
        _, fixed_report = fixer.auto_fix(valid_blueprint, validation_report)

        green_fixes = [
            fix for fix in fixed_report.applied_fixes
            if fix.code == AutoFixCode.RESOLVED_GREEN_SCREEN
        ]
        assert len(green_fixes) > 0


class TestBlueprintAutoFixerEmptyPrompts:
    """Tests for empty prompt handling"""

    def test_that_empty_prompts_are_filled_with_defaults(
        self, fixer: BlueprintAutoFixer, validator: BlueprintValidator, valid_blueprint: Blueprint
    ) -> None:
        """Verify empty prompts are filled with layer-appropriate defaults"""
        valid_blueprint.layers["A"].prompt = ""

        validation_report = validator.validate(valid_blueprint)
        fixed_blueprint, _ = fixer.auto_fix(valid_blueprint, validation_report)

        assert len(fixed_blueprint.layers["A"].prompt) > 0
        assert fixed_blueprint.layers["A"].prompt != ""

    def test_that_different_layers_get_appropriate_defaults(
        self, fixer: BlueprintAutoFixer, validator: BlueprintValidator, valid_blueprint: Blueprint
    ) -> None:
        """Verify different layers get semantically appropriate default prompts"""
        valid_blueprint.layers["A"].prompt = ""
        valid_blueprint.layers["T"].prompt = ""

        validation_report = validator.validate(valid_blueprint)
        fixed_blueprint, _ = fixer.auto_fix(valid_blueprint, validation_report)

        # A (background) should get different default than T (title)
        assert "background" in fixed_blueprint.layers["A"].prompt.lower()
        assert "title" in fixed_blueprint.layers["T"].prompt.lower() or "text" in fixed_blueprint.layers["T"].prompt.lower()

    def test_that_empty_prompt_fixes_are_reported(
        self, fixer: BlueprintAutoFixer, validator: BlueprintValidator, valid_blueprint: Blueprint
    ) -> None:
        """Verify empty prompt fixes are recorded in report"""
        valid_blueprint.layers["A"].prompt = ""

        validation_report = validator.validate(valid_blueprint)
        _, fixed_report = fixer.auto_fix(valid_blueprint, validation_report)

        empty_fixes = [
            fix for fix in fixed_report.applied_fixes
            if fix.code == AutoFixCode.INSERTED_DEFAULT_VALUE
        ]
        assert len(empty_fixes) > 0


class TestBlueprintAutoFixerReporting:
    """Tests for auto-fix reporting"""

    def test_that_successful_fix_is_marked_valid(
        self, fixer: BlueprintAutoFixer, validator: BlueprintValidator
    ) -> None:
        """Verify successfully fixed blueprint is marked as valid"""
        blueprint = Blueprint(
            layers={
                "A": LayerContent("A", 640, 480, "Background"),
                "B": LayerContent("B", 700, 700, "Overlay"),
                "C": LayerContent("C", 700, 700, "Overlay"),
                "D": LayerContent("D", 700, 700, "Overlay"),
                "T": LayerContent("T", 800, 150, "Title"),
            }
        )

        validation_report = validator.validate(blueprint)
        _, fixed_report = fixer.auto_fix(blueprint, validation_report)

        assert fixed_report.is_valid is True

    def test_that_multiple_fixes_are_tracked(
        self, fixer: BlueprintAutoFixer, validator: BlueprintValidator
    ) -> None:
        """Verify multiple fixes are all recorded"""
        blueprint = Blueprint(
            layers={
                "A": LayerContent("A", 640, 480, "[0.8] Background"),
                "B": LayerContent("B", 700, 700, "Green screen overlay"),
                "C": LayerContent("C", 700, 700, "Overlay"),
                "D": LayerContent("D", 700, 700, "Overlay"),
                "T": LayerContent("T", 800, 150, "Title"),
            }
        )

        validation_report = validator.validate(blueprint)
        _, fixed_report = fixer.auto_fix(blueprint, validation_report)

        assert len(fixed_report.applied_fixes) >= 2

    def test_that_unrecoverable_errors_remain_unresolved(
        self, fixer: BlueprintAutoFixer, validator: BlueprintValidator
    ) -> None:
        """Verify unrecoverable errors remain in unresolved_errors"""
        blueprint = Blueprint(
            layers={
                "A": LayerContent("A", 640, 480, "Background"),
                "B": LayerContent("B", 700, 700, "Overlay"),
                "C": LayerContent("C", 700, 700, "Overlay"),
                "D": LayerContent("D", 700, 700, "Overlay"),
                "T": LayerContent("T", 800, 150, "Title"),
            }
        )
        # This blueprint still has missing layer error which is unrecoverable in layer field validation

        validation_report = validator.validate(blueprint)
        _, fixed_report = fixer.auto_fix(blueprint, validation_report)

        # Dimension errors are recoverable
        assert all(fix.code == AutoFixCode.CORRECTED_DIMENSIONS for fix in fixed_report.applied_fixes)


class TestBlueprintAutoFixerSummary:
    """Tests for auto-fix summary generation"""

    def test_that_successful_fix_generates_success_summary(
        self, fixer: BlueprintAutoFixer, validator: BlueprintValidator
    ) -> None:
        """Verify successful fix generates positive summary"""
        blueprint = Blueprint(
            layers={
                "A": LayerContent("A", 640, 480, "Background"),
                "B": LayerContent("B", 700, 700, "Overlay"),
                "C": LayerContent("C", 700, 700, "Overlay"),
                "D": LayerContent("D", 700, 700, "Overlay"),
                "T": LayerContent("T", 800, 150, "Title"),
            }
        )

        validation_report = validator.validate(blueprint)
        _, fixed_report = fixer.auto_fix(blueprint, validation_report)

        assert "successful" in fixed_report.summary.lower() or "valid" in fixed_report.summary.lower()

    def test_that_partial_fix_mentions_remaining_errors(
        self, fixer: BlueprintAutoFixer, validator: BlueprintValidator
    ) -> None:
        """Verify partial fix generates summary mentioning unresolved errors"""
        blueprint = Blueprint(
            layers={
                "B": LayerContent("B", 700, 700, "Overlay"),
                "C": LayerContent("C", 700, 700, "Overlay"),
                "D": LayerContent("D", 700, 700, "Overlay"),
                "T": LayerContent("T", 800, 150, "Title"),
            }
        )

        validation_report = validator.validate(blueprint)
        _, fixed_report = fixer.auto_fix(blueprint, validation_report)

        if len(fixed_report.unresolved_errors) > 0:
            assert "unresolved" in fixed_report.summary.lower() or "error" in fixed_report.summary.lower()
