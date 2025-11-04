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

import re
from typing import Any

from parlant.applications.vfx_blueprint.constants import (
    LAYER_DIMENSIONS,
    REQUIRED_LAYERS,
    ValidationCode,
)
from parlant.applications.vfx_blueprint.models import (
    Blueprint,
    ValidationError,
    ValidationReport,
)


class BlueprintValidator:
    """Validates blueprint structures against layer sizing rules and content constraints"""

    # Green-screen keywords that indicate improper layer content
    GREEN_SCREEN_KEYWORDS = {"green screen", "chroma key", "solid color", "placeholder"}

    # Background layer IDs
    BACKGROUND_LAYERS = {"A"}

    # Bracket weight pattern: [number] at the start of prompt
    BRACKET_WEIGHT_PATTERN = re.compile(r"^\s*\[\d*\.?\d+\]\s*")

    def validate(self, blueprint: Blueprint) -> ValidationReport:
        """
        Validate a blueprint structure and return detailed report.

        Args:
            blueprint: Blueprint to validate

        Returns:
            ValidationReport with detected errors and validity status
        """
        errors: list[ValidationError] = []

        # Check for required layers
        errors.extend(self._validate_required_layers(blueprint))

        # Check layer dimensions
        errors.extend(self._validate_layer_dimensions(blueprint))

        # Check for green-screen conflicts
        errors.extend(self._validate_green_screen_conflicts(blueprint))

        # Check for improper background content
        errors.extend(self._validate_background_content(blueprint))

        # Check for bracket weights in prompts
        errors.extend(self._validate_bracket_weights(blueprint))

        # Check for required fields in layers
        errors.extend(self._validate_layer_fields(blueprint))

        # Create report
        is_valid = len(errors) == 0
        summary = self._generate_summary(errors, is_valid)

        report = ValidationReport(
            detected_errors=errors,
            unresolved_errors=errors,
            is_valid=is_valid,
            summary=summary,
        )

        return report

    def _validate_required_layers(self, blueprint: Blueprint) -> list[ValidationError]:
        """Check that all required layers are present"""
        errors: list[ValidationError] = []

        for layer_id in REQUIRED_LAYERS:
            if layer_id not in blueprint.layers:
                errors.append(
                    ValidationError(
                        code=ValidationCode.MISSING_REQUIRED_LAYER,
                        message=f"Required layer '{layer_id}' is missing",
                        layer_id=layer_id,
                        recoverable=False,
                    )
                )

        return errors

    def _validate_layer_dimensions(self, blueprint: Blueprint) -> list[ValidationError]:
        """Check that layer dimensions match expected sizes"""
        errors: list[ValidationError] = []

        for layer_id, layer in blueprint.layers.items():
            if layer_id not in LAYER_DIMENSIONS:
                continue

            expected_width, expected_height = LAYER_DIMENSIONS[layer_id]

            if layer.width != expected_width or layer.height != expected_height:
                errors.append(
                    ValidationError(
                        code=ValidationCode.INVALID_LAYER_DIMENSIONS,
                        message=(
                            f"Layer '{layer_id}' has dimensions "
                            f"{layer.width}x{layer.height}, "
                            f"expected {expected_width}x{expected_height}"
                        ),
                        layer_id=layer_id,
                        recoverable=True,
                    )
                )

        return errors

    def _validate_green_screen_conflicts(self, blueprint: Blueprint) -> list[ValidationError]:
        """Detect green-screen usage in non-background layers"""
        errors: list[ValidationError] = []

        for layer_id, layer in blueprint.layers.items():
            if layer_id not in self.BACKGROUND_LAYERS:
                prompt_lower = layer.prompt.lower()
                if any(
                    keyword in prompt_lower for keyword in self.GREEN_SCREEN_KEYWORDS
                ):
                    errors.append(
                        ValidationError(
                            code=ValidationCode.GREEN_SCREEN_CONFLICT,
                            message=(
                                f"Layer '{layer_id}' contains green-screen keywords, "
                                f"which should only be used in background layer"
                            ),
                            layer_id=layer_id,
                            recoverable=True,
                        )
                    )

        return errors

    def _validate_background_content(self, blueprint: Blueprint) -> list[ValidationError]:
        """Check for improper content in background layers"""
        errors: list[ValidationError] = []

        for layer_id in self.BACKGROUND_LAYERS:
            if layer_id not in blueprint.layers:
                continue

            layer = blueprint.layers[layer_id]
            # Background should not have generic placeholder content
            if not layer.prompt or layer.prompt.strip() == "":
                errors.append(
                    ValidationError(
                        code=ValidationCode.EMPTY_PROMPT,
                        message=f"Layer '{layer_id}' has empty prompt",
                        layer_id=layer_id,
                        recoverable=True,
                    )
                )

        return errors

    def _validate_bracket_weights(self, blueprint: Blueprint) -> list[ValidationError]:
        """Check for bracket-based prompt weights that need removal"""
        errors: list[ValidationError] = []

        for layer_id, layer in blueprint.layers.items():
            if self.BRACKET_WEIGHT_PATTERN.match(layer.prompt):
                errors.append(
                    ValidationError(
                        code=ValidationCode.BRACKET_WEIGHT_DETECTED,
                        message=(
                            f"Layer '{layer_id}' prompt contains bracket weights "
                            f"that need to be removed"
                        ),
                        layer_id=layer_id,
                        recoverable=True,
                    )
                )

        return errors

    def _validate_layer_fields(self, blueprint: Blueprint) -> list[ValidationError]:
        """Check for required fields in each layer"""
        errors: list[ValidationError] = []

        for layer_id, layer in blueprint.layers.items():
            # Check layer_id
            if not layer.layer_id:
                errors.append(
                    ValidationError(
                        code=ValidationCode.MISSING_REQUIRED_FIELD,
                        message=f"Layer '{layer_id}' missing layer_id field",
                        layer_id=layer_id,
                        recoverable=True,
                    )
                )

            # Check width and height are valid numbers
            if not isinstance(layer.width, int) or layer.width <= 0:
                errors.append(
                    ValidationError(
                        code=ValidationCode.INVALID_FIELD_TYPE,
                        message=f"Layer '{layer_id}' has invalid width value",
                        layer_id=layer_id,
                        recoverable=False,
                    )
                )

            if not isinstance(layer.height, int) or layer.height <= 0:
                errors.append(
                    ValidationError(
                        code=ValidationCode.INVALID_FIELD_TYPE,
                        message=f"Layer '{layer_id}' has invalid height value",
                        layer_id=layer_id,
                        recoverable=False,
                    )
                )

        return errors

    def _generate_summary(self, errors: list[ValidationError], is_valid: bool) -> str:
        """Generate a human-readable summary of validation results"""
        if is_valid:
            return "Blueprint validation passed - no issues detected"

        error_count = len(errors)
        unrecoverable = sum(1 for e in errors if not e.recoverable)

        if unrecoverable > 0:
            return (
                f"Blueprint validation failed: {error_count} error(s) detected, "
                f"{unrecoverable} unrecoverable"
            )
        else:
            return (
                f"Blueprint validation failed: {error_count} error(s) detected, "
                f"all recoverable via auto-fix"
            )
