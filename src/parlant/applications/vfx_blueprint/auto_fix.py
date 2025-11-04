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

import copy
import re
from typing import Any

from parlant.applications.vfx_blueprint.constants import (
    LAYER_DIMENSIONS,
    AutoFixCode,
)
from parlant.applications.vfx_blueprint.models import (
    AppliedFix,
    Blueprint,
    LayerContent,
    ValidationReport,
)


class BlueprintAutoFixer:
    """Automatically repairs common blueprint violations"""

    # Bracket weight pattern: [number] at the start of prompt
    BRACKET_WEIGHT_PATTERN = re.compile(r"^\s*\[\d*\.?\d+\]\s*")

    # Concrete object keywords that indicate overly specific prompts for backgrounds
    CONCRETE_KEYWORDS = {
        "person",
        "face",
        "man",
        "woman",
        "object",
        "car",
        "table",
        "chair",
    }

    def auto_fix(
        self,
        blueprint: Blueprint,
        validation_report: ValidationReport,
    ) -> tuple[Blueprint, ValidationReport]:
        """
        Attempt to auto-fix recoverable errors in blueprint.

        Creates an immutable copy of the original blueprint before making changes
        to support rollback if needed.

        Args:
            blueprint: Original blueprint
            validation_report: Validation report with detected errors

        Returns:
            Tuple of (fixed_blueprint, updated_report)
        """
        # Create immutable copy for rollback support
        fixed_blueprint = self._deep_copy_blueprint(blueprint)
        applied_fixes: list[AppliedFix] = []
        remaining_errors = list(validation_report.detected_errors)

        # Apply fixes for each error type
        from parlant.applications.vfx_blueprint.constants import ValidationCode

        # Fix dimension errors
        dims_fixes, dims_remaining = self._fix_dimensions(
            fixed_blueprint, remaining_errors
        )
        applied_fixes.extend(dims_fixes)
        remaining_errors = dims_remaining

        # Fix bracket weights
        bracket_fixes, bracket_remaining = self._fix_bracket_weights(
            fixed_blueprint, remaining_errors
        )
        applied_fixes.extend(bracket_fixes)
        remaining_errors = bracket_remaining

        # Fix background prompts with concrete objects
        prompt_fixes, prompt_remaining = self._fix_background_prompts(
            fixed_blueprint, remaining_errors
        )
        applied_fixes.extend(prompt_fixes)
        remaining_errors = prompt_remaining

        # Fix green-screen conflicts
        green_fixes, green_remaining = self._fix_green_screen_conflicts(
            fixed_blueprint, remaining_errors
        )
        applied_fixes.extend(green_fixes)
        remaining_errors = green_remaining

        # Fix empty prompts
        empty_fixes, empty_remaining = self._fix_empty_prompts(
            fixed_blueprint, remaining_errors
        )
        applied_fixes.extend(empty_fixes)
        remaining_errors = empty_remaining

        # Create updated report
        is_valid = len(remaining_errors) == 0
        summary = self._generate_summary(applied_fixes, remaining_errors, is_valid)

        updated_report = ValidationReport(
            detected_errors=list(validation_report.detected_errors),
            applied_fixes=applied_fixes,
            unresolved_errors=remaining_errors,
            is_valid=is_valid,
            summary=summary,
        )

        return fixed_blueprint, updated_report

    def _fix_dimensions(
        self,
        blueprint: Blueprint,
        errors: list[Any],
    ) -> tuple[list[AppliedFix], list[Any]]:
        """Fix invalid layer dimensions"""
        from parlant.applications.vfx_blueprint.constants import ValidationCode

        fixes: list[AppliedFix] = []
        remaining: list[Any] = []

        for error in errors:
            if error.code == ValidationCode.INVALID_LAYER_DIMENSIONS:
                layer_id = error.layer_id
                if layer_id in blueprint.layers:
                    expected_width, expected_height = LAYER_DIMENSIONS[layer_id]
                    layer = blueprint.layers[layer_id]

                    old_width, old_height = layer.width, layer.height

                    layer.width = expected_width
                    layer.height = expected_height

                    fixes.append(
                        AppliedFix(
                            code=AutoFixCode.CORRECTED_DIMENSIONS,
                            message=(
                                f"Corrected layer '{layer_id}' dimensions from "
                                f"{old_width}x{old_height} to {expected_width}x{expected_height}"
                            ),
                            layer_id=layer_id,
                            details={
                                "old_width": old_width,
                                "old_height": old_height,
                                "new_width": expected_width,
                                "new_height": expected_height,
                            },
                        )
                    )
            else:
                remaining.append(error)

        return fixes, remaining

    def _fix_bracket_weights(
        self,
        blueprint: Blueprint,
        errors: list[Any],
    ) -> tuple[list[AppliedFix], list[Any]]:
        """Remove bracket weights from prompts"""
        from parlant.applications.vfx_blueprint.constants import ValidationCode

        fixes: list[AppliedFix] = []
        remaining: list[Any] = []

        for error in errors:
            if error.code == ValidationCode.BRACKET_WEIGHT_DETECTED:
                layer_id = error.layer_id
                if layer_id in blueprint.layers:
                    layer = blueprint.layers[layer_id]
                    original_prompt = layer.prompt

                    # Remove bracket weights
                    layer.prompt = self.BRACKET_WEIGHT_PATTERN.sub("", layer.prompt)

                    fixes.append(
                        AppliedFix(
                            code=AutoFixCode.REMOVED_BRACKET_WEIGHTS,
                            message=(
                                f"Removed bracket weights from layer '{layer_id}' prompt"
                            ),
                            layer_id=layer_id,
                            details={
                                "original_prompt": original_prompt,
                                "new_prompt": layer.prompt,
                            },
                        )
                    )
            else:
                remaining.append(error)

        return fixes, remaining

    def _fix_background_prompts(
        self,
        blueprint: Blueprint,
        errors: list[Any],
    ) -> tuple[list[AppliedFix], list[Any]]:
        """Simplify background prompts when concrete objects are detected"""
        from parlant.applications.vfx_blueprint.constants import ValidationCode

        fixes: list[AppliedFix] = []
        remaining: list[Any] = []

        for error in errors:
            if error.code == ValidationCode.IMPROPER_BACKGROUND_CONTENT:
                layer_id = error.layer_id
                if layer_id in blueprint.layers and layer_id == "A":
                    layer = blueprint.layers[layer_id]
                    original_prompt = layer.prompt

                    # Check for concrete keywords
                    prompt_lower = layer.prompt.lower()
                    has_concrete = any(
                        keyword in prompt_lower for keyword in self.CONCRETE_KEYWORDS
                    )

                    if has_concrete:
                        # Simplify to generic background prompt
                        layer.prompt = "Clean, neutral background"

                        fixes.append(
                            AppliedFix(
                                code=AutoFixCode.SIMPLIFIED_BACKGROUND_PROMPT,
                                message=(
                                    f"Simplified layer '{layer_id}' prompt "
                                    f"from concrete objects to generic background"
                                ),
                                layer_id=layer_id,
                                details={
                                    "original_prompt": original_prompt,
                                    "new_prompt": layer.prompt,
                                },
                            )
                        )
            else:
                remaining.append(error)

        return fixes, remaining

    def _fix_green_screen_conflicts(
        self,
        blueprint: Blueprint,
        errors: list[Any],
    ) -> tuple[list[AppliedFix], list[Any]]:
        """Remove green-screen keywords from non-background layers"""
        from parlant.applications.vfx_blueprint.constants import ValidationCode

        GREEN_SCREEN_KEYWORDS = {"green screen", "chroma key", "solid color"}

        fixes: list[AppliedFix] = []
        remaining: list[Any] = []

        for error in errors:
            if error.code == ValidationCode.GREEN_SCREEN_CONFLICT:
                layer_id = error.layer_id
                if layer_id in blueprint.layers:
                    layer = blueprint.layers[layer_id]
                    original_prompt = layer.prompt

                    # Remove green-screen keywords
                    new_prompt = original_prompt
                    for keyword in GREEN_SCREEN_KEYWORDS:
                        new_prompt = new_prompt.replace(keyword, "").strip()

                    if new_prompt != original_prompt:
                        layer.prompt = new_prompt

                        fixes.append(
                            AppliedFix(
                                code=AutoFixCode.RESOLVED_GREEN_SCREEN,
                                message=(
                                    f"Removed green-screen keywords from layer '{layer_id}'"
                                ),
                                layer_id=layer_id,
                                details={
                                    "original_prompt": original_prompt,
                                    "new_prompt": layer.prompt,
                                },
                            )
                        )
            else:
                remaining.append(error)

        return fixes, remaining

    def _fix_empty_prompts(
        self,
        blueprint: Blueprint,
        errors: list[Any],
    ) -> tuple[list[AppliedFix], list[Any]]:
        """Insert default values for empty prompts"""
        from parlant.applications.vfx_blueprint.constants import ValidationCode

        fixes: list[AppliedFix] = []
        remaining: list[Any] = []

        for error in errors:
            if error.code == ValidationCode.EMPTY_PROMPT:
                layer_id = error.layer_id
                if layer_id in blueprint.layers:
                    layer = blueprint.layers[layer_id]

                    # Insert default prompt based on layer type
                    default_prompts = {
                        "A": "Clean, neutral background",
                        "B": "Content overlay",
                        "C": "Content overlay",
                        "D": "Content overlay",
                        "T": "Title or text display",
                    }

                    default_prompt = default_prompts.get(layer_id, "Placeholder content")
                    layer.prompt = default_prompt

                    fixes.append(
                        AppliedFix(
                            code=AutoFixCode.INSERTED_DEFAULT_VALUE,
                            message=(
                                f"Inserted default prompt for empty layer '{layer_id}'"
                            ),
                            layer_id=layer_id,
                            details={"default_prompt": default_prompt},
                        )
                    )
            else:
                remaining.append(error)

        return fixes, remaining

    def _deep_copy_blueprint(self, blueprint: Blueprint) -> Blueprint:
        """Create a deep copy of the blueprint for immutability support"""
        new_layers: dict[str, LayerContent] = {}

        for layer_id, layer in blueprint.layers.items():
            new_layer = LayerContent(
                layer_id=layer.layer_id,
                width=layer.width,
                height=layer.height,
                prompt=layer.prompt,
                metadata=copy.deepcopy(layer.metadata),
            )
            new_layers[layer_id] = new_layer

        return Blueprint(
            layers=new_layers,
            metadata=copy.deepcopy(blueprint.metadata),
            version=blueprint.version,
        )

    def _generate_summary(
        self,
        applied_fixes: list[AppliedFix],
        remaining_errors: list[Any],
        is_valid: bool,
    ) -> str:
        """Generate a human-readable summary of auto-fix results"""
        if is_valid:
            if applied_fixes:
                return (
                    f"Auto-fix successful: {len(applied_fixes)} fix(es) applied, "
                    f"blueprint is now valid"
                )
            else:
                return "No fixes needed - blueprint is valid"

        fix_count = len(applied_fixes)
        error_count = len(remaining_errors)

        return (
            f"Auto-fix completed: {fix_count} fix(es) applied, "
            f"{error_count} error(s) unresolved"
        )
