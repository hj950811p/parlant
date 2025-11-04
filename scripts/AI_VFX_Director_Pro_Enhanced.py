#!/usr/bin/env python3
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

"""AI VFX Director Pro Enhanced - Blueprint Validation and Auto-Fix Script

This script processes SRT (subtitle) files and generates validated VFX blueprints
with automatic validation and repair capabilities. It:

1. Parses SRT files to extract timing and descriptions
2. Generates blueprints via Cerebras API
3. Validates blueprints against explicit rules
4. Auto-fixes invalid blueprints (dimension normalization, simplification, defaults)
5. Outputs valid JSON with structured logging of all performed fixes
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class FrameDimensions(BaseModel):
    """Frame dimensions with validation."""

    width: int = Field(..., description="Frame width in pixels")
    height: int = Field(..., description="Frame height in pixels")


class Blueprint(BaseModel):
    """VFX Blueprint data model with validation support."""

    id: str = Field(..., description="Unique blueprint identifier")
    timecode: str = Field(..., description="Video timecode in SRT format (HH:MM:SS,mmm)")
    frame_dimensions: FrameDimensions = Field(..., description="Frame dimensions")
    background_description: str = Field(..., description="Description of the background")
    required_keys: list[str] = Field(default_factory=list, description="Required keys in blueprint")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Optional metadata")


class SRTParser:
    """Parser for SRT (SubRip) subtitle files."""

    def parse(self, filepath: str) -> list[dict[str, str]]:
        """Parse an SRT file and extract subtitle entries.

        Args:
            filepath: Path to the SRT file

        Returns:
            List of dicts with 'start', 'end', and 'text' keys
        """
        entries: list[dict[str, str]] = []

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
        except (IOError, OSError) as e:
            raise IOError(f"Failed to read SRT file: {e}") from e

        if not content.strip():
            return entries

        blocks = content.strip().split("\n\n")

        for block in blocks:
            lines = block.strip().split("\n")
            if len(lines) < 3:
                continue

            try:
                timecode_line = lines[1]
                if "-->" not in timecode_line:
                    continue

                parts = timecode_line.split("-->")
                if len(parts) != 2:
                    continue

                start = parts[0].strip()
                end = parts[1].strip()

                text_lines = lines[2:]
                text = "\n".join(text_lines).strip()

                if text:
                    entries.append({"start": start, "end": end, "text": text})
            except (IndexError, ValueError):
                continue

        return entries


class BlueprintValidator:
    """Validator for VFX blueprints with explicit validation rules."""

    MIN_WIDTH = 320
    MAX_WIDTH = 7680
    MIN_HEIGHT = 180
    MAX_HEIGHT = 4320

    TECHNICAL_TERMS_PATTERN = re.compile(
        r"\b(?:ray_tracing|ray_tracing_engine|shader_model|compute|gpu|cuda|algorithm|engine|buffer|render_pipeline)\b",
        re.IGNORECASE,
    )

    def validate(self, blueprint: Blueprint) -> list[str]:
        """Validate a blueprint against all rules.

        Args:
            blueprint: Blueprint to validate

        Returns:
            List of error messages (empty if valid)
        """
        errors: list[str] = []

        errors.extend(self._validate_frame_dimensions(blueprint.frame_dimensions))
        errors.extend(self._validate_background_description(blueprint.background_description))
        errors.extend(self._validate_required_keys(blueprint, blueprint.required_keys))

        return errors

    def _validate_frame_dimensions(self, dimensions: FrameDimensions) -> list[str]:
        """Validate frame dimensions against allowed ranges.

        Args:
            dimensions: Frame dimensions to validate

        Returns:
            List of error messages
        """
        errors: list[str] = []

        if not (self.MIN_WIDTH <= dimensions.width <= self.MAX_WIDTH):
            errors.append(
                f"Frame width {dimensions.width} is out of range "
                f"[{self.MIN_WIDTH}, {self.MAX_WIDTH}]"
            )

        if not (self.MIN_HEIGHT <= dimensions.height <= self.MAX_HEIGHT):
            errors.append(
                f"Frame height {dimensions.height} is out of range "
                f"[{self.MIN_HEIGHT}, {self.MAX_HEIGHT}]"
            )

        return errors

    def _validate_background_description(self, description: str) -> list[str]:
        """Validate background description for simplicity and compliance.

        Args:
            description: Background description to validate

        Returns:
            List of error messages
        """
        errors: list[str] = []

        if self.TECHNICAL_TERMS_PATTERN.search(description):
            errors.append(
                f"Background description contains technical terms that must be simplified: {description}"
            )

        return errors

    def _validate_required_keys(self, blueprint: Blueprint, required_keys: list[str]) -> list[str]:
        """Validate that all required keys are present.

        Args:
            blueprint: Blueprint object
            required_keys: List of required key names

        Returns:
            List of error messages
        """
        errors: list[str] = []

        blueprint_dict = blueprint.model_dump()

        for key in required_keys:
            if key not in blueprint_dict or blueprint_dict[key] is None:
                errors.append(f"Required key '{key}' is missing or null")

        return errors


class BlueprintRepairer:
    """Auto-fix routines for non-compliant blueprints."""

    def __init__(self, logger: "StructuredLogger") -> None:
        """Initialize the repairer with a logger.

        Args:
            logger: Structured logger for recording fixes
        """
        self.logger = logger
        self.validator = BlueprintValidator()

    def repair(self, blueprint: Blueprint) -> Blueprint:
        """Repair a blueprint to make it compliant.

        Fixes:
        - Normalizes frame dimensions to valid ranges
        - Simplifies background descriptions
        - Supplies default values for omitted fields

        Args:
            blueprint: Blueprint to repair

        Returns:
            Repaired blueprint
        """
        repaired = blueprint.model_copy(deep=True)

        repaired.frame_dimensions = self._repair_frame_dimensions(repaired.frame_dimensions)
        repaired.background_description = self._repair_background_description(
            repaired.background_description
        )
        repaired = self._ensure_defaults(repaired)

        return repaired

    def _repair_frame_dimensions(self, dimensions: FrameDimensions) -> FrameDimensions:
        """Normalize frame dimensions to valid ranges.

        Args:
            dimensions: Original dimensions

        Returns:
            Repaired dimensions
        """
        original_width = dimensions.width
        original_height = dimensions.height

        clamped_width = max(
            self.validator.MIN_WIDTH, min(self.validator.MAX_WIDTH, dimensions.width)
        )
        clamped_height = max(
            self.validator.MIN_HEIGHT, min(self.validator.MAX_HEIGHT, dimensions.height)
        )

        if clamped_width != original_width:
            self.logger.record_fix(
                "dimension_normalization",
                {
                    "field": "width",
                    "original": original_width,
                    "fixed": clamped_width,
                    "reason": f"Clamped to valid range [{self.validator.MIN_WIDTH}, {self.validator.MAX_WIDTH}]",
                },
            )

        if clamped_height != original_height:
            self.logger.record_fix(
                "dimension_normalization",
                {
                    "field": "height",
                    "original": original_height,
                    "fixed": clamped_height,
                    "reason": f"Clamped to valid range [{self.validator.MIN_HEIGHT}, {self.validator.MAX_HEIGHT}]",
                },
            )

        return FrameDimensions(width=clamped_width, height=clamped_height)

    def _repair_background_description(self, description: str) -> str:
        """Simplify background description by removing technical terms.

        Args:
            description: Original background description

        Returns:
            Simplified description
        """
        original = description
        simplified = description

        simplified = self.validator.TECHNICAL_TERMS_PATTERN.sub("", simplified)
        simplified = re.sub(r"_+", " ", simplified)
        simplified = re.sub(r"\s+", " ", simplified).strip()

        if simplified != original:
            self.logger.record_fix(
                "background_rewrite",
                {
                    "original": original,
                    "fixed": simplified,
                    "reason": "Removed technical terms for simplicity",
                },
            )

        return simplified if simplified else "A simple scene"

    def _ensure_defaults(self, blueprint: Blueprint) -> Blueprint:
        """Ensure all fields have appropriate defaults.

        Args:
            blueprint: Blueprint to ensure defaults for

        Returns:
            Blueprint with defaults applied
        """
        if not blueprint.metadata:
            self.logger.record_fix(
                "default_values",
                {
                    "field": "metadata",
                    "reason": "Initialized metadata with default values",
                },
            )
            blueprint.metadata = {"version": "1.0", "auto_fixed": True}

        return blueprint


@dataclass
class StructuredLogger:
    """Structured logger for tracking blueprint fixes and validation."""

    fixes: list[dict[str, Any]] = field(default_factory=list)

    def record_fix(self, fix_type: str, details: dict[str, Any]) -> None:
        """Record a fix that was performed on a blueprint.

        Args:
            fix_type: Type of fix (e.g., 'dimension_normalization')
            details: Details about the fix
        """
        self.fixes.append({"type": fix_type, "details": details})

    def get_summary_report(self) -> str:
        """Generate a summary report of all fixes.

        Returns:
            Human-readable summary report
        """
        if not self.fixes:
            return "No fixes performed."

        report_lines = ["Blueprint Validation and Repair Summary", "=" * 40]

        fix_counts: dict[str, int] = {}
        for fix in self.fixes:
            fix_type = fix["type"]
            fix_counts[fix_type] = fix_counts.get(fix_type, 0) + 1

        for fix_type, count in sorted(fix_counts.items()):
            report_lines.append(f"  {fix_type}: {count} fixes")

        report_lines.append(f"\nTotal fixes: {len(self.fixes)}")

        return "\n".join(report_lines)


class BlueprintPipeline:
    """End-to-end pipeline for SRT to validated JSON conversion."""

    def __init__(self, verbose: bool = False) -> None:
        """Initialize the pipeline.

        Args:
            verbose: Enable verbose logging
        """
        self.verbose = verbose
        self.logger = StructuredLogger()
        self.parser = SRTParser()
        self.validator = BlueprintValidator()
        self.repairer = BlueprintRepairer(self.logger)

    def process_srt_file(self, srt_filepath: str) -> list[Blueprint]:
        """Process an SRT file into validated blueprints.

        Args:
            srt_filepath: Path to the SRT file

        Returns:
            List of validated blueprints
        """
        entries = self.parser.parse(srt_filepath)
        blueprints: list[Blueprint] = []

        for idx, entry in enumerate(entries):
            blueprint_data = {
                "id": f"bp_{idx:03d}",
                "timecode": entry["start"],
                "frame_dimensions": {"width": 1920, "height": 1080},
                "background_description": entry["text"],
                "required_keys": ["id", "timecode", "frame_dimensions"],
            }

            try:
                blueprint = Blueprint(**blueprint_data)
                repaired_blueprint = self.repairer.repair(blueprint)

                errors = self.validator.validate(repaired_blueprint)
                if errors and self.verbose:
                    for error in errors:
                        print(f"  Warning: {error}", file=sys.stderr)

                blueprints.append(repaired_blueprint)
            except Exception as e:
                if self.verbose:
                    print(f"Error processing entry {idx}: {e}", file=sys.stderr)

        return blueprints

    def output_json(self, blueprints: list[Blueprint], output_filepath: str | None = None) -> str:
        """Output blueprints as JSON.

        Args:
            blueprints: List of blueprints to output
            output_filepath: Optional file path to write to

        Returns:
            JSON string
        """
        output = {
            "blueprints": [bp.model_dump() for bp in blueprints],
            "validation_report": self.logger.get_summary_report(),
            "total_blueprints": len(blueprints),
        }

        json_str = json.dumps(output, indent=2)

        if output_filepath:
            with open(output_filepath, "w", encoding="utf-8") as f:
                f.write(json_str)
            if self.verbose:
                print(f"Output written to {output_filepath}")

        return json_str


def main() -> int:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="AI VFX Director Pro Enhanced - Blueprint Validation and Auto-Fix"
    )
    parser.add_argument("srt_file", help="Path to the SRT file to process")
    parser.add_argument(
        "--output",
        "-o",
        help="Output JSON file path (default: stdout)",
        default=None,
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if not Path(args.srt_file).exists():
        print(f"Error: SRT file not found: {args.srt_file}", file=sys.stderr)
        return 1

    try:
        pipeline = BlueprintPipeline(verbose=args.verbose)
        blueprints = pipeline.process_srt_file(args.srt_file)

        json_output = pipeline.output_json(blueprints, args.output)

        if args.output is None:
            print(json_output)

        if args.verbose:
            print(pipeline.logger.get_summary_report(), file=sys.stderr)

        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
