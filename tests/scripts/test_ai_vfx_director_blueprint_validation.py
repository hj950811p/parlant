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
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from AI_VFX_Director_Pro_Enhanced import (
    SRTParser,
    BlueprintValidator,
    BlueprintRepairer,
    StructuredLogger,
    Blueprint,
)


class TestSRTParser:
    """Tests for SRT file parsing."""

    def test_srt_parser_extracts_timecodes_and_text_from_valid_srt_file(self) -> None:
        srt_content = """1
00:00:00,000 --> 00:00:05,000
First subtitle

2
00:00:05,000 --> 00:00:10,000
Second subtitle

3
00:00:10,000 --> 00:00:15,000
Third subtitle
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".srt", delete=False) as f:
            f.write(srt_content)
            f.flush()

            parser = SRTParser()
            entries = parser.parse(f.name)

            assert len(entries) == 3
            assert entries[0]["start"] == "00:00:00,000"
            assert entries[0]["end"] == "00:00:05,000"
            assert entries[0]["text"] == "First subtitle"
            assert entries[2]["text"] == "Third subtitle"

            Path(f.name).unlink()

    def test_srt_parser_handles_empty_srt_file(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".srt", delete=False) as f:
            f.write("")
            f.flush()

            parser = SRTParser()
            entries = parser.parse(f.name)

            assert len(entries) == 0

            Path(f.name).unlink()

    def test_srt_parser_handles_multiline_subtitles(self) -> None:
        srt_content = """1
00:00:00,000 --> 00:00:05,000
First line
Second line
Third line
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".srt", delete=False) as f:
            f.write(srt_content)
            f.flush()

            parser = SRTParser()
            entries = parser.parse(f.name)

            assert len(entries) == 1
            assert "First line" in entries[0]["text"]
            assert "Second line" in entries[0]["text"]
            assert "Third line" in entries[0]["text"]

            Path(f.name).unlink()


class TestBlueprintValidator:
    """Tests for blueprint validation."""

    def test_blueprint_validator_accepts_valid_blueprint(self) -> None:
        blueprint_data = {
            "id": "bp_001",
            "timecode": "00:00:00,000",
            "frame_dimensions": {"width": 1920, "height": 1080},
            "background_description": "A simple outdoor scene",
            "required_keys": ["id", "timecode", "frame_dimensions"],
        }
        blueprint = Blueprint(**blueprint_data)
        validator = BlueprintValidator()

        errors = validator.validate(blueprint)
        assert len(errors) == 0

    def test_blueprint_validator_detects_invalid_frame_dimensions_width(self) -> None:
        blueprint_data = {
            "id": "bp_001",
            "timecode": "00:00:00,000",
            "frame_dimensions": {"width": 100, "height": 1080},
            "background_description": "A simple outdoor scene",
            "required_keys": ["id", "timecode", "frame_dimensions"],
        }
        blueprint = Blueprint(**blueprint_data)
        validator = BlueprintValidator()

        errors = validator.validate(blueprint)
        assert len(errors) > 0
        assert any("width" in error.lower() for error in errors)

    def test_blueprint_validator_detects_invalid_frame_dimensions_height(self) -> None:
        blueprint_data = {
            "id": "bp_001",
            "timecode": "00:00:00,000",
            "frame_dimensions": {"width": 1920, "height": 20000},
            "background_description": "A simple outdoor scene",
            "required_keys": ["id", "timecode", "frame_dimensions"],
        }
        blueprint = Blueprint(**blueprint_data)
        validator = BlueprintValidator()

        errors = validator.validate(blueprint)
        assert len(errors) > 0
        assert any("height" in error.lower() for error in errors)

    def test_blueprint_validator_detects_missing_required_keys(self) -> None:
        blueprint_data = {
            "id": "bp_001",
            "timecode": "00:00:00,000",
            "frame_dimensions": {"width": 1920, "height": 1080},
            "background_description": "A simple outdoor scene",
            "required_keys": ["id", "timecode", "frame_dimensions", "missing_key"],
        }
        blueprint = Blueprint(**blueprint_data)
        validator = BlueprintValidator()

        errors = validator.validate(blueprint)
        assert len(errors) > 0

    def test_blueprint_validator_detects_non_compliant_background_description(self) -> None:
        blueprint_data = {
            "id": "bp_001",
            "timecode": "00:00:00,000",
            "frame_dimensions": {"width": 1920, "height": 1080},
            "background_description": "Use advanced algorithmic rendering with RAY_TRACING_ENGINE and compute shader_model_advanced",
            "required_keys": ["id", "timecode", "frame_dimensions"],
        }
        blueprint = Blueprint(**blueprint_data)
        validator = BlueprintValidator()

        errors = validator.validate(blueprint)
        assert len(errors) > 0


class TestBlueprintRepairer:
    """Tests for blueprint auto-fix routines."""

    def test_blueprint_auto_fixer_normalizes_incorrect_frame_width_dimensions(self) -> None:
        blueprint_data = {
            "id": "bp_001",
            "timecode": "00:00:00,000",
            "frame_dimensions": {"width": 100, "height": 1080},
            "background_description": "A simple scene",
            "required_keys": ["id", "timecode", "frame_dimensions"],
        }
        blueprint = Blueprint(**blueprint_data)
        logger = StructuredLogger()
        repairer = BlueprintRepairer(logger)

        fixed_blueprint = repairer.repair(blueprint)

        assert fixed_blueprint.frame_dimensions.width >= 320
        assert fixed_blueprint.frame_dimensions.width <= 7680
        assert len(logger.fixes) > 0

    def test_blueprint_auto_fixer_normalizes_incorrect_frame_height_dimensions(self) -> None:
        blueprint_data = {
            "id": "bp_001",
            "timecode": "00:00:00,000",
            "frame_dimensions": {"width": 1920, "height": 20000},
            "background_description": "A simple scene",
            "required_keys": ["id", "timecode", "frame_dimensions"],
        }
        blueprint = Blueprint(**blueprint_data)
        logger = StructuredLogger()
        repairer = BlueprintRepairer(logger)

        fixed_blueprint = repairer.repair(blueprint)

        assert fixed_blueprint.frame_dimensions.height >= 180
        assert fixed_blueprint.frame_dimensions.height <= 4320
        assert len(logger.fixes) > 0

    def test_blueprint_auto_fixer_rewrites_non_compliant_background(self) -> None:
        blueprint_data = {
            "id": "bp_001",
            "timecode": "00:00:00,000",
            "frame_dimensions": {"width": 1920, "height": 1080},
            "background_description": "Use advanced RAY_TRACING_ENGINE with shader_model_v3 rendering",
            "required_keys": ["id", "timecode", "frame_dimensions"],
        }
        blueprint = Blueprint(**blueprint_data)
        logger = StructuredLogger()
        repairer = BlueprintRepairer(logger)

        fixed_blueprint = repairer.repair(blueprint)

        assert "RAY_TRACING" not in fixed_blueprint.background_description.upper()
        assert "SHADER_MODEL" not in fixed_blueprint.background_description.upper()
        assert len(logger.fixes) > 0

    def test_blueprint_auto_fixer_supplies_default_values_for_omitted_fields(self) -> None:
        blueprint_data = {
            "id": "bp_001",
            "timecode": "00:00:00,000",
            "frame_dimensions": {"width": 1920, "height": 1080},
            "background_description": "A simple scene",
            "required_keys": ["id", "timecode", "frame_dimensions"],
        }
        blueprint = Blueprint(**blueprint_data)
        logger = StructuredLogger()
        repairer = BlueprintRepairer(logger)

        fixed_blueprint = repairer.repair(blueprint)

        assert fixed_blueprint.metadata is not None
        assert len(fixed_blueprint.metadata) > 0

    def test_blueprint_auto_fixer_maintains_json_schema_structure(self) -> None:
        blueprint_data = {
            "id": "bp_001",
            "timecode": "00:00:00,000",
            "frame_dimensions": {"width": 100, "height": 20000},
            "background_description": "Use RAY_TRACING rendering",
            "required_keys": ["id", "timecode"],
        }
        blueprint = Blueprint(**blueprint_data)
        logger = StructuredLogger()
        repairer = BlueprintRepairer(logger)

        repairer.repair(blueprint)

        schema = Blueprint.model_json_schema()
        assert "properties" in schema
        assert set(schema["properties"].keys()) == set(Blueprint.model_fields.keys())


class TestStructuredLogger:
    """Tests for structured logging and reporting."""

    def test_structured_logger_records_fixes_performed(self) -> None:
        logger = StructuredLogger()

        logger.record_fix(
            "dimension_normalization", {"field": "width", "original": 100, "fixed": 320}
        )
        logger.record_fix("background_rewrite", {"original": "complex", "fixed": "simple"})

        assert len(logger.fixes) == 2
        assert logger.fixes[0]["type"] == "dimension_normalization"
        assert logger.fixes[1]["type"] == "background_rewrite"

    def test_structured_logger_provides_summary_report(self) -> None:
        logger = StructuredLogger()

        logger.record_fix("dimension_normalization", {"field": "width"})
        logger.record_fix("background_rewrite", {"field": "background"})
        logger.record_fix("default_values", {"field": "metadata"})

        report = logger.get_summary_report()

        assert "dimension_normalization" in report
        assert "background_rewrite" in report
        assert "default_values" in report

    def test_structured_logger_tracks_multiple_fixes_per_blueprint(self) -> None:
        logger = StructuredLogger()

        logger.record_fix("type_1", {"detail": "a"})
        logger.record_fix("type_1", {"detail": "b"})
        logger.record_fix("type_2", {"detail": "c"})

        assert len(logger.fixes) == 3
        assert logger.fixes[0]["type"] == "type_1"
        assert logger.fixes[1]["type"] == "type_1"
        assert logger.fixes[2]["type"] == "type_2"


class TestEndToEndPipeline:
    """End-to-end tests for the complete pipeline."""

    def test_end_to_end_srt_to_validated_json_pipeline(self) -> None:
        srt_content = """1
00:00:00,000 --> 00:00:05,000
Frame with invalid dimensions
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".srt", delete=False) as f:
            f.write(srt_content)
            f.flush()

            parser = SRTParser()
            entries = parser.parse(f.name)

            blueprint_data = {
                "id": "bp_001",
                "timecode": entries[0]["start"],
                "frame_dimensions": {"width": 100, "height": 20000},
                "background_description": "Use RAY_TRACING rendering",
                "required_keys": ["id", "timecode"],
            }

            blueprint = Blueprint(**blueprint_data)
            logger = StructuredLogger()
            repairer = BlueprintRepairer(logger)
            fixed_blueprint = repairer.repair(blueprint)

            validator = BlueprintValidator()
            errors = validator.validate(fixed_blueprint)

            assert len(errors) == 0
            assert len(logger.fixes) > 0

            json_output = fixed_blueprint.model_dump_json()
            parsed_json = json.loads(json_output)
            assert parsed_json["id"] == "bp_001"

            Path(f.name).unlink()

    def test_end_to_end_multiple_srt_entries_produce_multiple_valid_blueprints(self) -> None:
        srt_content = """1
00:00:00,000 --> 00:00:05,000
First frame

2
00:00:05,000 --> 00:00:10,000
Second frame

3
00:00:10,000 --> 00:00:15,000
Third frame
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".srt", delete=False) as f:
            f.write(srt_content)
            f.flush()

            parser = SRTParser()
            entries = parser.parse(f.name)

            blueprints = []
            logger = StructuredLogger()
            validator = BlueprintValidator()
            repairer = BlueprintRepairer(logger)

            for idx, entry in enumerate(entries):
                blueprint_data = {
                    "id": f"bp_{idx:03d}",
                    "timecode": entry["start"],
                    "frame_dimensions": {"width": 1920 - idx * 100, "height": 1080},
                    "background_description": "A scene",
                    "required_keys": ["id", "timecode"],
                }
                blueprint = Blueprint(**blueprint_data)
                fixed_blueprint = repairer.repair(blueprint)
                errors = validator.validate(fixed_blueprint)
                assert len(errors) == 0
                blueprints.append(fixed_blueprint)

            assert len(blueprints) == 3

            Path(f.name).unlink()

    def test_end_to_end_output_maintains_required_json_schema(self) -> None:
        blueprint_data = {
            "id": "bp_001",
            "timecode": "00:00:00,000",
            "frame_dimensions": {"width": 1920, "height": 1080},
            "background_description": "A scene",
            "required_keys": ["id", "timecode"],
        }
        blueprint = Blueprint(**blueprint_data)
        json_output = blueprint.model_dump_json(indent=2)
        parsed = json.loads(json_output)

        assert "id" in parsed
        assert "timecode" in parsed
        assert "frame_dimensions" in parsed
        assert "width" in parsed["frame_dimensions"]
        assert "height" in parsed["frame_dimensions"]
        assert "background_description" in parsed
        assert "required_keys" in parsed
