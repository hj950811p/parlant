from pathlib import Path
import pytest
import json
import subprocess
import sys
import os


class TestCLI:
    @pytest.fixture
    def sample_srt_path(self) -> Path:
        return Path(__file__).parent / "fixtures" / "sample.srt"

    @pytest.fixture
    def script_path(self) -> Path:
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        return project_root / "examples" / "AI_VFX_Director_Pro_Enhanced.py"

    def test_that_cli_accepts_srt_file_input_and_outputs_json(
        self,
        sample_srt_path: Path,
        tmp_path: Path,
        script_path: Path,
    ) -> None:
        output_file = tmp_path / "blueprint.json"

        result = subprocess.run(
            [
                sys.executable,
                str(script_path),
                "--srt-file",
                str(sample_srt_path),
                "--output",
                str(output_file),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert output_file.exists()

        with open(output_file) as f:
            blueprint = json.load(f)

        assert "version" in blueprint
        assert "scenes" in blueprint
        assert "total_duration" in blueprint

    def test_that_cli_accepts_config_file_override(
        self,
        sample_srt_path: Path,
        tmp_path: Path,
        script_path: Path,
    ) -> None:
        config_file = tmp_path / "config.yaml"
        config_content = """
cerebras:
  model: llama3.3-70b
  temperature: 0.8
stages:
  max_retries: 2
"""
        with open(config_file, "w") as f:
            f.write(config_content)

        output_file = tmp_path / "blueprint.json"

        result = subprocess.run(
            [
                sys.executable,
                str(script_path),
                "--srt-file",
                str(sample_srt_path),
                "--output",
                str(output_file),
                "--config",
                str(config_file),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert output_file.exists()

    def test_that_cli_maintains_backward_compatible_output_schema(
        self,
        sample_srt_path: Path,
        tmp_path: Path,
        script_path: Path,
    ) -> None:
        output_file = tmp_path / "blueprint.json"

        result = subprocess.run(
            [
                sys.executable,
                str(script_path),
                "--srt-file",
                str(sample_srt_path),
                "--output",
                str(output_file),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0

        with open(output_file) as f:
            blueprint = json.load(f)

        assert blueprint["version"] == "1.0"
        assert isinstance(blueprint["scenes"], list)
        assert len(blueprint["scenes"]) > 0

        scene = blueprint["scenes"][0]
        assert "scene_id" in scene
        assert "visual_prompt" in scene
        assert "start_time" in scene
        assert "end_time" in scene
        assert "animation" in scene
