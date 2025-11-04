from pathlib import Path
import pytest
from io import StringIO
import yaml

from parlant.applications.vfx_blueprint.config import (
    VFXConfig,
    ConfigLoader,
)


class TestConfigLoader:
    def test_that_config_loader_parses_yaml_file(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_data = {
            "cerebras": {
                "model": "llama3.3-70b",
                "temperature": 0.8,
            },
            "stages": {
                "max_retries": 5,
                "retry_delay": 2.0,
            },
            "output": {
                "save_intermediate": True,
                "intermediate_dir": "/tmp/vfx",
            },
        }

        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        loader = ConfigLoader()
        config = loader.load_from_file(config_file)

        assert config.cerebras_model == "llama3.3-70b"
        assert config.cerebras_temperature == 0.8
        assert config.max_retries == 5
        assert config.retry_delay == 2.0
        assert config.save_intermediate is True
        assert config.intermediate_dir == Path("/tmp/vfx")

    def test_that_config_loader_falls_back_to_defaults_when_file_missing(
        self,
    ) -> None:
        loader = ConfigLoader()
        config = loader.load_from_file(Path("/nonexistent/config.yaml"))

        assert config.cerebras_model == "llama3.3-70b"
        assert config.cerebras_temperature == 0.7
        assert config.max_retries == 3
        assert config.retry_delay == 1.0
        assert config.save_intermediate is True

    def test_that_config_loader_merges_cli_overrides_with_file_config(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_data = {
            "cerebras": {
                "model": "llama3.1-8b",
                "temperature": 0.5,
            },
            "stages": {
                "max_retries": 2,
            },
        }

        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        loader = ConfigLoader()
        config = loader.load_from_file(
            config_file,
            overrides={
                "cerebras_temperature": 0.9,
                "max_retries": 10,
            },
        )

        assert config.cerebras_model == "llama3.1-8b"
        assert config.cerebras_temperature == 0.9
        assert config.max_retries == 10

    def test_that_config_loader_handles_partial_yaml(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_data = {
            "cerebras": {
                "temperature": 0.6,
            },
        }

        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        loader = ConfigLoader()
        config = loader.load_from_file(config_file)

        assert config.cerebras_model == "llama3.3-70b"
        assert config.cerebras_temperature == 0.6

    def test_that_config_supports_json_format(self, tmp_path: Path) -> None:
        import json

        config_file = tmp_path / "config.json"
        config_data = {
            "cerebras": {
                "model": "llama3.3-70b",
                "temperature": 0.7,
            },
            "stages": {
                "max_retries": 4,
            },
        }

        with open(config_file, "w") as f:
            json.dump(config_data, f)

        loader = ConfigLoader()
        config = loader.load_from_file(config_file)

        assert config.cerebras_model == "llama3.3-70b"
        assert config.max_retries == 4
