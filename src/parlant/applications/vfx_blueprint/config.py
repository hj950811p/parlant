import json
from pathlib import Path
from typing import Any
import yaml  # type: ignore
from pydantic import BaseModel, Field

from parlant.applications.vfx_blueprint.constants import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_DELAY,
    DEFAULT_CEREBRAS_MODEL,
    DEFAULT_TEMPERATURE,
)


class VFXConfig(BaseModel):
    cerebras_model: str = DEFAULT_CEREBRAS_MODEL
    cerebras_temperature: float = DEFAULT_TEMPERATURE
    max_retries: int = DEFAULT_MAX_RETRIES
    retry_delay: float = DEFAULT_RETRY_DELAY
    save_intermediate: bool = True
    intermediate_dir: Path = Field(default_factory=lambda: Path(".vfx_intermediate"))


class ConfigLoader:
    def load_from_file(
        self,
        config_path: Path,
        overrides: dict[str, Any] | None = None,
    ) -> VFXConfig:
        config_data: dict[str, Any] = {}

        if config_path.exists():
            with open(config_path, "r") as f:
                if config_path.suffix in [".yaml", ".yml"]:
                    raw_data = yaml.safe_load(f)
                elif config_path.suffix == ".json":
                    raw_data = json.load(f)
                else:
                    raise ValueError(f"Unsupported config format: {config_path.suffix}")

            config_data = self._flatten_config(raw_data or {})

        if overrides:
            config_data.update(overrides)

        return VFXConfig(**config_data)

    def _flatten_config(self, data: dict[str, Any]) -> dict[str, Any]:
        flattened: dict[str, Any] = {}

        if "cerebras" in data:
            if "model" in data["cerebras"]:
                flattened["cerebras_model"] = data["cerebras"]["model"]
            if "temperature" in data["cerebras"]:
                flattened["cerebras_temperature"] = data["cerebras"]["temperature"]

        if "stages" in data:
            if "max_retries" in data["stages"]:
                flattened["max_retries"] = data["stages"]["max_retries"]
            if "retry_delay" in data["stages"]:
                flattened["retry_delay"] = data["stages"]["retry_delay"]

        if "output" in data:
            if "save_intermediate" in data["output"]:
                flattened["save_intermediate"] = data["output"]["save_intermediate"]
            if "intermediate_dir" in data["output"]:
                flattened["intermediate_dir"] = Path(data["output"]["intermediate_dir"])

        return flattened
