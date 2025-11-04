from datetime import timedelta
from typing import Any
from pydantic import BaseModel, Field

from parlant.applications.vfx_blueprint.constants import StageName


class Subtitle(BaseModel):
    index: int
    start_time: timedelta
    end_time: timedelta
    text: str


class CameraParams(BaseModel):
    angle: str = "medium"
    movement: str = "static"
    transition: str | None = None


class AnimationParams(BaseModel):
    duration: float
    fps: int = 24
    camera: CameraParams = Field(default_factory=CameraParams)


class Scene(BaseModel):
    scene_id: int
    subtitle_indices: list[int]
    visual_prompt: str
    emotion: str | None = None
    start_time: float
    end_time: float
    animation: AnimationParams | None = None


class Blueprint(BaseModel):
    version: str = "1.0"
    total_duration: float
    scenes: list[Scene]
    metadata: dict[str, Any] = Field(default_factory=dict)


class StageResult(BaseModel):
    stage: StageName
    success: bool
    output: Any | None = None
    error: str | None = None
    duration: float
    retries: int = 0


class OrchestratorResult(BaseModel):
    success: bool
    blueprint: Blueprint | None = None
    stage_results: list[StageResult]
    total_duration: float
