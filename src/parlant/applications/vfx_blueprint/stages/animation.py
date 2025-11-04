from typing import Any

from parlant.applications.vfx_blueprint.stages.base import Stage
from parlant.applications.vfx_blueprint.stages.structure import SceneStructure
from parlant.applications.vfx_blueprint.models import Subtitle, AnimationParams, CameraParams


class AnimationStage(Stage):
    async def execute(
        self,
        subtitles: list[Subtitle],
        previous_output: Any | None = None,
    ) -> SceneStructure:
        if not isinstance(previous_output, SceneStructure):
            raise ValueError("AnimationStage requires SceneStructure from previous stage")

        for scene in previous_output.scenes:
            duration = scene.end_time - scene.start_time

            camera_params = await self._determine_camera_params(scene.emotion)

            scene.animation = AnimationParams(
                duration=duration,
                fps=24,
                camera=camera_params,
            )

        return previous_output

    async def _determine_camera_params(self, emotion: str | None) -> CameraParams:
        if emotion == "excited":
            return CameraParams(
                angle="wide",
                movement="dynamic",
                transition="quick_cut",
            )
        elif emotion == "sad":
            return CameraParams(
                angle="close",
                movement="slow_pan",
                transition="fade",
            )
        elif emotion == "calm":
            return CameraParams(
                angle="medium",
                movement="static",
                transition="dissolve",
            )
        else:
            return CameraParams(
                angle="medium",
                movement="static",
                transition=None,
            )

    async def validate_output(self, output: Any) -> bool:
        if not isinstance(output, SceneStructure):
            return False

        for scene in output.scenes:
            if scene.animation is None:
                return False
            if scene.animation.duration <= 0:
                return False
            if scene.animation.fps <= 0:
                return False

        return True
