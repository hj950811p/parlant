from typing import Any

from parlant.applications.vfx_blueprint.stages.base import Stage
from parlant.applications.vfx_blueprint.stages.structure import SceneStructure
from parlant.applications.vfx_blueprint.models import Subtitle, Blueprint


class VerifyStage(Stage):
    async def execute(
        self,
        subtitles: list[Subtitle],
        previous_output: Any | None = None,
    ) -> Blueprint:
        if not isinstance(previous_output, SceneStructure):
            raise ValueError("VerifyStage requires SceneStructure from previous stage")

        scenes = previous_output.scenes

        if not scenes:
            raise ValueError("No scenes to verify")

        total_duration = max(scene.end_time for scene in scenes)

        for scene in scenes:
            if not scene.visual_prompt:
                raise ValueError(f"Scene {scene.scene_id} missing visual_prompt")
            if scene.animation is None:
                raise ValueError(f"Scene {scene.scene_id} missing animation parameters")

        blueprint = Blueprint(
            total_duration=total_duration,
            scenes=scenes,
            metadata={
                "total_scenes": len(scenes),
                "total_subtitles": len(subtitles),
                "verified": True,
            },
        )

        return blueprint

    async def validate_output(self, output: Any) -> bool:
        if not isinstance(output, Blueprint):
            return False

        if output.total_duration <= 0:
            return False

        if not output.scenes:
            return False

        for scene in output.scenes:
            if not scene.visual_prompt:
                return False
            if scene.animation is None:
                return False

        return True
