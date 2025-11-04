from typing import Any
from pydantic import BaseModel

from parlant.applications.vfx_blueprint.stages.base import Stage
from parlant.applications.vfx_blueprint.models import Subtitle, Scene


class SceneStructure(BaseModel):
    scenes: list[Scene]


class StructureStage(Stage):
    async def execute(
        self,
        subtitles: list[Subtitle],
        previous_output: Any | None = None,
    ) -> SceneStructure:
        scenes: list[Scene] = []

        for i, subtitle in enumerate(subtitles):
            scene = Scene(
                scene_id=i + 1,
                subtitle_indices=[subtitle.index],
                visual_prompt="",
                start_time=subtitle.start_time.total_seconds(),
                end_time=subtitle.end_time.total_seconds(),
            )
            scenes.append(scene)

        return SceneStructure(scenes=scenes)

    async def validate_output(self, output: Any) -> bool:
        if not isinstance(output, SceneStructure):
            return False

        if not output.scenes:
            return False

        for scene in output.scenes:
            if scene.scene_id <= 0:
                return False
            if scene.start_time < 0 or scene.end_time <= scene.start_time:
                return False

        return True
