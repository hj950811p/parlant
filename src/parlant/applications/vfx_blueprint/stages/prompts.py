from typing import Any

from parlant.applications.vfx_blueprint.stages.base import Stage
from parlant.applications.vfx_blueprint.stages.structure import SceneStructure
from parlant.applications.vfx_blueprint.models import Subtitle


class PromptsStage(Stage):
    async def execute(
        self,
        subtitles: list[Subtitle],
        previous_output: Any | None = None,
    ) -> SceneStructure:
        if not isinstance(previous_output, SceneStructure):
            raise ValueError("PromptsStage requires SceneStructure from previous stage")

        subtitle_map = {s.index: s for s in subtitles}

        for scene in previous_output.scenes:
            subtitle_texts = [
                subtitle_map[idx].text for idx in scene.subtitle_indices if idx in subtitle_map
            ]

            combined_text = " ".join(subtitle_texts)

            visual_prompt = await self._generate_visual_prompt(combined_text)
            emotion = await self._analyze_emotion(combined_text)

            scene.visual_prompt = visual_prompt
            scene.emotion = emotion

        return previous_output

    async def _generate_visual_prompt(self, text: str) -> str:
        prompt = f"A cinematic scene showing: {text}"
        return prompt

    async def _analyze_emotion(self, text: str) -> str:
        emotions = ["neutral", "happy", "sad", "excited", "calm"]

        text_lower = text.lower()
        if any(word in text_lower for word in ["excited", "amazing", "wonderful"]):
            return "excited"
        elif any(word in text_lower for word in ["sad", "sorry", "unfortunate"]):
            return "sad"
        elif any(word in text_lower for word in ["happy", "joy", "great"]):
            return "happy"
        elif any(word in text_lower for word in ["calm", "peaceful", "serene"]):
            return "calm"

        return "neutral"

    async def validate_output(self, output: Any) -> bool:
        if not isinstance(output, SceneStructure):
            return False

        for scene in output.scenes:
            if not scene.visual_prompt:
                return False

        return True
