from datetime import timedelta
import pytest

from parlant.applications.vfx_blueprint.stages.structure import (
    StructureStage,
    SceneStructure,
)
from parlant.applications.vfx_blueprint.models import Subtitle
from parlant.applications.vfx_blueprint.config import VFXConfig
from parlant.core.loggers import Logger
from tests.test_utilities import _TestLogger


class TestStructureStage:
    @pytest.fixture
    def config(self) -> VFXConfig:
        return VFXConfig()

    @pytest.fixture
    def logger(self) -> Logger:
        return _TestLogger()

    @pytest.fixture
    def subtitles(self) -> list[Subtitle]:
        return [
            Subtitle(
                index=1,
                start_time=timedelta(seconds=0),
                end_time=timedelta(seconds=5),
                text="First subtitle",
            ),
            Subtitle(
                index=2,
                start_time=timedelta(seconds=5),
                end_time=timedelta(seconds=10),
                text="Second subtitle",
            ),
        ]

    @pytest.mark.asyncio
    async def test_that_structure_stage_extracts_scene_structure_from_subtitles(
        self,
        config: VFXConfig,
        logger: Logger,
        subtitles: list[Subtitle],
    ) -> None:
        stage = StructureStage(config, logger)
        result = await stage.execute(subtitles)

        assert isinstance(result, SceneStructure)
        assert len(result.scenes) == 2
        assert result.scenes[0].scene_id == 1
        assert result.scenes[0].start_time == 0.0
        assert result.scenes[0].end_time == 5.0
        assert result.scenes[1].scene_id == 2

    @pytest.mark.asyncio
    async def test_that_structure_stage_validates_output_schema(
        self,
        config: VFXConfig,
        logger: Logger,
        subtitles: list[Subtitle],
    ) -> None:
        stage = StructureStage(config, logger)
        result = await stage.execute(subtitles)

        is_valid = await stage.validate_output(result)
        assert is_valid is True

    @pytest.mark.asyncio
    async def test_that_structure_stage_rejects_invalid_output(
        self,
        config: VFXConfig,
        logger: Logger,
    ) -> None:
        stage = StructureStage(config, logger)

        is_valid = await stage.validate_output({"invalid": "data"})
        assert is_valid is False

        is_valid = await stage.validate_output(SceneStructure(scenes=[]))
        assert is_valid is False
