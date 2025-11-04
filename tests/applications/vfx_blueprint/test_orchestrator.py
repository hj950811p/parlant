from datetime import timedelta
from pathlib import Path
import pytest
import json

from parlant.applications.vfx_blueprint.orchestrator import Orchestrator
from parlant.applications.vfx_blueprint.models import Subtitle, Blueprint
from parlant.applications.vfx_blueprint.config import VFXConfig
from parlant.core.loggers import Logger
from tests.test_utilities import _TestLogger


class TestOrchestrator:
    @pytest.fixture
    def config(self, tmp_path: Path) -> VFXConfig:
        return VFXConfig(
            intermediate_dir=tmp_path / "intermediate",
            save_intermediate=True,
        )

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
    async def test_that_orchestrator_executes_all_stages_in_sequence(
        self,
        config: VFXConfig,
        logger: Logger,
        subtitles: list[Subtitle],
    ) -> None:
        orchestrator = Orchestrator(config, logger)
        result = await orchestrator.run(subtitles)

        assert result.success is True
        assert result.blueprint is not None
        assert len(result.stage_results) == 4
        assert all(stage_result.success for stage_result in result.stage_results)

    @pytest.mark.asyncio
    async def test_that_orchestrator_saves_intermediate_outputs(
        self,
        config: VFXConfig,
        logger: Logger,
        subtitles: list[Subtitle],
    ) -> None:
        orchestrator = Orchestrator(config, logger)
        result = await orchestrator.run(subtitles)

        assert result.success is True

        intermediate_dir = config.intermediate_dir
        assert intermediate_dir.exists()

        stage_files = list(intermediate_dir.glob("stage_*.json"))
        assert len(stage_files) == 4

    @pytest.mark.asyncio
    async def test_that_orchestrator_creates_valid_blueprint(
        self,
        config: VFXConfig,
        logger: Logger,
        subtitles: list[Subtitle],
    ) -> None:
        orchestrator = Orchestrator(config, logger)
        result = await orchestrator.run(subtitles)

        assert result.success is True
        assert result.blueprint is not None

        blueprint = result.blueprint
        assert blueprint.version == "1.0"
        assert len(blueprint.scenes) == 2
        assert blueprint.total_duration == 10.0

        for scene in blueprint.scenes:
            assert scene.visual_prompt
            assert scene.animation is not None
            assert scene.emotion is not None

    @pytest.mark.asyncio
    async def test_that_orchestrator_emits_structured_logs_for_each_stage(
        self,
        config: VFXConfig,
        logger: Logger,
        subtitles: list[Subtitle],
    ) -> None:
        orchestrator = Orchestrator(config, logger)
        result = await orchestrator.run(subtitles)

        assert result.success is True
        assert len(result.stage_results) == 4

        for stage_result in result.stage_results:
            assert stage_result.duration >= 0
            assert stage_result.retries >= 0
