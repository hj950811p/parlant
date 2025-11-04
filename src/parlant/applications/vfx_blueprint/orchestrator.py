import asyncio
import json
import time
from pathlib import Path
from typing import Any

from parlant.applications.vfx_blueprint.config import VFXConfig
from parlant.applications.vfx_blueprint.constants import STAGE_EXECUTION_ORDER, StageName
from parlant.applications.vfx_blueprint.models import (
    Subtitle,
    Blueprint,
    StageResult,
    OrchestratorResult,
)
from parlant.applications.vfx_blueprint.stages.base import Stage
from parlant.applications.vfx_blueprint.stages.structure import StructureStage
from parlant.applications.vfx_blueprint.stages.prompts import PromptsStage
from parlant.applications.vfx_blueprint.stages.animation import AnimationStage
from parlant.applications.vfx_blueprint.stages.verify import VerifyStage
from parlant.core.loggers import Logger


class OrchestratorError(Exception):
    pass


class Orchestrator:
    def __init__(
        self,
        config: VFXConfig,
        logger: Logger,
    ) -> None:
        self.config = config
        self.logger = logger

        self.stages: dict[StageName, Stage] = {
            StageName.STRUCTURE: StructureStage(config, logger),
            StageName.PROMPTS: PromptsStage(config, logger),
            StageName.ANIMATION: AnimationStage(config, logger),
            StageName.VERIFY: VerifyStage(config, logger),
        }

    async def run(self, subtitles: list[Subtitle]) -> OrchestratorResult:
        if self.config.save_intermediate:
            self.config.intermediate_dir.mkdir(parents=True, exist_ok=True)

        start_time = time.time()
        stage_results: list[StageResult] = []
        previous_output: Any = None
        final_blueprint: Blueprint | None = None

        for stage_name in STAGE_EXECUTION_ORDER:
            stage = self.stages[stage_name]

            with self.logger.scope(f"Stage:{stage_name.value}"):
                self.logger.info(f"Starting stage: {stage_name.value}")

                stage_result = await self._execute_stage_with_retry(
                    stage_name,
                    stage,
                    subtitles,
                    previous_output,
                )

                stage_results.append(stage_result)

                if not stage_result.success:
                    self.logger.error(f"Stage {stage_name.value} failed: {stage_result.error}")
                    break

                previous_output = stage_result.output

                if self.config.save_intermediate:
                    await self._save_intermediate_output(
                        stage_name,
                        stage_result.output,
                    )

                self.logger.info(
                    f"Stage {stage_name.value} completed in {stage_result.duration:.2f}s"
                )

        success = all(result.success for result in stage_results)

        if success and isinstance(previous_output, Blueprint):
            final_blueprint = previous_output

        total_duration = time.time() - start_time

        return OrchestratorResult(
            success=success,
            blueprint=final_blueprint,
            stage_results=stage_results,
            total_duration=total_duration,
        )

    async def _execute_stage_with_retry(
        self,
        stage_name: StageName,
        stage: Stage,
        subtitles: list[Subtitle],
        previous_output: Any,
    ) -> StageResult:
        retries = 0
        last_error: str | None = None

        for attempt in range(self.config.max_retries):
            start_time = time.time()

            try:
                output = await stage.execute(subtitles, previous_output)

                is_valid = await stage.validate_output(output)

                if not is_valid:
                    raise ValueError("Stage output validation failed")

                duration = time.time() - start_time

                return StageResult(
                    stage=stage_name,
                    success=True,
                    output=output,
                    duration=duration,
                    retries=retries,
                )

            except Exception as e:
                retries += 1
                last_error = str(e)
                self.logger.warning(f"Stage {stage_name.value} attempt {attempt + 1} failed: {e}")

                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay)

        duration = time.time() - start_time

        return StageResult(
            stage=stage_name,
            success=False,
            error=last_error,
            duration=duration,
            retries=retries,
        )

    async def _save_intermediate_output(
        self,
        stage_name: StageName,
        output: Any,
    ) -> None:
        output_file = self.config.intermediate_dir / f"stage_{stage_name.value}.json"

        try:
            if hasattr(output, "model_dump"):
                data = output.model_dump()
            else:
                data = output

            with open(output_file, "w") as f:
                json.dump(data, f, indent=2, default=str)

            self.logger.debug(f"Saved intermediate output to {output_file}")

        except Exception as e:
            self.logger.warning(f"Failed to save intermediate output: {e}")
