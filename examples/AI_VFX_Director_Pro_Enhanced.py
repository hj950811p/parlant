#!/usr/bin/env python3

import asyncio
import json
import sys
from pathlib import Path
import click

from parlant.applications.vfx_blueprint.utils.srt_parser import SRTParser
from parlant.applications.vfx_blueprint.config import ConfigLoader, VFXConfig
from parlant.applications.vfx_blueprint.orchestrator import Orchestrator
from parlant.core.loggers import StdoutLogger, LogLevel
from parlant.core.tracer import LocalTracer


@click.command()
@click.option(
    "--srt-file",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to the SRT subtitle file",
)
@click.option(
    "--output",
    required=True,
    type=click.Path(path_type=Path),
    help="Path to the output JSON blueprint file",
)
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to configuration file (YAML or JSON)",
)
@click.option(
    "--log-level",
    type=click.Choice(["trace", "debug", "info", "warning", "error", "critical"]),
    default="info",
    help="Logging level",
)
def main(
    srt_file: Path,
    output: Path,
    config: Path | None,
    log_level: str,
) -> None:
    """
    AI VFX Director Pro Enhanced

    Converts SRT subtitles into a blueprint JSON for Cerebras-driven video generation.

    This tool implements a multi-stage generation flow:
    - Stage 1: Structure extraction from subtitles
    - Stage 2: Visual prompt generation
    - Stage 3: Animation parameter calculation
    - Stage 4: Blueprint verification
    """
    asyncio.run(async_main(srt_file, output, config, log_level))


async def async_main(
    srt_file: Path,
    output: Path,
    config_path: Path | None,
    log_level_str: str,
) -> None:
    log_level_map = {
        "trace": LogLevel.TRACE,
        "debug": LogLevel.DEBUG,
        "info": LogLevel.INFO,
        "warning": LogLevel.WARNING,
        "error": LogLevel.ERROR,
        "critical": LogLevel.CRITICAL,
    }
    log_level = log_level_map[log_level_str]

    tracer = LocalTracer()
    logger = StdoutLogger(tracer, log_level)

    try:
        logger.info("Starting AI VFX Director Pro Enhanced")
        logger.info(f"Processing SRT file: {srt_file}")

        config_loader = ConfigLoader()
        if config_path:
            logger.info(f"Loading configuration from: {config_path}")
            config = config_loader.load_from_file(config_path)
        else:
            logger.info("Using default configuration")
            config = VFXConfig()

        parser = SRTParser()
        logger.info("Parsing SRT file...")
        subtitles = parser.parse_file(srt_file)
        logger.info(f"Parsed {len(subtitles)} subtitles")

        orchestrator = Orchestrator(config, logger)
        logger.info("Running VFX pipeline...")
        result = await orchestrator.run(subtitles)

        if not result.success:
            logger.error("Pipeline execution failed")
            failed_stages = [sr.stage.value for sr in result.stage_results if not sr.success]
            logger.error(f"Failed stages: {', '.join(failed_stages)}")
            sys.exit(1)

        if result.blueprint is None:
            logger.error("No blueprint generated")
            sys.exit(1)

        output.parent.mkdir(parents=True, exist_ok=True)

        with open(output, "w") as f:
            json.dump(result.blueprint.model_dump(), f, indent=2, default=str)

        logger.info(f"Blueprint saved to: {output}")
        logger.info(f"Total execution time: {result.total_duration:.2f}s")
        logger.info(f"Generated {len(result.blueprint.scenes)} scenes")
        logger.info("Pipeline completed successfully!")

    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
