# VFX Blueprint Generator

A modular pipeline for converting SRT subtitles into video blueprint JSONs for Cerebras-driven video generation.

## Architecture

The VFX Blueprint Generator follows a multi-stage pipeline architecture:

1. **Stage 1 (Structure)**: Extracts scene structure from subtitles
2. **Stage 2 (Prompts)**: Generates visual prompts and analyzes emotions
3. **Stage 3 (Animation)**: Adds camera parameters and timing information
4. **Stage 4 (Verify)**: Validates the complete blueprint

## Components

- **models.py**: Pydantic models for subtitles, scenes, blueprints, and stage results
- **config.py**: Configuration management with YAML/JSON support
- **constants.py**: Shared constants and defaults
- **orchestrator.py**: Stage orchestration with retry/rollback capabilities
- **stages/**: Individual stage implementations
- **utils/**: Utility modules (SRT parser, logging helpers)

## Usage

### Command Line

```bash
python examples/AI_VFX_Director_Pro_Enhanced.py \
    --srt-file input.srt \
    --output blueprint.json \
    --config config.yaml \
    --log-level info
```

### Programmatic

```python
from parlant.applications.vfx_blueprint.orchestrator import Orchestrator
from parlant.applications.vfx_blueprint.config import VFXConfig
from parlant.applications.vfx_blueprint.utils.srt_parser import SRTParser
from parlant.core.loggers import StdoutLogger, LogLevel
from parlant.core.tracer import LocalTracer

tracer = LocalTracer()
logger = StdoutLogger(tracer, LogLevel.INFO)
config = VFXConfig()

parser = SRTParser()
subtitles = parser.parse_file("input.srt")

orchestrator = Orchestrator(config, logger)
result = await orchestrator.run(subtitles)

if result.success:
    print(f"Generated {len(result.blueprint.scenes)} scenes")
```

## Configuration

Configuration can be provided via YAML or JSON files:

```yaml
cerebras:
  model: llama3.3-70b
  temperature: 0.7

stages:
  max_retries: 3
  retry_delay: 1.0

output:
  save_intermediate: true
  intermediate_dir: .vfx_intermediate
```

## Output Schema

The generated blueprint JSON follows this schema:

```json
{
  "version": "1.0",
  "total_duration": 120.5,
  "scenes": [
    {
      "scene_id": 1,
      "subtitle_indices": [1, 2],
      "visual_prompt": "A cinematic scene showing...",
      "emotion": "neutral",
      "start_time": 0.0,
      "end_time": 5.5,
      "animation": {
        "duration": 5.5,
        "fps": 24,
        "camera": {
          "angle": "medium",
          "movement": "static",
          "transition": null
        }
      }
    }
  ],
  "metadata": {
    "total_scenes": 10,
    "total_subtitles": 20,
    "verified": true
  }
}
```

## Testing

```bash
poetry run pytest tests/applications/vfx_blueprint/ -v
```
