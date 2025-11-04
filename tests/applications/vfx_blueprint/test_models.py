from datetime import timedelta
import pytest
from pydantic import ValidationError

from parlant.applications.vfx_blueprint.models import (
    Subtitle,
    Scene,
    Blueprint,
    StageResult,
    CameraParams,
    AnimationParams,
)
from parlant.applications.vfx_blueprint.constants import StageName


class TestSubtitleModel:
    def test_that_subtitle_model_validates_required_fields(self) -> None:
        subtitle = Subtitle(
            index=1,
            start_time=timedelta(seconds=0),
            end_time=timedelta(seconds=5),
            text="Hello world",
        )

        assert subtitle.index == 1
        assert subtitle.start_time == timedelta(seconds=0)
        assert subtitle.end_time == timedelta(seconds=5)
        assert subtitle.text == "Hello world"

    def test_that_subtitle_model_rejects_missing_fields(self) -> None:
        with pytest.raises(ValidationError):
            Subtitle(index=1, start_time=timedelta(seconds=0))  # type: ignore


class TestBlueprintModel:
    def test_that_blueprint_model_serializes_to_expected_json_schema(self) -> None:
        blueprint = Blueprint(
            total_duration=10.0,
            scenes=[
                Scene(
                    scene_id=1,
                    subtitle_indices=[0, 1],
                    visual_prompt="A beautiful sunset",
                    start_time=0.0,
                    end_time=5.0,
                )
            ],
        )

        json_data = blueprint.model_dump()

        assert json_data["version"] == "1.0"
        assert json_data["total_duration"] == 10.0
        assert len(json_data["scenes"]) == 1
        assert json_data["scenes"][0]["scene_id"] == 1
        assert json_data["scenes"][0]["visual_prompt"] == "A beautiful sunset"

    def test_that_blueprint_includes_default_version(self) -> None:
        blueprint = Blueprint(total_duration=0.0, scenes=[])
        assert blueprint.version == "1.0"


class TestStageResultModel:
    def test_that_stage_result_model_includes_metadata_and_output(self) -> None:
        result = StageResult(
            stage=StageName.STRUCTURE,
            success=True,
            output={"scenes": []},
            duration=1.5,
            retries=0,
        )

        assert result.stage == StageName.STRUCTURE
        assert result.success is True
        assert result.output == {"scenes": []}
        assert result.duration == 1.5
        assert result.retries == 0
        assert result.error is None

    def test_that_stage_result_captures_failures(self) -> None:
        result = StageResult(
            stage=StageName.VERIFY,
            success=False,
            error="Validation failed",
            duration=0.5,
            retries=2,
        )

        assert result.success is False
        assert result.error == "Validation failed"
        assert result.retries == 2


class TestAnimationParams:
    def test_that_animation_params_have_sensible_defaults(self) -> None:
        params = AnimationParams(duration=5.0)

        assert params.duration == 5.0
        assert params.fps == 24
        assert params.camera.angle == "medium"
        assert params.camera.movement == "static"

    def test_that_camera_params_can_be_customized(self) -> None:
        camera = CameraParams(
            angle="wide",
            movement="pan",
            transition="fade",
        )
        params = AnimationParams(duration=3.0, camera=camera)

        assert params.camera.angle == "wide"
        assert params.camera.movement == "pan"
        assert params.camera.transition == "fade"
