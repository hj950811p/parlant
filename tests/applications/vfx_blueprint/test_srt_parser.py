from datetime import timedelta
from pathlib import Path
import pytest

from parlant.applications.vfx_blueprint.utils.srt_parser import (
    SRTParser,
    SRTParseError,
)
from parlant.applications.vfx_blueprint.models import Subtitle


class TestSRTParser:
    @pytest.fixture
    def sample_srt_path(self) -> Path:
        return Path(__file__).parent / "fixtures" / "sample.srt"

    def test_that_srt_parser_extracts_subtitles_with_timestamps(
        self, sample_srt_path: Path
    ) -> None:
        parser = SRTParser()
        subtitles = parser.parse_file(sample_srt_path)

        assert len(subtitles) == 4

        assert subtitles[0].index == 1
        assert subtitles[0].start_time == timedelta(seconds=0)
        assert subtitles[0].end_time == timedelta(seconds=5)
        assert subtitles[0].text == "Welcome to the VFX pipeline."

        assert subtitles[1].index == 2
        assert subtitles[1].start_time == timedelta(seconds=5.5)
        assert subtitles[1].end_time == timedelta(seconds=10)
        assert subtitles[1].text == "This system generates video blueprints."

    def test_that_srt_parser_handles_malformed_srt_gracefully(self) -> None:
        parser = SRTParser()
        malformed_content = """
1
00:00:00,000 -> 00:00:05,000
Invalid format
"""
        with pytest.raises(SRTParseError) as exc_info:
            parser.parse_string(malformed_content)

        assert "timestamp" in str(exc_info.value).lower()

    def test_that_srt_parser_preserves_subtitle_order(self, sample_srt_path: Path) -> None:
        parser = SRTParser()
        subtitles = parser.parse_file(sample_srt_path)

        for i, subtitle in enumerate(subtitles, start=1):
            assert subtitle.index == i

    def test_that_srt_parser_handles_empty_file(self) -> None:
        parser = SRTParser()
        subtitles = parser.parse_string("")
        assert len(subtitles) == 0

    def test_that_srt_parser_parses_from_string(self) -> None:
        parser = SRTParser()
        content = """1
00:00:00,000 --> 00:00:05,000
Test subtitle

2
00:00:05,000 --> 00:00:10,000
Another subtitle
"""
        subtitles = parser.parse_string(content)

        assert len(subtitles) == 2
        assert subtitles[0].text == "Test subtitle"
        assert subtitles[1].text == "Another subtitle"
