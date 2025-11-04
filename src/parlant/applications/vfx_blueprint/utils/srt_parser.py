import re
from datetime import timedelta
from pathlib import Path

from parlant.applications.vfx_blueprint.models import Subtitle


class SRTParseError(Exception):
    pass


class SRTParser:
    TIMESTAMP_PATTERN = re.compile(
        r"(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})"
    )

    def parse_file(self, path: Path) -> list[Subtitle]:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        return self.parse_string(content)

    def parse_string(self, content: str) -> list[Subtitle]:
        if not content.strip():
            return []

        subtitles: list[Subtitle] = []
        blocks = content.strip().split("\n\n")

        for block in blocks:
            lines = block.strip().split("\n")
            if len(lines) < 3:
                continue

            try:
                index = int(lines[0].strip())
            except ValueError:
                continue

            timestamp_line = lines[1].strip()
            match = self.TIMESTAMP_PATTERN.match(timestamp_line)

            if not match:
                raise SRTParseError(
                    f"Invalid timestamp format in subtitle {index}: {timestamp_line}"
                )

            start_h, start_m, start_s, start_ms = map(int, match.groups()[:4])
            end_h, end_m, end_s, end_ms = map(int, match.groups()[4:])

            start_time = timedelta(
                hours=start_h,
                minutes=start_m,
                seconds=start_s,
                milliseconds=start_ms,
            )
            end_time = timedelta(
                hours=end_h,
                minutes=end_m,
                seconds=end_s,
                milliseconds=end_ms,
            )

            text = "\n".join(lines[2:]).strip()

            subtitles.append(
                Subtitle(
                    index=index,
                    start_time=start_time,
                    end_time=end_time,
                    text=text,
                )
            )

        return subtitles
