from abc import ABC, abstractmethod
from typing import Any

from parlant.applications.vfx_blueprint.config import VFXConfig
from parlant.applications.vfx_blueprint.models import Subtitle
from parlant.core.loggers import Logger


class Stage(ABC):
    def __init__(
        self,
        config: VFXConfig,
        logger: Logger,
    ) -> None:
        self.config = config
        self.logger = logger

    @abstractmethod
    async def execute(
        self,
        subtitles: list[Subtitle],
        previous_output: Any | None = None,
    ) -> Any:
        pass

    @abstractmethod
    async def validate_output(self, output: Any) -> bool:
        pass
