from enum import Enum


class StageName(str, Enum):
    STRUCTURE = "structure"
    PROMPTS = "prompts"
    ANIMATION = "animation"
    VERIFY = "verify"


DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 1.0
DEFAULT_CEREBRAS_MODEL = "llama3.3-70b"
DEFAULT_TEMPERATURE = 0.7


STAGE_EXECUTION_ORDER = [
    StageName.STRUCTURE,
    StageName.PROMPTS,
    StageName.ANIMATION,
    StageName.VERIFY,
]
