# Copyright 2025 Emcie Co Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
from pydantic import ValidationError
from cerebras.cloud.sdk import AsyncCerebras
from cerebras.cloud.sdk import (
    RateLimitError,
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
)
from typing import Any, Mapping
from typing_extensions import override
import jsonfinder  # type: ignore
import os
import tiktoken

from parlant.adapters.nlp.common import normalize_json_output, record_llm_metrics
from parlant.adapters.nlp.hugging_face import JinaAIEmbedder
from parlant.core.engines.alpha.prompt_builder import PromptBuilder
from parlant.core.meter import Meter
from parlant.core.nlp.embedding import Embedder
from parlant.core.nlp.generation import (
    T,
    BaseSchematicGenerator,
    SchematicGenerationResult,
)
from parlant.core.nlp.generation_info import GenerationInfo, UsageInfo
from parlant.core.loggers import Logger
from parlant.core.nlp.moderation import ModerationService, NoModeration
from parlant.core.nlp.policies import policy, retry
from parlant.core.nlp.service import NLPService
from parlant.core.nlp.tokenization import EstimatingTokenizer


class LlamaEstimatingTokenizer(EstimatingTokenizer):
    """Tokenizer that estimates token counts for Llama models using GPT-4o encoding.

    This tokenizer provides token count estimates for Cerebras Llama models by
    using OpenAI's GPT-4o tokenizer as a reference and applying a calibration
    factor specific to Llama models.

    Attributes:
        encoding: The tiktoken encoding used for token count estimation.
    """

    def __init__(self) -> None:
        """Initialize the tokenizer with GPT-4o encoding."""
        self.encoding = tiktoken.encoding_for_model("gpt-4o-2024-08-06")

    @override
    async def estimate_token_count(self, prompt: str) -> int:
        """Estimate the number of tokens in a prompt for Llama models.

        Args:
            prompt: The text to estimate token count for.

        Returns:
            Estimated number of tokens, adjusted for Llama model specifics.
        """
        tokens = self.encoding.encode(prompt)
        return len(tokens) + 36


class CerebrasSchematicGenerator(BaseSchematicGenerator[T]):
    """Generic schematic generator for Cerebras Llama models.

    This generator uses Cerebras' high-performance inference API to generate
    structured outputs matching a specified JSON schema. It handles retries for
    transient failures and provides comprehensive error handling.

    Attributes:
        supported_hints: List of supported generation hints (e.g., 'temperature').
        model_name: The name of the Cerebras model to use.
    """

    supported_hints = ["temperature"]

    def __init__(
        self,
        model_name: str,
        logger: Logger,
        meter: Meter,
    ) -> None:
        """Initialize the Cerebras schematic generator.

        Args:
            model_name: Name of the Cerebras model to use.
            logger: Logger instance for tracing and debugging.
            meter: Meter instance for recording metrics.

        Raises:
            ValueError: If CEREBRAS_API_KEY environment variable is not set.
        """
        self.model_name = model_name

        self._logger = logger
        self._meter = meter
        self._client = AsyncCerebras(api_key=os.environ.get("CEREBRAS_API_KEY"))

    @policy(
        [
            retry(
                exceptions=(
                    APIConnectionError,
                    APITimeoutError,
                    RateLimitError,
                ),
            ),
            retry(InternalServerError, max_exceptions=2, wait_times=(1.0, 5.0)),
        ]
    )
    @override
    async def do_generate(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[T]:
        """Generate structured output from a prompt with automatic retry logic.

        This method wraps the internal generation logic with retry policies for
        handling transient failures (rate limits, timeouts, connection errors).

        Args:
            prompt: The input prompt (string or PromptBuilder).
            hints: Optional generation hints like temperature.

        Returns:
            A SchematicGenerationResult containing the structured output and metadata.

        Raises:
            RateLimitError: If API rate limits are exceeded after retries.
            APIConnectionError: If unable to connect after retries.
            APITimeoutError: If request times out after retries.
            ValidationError: If generated content doesn't match the schema.
        """
        with self._logger.scope(f"Cerebras LLM Request ({self.schema.__name__})"):
            return await self._do_generate(prompt, hints)

    async def _do_generate(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[T]:
        """Internal generation logic without retry wrapping.

        Args:
            prompt: The input prompt (string or PromptBuilder).
            hints: Optional generation hints like temperature.

        Returns:
            A SchematicGenerationResult containing the structured output and metadata.
        """
        if isinstance(prompt, PromptBuilder):
            prompt = prompt.build()

        cerebras_api_arguments = {k: v for k, v in hints.items() if k in self.supported_hints}

        t_start = time.time()
        try:
            response = await self._client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_name,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "schema": self.schema.model_json_schema(),
                        "name": self.schema.__name__,
                        "strict": True,
                    },
                },
                **cerebras_api_arguments,
            )
        except RateLimitError:
            self._logger.error(
                "Cerebras API rate limit exceeded.\n"
                "Your account may have reached the maximum number of requests allowed per minute for the tier you are using.\n"
                "Please contact with Cerebras support for more information."
            )
            raise

        t_end = time.time()

        if response.usage:  # type: ignore
            self._logger.trace(response.usage.model_dump_json(indent=2))  # type: ignore

        raw_content = response.choices[0].message.content or "{}"  # type: ignore

        try:
            json_content = normalize_json_output(raw_content)
            json_object = jsonfinder.only_json(json_content)[2]
        except Exception:
            self._logger.error(
                f"Failed to extract JSON returned by {self.model_name}:\n{raw_content}"
            )
            raise

        try:
            model_content = self.schema.model_validate(json_object)

            await record_llm_metrics(
                self._meter,
                self.model_name,
                input_tokens=response.usage.prompt_tokens,  # type: ignore
                output_tokens=response.usage.completion_tokens,  # type: ignore
            )

            return SchematicGenerationResult(
                content=model_content,
                info=GenerationInfo(
                    schema_name=self.schema.__name__,
                    model=self.id,
                    duration=(t_end - t_start),
                    usage=UsageInfo(
                        input_tokens=response.usage.prompt_tokens,  # type: ignore
                        output_tokens=response.usage.completion_tokens,  # type: ignore
                        extra={},
                    ),
                ),
            )
        except ValidationError:
            self._logger.error(
                f"JSON content returned by {self.model_name} does not match expected schema:\n{raw_content}"
            )
            raise


class Llama3_3_8B(CerebrasSchematicGenerator[T]):
    """Cerebras Llama 3.3 8B model implementation.

    A lightweight model suitable for quick inference and development tasks.
    Provides 8K token context window.
    """

    def __init__(self, logger: Logger, meter: Meter) -> None:
        """Initialize the Llama 3.3 8B model.

        Args:
            logger: Logger instance for tracing and debugging.
            meter: Meter instance for recording metrics.
        """
        super().__init__(
            model_name="llama3.1-8b",
            logger=logger,
            meter=meter,
        )
        self._estimating_tokenizer = LlamaEstimatingTokenizer()

    @property
    @override
    def id(self) -> str:
        """Return the model identifier."""
        return self.model_name

    @property
    @override
    def max_tokens(self) -> int:
        """Return the maximum tokens supported by this model."""
        return 8192

    @property
    @override
    def tokenizer(self) -> LlamaEstimatingTokenizer:
        """Return the tokenizer for this model."""
        return self._estimating_tokenizer


class Llama3_3_70B(CerebrasSchematicGenerator[T]):
    """Cerebras Llama 3.3 70B model implementation.

    A large, powerful model suitable for complex reasoning and generation tasks.
    Provides 32K token context window. Recommended for production deployments.
    """

    def __init__(self, logger: Logger, meter: Meter) -> None:
        """Initialize the Llama 3.3 70B model.

        Args:
            logger: Logger instance for tracing and debugging.
            meter: Meter instance for recording metrics.
        """
        super().__init__(
            model_name="llama3.3-70b",
            logger=logger,
            meter=meter,
        )

        self._estimating_tokenizer = LlamaEstimatingTokenizer()

    @property
    @override
    def id(self) -> str:
        """Return the model identifier."""
        return self.model_name

    @property
    @override
    def tokenizer(self) -> LlamaEstimatingTokenizer:
        """Return the tokenizer for this model."""
        return self._estimating_tokenizer

    @property
    @override
    def max_tokens(self) -> int:
        """Return the maximum tokens supported by this model."""
        return 32 * 1024


class CerebrasService(NLPService):
    """NLP Service implementation using Cerebras API.

    Provides integration with Cerebras' high-performance Llama models,
    handling schema-based generation, embeddings, and moderation.
    """

    @staticmethod
    def verify_environment() -> str | None:
        """Verify that the environment is configured correctly for Cerebras.

        Returns:
            An error message if configuration is missing, None if OK.
        """
        if not os.environ.get("CEREBRAS_API_KEY"):
            return """\
You're using the Cerebras NLP service, but CEREBRAS_API_KEY is not set.
Please set CEREBRAS_API_KEY in your environment before running Parlant.
"""

        return None

    def __init__(
        self,
        logger: Logger,
        meter: Meter,
    ) -> None:
        """Initialize the Cerebras NLP Service.

        Args:
            logger: Logger instance for tracing and debugging.
            meter: Meter instance for recording metrics.
        """
        self._logger = logger
        self._meter = meter
        self._logger.info("Initialized CerebrasService")

    @override
    async def get_schematic_generator(self, t: type[T]) -> CerebrasSchematicGenerator[T]:
        """Get a schematic generator for the given schema type.

        Args:
            t: The schema type to generate for.

        Returns:
            A CerebrasSchematicGenerator configured with Llama 3.3 70B model.
        """
        return Llama3_3_70B[t](self._logger, self._meter)  # type: ignore

    @override
    async def get_embedder(self) -> Embedder:
        """Get an embedder for text vectorization.

        Returns:
            A JinaAIEmbedder configured with appropriate settings.
        """
        return JinaAIEmbedder(self._logger, self._meter)

    @override
    async def get_moderation_service(self) -> ModerationService:
        """Get a moderation service for content filtering.

        Returns:
            A NoModeration service (no moderation applied).
        """
        return NoModeration()
