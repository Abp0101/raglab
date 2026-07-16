"""Local LlamaIndex Ollama structured generation with normalized usage."""

from collections.abc import Callable
from typing import Any, Protocol

import httpx
from llama_index.core.callbacks import CallbackManager, CBEventType, EventPayload
from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.ollama import Ollama
from ollama import ResponseError
from pydantic import ValidationError

from raglab.core.exceptions import MalformedProviderResponseError, ProviderUnavailableError
from raglab.core.schemas import UsageMetrics
from raglab.generation.output import GroundedAnswer


class LlamaIndexStructuredPredictor(Protocol):
    """Injectable boundary around LlamaIndex structured prediction."""

    async def predict(
        self,
        prompt: PromptTemplate,
        **prompt_args: str,
    ) -> tuple[GroundedAnswer, UsageMetrics, str]: ...


LlamaIndexModelFactory = Callable[[str, float], LlamaIndexStructuredPredictor]


class OllamaUsageCapture(BaseCallbackHandler):
    """Capture token counts exposed by the local Ollama ChatResponse callback."""

    def __init__(self, model: str) -> None:
        super().__init__(event_starts_to_ignore=[], event_ends_to_ignore=[])
        self.model = model
        self.reset()

    def reset(self) -> None:
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.llm_calls = 0

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: dict[str, Any] | None = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        return event_id

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: dict[str, Any] | None = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        if event_type is not CBEventType.LLM or payload is None:
            return
        response = payload.get(EventPayload.RESPONSE) or payload.get(EventPayload.COMPLETION)
        raw = getattr(response, "raw", None)
        if not isinstance(raw, dict):
            return
        usage = raw.get("usage")
        if not isinstance(usage, dict):
            return
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        if not isinstance(prompt_tokens, int) or not isinstance(completion_tokens, int):
            return
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.llm_calls += 1
        resolved_model = raw.get("model")
        if isinstance(resolved_model, str) and resolved_model:
            self.model = resolved_model

    def start_trace(self, trace_id: str | None = None) -> None:
        return None

    def end_trace(
        self,
        trace_id: str | None = None,
        trace_map: dict[str, list[str]] | None = None,
    ) -> None:
        return None

    def usage(self) -> UsageMetrics:
        total_tokens = self.prompt_tokens + self.completion_tokens
        return UsageMetrics(
            prompt_tokens=self.prompt_tokens if self.llm_calls else None,
            completion_tokens=self.completion_tokens if self.llm_calls else None,
            total_tokens=total_tokens if self.llm_calls else None,
            estimated_cost_usd=0,
            llm_calls=self.llm_calls,
        )


class LlamaIndexOllamaPredictor:
    """Use LlamaIndex's JSON-schema Ollama path without any cloud provider."""

    def __init__(
        self,
        *,
        model: str,
        temperature: float,
        base_url: str,
        timeout_seconds: float,
    ) -> None:
        self._usage = OllamaUsageCapture(model)
        self._llm = Ollama(
            model=model,
            temperature=temperature,
            base_url=base_url,
            request_timeout=timeout_seconds,
            callback_manager=CallbackManager([self._usage]),
        )

    async def predict(
        self,
        prompt: PromptTemplate,
        **prompt_args: str,
    ) -> tuple[GroundedAnswer, UsageMetrics, str]:
        self._usage.reset()
        try:
            generated = await self._llm.astructured_predict(
                output_cls=GroundedAnswer,
                prompt=prompt,
                **prompt_args,
            )
        except ValidationError as error:
            raise MalformedProviderResponseError(
                "LlamaIndex output did not match the grounded answer schema"
            ) from error
        except (httpx.HTTPError, ResponseError, TimeoutError) as error:
            raise ProviderUnavailableError("local LlamaIndex Ollama request failed") from error
        return generated, self._usage.usage(), self._usage.model


def create_llamaindex_ollama_factory(
    base_url: str,
    *,
    timeout_seconds: float,
) -> LlamaIndexModelFactory:
    """Create request-scoped LlamaIndex predictors backed only by local Ollama."""

    def factory(model: str, temperature: float) -> LlamaIndexStructuredPredictor:
        return LlamaIndexOllamaPredictor(
            model=model,
            temperature=temperature,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
        )

    return factory
