"""Shared local ChatOllama structured-output helpers for framework adapters."""

from collections.abc import Callable, Mapping
from typing import Any, cast

from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama

from raglab.core.exceptions import MalformedProviderResponseError
from raglab.core.schemas import UsageMetrics
from raglab.generation.output import GroundedAnswer

StructuredModelFactory = Callable[[str, float], Runnable[Any, dict[str, Any]]]


def create_ollama_structured_model_factory(
    base_url: str,
    *,
    timeout_seconds: float,
) -> StructuredModelFactory:
    """Build only local ChatOllama models; no metered provider is accepted."""

    def factory(model: str, temperature: float) -> Runnable[Any, dict[str, Any]]:
        chat = ChatOllama(
            model=model,
            temperature=temperature,
            base_url=base_url,
            async_client_kwargs={"timeout": timeout_seconds},
        )
        runnable = chat.with_structured_output(
            GroundedAnswer,
            method="json_schema",
            include_raw=True,
        )
        return cast(Runnable[Any, dict[str, Any]], runnable)

    return factory


def parse_structured_result(output: dict[str, Any]) -> tuple[GroundedAnswer, AIMessage]:
    """Validate LangChain's parsed/raw structured-output envelope."""
    if output.get("parsing_error") is not None:
        raise MalformedProviderResponseError("structured output validation failed")
    parsed = output.get("parsed")
    raw = output.get("raw")
    try:
        generated = (
            parsed if isinstance(parsed, GroundedAnswer) else GroundedAnswer.model_validate(parsed)
        )
    except Exception as error:
        raise MalformedProviderResponseError(
            "model output did not match the grounded answer schema"
        ) from error
    if not isinstance(raw, AIMessage):
        raise MalformedProviderResponseError("structured output did not include an AI message")
    return generated, raw


def usage_from_message(message: AIMessage) -> UsageMetrics:
    """Normalize local model usage and explicitly record zero API cost."""
    metadata: Mapping[str, Any] = message.usage_metadata or {}
    prompt_tokens = int(metadata["input_tokens"]) if "input_tokens" in metadata else None
    completion_tokens = int(metadata["output_tokens"]) if "output_tokens" in metadata else None
    total_tokens = int(metadata["total_tokens"]) if "total_tokens" in metadata else None
    return UsageMetrics(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        estimated_cost_usd=0,
        llm_calls=1,
    )


def add_usage(left: UsageMetrics, right: UsageMetrics) -> UsageMetrics:
    """Accumulate usage across a bounded graph repair loop."""
    prompt_tokens = _add_optional(left.prompt_tokens, right.prompt_tokens)
    completion_tokens = _add_optional(left.completion_tokens, right.completion_tokens)
    total_tokens = _add_optional(left.total_tokens, right.total_tokens)
    return UsageMetrics(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        estimated_cost_usd=0,
        llm_calls=left.llm_calls + right.llm_calls,
        retrieval_iterations=left.retrieval_iterations,
    )


def _add_optional(left: int | None, right: int | None) -> int | None:
    if left is None and right is None:
        return None
    return (left or 0) + (right or 0)
