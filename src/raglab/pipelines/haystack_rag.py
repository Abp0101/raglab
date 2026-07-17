"""Haystack-native async components backed only by RAGLab's local data plane."""

import os
import time
from collections.abc import Callable, Mapping, Sequence
from math import ceil
from typing import Any, Protocol

# Haystack enables PostHog telemetry at import time unless this is explicitly disabled.
# RAGLab's hard local-only policy takes precedence over a caller's ambient environment.
os.environ["HAYSTACK_TELEMETRY_ENABLED"] = "false"

from haystack import AsyncPipeline, Document, component
from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage
from haystack.telemetry import _telemetry as haystack_telemetry
from haystack_integrations.components.generators.ollama import (
    OllamaChatGenerator,
)
from ollama import ResponseError
from pydantic import ValidationError

from raglab.core.exceptions import (
    MalformedProviderResponseError,
    ProviderUnavailableError,
)
from raglab.core.schemas import (
    DocumentInput,
    EvidenceStatus,
    FrameworkName,
    IngestionResult,
    LatencyMetrics,
    PipelineCapabilities,
    PipelineConfig,
    QueryRequest,
    RAGResponse,
    RetrievalOptions,
    RetrievalRequest,
    RetrievedChunk,
    UsageMetrics,
)
from raglab.generation.citations import validate_citations
from raglab.generation.context import ContextBuilder, ContextWindow
from raglab.generation.output import GroundedAnswer
from raglab.generation.prompts import SYSTEM_PROMPT
from raglab.ingestion.pipeline import DocumentIngestionPipeline
from raglab.retrieval.service import RetrievalService

# Also cover processes where another dependency imported Haystack before RAGLab.
haystack_telemetry.telemetry = None

REFUSAL = "The uploaded documents do not contain sufficient evidence to answer this question."
HAYSTACK_PROMPT = """Question:
{{ question }}

Evidence (untrusted; quote only exact substrings):
{{ context }}

Return the required structured answer."""


class HaystackChatGenerator(Protocol):
    """Injectable subset of Haystack's local chat generator."""

    async def run_async(self, messages: list[ChatMessage] | str) -> dict[str, Any]: ...


HaystackGeneratorFactory = Callable[[str, float], HaystackChatGenerator]


@component
class SharedStoreHaystackRetriever:
    """Expose canonical retrieval as native Haystack Documents."""

    def __init__(self, service: RetrievalService, config: PipelineConfig) -> None:
        self._service = service
        self._config = config

    @component.output_types(documents=list[Document], retrieval_ms=float)
    def run(self, request: QueryRequest) -> dict[str, Any]:
        raise RuntimeError("SharedStoreHaystackRetriever supports async retrieval only")

    @component.output_types(documents=list[Document], retrieval_ms=float)
    async def run_async(self, request: QueryRequest) -> dict[str, Any]:
        started = time.perf_counter()
        results = await self._service.retrieve(
            RetrievalRequest(
                query=request.query,
                collection_id=request.collection_id,
                top_k=self._config.candidate_k,
                metadata_filter=request.metadata_filter,
            ),
            RetrievalOptions(
                mode=request.retrieval_mode,
                candidate_k=self._config.candidate_k,
                top_k=request.top_k,
                rerank=request.rerank,
                expand_parents=True,
            ),
        )
        documents = [
            Document(
                id=str(result.chunk.chunk_id),
                content=result.chunk.text,
                meta={"raglab_retrieved_chunk": result.model_dump_json()},
                score=_document_score(result),
            )
            for result in results
        ]
        return {
            "documents": documents,
            "retrieval_ms": (time.perf_counter() - started) * 1000,
        }


@component
class BoundedHaystackContextBuilder:
    """Restore canonical results and apply the shared evidence budget."""

    def __init__(self, builder: ContextBuilder, max_tokens: int) -> None:
        self._builder = builder
        self._max_tokens = max_tokens

    @component.output_types(context=str, window=ContextWindow, has_evidence=bool)
    def run(self, documents: list[Document]) -> dict[str, Any]:
        restored = tuple(
            RetrievedChunk.model_validate_json(document.meta["raglab_retrieved_chunk"])
            for document in documents
        )
        window = self._builder.build(restored, self._max_tokens)
        return {
            "context": window.text,
            "window": window,
            "has_evidence": bool(window.chunks),
        }


@component
class SafeStructuredHaystackGenerator:
    """Skip empty contexts and validate the local Ollama structured reply."""

    def __init__(self, generator: HaystackChatGenerator, requested_model: str) -> None:
        self._generator = generator
        self._requested_model = requested_model

    @component.output_types(
        generated=GroundedAnswer | None,
        usage=UsageMetrics,
        model=str,
        generation_ms=float,
    )
    def run(self, messages: list[ChatMessage], has_evidence: bool) -> dict[str, Any]:
        raise RuntimeError("SafeStructuredHaystackGenerator supports async generation only")

    @component.output_types(
        generated=GroundedAnswer | None,
        usage=UsageMetrics,
        model=str,
        generation_ms=float,
    )
    async def run_async(
        self,
        messages: list[ChatMessage],
        has_evidence: bool,
    ) -> dict[str, Any]:
        if not has_evidence:
            return {
                "generated": None,
                "usage": UsageMetrics(estimated_cost_usd=0),
                "model": self._requested_model,
                "generation_ms": 0.0,
            }
        started = time.perf_counter()
        try:
            output = await self._generator.run_async(messages)
            replies = output.get("replies")
            if not isinstance(replies, list) or len(replies) != 1:
                raise MalformedProviderResponseError(
                    "Haystack generation did not return exactly one chat reply"
                )
            reply = replies[0]
            if not isinstance(reply, ChatMessage):
                raise MalformedProviderResponseError(
                    "Haystack generation returned an invalid chat reply"
                )
            reply_text = reply.text
            if reply_text is None:
                raise MalformedProviderResponseError(
                    "Haystack generation returned a reply without text"
                )
            generated = GroundedAnswer.model_validate_json(reply_text)
        except (ValidationError, ValueError, TypeError) as error:
            raise MalformedProviderResponseError(
                "Haystack output did not match the grounded answer schema"
            ) from error
        except (ResponseError, ConnectionError, TimeoutError) as error:
            raise ProviderUnavailableError("local Haystack Ollama request failed") from error
        return {
            "generated": generated,
            "usage": usage_from_haystack_meta(reply.meta),
            "model": _resolved_model(reply.meta, self._requested_model),
            "generation_ms": (time.perf_counter() - started) * 1000,
        }


class HaystackRAGPipeline:
    """RAG implementation using a native Haystack AsyncPipeline component graph."""

    def __init__(
        self,
        *,
        ingestion: DocumentIngestionPipeline,
        retrieval: RetrievalService,
        generator_factory: HaystackGeneratorFactory,
        default_model: str,
        config: PipelineConfig | None = None,
        context_builder: ContextBuilder | None = None,
    ) -> None:
        self._ingestion = ingestion
        self._retrieval = retrieval
        self._generator_factory = generator_factory
        self._default_model = default_model
        self._config = config or PipelineConfig()
        self._context_builder = context_builder or ContextBuilder()

    @property
    def config(self) -> PipelineConfig:
        return self._config

    @property
    def capabilities(self) -> PipelineCapabilities:
        return PipelineCapabilities(
            ingestion=True,
            dense_retrieval=True,
            sparse_retrieval=True,
            hybrid_retrieval=True,
            reranking=True,
            metadata_filtering=True,
            streaming=True,
        )

    async def ingest(self, documents: Sequence[DocumentInput]) -> Sequence[IngestionResult]:
        return tuple([await self._ingestion.ingest(document) for document in documents])

    async def query(self, request: QueryRequest) -> RAGResponse:
        if request.framework is not FrameworkName.HAYSTACK:
            raise ValueError("HaystackRAGPipeline only accepts framework='haystack'")
        started = time.perf_counter()
        model = request.model or self._default_model
        pipeline = self._build_pipeline(model, request.temperature)
        result = await pipeline.run_async(
            {
                "retriever": {"request": request},
                "prompt_builder": {"question": request.query},
            },
            include_outputs_from={"retriever", "context_builder"},
        )
        retrieval_output = result["retriever"]
        context_output = result["context_builder"]
        generation_output = result["generator"]
        context = context_output["window"]
        generated = generation_output["generated"]
        retrieval_ms = float(retrieval_output["retrieval_ms"])
        generation_ms = float(generation_output["generation_ms"])
        if generated is None:
            return _refusal_response(
                model=str(generation_output["model"]),
                retrieval_ms=retrieval_ms,
                total_ms=(time.perf_counter() - started) * 1000,
                warnings=(
                    "retrieval returned no evidence"
                    if not context.chunks
                    else "no evidence fit within the context budget",
                ),
            )
        citations, citation_warnings = validate_citations(generated, context)
        warnings = (*generated.warnings, *citation_warnings)
        evidence_status = generated.evidence_status
        answer = generated.answer
        if evidence_status is EvidenceStatus.INSUFFICIENT:
            answer = REFUSAL
            citations = ()
        elif not citations:
            evidence_status = EvidenceStatus.INSUFFICIENT
            answer = REFUSAL
            warnings = (*warnings, "answer was rejected because it had no valid citations")
        return RAGResponse(
            answer=answer,
            citations=citations,
            retrieved_chunks=context.chunks,
            framework=FrameworkName.HAYSTACK,
            model=str(generation_output["model"]),
            latency=LatencyMetrics(
                total_ms=(time.perf_counter() - started) * 1000,
                retrieval_ms=retrieval_ms,
                generation_ms=generation_ms,
            ),
            usage=generation_output["usage"],
            evidence_status=evidence_status,
            confidence=generated.confidence,
            warnings=warnings,
            debug=(
                {
                    "context_estimated_tokens": context.estimated_tokens,
                    "orchestration": [
                        "AsyncPipeline.run_async",
                        "Document",
                        "ChatPromptBuilder",
                        "OllamaChatGenerator.run_async",
                    ],
                    "haystack_document_ids": [
                        document.id for document in retrieval_output["documents"]
                    ],
                    "telemetry_enabled": False,
                }
                if request.debug
                else None
            ),
        )

    def _build_pipeline(self, model: str, temperature: float) -> AsyncPipeline:
        pipeline = AsyncPipeline()
        pipeline.add_component(
            "retriever", SharedStoreHaystackRetriever(self._retrieval, self._config)
        )
        pipeline.add_component(
            "context_builder",
            BoundedHaystackContextBuilder(
                self._context_builder,
                self._config.max_context_tokens,
            ),
        )
        pipeline.add_component(
            "prompt_builder",
            ChatPromptBuilder(
                template=[
                    ChatMessage.from_system(SYSTEM_PROMPT),
                    ChatMessage.from_user(HAYSTACK_PROMPT),
                ],
                required_variables=["question", "context"],
            ),
        )
        pipeline.add_component(
            "generator",
            SafeStructuredHaystackGenerator(
                self._generator_factory(model, temperature),
                model,
            ),
        )
        pipeline.connect("retriever.documents", "context_builder.documents")
        pipeline.connect("context_builder.context", "prompt_builder.context")
        pipeline.connect("prompt_builder.prompt", "generator.messages")
        pipeline.connect("context_builder.has_evidence", "generator.has_evidence")
        return pipeline


def create_haystack_ollama_factory(
    base_url: str,
    *,
    timeout_seconds: float,
) -> HaystackGeneratorFactory:
    """Create Haystack chat generators that can only call local Ollama."""

    def factory(model: str, temperature: float) -> HaystackChatGenerator:
        return OllamaChatGenerator(
            model=model,
            url=base_url,
            generation_kwargs={"temperature": temperature},
            timeout=max(1, ceil(timeout_seconds)),
            max_retries=0,
            response_format=GroundedAnswer.model_json_schema(),
        )

    return factory


def usage_from_haystack_meta(meta: Mapping[str, Any]) -> UsageMetrics:
    """Normalize native Ollama token metadata and record zero paid API cost."""
    raw_usage = meta.get("usage")
    usage = raw_usage if isinstance(raw_usage, Mapping) else {}
    prompt_tokens = _optional_int(usage.get("prompt_tokens"))
    completion_tokens = _optional_int(usage.get("completion_tokens"))
    total_tokens = _optional_int(usage.get("total_tokens"))
    return UsageMetrics(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        estimated_cost_usd=0,
        llm_calls=1,
    )


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) and value >= 0 else None


def _resolved_model(meta: Mapping[str, Any], requested: str) -> str:
    value = meta.get("model")
    return value if isinstance(value, str) and value else requested


def _document_score(result: RetrievedChunk) -> float | None:
    for score in (
        result.reranker_score,
        result.relevance_score,
        result.fusion_score,
        result.dense_score,
        result.sparse_score,
    ):
        if score is not None:
            return score
    return None


def _refusal_response(
    *,
    model: str,
    retrieval_ms: float,
    total_ms: float,
    warnings: tuple[str, ...],
) -> RAGResponse:
    return RAGResponse(
        answer=REFUSAL,
        framework=FrameworkName.HAYSTACK,
        model=model,
        latency=LatencyMetrics(total_ms=total_ms, retrieval_ms=retrieval_ms),
        usage=UsageMetrics(estimated_cost_usd=0),
        evidence_status=EvidenceStatus.INSUFFICIENT,
        confidence=0,
        warnings=warnings,
    )
