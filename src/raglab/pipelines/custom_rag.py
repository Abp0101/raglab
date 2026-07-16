"""Framework-free retrieval and grounded generation pipeline."""

import time
from collections.abc import Sequence

from pydantic import ValidationError

from raglab.core.exceptions import MalformedProviderResponseError
from raglab.core.interfaces import LLMProvider
from raglab.core.schemas import (
    DocumentInput,
    EvidenceStatus,
    FrameworkName,
    GenerationRequest,
    IngestionResult,
    LatencyMetrics,
    PipelineCapabilities,
    PipelineConfig,
    QueryRequest,
    RAGResponse,
    RetrievalOptions,
    RetrievalRequest,
)
from raglab.generation.citations import validate_citations
from raglab.generation.context import ContextBuilder
from raglab.generation.output import GroundedAnswer
from raglab.generation.prompts import SYSTEM_PROMPT, build_user_prompt
from raglab.ingestion.pipeline import DocumentIngestionPipeline
from raglab.retrieval.service import RetrievalService

REFUSAL = "The uploaded documents do not contain sufficient evidence to answer this question."


class CustomRAGPipeline:
    """Explicit RAG baseline with no orchestration framework dependencies."""

    def __init__(
        self,
        *,
        ingestion: DocumentIngestionPipeline,
        retrieval: RetrievalService,
        llm: LLMProvider,
        default_model: str,
        config: PipelineConfig | None = None,
        context_builder: ContextBuilder | None = None,
    ) -> None:
        self._ingestion = ingestion
        self._retrieval = retrieval
        self._llm = llm
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
        )

    async def ingest(self, documents: Sequence[DocumentInput]) -> Sequence[IngestionResult]:
        return tuple([await self._ingestion.ingest(document) for document in documents])

    async def query(self, request: QueryRequest) -> RAGResponse:
        if request.framework is not FrameworkName.CUSTOM:
            raise ValueError("CustomRAGPipeline only accepts framework='custom'")
        started = time.perf_counter()
        retrieval_started = time.perf_counter()
        retrieved = tuple(
            await self._retrieval.retrieve(
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
        )
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000
        model = request.model or self._default_model
        if not retrieved:
            return _refusal_response(
                model=model,
                retrieval_ms=retrieval_ms,
                total_ms=(time.perf_counter() - started) * 1000,
                warnings=("retrieval returned no evidence",),
            )
        context = self._context_builder.build(retrieved, self._config.max_context_tokens)
        if not context.chunks:
            return _refusal_response(
                model=model,
                retrieval_ms=retrieval_ms,
                total_ms=(time.perf_counter() - started) * 1000,
                warnings=("no evidence fit within the context budget",),
            )
        generation_started = time.perf_counter()
        result = await self._llm.generate(
            GenerationRequest(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=build_user_prompt(request.query, context),
                model=model,
                temperature=request.temperature,
                response_schema=GroundedAnswer.model_json_schema(),
            )
        )
        generation_ms = (time.perf_counter() - generation_started) * 1000
        try:
            generated = GroundedAnswer.model_validate_json(result.text)
        except ValidationError as error:
            raise MalformedProviderResponseError(
                "model output did not match the grounded answer schema"
            ) from error
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
            framework=FrameworkName.CUSTOM,
            model=result.model,
            latency=LatencyMetrics(
                total_ms=(time.perf_counter() - started) * 1000,
                retrieval_ms=retrieval_ms,
                generation_ms=generation_ms,
            ),
            usage=result.usage,
            evidence_status=evidence_status,
            confidence=generated.confidence,
            warnings=warnings,
            debug=(
                {"context_estimated_tokens": context.estimated_tokens} if request.debug else None
            ),
        )


def _refusal_response(
    *,
    model: str,
    retrieval_ms: float,
    total_ms: float,
    warnings: tuple[str, ...],
) -> RAGResponse:
    return RAGResponse(
        answer=REFUSAL,
        framework=FrameworkName.CUSTOM,
        model=model,
        latency=LatencyMetrics(total_ms=total_ms, retrieval_ms=retrieval_ms),
        evidence_status=EvidenceStatus.INSUFFICIENT,
        confidence=0,
        warnings=warnings,
    )
