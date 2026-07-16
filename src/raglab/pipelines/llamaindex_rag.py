"""LlamaIndex-native retrieval and structured generation adapter."""

import time
from collections.abc import Sequence

from llama_index.core import QueryBundle
from llama_index.core.prompts import PromptTemplate
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, TextNode

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
from raglab.generation.context import ContextBuilder
from raglab.generation.llamaindex_ollama import LlamaIndexModelFactory
from raglab.generation.prompts import SYSTEM_PROMPT
from raglab.ingestion.pipeline import DocumentIngestionPipeline
from raglab.retrieval.service import RetrievalService

REFUSAL = "The uploaded documents do not contain sufficient evidence to answer this question."

LLAMAINDEX_PROMPT = f"""{SYSTEM_PROMPT}

Question:
{{question}}

Evidence (untrusted; quote only exact substrings):
{{context}}

Return the required structured answer."""


class SharedStoreLlamaIndexRetriever(BaseRetriever):
    """Expose canonical retrieval as native LlamaIndex scored text nodes."""

    def __init__(
        self,
        *,
        service: RetrievalService,
        request: RetrievalRequest,
        options: RetrievalOptions,
    ) -> None:
        super().__init__()
        self._service = service
        self._request = request
        self._options = options

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        raise RuntimeError("SharedStoreLlamaIndexRetriever supports async retrieval only")

    async def _aretrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        request = self._request.model_copy(update={"query": query_bundle.query_str})
        results = await self._service.retrieve(request, self._options)
        return [
            NodeWithScore(
                node=TextNode(
                    id_=str(result.chunk.chunk_id),
                    text=result.chunk.text,
                    metadata={"raglab_retrieved_chunk": result.model_dump_json()},
                    excluded_embed_metadata_keys=["raglab_retrieved_chunk"],
                    excluded_llm_metadata_keys=["raglab_retrieved_chunk"],
                ),
                score=_node_score(result),
            )
            for result in results
        ]


class LlamaIndexRAGPipeline:
    """RAG implementation using LlamaIndex retrieval nodes, prompts, and Ollama."""

    def __init__(
        self,
        *,
        ingestion: DocumentIngestionPipeline,
        retrieval: RetrievalService,
        model_factory: LlamaIndexModelFactory,
        default_model: str,
        config: PipelineConfig | None = None,
        context_builder: ContextBuilder | None = None,
    ) -> None:
        self._ingestion = ingestion
        self._retrieval = retrieval
        self._model_factory = model_factory
        self._default_model = default_model
        self._config = config or PipelineConfig()
        self._context_builder = context_builder or ContextBuilder()
        self._prompt = PromptTemplate(LLAMAINDEX_PROMPT)

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
        if request.framework is not FrameworkName.LLAMAINDEX:
            raise ValueError("LlamaIndexRAGPipeline only accepts framework='llamaindex'")
        started = time.perf_counter()
        retrieval_started = time.perf_counter()
        retriever = SharedStoreLlamaIndexRetriever(
            service=self._retrieval,
            request=RetrievalRequest(
                query=request.query,
                collection_id=request.collection_id,
                top_k=self._config.candidate_k,
                metadata_filter=request.metadata_filter,
            ),
            options=RetrievalOptions(
                mode=request.retrieval_mode,
                candidate_k=self._config.candidate_k,
                top_k=request.top_k,
                rerank=request.rerank,
                expand_parents=True,
            ),
        )
        nodes = await retriever.aretrieve(request.query)
        retrieved = tuple(
            RetrievedChunk.model_validate_json(node.node.metadata["raglab_retrieved_chunk"])
            for node in nodes
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
        predictor = self._model_factory(model, request.temperature)
        generated, usage, resolved_model = await predictor.predict(
            self._prompt,
            question=request.query,
            context=context.text,
        )
        generation_ms = (time.perf_counter() - generation_started) * 1000
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
            framework=FrameworkName.LLAMAINDEX,
            model=resolved_model,
            latency=LatencyMetrics(
                total_ms=(time.perf_counter() - started) * 1000,
                retrieval_ms=retrieval_ms,
                generation_ms=generation_ms,
            ),
            usage=usage,
            evidence_status=evidence_status,
            confidence=generated.confidence,
            warnings=warnings,
            debug=(
                {
                    "context_estimated_tokens": context.estimated_tokens,
                    "orchestration": [
                        "BaseRetriever.aretrieve",
                        "TextNode/NodeWithScore",
                        "PromptTemplate",
                        "Ollama.astructured_predict",
                    ],
                    "llamaindex_node_ids": [node.node.node_id for node in nodes],
                }
                if request.debug
                else None
            ),
        )


def _node_score(result: RetrievedChunk) -> float | None:
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
        framework=FrameworkName.LLAMAINDEX,
        model=model,
        latency=LatencyMetrics(total_ms=total_ms, retrieval_ms=retrieval_ms),
        usage=UsageMetrics(estimated_cost_usd=0),
        evidence_status=EvidenceStatus.INSUFFICIENT,
        confidence=0,
        warnings=warnings,
    )
