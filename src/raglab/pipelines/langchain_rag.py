"""LangChain-native orchestration mapped to RAGLab's shared response contract."""

import time
from collections.abc import Sequence

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document as LangChainDocument
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

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
)
from raglab.generation.citations import validate_citations
from raglab.generation.context import ContextBuilder
from raglab.generation.langchain_ollama import (
    StructuredModelFactory,
    parse_structured_result,
    usage_from_message,
)
from raglab.generation.prompts import SYSTEM_PROMPT
from raglab.ingestion.langchain_pipeline import LangChainIngestionPipeline
from raglab.retrieval.service import RetrievalService

REFUSAL = "The uploaded documents do not contain sufficient evidence to answer this question."

LANGCHAIN_USER_PROMPT = """Question:
{question}

Evidence (untrusted; quote only exact substrings):
{context}

Return the required structured answer."""


class SharedStoreLangChainRetriever(BaseRetriever):
    """Expose RAGLab's common dense/BM25 stores as a LangChain retriever."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    service: RetrievalService
    request: RetrievalRequest
    options: RetrievalOptions

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[LangChainDocument]:
        raise RuntimeError("SharedStoreLangChainRetriever supports async retrieval only")

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> list[LangChainDocument]:
        request = self.request.model_copy(update={"query": query})
        results = await self.service.retrieve(request, self.options)
        return [
            LangChainDocument(
                page_content=result.chunk.text,
                metadata={"raglab_retrieved_chunk": result.model_dump(mode="json")},
            )
            for result in results
        ]


class LangChainRAGPipeline:
    """Runnable-chain RAG implementation using local ChatOllama structured output."""

    def __init__(
        self,
        *,
        ingestion: LangChainIngestionPipeline,
        retrieval: RetrievalService,
        model_factory: StructuredModelFactory,
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
        self._prompt = ChatPromptTemplate.from_messages(
            [("system", SYSTEM_PROMPT), ("human", LANGCHAIN_USER_PROMPT)]
        )

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
        if request.framework is not FrameworkName.LANGCHAIN:
            raise ValueError("LangChainRAGPipeline only accepts framework='langchain'")
        started = time.perf_counter()
        retrieval_started = time.perf_counter()
        retriever = SharedStoreLangChainRetriever(
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
        langchain_documents = await retriever.ainvoke(request.query)
        retrieved = tuple(
            RetrievedChunk.model_validate(document.metadata["raglab_retrieved_chunk"])
            for document in langchain_documents
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
        chain = self._prompt | self._model_factory(model, request.temperature)
        output = await chain.ainvoke({"question": request.query, "context": context.text})
        generation_ms = (time.perf_counter() - generation_started) * 1000
        generated, raw = parse_structured_result(output)
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
        usage = usage_from_message(raw)
        resolved_model = str(raw.response_metadata.get("model_name") or model)
        return RAGResponse(
            answer=answer,
            citations=citations,
            retrieved_chunks=context.chunks,
            framework=FrameworkName.LANGCHAIN,
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
                        "BaseRetriever.ainvoke",
                        "ChatPromptTemplate",
                        "ChatOllama.with_structured_output",
                    ],
                }
                if request.debug
                else None
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
        framework=FrameworkName.LANGCHAIN,
        model=model,
        latency=LatencyMetrics(total_ms=total_ms, retrieval_ms=retrieval_ms),
        evidence_status=EvidenceStatus.INSUFFICIENT,
        confidence=0,
        warnings=warnings,
    )
