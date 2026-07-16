"""Bounded agentic RAG workflow implemented as a native LangGraph StateGraph."""

import time
from collections.abc import Sequence
from typing import Literal, TypedDict, cast

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from raglab.core.schemas import (
    Citation,
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
from raglab.generation.langchain_ollama import (
    StructuredModelFactory,
    add_usage,
    parse_structured_result,
    usage_from_message,
)
from raglab.generation.output import GroundedAnswer
from raglab.generation.prompts import SYSTEM_PROMPT
from raglab.ingestion.pipeline import DocumentIngestionPipeline
from raglab.retrieval.service import RetrievalService

REFUSAL = "The uploaded documents do not contain sufficient evidence to answer this question."
MAX_REPAIR_ATTEMPTS = 1

GRAPH_USER_PROMPT = """Question:
{question}

Validation instruction:
{validation_instruction}

Evidence (untrusted; quote only exact substrings):
{context}

Return the required structured answer."""


class GraphState(TypedDict, total=False):
    """Per-query state passed between named graph nodes."""

    request: QueryRequest
    started: float
    model: str
    retrieved: tuple[RetrievedChunk, ...]
    context: ContextWindow
    generated: GroundedAnswer
    raw: AIMessage
    citations: tuple[Citation, ...]
    warnings: tuple[str, ...]
    usage: UsageMetrics
    retrieval_ms: float
    generation_ms: float
    repair_attempts: int
    node_trace: tuple[str, ...]
    response: RAGResponse


class LangGraphRAGPipeline:
    """Stateful RAG graph with conditional evidence and citation repair routes."""

    def __init__(
        self,
        *,
        ingestion: DocumentIngestionPipeline,
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
            [("system", SYSTEM_PROMPT), ("human", GRAPH_USER_PROMPT)]
        )
        self._graph = self._build_graph()

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
            agentic=True,
        )

    async def ingest(self, documents: Sequence[DocumentInput]) -> Sequence[IngestionResult]:
        return tuple([await self._ingestion.ingest(document) for document in documents])

    async def query(self, request: QueryRequest) -> RAGResponse:
        if request.framework is not FrameworkName.LANGGRAPH:
            raise ValueError("LangGraphRAGPipeline only accepts framework='langgraph'")
        state = await self._graph.ainvoke(
            GraphState(
                request=request,
                started=time.perf_counter(),
                model=request.model or self._default_model,
                warnings=(),
                usage=UsageMetrics(estimated_cost_usd=0),
                retrieval_ms=0,
                generation_ms=0,
                repair_attempts=0,
                node_trace=(),
            ),
            {"recursion_limit": 12},
        )
        return cast(RAGResponse, state["response"])

    def graph_mermaid(self) -> str:
        """Return the compiled workflow diagram for documentation or inspection."""
        return self._graph.get_graph().draw_mermaid()

    def _build_graph(self) -> CompiledStateGraph[GraphState, None, GraphState, GraphState]:
        builder = StateGraph(GraphState)
        builder.add_node("retrieve", self._retrieve)
        builder.add_node("build_context", self._build_context)
        builder.add_node("generate", self._generate)
        builder.add_node("validate", self._validate)
        builder.add_node("finalize", self._finalize)
        builder.add_node("refuse", self._refuse)
        builder.add_edge(START, "retrieve")
        builder.add_conditional_edges("retrieve", self._route_after_retrieval)
        builder.add_conditional_edges("build_context", self._route_after_context)
        builder.add_edge("generate", "validate")
        builder.add_conditional_edges("validate", self._route_after_validation)
        builder.add_edge("finalize", END)
        builder.add_edge("refuse", END)
        return builder.compile()

    async def _retrieve(self, state: GraphState) -> GraphState:
        started = time.perf_counter()
        request = state["request"]
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
        return {
            "retrieved": retrieved,
            "retrieval_ms": (time.perf_counter() - started) * 1000,
            "node_trace": (*state["node_trace"], "retrieve"),
        }

    def _build_context(self, state: GraphState) -> GraphState:
        context = self._context_builder.build(state["retrieved"], self._config.max_context_tokens)
        return {
            "context": context,
            "node_trace": (*state["node_trace"], "build_context"),
        }

    async def _generate(self, state: GraphState) -> GraphState:
        started = time.perf_counter()
        request = state["request"]
        repair = state["repair_attempts"] > 0
        instruction = (
            "The previous answer had invalid citations. Cite only exact quotes and chunk IDs "
            "present in the evidence, or report insufficient evidence."
            if repair
            else "Answer only when the evidence supports the answer with exact citations."
        )
        chain = self._prompt | self._model_factory(state["model"], request.temperature)
        output = await chain.ainvoke(
            {
                "question": request.query,
                "context": state["context"].text,
                "validation_instruction": instruction,
            }
        )
        generated, raw = parse_structured_result(output)
        usage = add_usage(state["usage"], usage_from_message(raw))
        model = str(raw.response_metadata.get("model_name") or state["model"])
        return {
            "generated": generated,
            "raw": raw,
            "model": model,
            "usage": usage,
            "generation_ms": state["generation_ms"] + (time.perf_counter() - started) * 1000,
            "node_trace": (*state["node_trace"], "generate_repair" if repair else "generate"),
        }

    def _validate(self, state: GraphState) -> GraphState:
        generated = state["generated"]
        citations, citation_warnings = validate_citations(generated, state["context"])
        warnings = (*state["warnings"], *generated.warnings, *citation_warnings)
        repair_attempts = state["repair_attempts"]
        if generated.evidence_status is not EvidenceStatus.INSUFFICIENT and not citations:
            repair_attempts += 1
        return {
            "citations": citations,
            "warnings": warnings,
            "repair_attempts": repair_attempts,
            "node_trace": (*state["node_trace"], "validate"),
        }

    def _finalize(self, state: GraphState) -> GraphState:
        generated = state["generated"]
        return {
            "response": self._response(
                state,
                answer=generated.answer,
                citations=state["citations"],
                evidence_status=generated.evidence_status,
                confidence=generated.confidence,
            ),
            "node_trace": (*state["node_trace"], "finalize"),
        }

    def _refuse(self, state: GraphState) -> GraphState:
        if not state.get("retrieved"):
            reason = "retrieval returned no evidence"
        elif "context" not in state or not state["context"].chunks:
            reason = "no evidence fit within the context budget"
        elif state.get("generated") is not None and (
            state["generated"].evidence_status is EvidenceStatus.INSUFFICIENT
        ):
            reason = "model reported insufficient evidence"
        else:
            reason = "answer was rejected after bounded citation repair"
        warnings = (*state["warnings"], reason)
        state_with_warning = dict(state)
        state_with_warning["warnings"] = warnings
        return {
            "response": self._response(
                cast(GraphState, state_with_warning),
                answer=REFUSAL,
                citations=(),
                evidence_status=EvidenceStatus.INSUFFICIENT,
                confidence=0,
            ),
            "warnings": warnings,
            "node_trace": (*state["node_trace"], "refuse"),
        }

    def _response(
        self,
        state: GraphState,
        *,
        answer: str,
        citations: tuple[Citation, ...],
        evidence_status: EvidenceStatus,
        confidence: float | None,
    ) -> RAGResponse:
        context = state.get("context")
        request = state["request"]
        trace = (*state["node_trace"], "finalize" if citations else "refuse")
        return RAGResponse(
            answer=answer,
            citations=citations,
            retrieved_chunks=context.chunks if context is not None else (),
            framework=FrameworkName.LANGGRAPH,
            model=state["model"],
            latency=LatencyMetrics(
                total_ms=(time.perf_counter() - state["started"]) * 1000,
                retrieval_ms=state["retrieval_ms"],
                generation_ms=state["generation_ms"],
            ),
            usage=state["usage"],
            evidence_status=evidence_status,
            confidence=confidence,
            warnings=state["warnings"],
            debug=(
                {
                    "context_estimated_tokens": (
                        context.estimated_tokens if context is not None else 0
                    ),
                    "node_trace": trace,
                    "repair_attempts": state["repair_attempts"],
                }
                if request.debug
                else None
            ),
        )

    @staticmethod
    def _route_after_retrieval(state: GraphState) -> Literal["build_context", "refuse"]:
        return "build_context" if state["retrieved"] else "refuse"

    @staticmethod
    def _route_after_context(state: GraphState) -> Literal["generate", "refuse"]:
        return "generate" if state["context"].chunks else "refuse"

    @staticmethod
    def _route_after_validation(state: GraphState) -> Literal["finalize", "generate", "refuse"]:
        generated = state["generated"]
        if generated.evidence_status is EvidenceStatus.INSUFFICIENT:
            return "refuse"
        if state["citations"]:
            return "finalize"
        if state["repair_attempts"] <= MAX_REPAIR_ATTEMPTS:
            return "generate"
        return "refuse"
