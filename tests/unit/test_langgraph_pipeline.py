from collections.abc import Sequence
from typing import Any
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from raglab.core.interfaces import RAGPipeline
from raglab.core.schemas import (
    DocumentInput,
    EvidenceStatus,
    FrameworkName,
    PipelineConfig,
    QueryRequest,
    RetrievalRequest,
    RetrievedChunk,
)
from raglab.generation.output import GeneratedCitation, GroundedAnswer
from raglab.pipelines.langgraph_rag import LangGraphRAGPipeline
from raglab.retrieval.service import RetrievalService
from tests.unit.retrieval_fixtures import make_chunk


class FakeIngestion:
    async def ingest(self, document: DocumentInput) -> object:
        raise AssertionError("ingestion is not used in query tests")


class FakeEmbeddings:
    model_name = "local-test"

    async def embed_chunks(self, chunks: Sequence[object]) -> Sequence[object]:
        return ()

    async def embed_query(self, query: str) -> Sequence[float]:
        return (0.1, 0.2)


class FakeDenseRetriever:
    def __init__(self, results: Sequence[RetrievedChunk]) -> None:
        self.results = results

    async def retrieve(
        self,
        request: RetrievalRequest,
        query_vector: Sequence[float],
    ) -> Sequence[RetrievedChunk]:
        return self.results


class FakeSparseRetriever:
    async def retrieve(self, request: RetrievalRequest) -> Sequence[RetrievedChunk]:
        return ()


def retrieval(results: Sequence[RetrievedChunk]) -> RetrievalService:
    return RetrievalService(
        embedding_provider=FakeEmbeddings(),  # type: ignore[arg-type]
        dense_retriever=FakeDenseRetriever(results),
        sparse_retriever=FakeSparseRetriever(),
    )


def query() -> QueryRequest:
    return QueryRequest(
        query="What was the sampling rate?",
        framework=FrameworkName.LANGGRAPH,
        collection_id=uuid4(),
        retrieval_mode="dense",
        rerank=False,
        debug=True,
    )


def output(answer: GroundedAnswer, model: str) -> dict[str, Any]:
    return {
        "parsed": answer,
        "raw": AIMessage(
            content="",
            response_metadata={"model_name": model},
            usage_metadata={
                "input_tokens": 40,
                "output_tokens": 12,
                "total_tokens": 52,
            },
        ),
        "parsing_error": None,
    }


def model_factory(answers: Sequence[GroundedAnswer]) -> Any:
    remaining = list(answers)

    def factory(model: str, temperature: float) -> RunnableLambda[Any, dict[str, Any]]:
        return RunnableLambda(lambda _: output(remaining.pop(0), model))

    return factory


def pipeline(
    results: Sequence[RetrievedChunk],
    answers: Sequence[GroundedAnswer],
) -> LangGraphRAGPipeline:
    return LangGraphRAGPipeline(
        ingestion=FakeIngestion(),  # type: ignore[arg-type]
        retrieval=retrieval(results),
        model_factory=model_factory(answers),
        default_model="local-test",
        config=PipelineConfig(candidate_k=5, top_k=3, max_context_tokens=1000),
    )


@pytest.mark.asyncio
async def test_langgraph_pipeline_executes_explicit_valid_path() -> None:
    chunk = make_chunk("The rehabilitation IMU sampled motion at 100 Hz.")
    generated = GroundedAnswer(
        answer="The IMU sampled motion at 100 Hz.",
        citations=(
            GeneratedCitation(chunk_id=chunk.chunk_id, quoted_text="sampled motion at 100 Hz"),
        ),
        evidence_status=EvidenceStatus.SUFFICIENT,
        confidence=0.9,
    )
    graph_pipeline = pipeline(
        (RetrievedChunk(chunk=chunk, rank=1, relevance_score=0.9),),
        (generated,),
    )

    response = await graph_pipeline.query(query())

    assert isinstance(graph_pipeline, RAGPipeline)
    assert graph_pipeline.capabilities.agentic is True
    assert response.framework is FrameworkName.LANGGRAPH
    assert response.usage.estimated_cost_usd == 0
    assert response.usage.llm_calls == 1
    assert response.debug is not None
    assert response.debug["node_trace"] == (
        "retrieve",
        "build_context",
        "generate",
        "validate",
        "finalize",
    )
    assert "generate" in graph_pipeline.graph_mermaid()


@pytest.mark.asyncio
async def test_langgraph_pipeline_refuses_without_evidence_and_skips_model() -> None:
    graph_pipeline = pipeline((), ())

    response = await graph_pipeline.query(query())

    assert response.evidence_status is EvidenceStatus.INSUFFICIENT
    assert response.usage.llm_calls == 0
    assert response.usage.estimated_cost_usd == 0
    assert response.debug is not None
    assert response.debug["node_trace"] == ("retrieve", "refuse")


@pytest.mark.asyncio
async def test_langgraph_pipeline_repairs_invalid_citation_once() -> None:
    chunk = make_chunk("Evidence says the sampling rate was 100 Hz.")
    invalid = GroundedAnswer(
        answer="It was 200 Hz.",
        citations=(GeneratedCitation(chunk_id=chunk.chunk_id, quoted_text="200 Hz"),),
        evidence_status=EvidenceStatus.SUFFICIENT,
        confidence=0.5,
    )
    repaired = GroundedAnswer(
        answer="The sampling rate was 100 Hz.",
        citations=(GeneratedCitation(chunk_id=chunk.chunk_id, quoted_text="100 Hz"),),
        evidence_status=EvidenceStatus.SUFFICIENT,
        confidence=0.9,
    )
    graph_pipeline = pipeline((RetrievedChunk(chunk=chunk, rank=1),), (invalid, repaired))

    response = await graph_pipeline.query(query())

    assert response.answer == repaired.answer
    assert response.usage.llm_calls == 2
    assert response.usage.total_tokens == 104
    assert response.debug is not None
    assert response.debug["repair_attempts"] == 1
    assert "generate_repair" in response.debug["node_trace"]


@pytest.mark.asyncio
async def test_langgraph_pipeline_refuses_after_bounded_repair() -> None:
    chunk = make_chunk("Evidence says 100 Hz.")
    invalid = GroundedAnswer(
        answer="It was 200 Hz.",
        citations=(GeneratedCitation(chunk_id=chunk.chunk_id, quoted_text="200 Hz"),),
        evidence_status=EvidenceStatus.SUFFICIENT,
        confidence=0.5,
    )
    graph_pipeline = pipeline((RetrievedChunk(chunk=chunk, rank=1),), (invalid, invalid))

    response = await graph_pipeline.query(query())

    assert response.evidence_status is EvidenceStatus.INSUFFICIENT
    assert response.citations == ()
    assert response.usage.llm_calls == 2
    assert response.debug is not None
    assert response.debug["repair_attempts"] == 2
    assert response.debug["node_trace"][-1] == "refuse"
    assert response.warnings[-1] == "answer was rejected after bounded citation repair"
