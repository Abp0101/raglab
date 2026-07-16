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
from raglab.pipelines.langchain_rag import LangChainRAGPipeline
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
        framework=FrameworkName.LANGCHAIN,
        collection_id=uuid4(),
        retrieval_mode="dense",
        rerank=False,
        debug=True,
    )


def model_factory(answer: GroundedAnswer) -> Any:
    def factory(model: str, temperature: float) -> RunnableLambda[Any, dict[str, Any]]:
        return RunnableLambda(
            lambda _: {
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
        )

    return factory


@pytest.mark.asyncio
async def test_langchain_pipeline_uses_native_chain_and_shared_contract() -> None:
    chunk = make_chunk("The rehabilitation IMU sampled motion at 100 Hz.")
    result = RetrievedChunk(chunk=chunk, rank=1, relevance_score=0.9, dense_score=0.9)
    generated = GroundedAnswer(
        answer="The IMU sampled motion at 100 Hz.",
        citations=(
            GeneratedCitation(
                chunk_id=chunk.chunk_id,
                quoted_text="sampled motion at 100 Hz",
            ),
        ),
        evidence_status=EvidenceStatus.SUFFICIENT,
        confidence=0.9,
        warnings=(),
    )
    pipeline = LangChainRAGPipeline(
        ingestion=FakeIngestion(),  # type: ignore[arg-type]
        retrieval=retrieval((result,)),
        model_factory=model_factory(generated),
        default_model="local-test",
        config=PipelineConfig(candidate_k=5, top_k=3, max_context_tokens=1000),
    )

    response = await pipeline.query(query())

    assert isinstance(pipeline, RAGPipeline)
    assert response.framework is FrameworkName.LANGCHAIN
    assert response.citations[0].chunk_id == chunk.chunk_id
    assert response.usage.total_tokens == 52
    assert response.usage.estimated_cost_usd == 0
    assert response.debug is not None
    assert response.debug["orchestration"][0] == "BaseRetriever.ainvoke"


@pytest.mark.asyncio
async def test_langchain_pipeline_refuses_without_retrieval_and_skips_model() -> None:
    calls = 0

    def factory(model: str, temperature: float) -> Any:
        nonlocal calls
        calls += 1
        raise AssertionError("model should not be constructed")

    pipeline = LangChainRAGPipeline(
        ingestion=FakeIngestion(),  # type: ignore[arg-type]
        retrieval=retrieval(()),
        model_factory=factory,
        default_model="local-test",
    )

    response = await pipeline.query(query())

    assert response.evidence_status is EvidenceStatus.INSUFFICIENT
    assert response.citations == ()
    assert calls == 0


@pytest.mark.asyncio
async def test_langchain_pipeline_rejects_invalid_citation() -> None:
    chunk = make_chunk("Evidence says 100 Hz.")
    generated = GroundedAnswer(
        answer="It was 200 Hz.",
        citations=(GeneratedCitation(chunk_id=chunk.chunk_id, quoted_text="Evidence says 200 Hz"),),
        evidence_status=EvidenceStatus.SUFFICIENT,
        confidence=0.7,
        warnings=(),
    )
    pipeline = LangChainRAGPipeline(
        ingestion=FakeIngestion(),  # type: ignore[arg-type]
        retrieval=retrieval((RetrievedChunk(chunk=chunk, rank=1),)),
        model_factory=model_factory(generated),
        default_model="local-test",
    )

    response = await pipeline.query(query())

    assert response.evidence_status is EvidenceStatus.INSUFFICIENT
    assert response.citations == ()
    assert "no valid citations" in response.warnings[-1]
