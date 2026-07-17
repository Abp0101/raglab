import os
from collections.abc import Sequence
from typing import Any
from uuid import uuid4

import pytest
from haystack import Document
from haystack.dataclasses import ChatMessage
from haystack.telemetry import _telemetry as haystack_telemetry

from raglab.core.interfaces import RAGPipeline
from raglab.core.schemas import (
    DocumentInput,
    EvidenceStatus,
    FrameworkName,
    PipelineConfig,
    QueryRequest,
    RetrievedChunk,
)
from raglab.generation.output import GeneratedCitation, GroundedAnswer
from raglab.pipelines.haystack_rag import (
    HaystackRAGPipeline,
    SharedStoreHaystackRetriever,
    usage_from_haystack_meta,
)
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
        request: object,
        query_vector: Sequence[float],
    ) -> Sequence[RetrievedChunk]:
        return self.results


class FakeSparseRetriever:
    async def retrieve(self, request: object) -> Sequence[RetrievedChunk]:
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
        framework=FrameworkName.HAYSTACK,
        collection_id=uuid4(),
        retrieval_mode="dense",
        rerank=False,
        debug=True,
    )


class FakeGenerator:
    def __init__(self, answer: GroundedAnswer) -> None:
        self.answer = answer
        self.messages: list[ChatMessage] = []

    async def run_async(self, messages: list[ChatMessage] | str) -> dict[str, Any]:
        assert isinstance(messages, list)
        self.messages = messages
        return {
            "replies": [
                ChatMessage.from_assistant(
                    text=self.answer.model_dump_json(),
                    meta={
                        "model": "resolved-local-test",
                        "usage": {
                            "prompt_tokens": 40,
                            "completion_tokens": 12,
                            "total_tokens": 52,
                        },
                    },
                )
            ]
        }


def pipeline(
    results: Sequence[RetrievedChunk],
    answer: GroundedAnswer,
) -> tuple[HaystackRAGPipeline, FakeGenerator]:
    generator = FakeGenerator(answer)
    adapter = HaystackRAGPipeline(
        ingestion=FakeIngestion(),  # type: ignore[arg-type]
        retrieval=retrieval(results),
        generator_factory=lambda model, temperature: generator,
        default_model="local-test",
        config=PipelineConfig(candidate_k=5, top_k=3, max_context_tokens=1000),
    )
    return adapter, generator


@pytest.mark.asyncio
async def test_haystack_pipeline_runs_native_async_graph_and_shared_contract() -> None:
    chunk = make_chunk("The rehabilitation IMU sampled motion at 100 Hz.")
    result = RetrievedChunk(
        chunk=chunk,
        rank=1,
        relevance_score=0.8,
        reranker_score=0.95,
    )
    generated = GroundedAnswer(
        answer="The IMU sampled motion at 100 Hz.",
        citations=(
            GeneratedCitation(chunk_id=chunk.chunk_id, quoted_text="sampled motion at 100 Hz"),
        ),
        evidence_status=EvidenceStatus.SUFFICIENT,
        confidence=0.9,
    )
    adapter, generator = pipeline((result,), generated)

    response = await adapter.query(query())

    assert isinstance(adapter, RAGPipeline)
    assert response.framework is FrameworkName.HAYSTACK
    assert response.citations[0].chunk_id == chunk.chunk_id
    assert response.usage.total_tokens == 52
    assert response.usage.estimated_cost_usd == 0
    assert response.model == "resolved-local-test"
    prompt_text = generator.messages[1].text
    assert prompt_text is not None
    assert "What was the sampling rate?" in prompt_text
    assert response.debug is not None
    assert response.debug["orchestration"][0] == "AsyncPipeline.run_async"
    assert response.debug["haystack_document_ids"] == [str(chunk.chunk_id)]
    assert response.debug["telemetry_enabled"] is False


@pytest.mark.asyncio
async def test_haystack_retriever_maps_document_score_and_provenance() -> None:
    chunk = make_chunk("Evidence says 100 Hz.")
    result = RetrievedChunk(chunk=chunk, rank=1, relevance_score=0.7, reranker_score=0.91)
    request = query().model_copy(update={"collection_id": chunk.metadata.collection_id})
    retriever = SharedStoreHaystackRetriever(
        retrieval((result,)),
        PipelineConfig(candidate_k=5, top_k=5),
    )

    output = await retriever.run_async(request)

    document = output["documents"][0]
    assert isinstance(document, Document)
    assert document.id == str(chunk.chunk_id)
    assert document.score == 0.91
    restored = RetrievedChunk.model_validate_json(document.meta["raglab_retrieved_chunk"])
    assert restored == result


@pytest.mark.asyncio
async def test_haystack_pipeline_refuses_without_evidence_and_skips_model() -> None:
    generated = GroundedAnswer(
        answer="unused",
        citations=(),
        evidence_status=EvidenceStatus.INSUFFICIENT,
        confidence=0,
    )
    adapter, generator = pipeline((), generated)

    response = await adapter.query(query())

    assert response.evidence_status is EvidenceStatus.INSUFFICIENT
    assert response.usage.llm_calls == 0
    assert response.usage.estimated_cost_usd == 0
    assert generator.messages == []


@pytest.mark.asyncio
async def test_haystack_pipeline_rejects_invalid_citation() -> None:
    chunk = make_chunk("Evidence says 100 Hz.")
    generated = GroundedAnswer(
        answer="It was 200 Hz.",
        citations=(GeneratedCitation(chunk_id=chunk.chunk_id, quoted_text="200 Hz"),),
        evidence_status=EvidenceStatus.SUFFICIENT,
        confidence=0.5,
    )
    adapter, _ = pipeline((RetrievedChunk(chunk=chunk, rank=1),), generated)

    response = await adapter.query(query())

    assert response.evidence_status is EvidenceStatus.INSUFFICIENT
    assert response.citations == ()
    assert response.warnings[-1] == "answer was rejected because it had no valid citations"


def test_haystack_usage_and_telemetry_are_local_only() -> None:
    usage = usage_from_haystack_meta(
        {
            "usage": {
                "prompt_tokens": 17,
                "completion_tokens": 5,
                "total_tokens": 22,
            }
        }
    )

    assert usage.prompt_tokens == 17
    assert usage.completion_tokens == 5
    assert usage.total_tokens == 22
    assert usage.estimated_cost_usd == 0
    assert usage.llm_calls == 1
    assert os.environ["HAYSTACK_TELEMETRY_ENABLED"] == "false"
    assert haystack_telemetry.telemetry is None
