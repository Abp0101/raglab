from collections.abc import Sequence
from uuid import uuid4

import pytest

from raglab.core.exceptions import ProviderUnavailableError
from raglab.core.schemas import (
    Embedding,
    RetrievalMode,
    RetrievalOptions,
    RetrievalRequest,
    RetrievedChunk,
)
from raglab.retrieval.service import RetrievalService
from tests.unit.retrieval_fixtures import make_chunk


class FakeEmbeddingProvider:
    model_name = "test"

    def __init__(self) -> None:
        self.query_calls = 0

    async def embed_chunks(self, chunks: Sequence[object]) -> Sequence[Embedding]:
        return ()

    async def embed_query(self, query: str) -> Sequence[float]:
        self.query_calls += 1
        return (1.0, 0.0)


class FakeDenseRetriever:
    def __init__(self, results: Sequence[RetrievedChunk]) -> None:
        self.results = results

    async def retrieve(
        self, request: RetrievalRequest, query_vector: Sequence[float]
    ) -> Sequence[RetrievedChunk]:
        return self.results


class FakeSparseRetriever:
    def __init__(self, results: Sequence[RetrievedChunk]) -> None:
        self.results = results

    async def retrieve(self, request: RetrievalRequest) -> Sequence[RetrievedChunk]:
        return self.results


class FailingSparseRetriever:
    async def retrieve(self, request: RetrievalRequest) -> Sequence[RetrievedChunk]:
        raise ConnectionError("redis connection details must remain internal")


class FakeReranker:
    async def rerank(
        self, query: str, candidates: Sequence[RetrievedChunk], top_k: int
    ) -> Sequence[RetrievedChunk]:
        return tuple(reversed(candidates[:top_k]))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "expected_embedding_calls"),
    [(RetrievalMode.DENSE, 1), (RetrievalMode.SPARSE, 0), (RetrievalMode.HYBRID, 1)],
)
async def test_service_executes_selected_first_stage_mode(
    mode: RetrievalMode, expected_embedding_calls: int
) -> None:
    shared = make_chunk("shared")
    dense = RetrievedChunk(chunk=shared, rank=1, dense_score=0.9)
    sparse = RetrievedChunk(chunk=shared, rank=1, sparse_score=2.0)
    embeddings = FakeEmbeddingProvider()
    service = RetrievalService(
        embedding_provider=embeddings,  # type: ignore[arg-type]
        dense_retriever=FakeDenseRetriever([dense]),
        sparse_retriever=FakeSparseRetriever([sparse]),
    )

    results = await service.retrieve(
        RetrievalRequest(query="question", collection_id=uuid4()),
        RetrievalOptions(mode=mode, rerank=False, expand_parents=False),
    )

    assert len(results) == 1
    assert embeddings.query_calls == expected_embedding_calls
    if mode is RetrievalMode.HYBRID:
        assert results[0].fusion_score is not None


@pytest.mark.asyncio
async def test_service_requires_configured_reranker_when_requested() -> None:
    chunk = RetrievedChunk(chunk=make_chunk("evidence"), rank=1, dense_score=0.9)
    service = RetrievalService(
        embedding_provider=FakeEmbeddingProvider(),  # type: ignore[arg-type]
        dense_retriever=FakeDenseRetriever([chunk]),
        sparse_retriever=FakeSparseRetriever([]),
    )

    with pytest.raises(ValueError, match="no reranker"):
        await service.retrieve(
            RetrievalRequest(query="question", collection_id=uuid4()),
            RetrievalOptions(mode=RetrievalMode.DENSE, rerank=True),
        )


@pytest.mark.asyncio
async def test_service_translates_storage_failure_to_safe_provider_error() -> None:
    service = RetrievalService(
        embedding_provider=FakeEmbeddingProvider(),  # type: ignore[arg-type]
        dense_retriever=FakeDenseRetriever([]),
        sparse_retriever=FailingSparseRetriever(),
    )

    with pytest.raises(
        ProviderUnavailableError,
        match="retrieval provider request failed",
    ) as captured:
        await service.retrieve(
            RetrievalRequest(query="question", collection_id=uuid4()),
            RetrievalOptions(mode=RetrievalMode.SPARSE, rerank=False),
        )

    assert "redis connection details" not in str(captured.value)


@pytest.mark.asyncio
async def test_service_reranks_and_limits_final_results() -> None:
    first = RetrievedChunk(chunk=make_chunk("first"), rank=1, dense_score=0.9)
    second = RetrievedChunk(chunk=make_chunk("second"), rank=2, dense_score=0.8)
    service = RetrievalService(
        embedding_provider=FakeEmbeddingProvider(),  # type: ignore[arg-type]
        dense_retriever=FakeDenseRetriever([first, second]),
        sparse_retriever=FakeSparseRetriever([]),
        reranker=FakeReranker(),
    )

    results = await service.retrieve(
        RetrievalRequest(query="question", collection_id=uuid4()),
        RetrievalOptions(
            mode=RetrievalMode.DENSE,
            candidate_k=2,
            top_k=1,
            rerank=True,
            expand_parents=False,
        ),
    )

    assert [result.chunk.text for result in results] == ["second"]
    assert results[0].rank == 1
