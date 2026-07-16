from datetime import date
from uuid import uuid4

import pytest
from qdrant_client import AsyncQdrantClient

from raglab.core.schemas import Embedding, MetadataFilter, RetrievalRequest
from raglab.retrieval import QdrantDenseRetriever, QdrantVectorIndexer
from tests.unit.retrieval_fixtures import make_chunk


@pytest.mark.asyncio
async def test_dense_retrieval_applies_collection_and_metadata_filters() -> None:
    client = AsyncQdrantClient(location=":memory:")
    collection_id = uuid4()
    other_collection = uuid4()
    methods = make_chunk("IMU calibration procedure", collection_id=collection_id)
    results = make_chunk(
        "IMU validation outcome",
        collection_id=collection_id,
        section_heading="RESULTS",
        chunk_index=1,
    )
    unrelated = make_chunk("IMU elsewhere", collection_id=other_collection)
    chunks = (methods, results, unrelated)
    embeddings = tuple(
        Embedding(
            chunk_id=chunk.chunk_id,
            vector=vector,
            model="test",
            dimensions=2,
        )
        for chunk, vector in zip(chunks, ((1.0, 0.0), (0.9, 0.1), (1.0, 0.0)), strict=True)
    )
    indexer = QdrantVectorIndexer(client, "dense_test")
    await indexer.upsert(chunks, embeddings)
    retriever = QdrantDenseRetriever(client, "dense_test")

    retrieved = await retriever.retrieve(
        RetrievalRequest(
            query="calibration",
            collection_id=collection_id,
            top_k=5,
            metadata_filter=MetadataFilter(
                section_headings=("METHODS",),
                published_from=date(2025, 1, 1),
                attributes={"page_number": 1},
            ),
        ),
        (1.0, 0.0),
    )

    assert [result.chunk.chunk_id for result in retrieved] == [methods.chunk_id]
    assert retrieved[0].dense_score is not None
    await client.close()


@pytest.mark.asyncio
async def test_dense_retriever_returns_empty_for_missing_index() -> None:
    client = AsyncQdrantClient(location=":memory:")
    retriever = QdrantDenseRetriever(client, "missing")

    result = await retriever.retrieve(
        RetrievalRequest(query="question", collection_id=uuid4()), (1.0, 0.0)
    )

    assert result == ()
    await client.close()
