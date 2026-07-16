from datetime import UTC, datetime
from uuid import uuid4

import pytest
from qdrant_client import AsyncQdrantClient

from raglab.core.schemas import Chunk, DocumentMetadata, Embedding
from raglab.retrieval import QdrantVectorIndexer


def make_chunk() -> Chunk:
    return Chunk(
        chunk_id=uuid4(),
        text="Force sensors measured load.",
        metadata=DocumentMetadata(
            document_id=uuid4(),
            collection_id=uuid4(),
            file_name="fsr.pdf",
            display_title="FSR Study",
            uploaded_at=datetime.now(UTC),
            file_type="application/pdf",
            page_number=2,
            chunk_index=0,
            content_hash="e" * 64,
        ),
    )


@pytest.mark.asyncio
async def test_qdrant_indexer_round_trip_in_local_mode() -> None:
    client = AsyncQdrantClient(location=":memory:")
    indexer = QdrantVectorIndexer(client, "test_chunks")
    chunk = make_chunk()
    embedding = Embedding(
        chunk_id=chunk.chunk_id,
        vector=(0.1, 0.2, 0.3),
        model="test-model",
        dimensions=3,
    )

    await indexer.upsert([chunk], [embedding])
    records, _ = await client.scroll("test_chunks", limit=10, with_payload=True)
    assert records[0].payload is not None
    assert records[0].payload["collection_id"] == str(chunk.metadata.collection_id)

    await indexer.delete([chunk.chunk_id])
    records, _ = await client.scroll("test_chunks", limit=10)
    assert records == []
    await client.close()


@pytest.mark.asyncio
async def test_qdrant_indexer_rejects_dimension_change() -> None:
    client = AsyncQdrantClient(location=":memory:")
    indexer = QdrantVectorIndexer(client, "test_chunks")
    chunk = make_chunk()
    await indexer.upsert(
        [chunk],
        [Embedding(chunk_id=chunk.chunk_id, vector=(0.1, 0.2), model="one", dimensions=2)],
    )

    with pytest.raises(ValueError, match="dimensions"):
        await indexer.upsert(
            [chunk],
            [
                Embedding(
                    chunk_id=chunk.chunk_id,
                    vector=(0.1, 0.2, 0.3),
                    model="two",
                    dimensions=3,
                )
            ],
        )
    await client.close()
