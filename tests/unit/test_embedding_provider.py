from datetime import UTC, datetime
from uuid import uuid4

import numpy as np
import pytest

from raglab.core.schemas import Chunk, DocumentMetadata
from raglab.embeddings import SentenceTransformerEmbeddingProvider


class FakeEncoder:
    def encode(self, sentences: list[str], **kwargs: object) -> np.ndarray:
        assert kwargs["normalize_embeddings"] is True
        return np.array([[float(index), 1.0] for index, _ in enumerate(sentences)])


def make_chunk(text: str) -> Chunk:
    return Chunk(
        chunk_id=uuid4(),
        text=text,
        metadata=DocumentMetadata(
            document_id=uuid4(),
            collection_id=uuid4(),
            file_name="study.pdf",
            display_title="Study",
            uploaded_at=datetime.now(UTC),
            file_type="application/pdf",
            chunk_index=0,
            content_hash="c" * 64,
        ),
    )


@pytest.mark.asyncio
async def test_provider_maps_encoder_vectors_to_chunk_embeddings() -> None:
    chunks = [make_chunk("first"), make_chunk("second")]
    provider = SentenceTransformerEmbeddingProvider("test-model", model=FakeEncoder())

    embeddings = await provider.embed_chunks(chunks)
    query = await provider.embed_query("question")

    assert [embedding.chunk_id for embedding in embeddings] == [chunk.chunk_id for chunk in chunks]
    assert embeddings[0].dimensions == 2
    assert embeddings[1].vector == (1.0, 1.0)
    assert query == (0.0, 1.0)


@pytest.mark.asyncio
async def test_provider_does_not_load_model_for_empty_batch() -> None:
    provider = SentenceTransformerEmbeddingProvider("not-downloaded")

    assert await provider.embed_chunks([]) == ()


@pytest.mark.live_model
@pytest.mark.asyncio
async def test_default_local_model_generates_normalized_query_embedding() -> None:
    provider = SentenceTransformerEmbeddingProvider("sentence-transformers/all-MiniLM-L6-v2")

    vector = await provider.embed_query("wearable rehabilitation sensor")

    assert len(vector) == 384
    assert sum(value * value for value in vector) == pytest.approx(1.0, rel=1e-4)
