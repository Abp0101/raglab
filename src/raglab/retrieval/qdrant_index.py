"""Qdrant adapter for dense ingestion vectors."""

import asyncio
from collections.abc import Sequence
from uuid import UUID

from qdrant_client import AsyncQdrantClient, models

from raglab.core.schemas import Chunk, Embedding


class QdrantVectorIndexer:
    """Maintain one shared cosine collection with filterable payloads."""

    def __init__(self, client: AsyncQdrantClient, collection_name: str) -> None:
        self._client = client
        self._collection_name = collection_name
        self._init_lock = asyncio.Lock()

    async def upsert(self, chunks: Sequence[Chunk], embeddings: Sequence[Embedding]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("each chunk must have one embedding")
        if not chunks:
            return
        dimensions = embeddings[0].dimensions
        if any(embedding.dimensions != dimensions for embedding in embeddings):
            raise ValueError("all embeddings must have the same dimensions")
        await self._ensure_collection(dimensions)
        await self._client.upsert(
            collection_name=self._collection_name,
            points=[
                models.PointStruct(
                    id=str(chunk.chunk_id),
                    vector=list(embedding.vector),
                    payload=_payload(chunk, embedding.model),
                )
                for chunk, embedding in zip(chunks, embeddings, strict=True)
            ],
            wait=True,
        )

    async def delete(self, chunk_ids: Sequence[UUID]) -> None:
        if not chunk_ids or not await self._client.collection_exists(self._collection_name):
            return
        await self._client.delete(
            collection_name=self._collection_name,
            points_selector=models.PointIdsList(points=[str(chunk_id) for chunk_id in chunk_ids]),
            wait=True,
        )

    async def _ensure_collection(self, dimensions: int) -> None:
        if await self._client.collection_exists(self._collection_name):
            await self._validate_dimensions(dimensions)
            return
        async with self._init_lock:
            if not await self._client.collection_exists(self._collection_name):
                await self._client.create_collection(
                    collection_name=self._collection_name,
                    vectors_config=models.VectorParams(
                        size=dimensions, distance=models.Distance.COSINE
                    ),
                )
                for field in ("collection_id", "document_id", "file_type"):
                    await self._client.create_payload_index(
                        collection_name=self._collection_name,
                        field_name=field,
                        field_schema=models.PayloadSchemaType.KEYWORD,
                    )
            else:
                await self._validate_dimensions(dimensions)

    async def _validate_dimensions(self, dimensions: int) -> None:
        info = await self._client.get_collection(self._collection_name)
        vector_config = info.config.params.vectors
        if not isinstance(vector_config, models.VectorParams) or vector_config.size != dimensions:
            raise ValueError("embedding dimensions do not match the existing Qdrant collection")


def _payload(chunk: Chunk, model: str) -> dict[str, object]:
    metadata = chunk.metadata
    return {
        "chunk_id": str(chunk.chunk_id),
        "document_id": str(metadata.document_id),
        "collection_id": str(metadata.collection_id),
        "file_name": metadata.file_name,
        "display_title": metadata.display_title,
        "authors": list(metadata.authors),
        "file_type": metadata.file_type,
        "page_number": metadata.page_number,
        "section_heading": metadata.section_heading,
        "chunk_index": metadata.chunk_index,
        "parent_chunk_id": str(metadata.parent_chunk_id) if metadata.parent_chunk_id else None,
        "content_hash": metadata.content_hash,
        "embedding_model": model,
        "text": chunk.text,
    }
