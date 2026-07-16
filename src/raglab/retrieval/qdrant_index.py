"""Qdrant adapter for dense ingestion vectors."""

import asyncio
from collections.abc import Sequence
from uuid import UUID

from qdrant_client import AsyncQdrantClient, models

from raglab.core.schemas import Chunk, Embedding, RetrievalRequest, RetrievedChunk
from raglab.retrieval.serialization import chunk_to_payload, payload_to_chunk


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
                    payload={**chunk_to_payload(chunk), "embedding_model": embedding.model},
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
                for field in (
                    "collection_id",
                    "document_id",
                    "file_type",
                    "authors",
                    "section_heading",
                ):
                    await self._client.create_payload_index(
                        collection_name=self._collection_name,
                        field_name=field,
                        field_schema=models.PayloadSchemaType.KEYWORD,
                    )
                await self._client.create_payload_index(
                    collection_name=self._collection_name,
                    field_name="publication_ordinal",
                    field_schema=models.PayloadSchemaType.INTEGER,
                )
            else:
                await self._validate_dimensions(dimensions)

    async def _validate_dimensions(self, dimensions: int) -> None:
        info = await self._client.get_collection(self._collection_name)
        vector_config = info.config.params.vectors
        if not isinstance(vector_config, models.VectorParams) or vector_config.size != dimensions:
            raise ValueError("embedding dimensions do not match the existing Qdrant collection")


class QdrantDenseRetriever:
    """Search normalized vectors with server-side collection and metadata filters."""

    def __init__(self, client: AsyncQdrantClient, collection_name: str) -> None:
        self._client = client
        self._collection_name = collection_name

    async def retrieve(
        self,
        request: RetrievalRequest,
        query_vector: Sequence[float],
    ) -> Sequence[RetrievedChunk]:
        if not await self._client.collection_exists(self._collection_name):
            return ()
        response = await self._client.query_points(
            collection_name=self._collection_name,
            query=list(query_vector),
            query_filter=_qdrant_filter(request),
            limit=request.top_k,
            with_payload=True,
        )
        results: list[RetrievedChunk] = []
        for rank, point in enumerate(response.points, start=1):
            if point.payload is None:
                continue
            score = float(point.score)
            results.append(
                RetrievedChunk(
                    chunk=payload_to_chunk(point.payload),
                    rank=rank,
                    relevance_score=score,
                    dense_score=score,
                )
            )
        return tuple(results)


def _qdrant_filter(request: RetrievalRequest) -> models.Filter:
    must: list[models.Condition] = [
        models.FieldCondition(
            key="collection_id", match=models.MatchValue(value=str(request.collection_id))
        )
    ]
    metadata = request.metadata_filter
    if metadata is None:
        return models.Filter(must=must)
    if metadata.document_ids:
        must.append(
            models.FieldCondition(
                key="document_id",
                match=models.MatchAny(any=[str(value) for value in metadata.document_ids]),
            )
        )
    if metadata.authors:
        must.append(
            models.FieldCondition(key="authors", match=models.MatchAny(any=list(metadata.authors)))
        )
    if metadata.file_types:
        must.append(
            models.FieldCondition(
                key="file_type", match=models.MatchAny(any=list(metadata.file_types))
            )
        )
    if metadata.section_headings:
        must.append(
            models.FieldCondition(
                key="section_heading",
                match=models.MatchAny(any=list(metadata.section_headings)),
            )
        )
    if metadata.published_from or metadata.published_to:
        must.append(
            models.FieldCondition(
                key="publication_ordinal",
                range=models.Range(
                    gte=(metadata.published_from.toordinal() if metadata.published_from else None),
                    lte=(metadata.published_to.toordinal() if metadata.published_to else None),
                ),
            )
        )
    for key, value in metadata.attributes.items():
        must.append(models.FieldCondition(key=key, match=models.MatchValue(value=value)))
    return models.Filter(must=must)
