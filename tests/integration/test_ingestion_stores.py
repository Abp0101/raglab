from collections.abc import Sequence
from contextlib import suppress
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis
from sqlalchemy import delete, select

from raglab.core.config import Settings
from raglab.core.schemas import (
    Chunk,
    CollectionCreate,
    Document,
    DocumentMetadata,
    DocumentStatus,
    Embedding,
    RetrievalRequest,
)
from raglab.database.models import CollectionRecord, DocumentRecord
from raglab.database.repositories import (
    SQLAlchemyCatalogRepository,
    SQLAlchemyChunkRepository,
    SQLAlchemyDocumentRepository,
)
from raglab.database.session import create_engine, create_session_factory
from raglab.retrieval import (
    QdrantDenseRetriever,
    QdrantVectorIndexer,
    RedisBM25Indexer,
    RedisBM25Retriever,
)

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_collection_catalog_round_trip() -> None:
    settings = Settings(_env_file=None)
    engine = create_engine(settings.postgres_dsn)
    sessions = create_session_factory(engine)
    catalog = SQLAlchemyCatalogRepository(sessions)
    collection_id = None

    try:
        created = await catalog.create_collection(
            CollectionCreate(name="API integration", description="Local services only")
        )
        collection_id = created.collection_id

        fetched = await catalog.get_collection(created.collection_id)
        listed = await catalog.list_collections()

        assert fetched.name == "API integration"
        assert fetched.document_count == 0
        assert created.collection_id in {item.collection_id for item in listed}
    finally:
        if collection_id is not None:
            with suppress(Exception):
                async with sessions() as session, session.begin():
                    await session.execute(
                        delete(CollectionRecord).where(CollectionRecord.id == collection_id)
                    )
        await engine.dispose()


def make_document_and_chunks() -> tuple[Document, Sequence[Chunk]]:
    collection_id = uuid4()
    document = Document(
        document_id=uuid4(),
        collection_id=collection_id,
        file_name="integration.pdf",
        display_title="Integration Study",
        uploaded_at=datetime.now(UTC),
        file_type="application/pdf",
        content_hash="f" * 64,
        page_count=1,
        status=DocumentStatus.PROCESSING,
    )
    metadata = DocumentMetadata(
        document_id=document.document_id,
        collection_id=collection_id,
        file_name=document.file_name,
        display_title=document.display_title,
        uploaded_at=document.uploaded_at,
        file_type=document.file_type,
        page_number=1,
        chunk_index=0,
        content_hash=document.content_hash,
    )
    return document, [Chunk(chunk_id=uuid4(), text="The IMU sampled at 100 Hz.", metadata=metadata)]


@pytest.mark.asyncio
async def test_document_dense_and_sparse_stores_round_trip() -> None:
    settings = Settings(_env_file=None)
    document, chunks = make_document_and_chunks()
    engine = create_engine(settings.postgres_dsn)
    sessions = create_session_factory(engine)
    repository = SQLAlchemyDocumentRepository(sessions)
    chunk_repository = SQLAlchemyChunkRepository(sessions)
    qdrant = AsyncQdrantClient(url=str(settings.qdrant_url), check_compatibility=False)
    collection_name = f"integration_{uuid4().hex}"
    vectors = QdrantVectorIndexer(qdrant, collection_name)
    redis = Redis.from_url(str(settings.redis_dsn), decode_responses=True)
    key_prefix = f"integration:{uuid4().hex}"
    sparse = RedisBM25Indexer(redis, key_prefix=key_prefix)

    try:
        now = datetime.now(UTC)
        async with sessions() as session, session.begin():
            session.add(
                CollectionRecord(
                    id=document.collection_id,
                    name="Integration",
                    description=None,
                    created_at=now,
                    updated_at=now,
                )
            )

        await repository.save(document, chunks)
        await repository.set_status(document.document_id, DocumentStatus.READY)
        found = await repository.find_by_hash(document.collection_id, document.content_hash)
        assert found is not None
        assert found.status is DocumentStatus.READY
        loaded_chunks = await chunk_repository.get_by_ids([chunks[0].chunk_id])
        assert loaded_chunks[0].text == chunks[0].text

        embedding = Embedding(
            chunk_id=chunks[0].chunk_id,
            vector=(0.1, 0.2, 0.3),
            model="integration-model",
            dimensions=3,
        )
        await vectors.upsert(chunks, [embedding])
        points, _ = await qdrant.scroll(collection_name, limit=10, with_payload=True)
        assert points[0].payload is not None
        assert points[0].payload["document_id"] == str(document.document_id)
        dense_results = await QdrantDenseRetriever(qdrant, collection_name).retrieve(
            RetrievalRequest(query="sampling rate", collection_id=document.collection_id, top_k=1),
            (0.1, 0.2, 0.3),
        )
        assert dense_results[0].chunk.chunk_id == chunks[0].chunk_id

        await sparse.upsert(chunks)
        redis_key = f"{key_prefix}:collection:{document.collection_id}:chunks"
        assert await redis.hlen(redis_key) == 1
        sparse_results = await RedisBM25Retriever(redis, key_prefix=key_prefix).retrieve(
            RetrievalRequest(query="sampled 100 Hz", collection_id=document.collection_id, top_k=1)
        )
        assert sparse_results[0].chunk.chunk_id == chunks[0].chunk_id

        await sparse.delete(chunks)
        await vectors.delete([chunks[0].chunk_id])
        await repository.delete(document.document_id)
        async with sessions() as session:
            assert (
                await session.scalar(
                    select(DocumentRecord.id).where(DocumentRecord.id == document.document_id)
                )
                is None
            )
    finally:
        with suppress(Exception):
            await qdrant.delete_collection(collection_name)
        await qdrant.close()
        with suppress(Exception):
            await redis.delete(
                f"{key_prefix}:collection:{document.collection_id}:chunks",
                f"{key_prefix}:document:{document.document_id}:chunks",
            )
        await redis.aclose()
        with suppress(Exception):
            async with sessions() as session, session.begin():
                await session.execute(
                    delete(CollectionRecord).where(CollectionRecord.id == document.collection_id)
                )
        await engine.dispose()
