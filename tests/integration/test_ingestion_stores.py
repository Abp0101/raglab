import asyncio
from collections.abc import Sequence
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest
from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis
from sqlalchemy import delete, select, update

from raglab.core.config import Settings
from raglab.core.pagination import CursorKind, encode_cursor
from raglab.core.schemas import (
    Chunk,
    CollectionCreate,
    Document,
    DocumentInput,
    DocumentMetadata,
    DocumentStatus,
    Embedding,
    IngestionJobStatus,
    IngestionResult,
    RetrievalRequest,
)
from raglab.database.models import CollectionRecord, DocumentRecord, IngestionJobRecord
from raglab.database.repositories import (
    SQLAlchemyCatalogRepository,
    SQLAlchemyChunkRepository,
    SQLAlchemyDocumentRepository,
    SQLAlchemyIngestionJobRepository,
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
        assert created.collection_id in {item.collection_id for item in listed.items}
    finally:
        if collection_id is not None:
            with suppress(Exception):
                async with sessions() as session, session.begin():
                    await session.execute(
                        delete(CollectionRecord).where(CollectionRecord.id == collection_id)
                    )
        await engine.dispose()


@pytest.mark.asyncio
async def test_postgres_keyset_pages_use_uuid_tiebreakers() -> None:
    settings = Settings(_env_file=None)
    engine = create_engine(settings.postgres_dsn)
    sessions = create_session_factory(engine)
    catalog = SQLAlchemyCatalogRepository(sessions)
    jobs = SQLAlchemyIngestionJobRepository(sessions)
    collection_ids = []
    anchor = datetime(2099, 1, 1, tzinfo=UTC)

    try:
        collections = [
            await catalog.create_collection(CollectionCreate(name=f"Page {index}"))
            for index in range(3)
        ]
        collection_ids = [item.collection_id for item in collections]
        primary_id = collection_ids[0]
        document_ids = [uuid4() for _ in range(3)]
        now = datetime.now(UTC)
        async with sessions() as session, session.begin():
            await session.execute(
                update(CollectionRecord)
                .where(CollectionRecord.id.in_(collection_ids))
                .values(created_at=anchor, updated_at=anchor)
            )
            session.add_all(
                DocumentRecord(
                    id=document_id,
                    collection_id=primary_id,
                    file_name=f"{index}.pdf",
                    display_title=f"Document {index}",
                    authors=[],
                    source_url=None,
                    uploaded_at=anchor,
                    publication_date=None,
                    file_type="application/pdf",
                    content_hash=f"{index + 1:064x}",
                    page_count=1,
                    status=DocumentStatus.READY.value,
                    created_at=now,
                    updated_at=now,
                )
                for index, document_id in enumerate(document_ids)
            )
        created_jobs = [
            await jobs.create(
                DocumentInput(
                    file_name=f"job-{index}.pdf",
                    content=f"%PDF-job-{index}".encode(),
                    collection_id=primary_id,
                )
            )
            for index in range(3)
        ]
        job_ids = [job.job_id for job in created_jobs]
        async with sessions() as session, session.begin():
            await session.execute(
                update(IngestionJobRecord)
                .where(IngestionJobRecord.id.in_(job_ids))
                .values(created_at=anchor, updated_at=anchor)
            )

        collection_start = encode_cursor(
            kind=CursorKind.COLLECTIONS,
            scope=None,
            ordered_at=anchor,
            item_id=UUID(int=0),
        )
        collection_first = await catalog.list_collections(limit=2, cursor=collection_start)
        collection_second = await catalog.list_collections(
            limit=2,
            cursor=collection_first.next_cursor,
        )
        document_first = await catalog.list_documents(primary_id, limit=2)
        document_second = await catalog.list_documents(
            primary_id,
            limit=2,
            cursor=document_first.next_cursor,
        )
        job_first = await jobs.list_for_collection(primary_id, limit=2)
        job_second = await jobs.list_for_collection(
            primary_id,
            limit=2,
            cursor=job_first.next_cursor,
        )

        assert [
            item.collection_id for item in (*collection_first.items, *collection_second.items)
        ] == sorted(collection_ids)
        assert [
            item.document_id for item in (*document_first.items, *document_second.items)
        ] == sorted(document_ids)
        assert [item.job_id for item in (*job_first.items, *job_second.items)] == sorted(job_ids)
        assert collection_second.next_cursor is None
        assert document_second.next_cursor is None
        assert job_second.next_cursor is None
    finally:
        if collection_ids:
            with suppress(Exception):
                async with sessions() as session, session.begin():
                    await session.execute(
                        delete(CollectionRecord).where(CollectionRecord.id.in_(collection_ids))
                    )
        await engine.dispose()


@pytest.mark.asyncio
async def test_persistent_ingestion_job_clears_upload_after_completion() -> None:
    settings = Settings(_env_file=None)
    engine = create_engine(settings.postgres_dsn)
    sessions = create_session_factory(engine)
    catalog = SQLAlchemyCatalogRepository(sessions)
    jobs = SQLAlchemyIngestionJobRepository(sessions)
    collection_id = None

    try:
        collection = await catalog.create_collection(CollectionCreate(name="Job integration"))
        collection_id = collection.collection_id
        queued = await jobs.create(
            DocumentInput(
                file_name="queued.pdf",
                content=b"%PDF-integration",
                collection_id=collection.collection_id,
            )
        )

        owner_id = uuid4()
        claimed = await jobs.claim_next(owner_id, timedelta(seconds=30))
        assert claimed is not None
        assert claimed.job_id == queued.job_id
        assert (await jobs.get(queued.job_id)).status is IngestionJobStatus.PROCESSING
        assert (await jobs.get(queued.job_id)).attempt_count == 1

        result = IngestionResult(
            document_id=uuid4(),
            collection_id=collection.collection_id,
            page_count=1,
            chunk_count=1,
            duration_ms=1,
            parser="integration",
            chunking_strategy="integration",
            embedding_model="local",
        )
        assert await jobs.complete(queued.job_id, owner_id, result) is True
        completed = await jobs.get(queued.job_id)

        assert completed.status is IngestionJobStatus.COMPLETED
        assert completed.result == result
        async with sessions() as session:
            record = await session.get(IngestionJobRecord, queued.job_id)
            assert record is not None
            assert record.content is None
            assert record.lease_owner is None
            assert record.lease_expires_at is None
    finally:
        if collection_id is not None:
            with suppress(Exception):
                async with sessions() as session, session.begin():
                    await session.execute(
                        delete(CollectionRecord).where(CollectionRecord.id == collection_id)
                    )
        await engine.dispose()


@pytest.mark.asyncio
async def test_expired_postgres_lease_is_reclaimed_once() -> None:
    settings = Settings(_env_file=None)
    engine = create_engine(settings.postgres_dsn)
    sessions = create_session_factory(engine)
    catalog = SQLAlchemyCatalogRepository(sessions)
    jobs = SQLAlchemyIngestionJobRepository(sessions)
    collection_id = None

    try:
        collection = await catalog.create_collection(CollectionCreate(name="Lease integration"))
        collection_id = collection.collection_id
        queued = await jobs.create(
            DocumentInput(
                file_name="leased.pdf",
                content=b"%PDF-lease",
                collection_id=collection.collection_id,
            )
        )
        first_owner = uuid4()
        second_owner = uuid4()
        first = await jobs.claim_next(first_owner, timedelta(seconds=30))
        assert first is not None
        async with sessions() as session, session.begin():
            await session.execute(
                update(IngestionJobRecord)
                .where(IngestionJobRecord.id == queued.job_id)
                .values(lease_expires_at=datetime.now(UTC) - timedelta(seconds=1))
            )

        second = await jobs.claim_next(second_owner, timedelta(seconds=30))

        assert second is not None
        assert second.job_id == queued.job_id
        assert second.attempt_count == 2
        result = IngestionResult(
            document_id=uuid4(),
            collection_id=collection.collection_id,
            page_count=1,
            chunk_count=1,
            duration_ms=1,
            parser="integration",
            chunking_strategy="integration",
            embedding_model="local",
        )
        assert await jobs.complete(queued.job_id, first_owner, result) is False
        assert await jobs.complete(queued.job_id, second_owner, result) is True
    finally:
        if collection_id is not None:
            with suppress(Exception):
                async with sessions() as session, session.begin():
                    await session.execute(
                        delete(CollectionRecord).where(CollectionRecord.id == collection_id)
                    )
        await engine.dispose()


@pytest.mark.asyncio
async def test_concurrent_postgres_workers_claim_one_job_once() -> None:
    settings = Settings(_env_file=None)
    engine = create_engine(settings.postgres_dsn)
    sessions = create_session_factory(engine)
    catalog = SQLAlchemyCatalogRepository(sessions)
    jobs = SQLAlchemyIngestionJobRepository(sessions)
    collection_id = None

    try:
        collection = await catalog.create_collection(CollectionCreate(name="Claim integration"))
        collection_id = collection.collection_id
        queued = await jobs.create(
            DocumentInput(
                file_name="competing.pdf",
                content=b"%PDF-competing",
                collection_id=collection.collection_id,
            )
        )

        claims = await asyncio.gather(
            jobs.claim_next(uuid4(), timedelta(seconds=30)),
            jobs.claim_next(uuid4(), timedelta(seconds=30)),
        )

        claimed = [claim for claim in claims if claim is not None]
        assert len(claimed) == 1
        assert claimed[0].job_id == queued.job_id
        assert claimed[0].attempt_count == 1
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
