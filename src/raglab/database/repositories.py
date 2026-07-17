"""SQLAlchemy persistence adapters for ingestion."""

import hashlib
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from sqlalchemy import and_, delete, func, or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.sql.elements import ColumnElement

from raglab.core.exceptions import (
    CollectionNotFoundError,
    DocumentNotFoundError,
    DuplicateDocumentError,
    IngestionJobNotFoundError,
)
from raglab.core.schemas import (
    Chunk,
    Collection,
    CollectionCreate,
    Document,
    DocumentInput,
    DocumentMetadata,
    DocumentStatus,
    IngestionJob,
    IngestionJobClaim,
    IngestionJobError,
    IngestionJobStatus,
    IngestionResult,
    TextSpan,
)
from raglab.database.models import (
    ChunkRecord,
    CollectionRecord,
    DocumentRecord,
    IngestionJobRecord,
)


class SQLAlchemyCatalogRepository:
    """Manage collection records and read document metadata for the API."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def create_collection(self, request: CollectionCreate) -> Collection:
        now = datetime.now(UTC)
        record = CollectionRecord(
            id=uuid4(),
            name=request.name,
            description=request.description,
            created_at=now,
            updated_at=now,
        )
        async with self._sessions() as session, session.begin():
            session.add(record)
            await session.flush()
        return _to_collection(record, document_count=0)

    async def list_collections(self) -> Sequence[Collection]:
        async with self._sessions() as session:
            rows = (
                await session.execute(
                    select(CollectionRecord, func.count(DocumentRecord.id))
                    .outerjoin(DocumentRecord)
                    .group_by(CollectionRecord.id)
                    .order_by(CollectionRecord.created_at, CollectionRecord.id)
                )
            ).all()
        return tuple(_to_collection(record, document_count=count) for record, count in rows)

    async def get_collection(self, collection_id: UUID) -> Collection:
        async with self._sessions() as session:
            row = (
                await session.execute(
                    select(CollectionRecord, func.count(DocumentRecord.id))
                    .outerjoin(DocumentRecord)
                    .where(CollectionRecord.id == collection_id)
                    .group_by(CollectionRecord.id)
                )
            ).one_or_none()
        if row is None:
            raise CollectionNotFoundError(f"collection {collection_id} does not exist")
        return _to_collection(row[0], document_count=row[1])

    async def list_documents(self, collection_id: UUID) -> Sequence[Document]:
        await self.get_collection(collection_id)
        async with self._sessions() as session:
            records = (
                await session.scalars(
                    select(DocumentRecord)
                    .where(DocumentRecord.collection_id == collection_id)
                    .order_by(DocumentRecord.uploaded_at, DocumentRecord.id)
                )
            ).all()
        return tuple(_to_document(record) for record in records)

    async def get_document(self, document_id: UUID) -> Document:
        async with self._sessions() as session:
            record = await session.get(DocumentRecord, document_id)
        if record is None:
            raise DocumentNotFoundError(f"document {document_id} does not exist")
        return _to_document(record)


class SQLAlchemyIngestionJobRepository:
    """Persist upload bytes until a background ingestion job reaches a terminal state."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def create(self, document: DocumentInput) -> IngestionJob:
        now = datetime.now(UTC)
        record = IngestionJobRecord(
            id=uuid4(),
            collection_id=document.collection_id,
            file_name=document.file_name,
            display_title=document.display_title,
            source_url=str(document.source_url) if document.source_url else None,
            content=document.content,
            status=IngestionJobStatus.QUEUED.value,
            result=None,
            error_type=None,
            error_message=None,
            attempt_count=0,
            lease_owner=None,
            lease_expires_at=None,
            created_at=now,
            updated_at=now,
        )
        async with self._sessions() as session, session.begin():
            session.add(record)
            await session.flush()
        return _to_ingestion_job(record)

    async def get(self, job_id: UUID) -> IngestionJob:
        async with self._sessions() as session:
            record = await session.get(IngestionJobRecord, job_id)
        if record is None:
            raise IngestionJobNotFoundError(f"ingestion job {job_id} does not exist")
        return _to_ingestion_job(record)

    async def claim_next(
        self,
        owner_id: UUID,
        lease_duration: timedelta,
    ) -> IngestionJobClaim | None:
        """Atomically claim the oldest queued or expired job for one worker."""
        async with self._sessions() as session, session.begin():
            now = await _database_now(session)
            record = await session.scalar(
                select(IngestionJobRecord)
                .where(
                    IngestionJobRecord.content.is_not(None),
                    or_(
                        IngestionJobRecord.status == IngestionJobStatus.QUEUED.value,
                        and_(
                            IngestionJobRecord.status == IngestionJobStatus.PROCESSING.value,
                            or_(
                                IngestionJobRecord.lease_expires_at.is_(None),
                                IngestionJobRecord.lease_expires_at <= now,
                            ),
                        ),
                    ),
                )
                .order_by(IngestionJobRecord.created_at, IngestionJobRecord.id)
                .limit(1)
                .with_for_update(skip_locked=True)
            )
            if record is None or record.content is None:
                return None
            record.status = IngestionJobStatus.PROCESSING.value
            record.attempt_count += 1
            record.lease_owner = owner_id
            record.lease_expires_at = now + lease_duration
            record.updated_at = now
            return IngestionJobClaim(
                job_id=record.id,
                document=DocumentInput(
                    file_name=record.file_name,
                    content=record.content,
                    collection_id=record.collection_id,
                    display_title=record.display_title,
                    source_url=record.source_url,
                ),
                attempt_count=record.attempt_count,
                lease_expires_at=record.lease_expires_at,
            )

    async def renew(
        self,
        job_id: UUID,
        owner_id: UUID,
        lease_duration: timedelta,
    ) -> bool:
        """Extend a live lease only while it is still owned and unexpired."""
        async with self._sessions() as session, session.begin():
            now = await _database_now(session)
            updated_id = await session.scalar(
                update(IngestionJobRecord)
                .where(*_active_lease_conditions(job_id, owner_id, now))
                .values(lease_expires_at=now + lease_duration, updated_at=now)
                .returning(IngestionJobRecord.id)
            )
        return updated_id is not None

    async def complete(
        self,
        job_id: UUID,
        owner_id: UUID,
        result: IngestionResult,
    ) -> bool:
        return await self._finish(
            job_id,
            owner_id,
            status=IngestionJobStatus.COMPLETED,
            result=result.model_dump(mode="json"),
        )

    async def fail(
        self,
        job_id: UUID,
        owner_id: UUID,
        error_type: str,
        message: str,
    ) -> bool:
        return await self._finish(
            job_id,
            owner_id,
            status=IngestionJobStatus.FAILED,
            error_type=error_type,
            error_message=message,
        )

    async def release(self, job_id: UUID, owner_id: UUID) -> bool:
        """Return owned work to the queue during graceful cancellation."""
        async with self._sessions() as session, session.begin():
            now = await _database_now(session)
            updated_id = await session.scalar(
                update(IngestionJobRecord)
                .where(
                    IngestionJobRecord.id == job_id,
                    IngestionJobRecord.status == IngestionJobStatus.PROCESSING.value,
                    IngestionJobRecord.lease_owner == owner_id,
                )
                .values(
                    status=IngestionJobStatus.QUEUED.value,
                    lease_owner=None,
                    lease_expires_at=None,
                    updated_at=now,
                )
                .returning(IngestionJobRecord.id)
            )
        return updated_id is not None

    async def _finish(
        self,
        job_id: UUID,
        owner_id: UUID,
        *,
        status: IngestionJobStatus,
        result: dict[str, object] | None = None,
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> bool:
        async with self._sessions() as session, session.begin():
            now = await _database_now(session)
            updated_id = await session.scalar(
                update(IngestionJobRecord)
                .where(*_active_lease_conditions(job_id, owner_id, now))
                .values(
                    status=status.value,
                    result=result,
                    error_type=error_type,
                    error_message=error_message,
                    content=None,
                    lease_owner=None,
                    lease_expires_at=None,
                    updated_at=now,
                )
                .returning(IngestionJobRecord.id)
            )
            if updated_id is None:
                exists = await session.scalar(
                    select(IngestionJobRecord.id).where(IngestionJobRecord.id == job_id)
                )
                if exists is None:
                    raise IngestionJobNotFoundError(f"ingestion job {job_id} does not exist")
                return False
        return True


class SQLAlchemyDocumentRepository:
    """Store one document and all chunks in a relational transaction."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def find_by_hash(self, collection_id: UUID, content_hash: str) -> Document | None:
        async with self._sessions() as session:
            record = await session.scalar(
                select(DocumentRecord).where(
                    DocumentRecord.collection_id == collection_id,
                    DocumentRecord.content_hash == content_hash,
                )
            )
            return _to_document(record) if record is not None else None

    async def save(self, document: Document, chunks: Sequence[Chunk]) -> None:
        now = datetime.now(UTC)
        async with self._sessions() as session, session.begin():
            exists = await session.scalar(
                select(CollectionRecord.id).where(CollectionRecord.id == document.collection_id)
            )
            if exists is None:
                raise CollectionNotFoundError(f"collection {document.collection_id} does not exist")
            session.add(_document_record(document, now))
            session.add_all(_chunk_record(chunk, now) for chunk in chunks)
            try:
                await session.flush()
            except IntegrityError as error:
                raise DuplicateDocumentError(
                    "document content already exists in collection"
                ) from error

    async def set_status(self, document_id: UUID, status: DocumentStatus) -> None:
        async with self._sessions() as session, session.begin():
            await session.execute(
                update(DocumentRecord)
                .where(DocumentRecord.id == document_id)
                .values(status=status.value, updated_at=datetime.now(UTC))
            )

    async def delete(self, document_id: UUID) -> None:
        async with self._sessions() as session, session.begin():
            await session.execute(delete(DocumentRecord).where(DocumentRecord.id == document_id))


class SQLAlchemyChunkRepository:
    """Load chunks with document metadata for context expansion."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def get_by_ids(self, chunk_ids: Sequence[UUID]) -> Sequence[Chunk]:
        if not chunk_ids:
            return ()
        async with self._sessions() as session:
            rows = (
                await session.execute(
                    select(ChunkRecord, DocumentRecord)
                    .join(DocumentRecord, DocumentRecord.id == ChunkRecord.document_id)
                    .where(ChunkRecord.id.in_(chunk_ids))
                )
            ).all()
        chunks = {
            chunk_record.id: _to_chunk(chunk_record, document) for chunk_record, document in rows
        }
        return tuple(chunks[chunk_id] for chunk_id in chunk_ids if chunk_id in chunks)


def _document_record(document: Document, now: datetime) -> DocumentRecord:
    return DocumentRecord(
        id=document.document_id,
        collection_id=document.collection_id,
        file_name=document.file_name,
        display_title=document.display_title,
        authors=list(document.authors),
        source_url=str(document.source_url) if document.source_url else None,
        uploaded_at=document.uploaded_at,
        publication_date=document.publication_date,
        file_type=document.file_type,
        content_hash=document.content_hash,
        page_count=document.page_count,
        status=document.status.value,
        created_at=now,
        updated_at=now,
    )


def _to_collection(record: CollectionRecord, *, document_count: int) -> Collection:
    return Collection(
        collection_id=record.id,
        name=record.name,
        description=record.description,
        created_at=record.created_at,
        updated_at=record.updated_at,
        document_count=document_count,
    )


def _to_ingestion_job(record: IngestionJobRecord) -> IngestionJob:
    return IngestionJob(
        job_id=record.id,
        collection_id=record.collection_id,
        file_name=record.file_name,
        status=IngestionJobStatus(record.status),
        created_at=record.created_at,
        updated_at=record.updated_at,
        attempt_count=record.attempt_count,
        lease_expires_at=record.lease_expires_at,
        result=(IngestionResult.model_validate(record.result) if record.result else None),
        error=(
            IngestionJobError(type=record.error_type, message=record.error_message)
            if record.error_type and record.error_message
            else None
        ),
    )


def _active_lease_conditions(
    job_id: UUID,
    owner_id: UUID,
    now: datetime,
) -> tuple[ColumnElement[bool], ...]:
    """Prevent expired or superseded workers from mutating terminal state."""
    return (
        IngestionJobRecord.id == job_id,
        IngestionJobRecord.status == IngestionJobStatus.PROCESSING.value,
        IngestionJobRecord.lease_owner == owner_id,
        IngestionJobRecord.lease_expires_at.is_not(None),
        IngestionJobRecord.lease_expires_at > now,
    )


async def _database_now(session: AsyncSession) -> datetime:
    """Use one authoritative clock for lease acquisition and expiry decisions."""
    value = await session.scalar(select(func.now()))
    if not isinstance(value, datetime):
        raise RuntimeError("database did not return a timestamp")
    return value


def _chunk_record(chunk: Chunk, now: datetime) -> ChunkRecord:
    span = chunk.text_span
    return ChunkRecord(
        id=chunk.chunk_id,
        document_id=chunk.metadata.document_id,
        collection_id=chunk.metadata.collection_id,
        text=chunk.text,
        page_number=chunk.metadata.page_number,
        section_heading=chunk.metadata.section_heading,
        chunk_index=chunk.metadata.chunk_index,
        parent_chunk_id=chunk.metadata.parent_chunk_id,
        content_hash=chunk.metadata.content_hash,
        token_count=chunk.token_count,
        text_start=span.start if span else None,
        text_end=span.end if span else None,
        text_sha256=hashlib.sha256(chunk.text.encode()).digest(),
        created_at=now,
        updated_at=now,
    )


def _to_document(record: DocumentRecord) -> Document:
    return Document(
        document_id=record.id,
        collection_id=record.collection_id,
        file_name=record.file_name,
        display_title=record.display_title,
        authors=tuple(record.authors),
        source_url=record.source_url,
        uploaded_at=record.uploaded_at,
        publication_date=record.publication_date,
        file_type=record.file_type,
        content_hash=record.content_hash,
        page_count=record.page_count,
        status=DocumentStatus(record.status),
    )


def _to_chunk(record: ChunkRecord, document: DocumentRecord) -> Chunk:
    return Chunk(
        chunk_id=record.id,
        text=record.text,
        metadata=DocumentMetadata(
            document_id=document.id,
            collection_id=document.collection_id,
            file_name=document.file_name,
            display_title=document.display_title,
            authors=tuple(document.authors),
            source_url=document.source_url,
            uploaded_at=document.uploaded_at,
            publication_date=document.publication_date,
            file_type=document.file_type,
            page_number=record.page_number,
            section_heading=record.section_heading,
            chunk_index=record.chunk_index,
            parent_chunk_id=record.parent_chunk_id,
            content_hash=record.content_hash,
        ),
        token_count=record.token_count,
        text_span=(
            TextSpan(start=record.text_start, end=record.text_end)
            if record.text_start is not None and record.text_end is not None
            else None
        ),
    )
