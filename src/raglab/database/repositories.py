"""SQLAlchemy persistence adapters for ingestion."""

import hashlib
from collections.abc import Sequence
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import delete, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from raglab.core.exceptions import CollectionNotFoundError, DuplicateDocumentError
from raglab.core.schemas import Chunk, Document, DocumentStatus
from raglab.database.models import ChunkRecord, CollectionRecord, DocumentRecord


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
