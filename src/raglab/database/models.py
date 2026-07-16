"""Relational models for collections, documents, and chunks."""

from datetime import date, datetime
from uuid import UUID

from sqlalchemy import JSON, Date, DateTime, ForeignKey, Index, Integer, LargeBinary, String, Text
from sqlalchemy import Uuid as SQLUuid
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column, relationship

from raglab.core.schemas import DocumentStatus
from raglab.database.base import Base


class TimestampMixin:
    """Application-managed UTC timestamps."""

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class CollectionRecord(TimestampMixin, Base):
    """Named logical collection used by all retrieval backends."""

    __tablename__ = "collections"

    id: Mapped[UUID] = mapped_column(SQLUuid(), primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str | None] = mapped_column(Text(), nullable=True)
    documents: Mapped[list["DocumentRecord"]] = relationship(
        back_populates="collection", cascade="all, delete-orphan"
    )


class DocumentRecord(TimestampMixin, Base):
    """Source metadata and ingestion lifecycle state."""

    __tablename__ = "documents"
    __table_args__ = (
        Index("uq_documents_collection_hash", "collection_id", "content_hash", unique=True),
        Index("ix_documents_collection_status", "collection_id", "status"),
    )

    id: Mapped[UUID] = mapped_column(SQLUuid(), primary_key=True)
    collection_id: Mapped[UUID] = mapped_column(
        SQLUuid(), ForeignKey("collections.id", ondelete="CASCADE"), index=True
    )
    file_name: Mapped[str] = mapped_column(String(255))
    display_title: Mapped[str] = mapped_column(String(500))
    authors: Mapped[list[str]] = mapped_column(ARRAY(String()), default=list)
    source_url: Mapped[str | None] = mapped_column(Text(), nullable=True)
    uploaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    publication_date: Mapped[date | None] = mapped_column(Date(), nullable=True)
    file_type: Mapped[str] = mapped_column(String(100))
    content_hash: Mapped[str] = mapped_column(String(64))
    page_count: Mapped[int | None] = mapped_column(Integer(), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default=DocumentStatus.PENDING.value)
    collection: Mapped[CollectionRecord] = relationship(back_populates="documents")
    chunks: Mapped[list["ChunkRecord"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class ChunkRecord(TimestampMixin, Base):
    """Normalized retrievable text and provenance."""

    __tablename__ = "chunks"
    __table_args__ = (
        Index("uq_chunks_document_index", "document_id", "chunk_index", unique=True),
        Index("ix_chunks_collection_document", "collection_id", "document_id"),
    )

    id: Mapped[UUID] = mapped_column(SQLUuid(), primary_key=True)
    document_id: Mapped[UUID] = mapped_column(
        SQLUuid(), ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    collection_id: Mapped[UUID] = mapped_column(
        SQLUuid(), ForeignKey("collections.id", ondelete="CASCADE"), index=True
    )
    text: Mapped[str] = mapped_column(Text())
    page_number: Mapped[int | None] = mapped_column(Integer(), nullable=True)
    section_heading: Mapped[str | None] = mapped_column(String(500), nullable=True)
    chunk_index: Mapped[int] = mapped_column(Integer())
    parent_chunk_id: Mapped[UUID | None] = mapped_column(SQLUuid(), nullable=True)
    content_hash: Mapped[str] = mapped_column(String(64))
    token_count: Mapped[int | None] = mapped_column(Integer(), nullable=True)
    text_start: Mapped[int | None] = mapped_column(Integer(), nullable=True)
    text_end: Mapped[int | None] = mapped_column(Integer(), nullable=True)
    text_sha256: Mapped[bytes] = mapped_column(LargeBinary(32))
    document: Mapped[DocumentRecord] = relationship(back_populates="chunks")


class IngestionJobRecord(TimestampMixin, Base):
    """Durable queued upload, cleared of source bytes after terminal completion."""

    __tablename__ = "ingestion_jobs"
    __table_args__ = (Index("ix_ingestion_jobs_collection_status", "collection_id", "status"),)

    id: Mapped[UUID] = mapped_column(SQLUuid(), primary_key=True)
    collection_id: Mapped[UUID] = mapped_column(
        SQLUuid(), ForeignKey("collections.id", ondelete="CASCADE"), index=True
    )
    file_name: Mapped[str] = mapped_column(String(255))
    display_title: Mapped[str | None] = mapped_column(String(500), nullable=True)
    source_url: Mapped[str | None] = mapped_column(Text(), nullable=True)
    content: Mapped[bytes | None] = mapped_column(LargeBinary(), nullable=True)
    status: Mapped[str] = mapped_column(String(32))
    result: Mapped[dict[str, object] | None] = mapped_column(JSON(), nullable=True)
    error_type: Mapped[str | None] = mapped_column(String(100), nullable=True)
    error_message: Mapped[str | None] = mapped_column(String(1000), nullable=True)
