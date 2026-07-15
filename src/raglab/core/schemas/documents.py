"""Document, chunk, embedding, and ingestion boundary models."""

from datetime import date, datetime
from enum import StrEnum
from typing import Self
from uuid import UUID

from pydantic import Field, HttpUrl, model_validator

from raglab.core.schemas.common import RAGLabModel


class DocumentStatus(StrEnum):
    """Lifecycle state of an uploaded document."""

    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class DocumentInput(RAGLabModel):
    """Untrusted document bytes accepted by an ingestion pipeline."""

    file_name: str = Field(min_length=1, max_length=255)
    content: bytes = Field(min_length=1)
    collection_id: UUID
    display_title: str | None = Field(default=None, max_length=500)
    source_url: HttpUrl | None = None


class Document(RAGLabModel):
    """Collection-level document record."""

    document_id: UUID
    collection_id: UUID
    file_name: str = Field(min_length=1, max_length=255)
    display_title: str = Field(min_length=1, max_length=500)
    authors: tuple[str, ...] = ()
    source_url: HttpUrl | None = None
    uploaded_at: datetime
    publication_date: date | None = None
    file_type: str = Field(min_length=1, max_length=100)
    content_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    page_count: int | None = Field(default=None, ge=0)
    status: DocumentStatus = DocumentStatus.PENDING


class DocumentMetadata(RAGLabModel):
    """Metadata copied onto each chunk so retrieval results remain traceable."""

    document_id: UUID
    collection_id: UUID
    file_name: str = Field(min_length=1, max_length=255)
    display_title: str = Field(min_length=1, max_length=500)
    authors: tuple[str, ...] = ()
    source_url: HttpUrl | None = None
    uploaded_at: datetime
    publication_date: date | None = None
    file_type: str = Field(min_length=1, max_length=100)
    page_number: int | None = Field(default=None, ge=1)
    section_heading: str | None = Field(default=None, max_length=500)
    chunk_index: int = Field(ge=0)
    parent_chunk_id: UUID | None = None
    content_hash: str = Field(pattern=r"^[a-f0-9]{64}$")


class TextSpan(RAGLabModel):
    """Half-open character offsets into the parser's normalized page text."""

    start: int = Field(ge=0)
    end: int = Field(gt=0)

    @model_validator(mode="after")
    def end_follows_start(self) -> Self:
        if self.end <= self.start:
            raise ValueError("text span end must be greater than start")
        return self


class SectionHeading(RAGLabModel):
    """Detected section heading and its offset in normalized page text."""

    text: str = Field(min_length=1, max_length=500)
    start: int = Field(ge=0)


class DocumentPage(RAGLabModel):
    """Normalized text extracted from one source page."""

    page_number: int = Field(ge=1)
    text: str = Field(min_length=1)
    section_headings: tuple[SectionHeading, ...] = ()


class ParsedDocument(RAGLabModel):
    """Parser output before chunking and embedding."""

    document: Document
    pages: tuple[DocumentPage, ...] = Field(min_length=1)
    parser_name: str = Field(min_length=1, max_length=100)
    warnings: tuple[str, ...] = ()


class Chunk(RAGLabModel):
    """Retrievable text unit with source provenance."""

    chunk_id: UUID
    text: str = Field(min_length=1)
    metadata: DocumentMetadata
    token_count: int | None = Field(default=None, ge=1)
    text_span: TextSpan | None = None


class Embedding(RAGLabModel):
    """Embedding vector associated with a chunk and provider model."""

    chunk_id: UUID
    vector: tuple[float, ...] = Field(min_length=1)
    model: str = Field(min_length=1, max_length=255)
    dimensions: int = Field(gt=0)

    @model_validator(mode="after")
    def dimensions_match_vector(self) -> Self:
        if len(self.vector) != self.dimensions:
            raise ValueError("embedding dimensions must match vector length")
        return self


class ChunkingConfig(RAGLabModel):
    """Framework-neutral configuration for a chunking strategy."""

    strategy: str = Field(default="recursive", min_length=1, max_length=100)
    chunk_size: int = Field(default=512, ge=32, le=8192)
    chunk_overlap: int = Field(default=64, ge=0)

    @model_validator(mode="after")
    def overlap_is_smaller_than_chunk(self) -> Self:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk overlap must be smaller than chunk size")
        return self


class IngestionResult(RAGLabModel):
    """Standard report returned by every pipeline after ingestion."""

    document_id: UUID
    collection_id: UUID
    page_count: int = Field(ge=0)
    chunk_count: int = Field(ge=0)
    duration_ms: float = Field(ge=0)
    parser: str = Field(min_length=1)
    chunking_strategy: str = Field(min_length=1)
    embedding_model: str = Field(min_length=1)
    duplicate: bool = False
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
