"""Stable chunk payload serialization shared by retrieval backends."""

from collections.abc import Mapping
from datetime import date, datetime
from typing import Any
from uuid import UUID

from raglab.core.schemas import Chunk, DocumentMetadata, TextSpan


def chunk_to_payload(chunk: Chunk) -> dict[str, object]:
    """Serialize a chunk into JSON-compatible retrieval metadata."""
    metadata = chunk.metadata
    span = chunk.text_span
    return {
        "chunk_id": str(chunk.chunk_id),
        "document_id": str(metadata.document_id),
        "collection_id": str(metadata.collection_id),
        "file_name": metadata.file_name,
        "display_title": metadata.display_title,
        "authors": list(metadata.authors),
        "source_url": str(metadata.source_url) if metadata.source_url else None,
        "uploaded_at": metadata.uploaded_at.isoformat(),
        "publication_date": (
            metadata.publication_date.isoformat() if metadata.publication_date else None
        ),
        "publication_ordinal": (
            metadata.publication_date.toordinal() if metadata.publication_date else None
        ),
        "file_type": metadata.file_type,
        "page_number": metadata.page_number,
        "section_heading": metadata.section_heading,
        "chunk_index": metadata.chunk_index,
        "parent_chunk_id": str(metadata.parent_chunk_id) if metadata.parent_chunk_id else None,
        "content_hash": metadata.content_hash,
        "token_count": chunk.token_count,
        "text_start": span.start if span else None,
        "text_end": span.end if span else None,
        "text": chunk.text,
    }


def payload_to_chunk(payload: Mapping[str, Any]) -> Chunk:
    """Rebuild the shared chunk model from a trusted index payload."""
    parent_id = payload.get("parent_chunk_id")
    publication_date = payload.get("publication_date")
    text_start = payload.get("text_start")
    text_end = payload.get("text_end")
    return Chunk(
        chunk_id=UUID(str(payload["chunk_id"])),
        text=str(payload["text"]),
        metadata=DocumentMetadata(
            document_id=UUID(str(payload["document_id"])),
            collection_id=UUID(str(payload["collection_id"])),
            file_name=str(payload["file_name"]),
            display_title=str(payload["display_title"]),
            authors=tuple(str(author) for author in payload.get("authors", [])),
            source_url=payload.get("source_url"),
            uploaded_at=datetime.fromisoformat(str(payload["uploaded_at"])),
            publication_date=(
                date.fromisoformat(str(publication_date)) if publication_date else None
            ),
            file_type=str(payload["file_type"]),
            page_number=_optional_int(payload.get("page_number")),
            section_heading=(
                str(payload["section_heading"]) if payload.get("section_heading") else None
            ),
            chunk_index=int(payload["chunk_index"]),
            parent_chunk_id=UUID(str(parent_id)) if parent_id else None,
            content_hash=str(payload["content_hash"]),
        ),
        token_count=_optional_int(payload.get("token_count")),
        text_span=(
            TextSpan(start=int(text_start), end=int(text_end))
            if text_start is not None and text_end is not None
            else None
        ),
    )


def _optional_int(value: object) -> int | None:
    return int(str(value)) if value is not None else None
