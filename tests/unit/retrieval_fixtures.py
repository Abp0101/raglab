"""Test data builders shared by retrieval unit tests."""

from datetime import UTC, date, datetime
from uuid import UUID, uuid4

from raglab.core.schemas import Chunk, DocumentMetadata


def make_chunk(
    text: str,
    *,
    collection_id: UUID | None = None,
    document_id: UUID | None = None,
    chunk_id: UUID | None = None,
    parent_chunk_id: UUID | None = None,
    page_number: int = 1,
    section_heading: str | None = "METHODS",
    authors: tuple[str, ...] = ("Ada Engineer",),
    publication_date: date | None = date(2025, 1, 2),
    chunk_index: int = 0,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id or uuid4(),
        text=text,
        metadata=DocumentMetadata(
            document_id=document_id or uuid4(),
            collection_id=collection_id or uuid4(),
            file_name="study.pdf",
            display_title="Sensor Study",
            authors=authors,
            uploaded_at=datetime.now(UTC),
            publication_date=publication_date,
            file_type="application/pdf",
            page_number=page_number,
            section_heading=section_heading,
            chunk_index=chunk_index,
            parent_chunk_id=parent_chunk_id,
            content_hash="a" * 64,
        ),
    )
