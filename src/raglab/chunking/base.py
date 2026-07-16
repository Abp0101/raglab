"""Shared deterministic chunk construction and span utilities."""

from uuid import UUID, uuid5

from raglab.core.schemas import (
    Chunk,
    DocumentMetadata,
    DocumentPage,
    ParsedDocument,
    TextSpan,
)


def trim_span(text: str, start: int, end: int) -> tuple[int, int] | None:
    """Trim boundary whitespace while retaining source-relative offsets."""
    candidate = text[start:end]
    normalized_start = start + len(candidate) - len(candidate.lstrip())
    normalized_end = start + len(candidate.rstrip())
    if normalized_end <= normalized_start:
        return None
    return normalized_start, normalized_end


def active_heading(page: DocumentPage, start: int) -> str | None:
    """Return the closest preceding detected section heading."""
    return next(
        (heading.text for heading in reversed(page.section_headings) if heading.start <= start),
        None,
    )


def build_chunk(
    document: ParsedDocument,
    page: DocumentPage,
    start: int,
    end: int,
    chunk_index: int,
    *,
    namespace: str,
    parent_chunk_id: UUID | None = None,
    token_count: int | None = None,
) -> Chunk:
    """Build a chunk with deterministic identity and complete provenance."""
    source = document.document
    chunk_id = uuid5(
        source.document_id,
        f"{namespace}:{page.page_number}:{start}:{end}:{parent_chunk_id or ''}",
    )
    return Chunk(
        chunk_id=chunk_id,
        text=page.text[start:end],
        metadata=DocumentMetadata(
            document_id=source.document_id,
            collection_id=source.collection_id,
            file_name=source.file_name,
            display_title=source.display_title,
            authors=source.authors,
            source_url=source.source_url,
            uploaded_at=source.uploaded_at,
            publication_date=source.publication_date,
            file_type=source.file_type,
            page_number=page.page_number,
            section_heading=active_heading(page, start),
            chunk_index=chunk_index,
            parent_chunk_id=parent_chunk_id,
            content_hash=source.content_hash,
        ),
        token_count=token_count,
        text_span=TextSpan(start=start, end=end),
    )
