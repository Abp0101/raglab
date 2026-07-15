"""Boundary-aware recursive character chunking with source offsets."""

from collections.abc import Sequence
from uuid import uuid5

from raglab.core.schemas import (
    Chunk,
    ChunkingConfig,
    DocumentMetadata,
    DocumentPage,
    ParsedDocument,
    TextSpan,
)


class RecursiveCharacterChunker:
    """Split page text near natural boundaries with deterministic chunk IDs."""

    name = "recursive-character"
    _separators = ("\n\n", "\n", ". ", " ")

    def chunk(self, document: ParsedDocument, config: ChunkingConfig) -> Sequence[Chunk]:
        chunks: list[Chunk] = []
        for page in document.pages:
            for start, end in self._page_spans(page.text, config):
                text = page.text[start:end]
                leading_whitespace = len(text) - len(text.lstrip())
                trailing_end = len(text.rstrip())
                normalized_start = start + leading_whitespace
                normalized_end = start + trailing_end
                if normalized_end <= normalized_start:
                    continue
                chunks.append(
                    self._build_chunk(
                        document,
                        page,
                        normalized_start,
                        normalized_end,
                        len(chunks),
                    )
                )
        return chunks

    def _page_spans(self, text: str, config: ChunkingConfig) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        start = 0
        while start < len(text):
            maximum_end = min(start + config.chunk_size, len(text))
            end = maximum_end
            if maximum_end < len(text):
                minimum_boundary = start + max(config.chunk_size // 2, 1)
                for separator in self._separators:
                    boundary = text.rfind(separator, minimum_boundary, maximum_end)
                    if boundary >= minimum_boundary:
                        end = boundary + (1 if separator == ". " else len(separator))
                        break
            spans.append((start, end))
            if end >= len(text):
                break
            next_start = max(end - config.chunk_overlap, start + 1)
            start = next_start
        return spans

    @staticmethod
    def _build_chunk(
        document: ParsedDocument,
        page: DocumentPage,
        start: int,
        end: int,
        chunk_index: int,
    ) -> Chunk:
        source = document.document
        heading = next(
            (
                candidate.text
                for candidate in reversed(page.section_headings)
                if candidate.start <= start
            ),
            None,
        )
        chunk_id = uuid5(source.document_id, f"{page.page_number}:{start}:{end}")
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
                section_heading=heading,
                chunk_index=chunk_index,
                content_hash=source.content_hash,
            ),
            text_span=TextSpan(start=start, end=end),
        )
