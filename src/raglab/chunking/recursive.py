"""Boundary-aware recursive character chunking."""

from collections.abc import Sequence

from raglab.chunking.base import build_chunk, trim_span
from raglab.chunking.spans import recursive_spans
from raglab.core.schemas import Chunk, ChunkingConfig, ParsedDocument


class RecursiveCharacterChunker:
    """Split page text near natural boundaries with deterministic chunk IDs."""

    name = "recursive-character"

    def chunk(self, document: ParsedDocument, config: ChunkingConfig) -> Sequence[Chunk]:
        chunks: list[Chunk] = []
        for page in document.pages:
            for start, end in recursive_spans(
                page.text,
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
            ):
                span = trim_span(page.text, start, end)
                if span is None:
                    continue
                normalized_start, normalized_end = span
                chunks.append(
                    build_chunk(
                        document,
                        page,
                        normalized_start,
                        normalized_end,
                        len(chunks),
                        namespace=self.name,
                    )
                )
        return chunks
