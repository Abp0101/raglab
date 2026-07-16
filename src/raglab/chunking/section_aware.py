"""Section-preserving recursive chunking."""

from collections.abc import Sequence
from itertools import pairwise

from raglab.chunking.base import build_chunk, trim_span
from raglab.chunking.spans import recursive_spans
from raglab.chunking.tokenization import count_lexical_tokens
from raglab.core.schemas import Chunk, ChunkingConfig, DocumentPage, ParsedDocument


class SectionAwareChunker:
    """Keep detected section boundaries intact, splitting long sections recursively."""

    name = "section-aware"

    def chunk(self, document: ParsedDocument, config: ChunkingConfig) -> Sequence[Chunk]:
        chunks: list[Chunk] = []
        for page in document.pages:
            for section_start, section_end in _section_spans(page):
                section_text = page.text[section_start:section_end]
                for start, end in recursive_spans(
                    section_text,
                    chunk_size=config.chunk_size,
                    chunk_overlap=config.chunk_overlap,
                    offset=section_start,
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
                            token_count=count_lexical_tokens(
                                page.text[normalized_start:normalized_end]
                            ),
                        )
                    )
        return chunks


def _section_spans(page: DocumentPage) -> tuple[tuple[int, int], ...]:
    boundaries = sorted({0, *(heading.start for heading in page.section_headings), len(page.text)})
    return tuple((start, end) for start, end in pairwise(boundaries) if end > start)
