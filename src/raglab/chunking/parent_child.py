"""Hierarchical parent-child chunking for small-vector/large-context retrieval."""

from collections.abc import Sequence

from raglab.chunking.base import build_chunk, trim_span
from raglab.chunking.spans import recursive_spans
from raglab.chunking.tokenization import count_lexical_tokens
from raglab.core.schemas import Chunk, ChunkingConfig, ParsedDocument


class ParentChildChunker:
    """Emit large parents and smaller overlapping children linked by UUID."""

    name = "parent-child"

    def chunk(self, document: ParsedDocument, config: ChunkingConfig) -> Sequence[Chunk]:
        chunks: list[Chunk] = []
        for page in document.pages:
            parent_spans = recursive_spans(
                page.text,
                chunk_size=config.parent_chunk_size,
                chunk_overlap=config.parent_chunk_overlap,
            )
            for parent_start, parent_end in parent_spans:
                parent_span = trim_span(page.text, parent_start, parent_end)
                if parent_span is None:
                    continue
                normalized_parent_start, normalized_parent_end = parent_span
                parent = build_chunk(
                    document,
                    page,
                    normalized_parent_start,
                    normalized_parent_end,
                    len(chunks),
                    namespace=f"{self.name}:parent",
                    token_count=count_lexical_tokens(
                        page.text[normalized_parent_start:normalized_parent_end]
                    ),
                )
                chunks.append(parent)
                parent_text = page.text[normalized_parent_start:normalized_parent_end]
                for child_start, child_end in recursive_spans(
                    parent_text,
                    chunk_size=config.chunk_size,
                    chunk_overlap=config.chunk_overlap,
                    offset=normalized_parent_start,
                ):
                    child_span = trim_span(page.text, child_start, child_end)
                    if child_span is None:
                        continue
                    normalized_child_start, normalized_child_end = child_span
                    chunks.append(
                        build_chunk(
                            document,
                            page,
                            normalized_child_start,
                            normalized_child_end,
                            len(chunks),
                            namespace=f"{self.name}:child",
                            parent_chunk_id=parent.chunk_id,
                            token_count=count_lexical_tokens(
                                page.text[normalized_child_start:normalized_child_end]
                            ),
                        )
                    )
        return chunks
