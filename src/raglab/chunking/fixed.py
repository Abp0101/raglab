"""Fixed lexical-token chunking with exact character provenance."""

from collections.abc import Sequence

from raglab.chunking.base import build_chunk
from raglab.chunking.tokenization import lexical_token_spans
from raglab.core.schemas import Chunk, ChunkingConfig, ParsedDocument


class FixedTokenChunker:
    """Create fixed-size token windows with configurable token overlap."""

    name = "fixed-token"

    def chunk(self, document: ParsedDocument, config: ChunkingConfig) -> Sequence[Chunk]:
        chunks: list[Chunk] = []
        for page in document.pages:
            tokens = lexical_token_spans(page.text)
            step = config.chunk_size - config.chunk_overlap
            for token_start in range(0, len(tokens), step):
                window = tokens[token_start : token_start + config.chunk_size]
                if not window:
                    break
                chunks.append(
                    build_chunk(
                        document,
                        page,
                        window[0].start,
                        window[-1].end,
                        len(chunks),
                        namespace=self.name,
                        token_count=len(window),
                    )
                )
                if token_start + config.chunk_size >= len(tokens):
                    break
        return chunks
