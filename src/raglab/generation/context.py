"""Bounded context assembly with explicit untrusted-data delimiters."""

import json
from dataclasses import dataclass
from uuid import UUID

from raglab.chunking.tokenization import lexical_token_spans
from raglab.core.schemas import RetrievedChunk


@dataclass(frozen=True, slots=True)
class ContextWindow:
    """Serialized evidence and the exact chunks available for citation."""

    text: str
    chunks: tuple[RetrievedChunk, ...]
    estimated_tokens: int


class ContextBuilder:
    """Fit complete or truncated chunks into a provider-neutral token estimate."""

    def build(self, chunks: tuple[RetrievedChunk, ...], max_tokens: int) -> ContextWindow:
        selected: list[RetrievedChunk] = []
        records: list[dict[str, object]] = []
        used = 0
        for result in chunks:
            remaining = max_tokens - used
            if remaining <= 0:
                break
            text, count = _bounded_text(result.chunk.text, remaining)
            if not text:
                continue
            records.append(
                {
                    "chunk_id": str(result.chunk.chunk_id),
                    "document_title": result.chunk.metadata.display_title,
                    "page_number": result.chunk.metadata.page_number,
                    "section_heading": result.chunk.metadata.section_heading,
                    "text": text,
                }
            )
            selected.append(
                result.model_copy(update={"chunk": result.chunk.model_copy(update={"text": text})})
            )
            used += count
        payload = json.dumps(records, ensure_ascii=False, separators=(",", ":"))
        payload = payload.replace("<", "\\u003c").replace(">", "\\u003e")
        return ContextWindow(
            text=f"<untrusted_evidence_json>{payload}</untrusted_evidence_json>",
            chunks=tuple(selected),
            estimated_tokens=used,
        )


def _bounded_text(text: str, token_budget: int) -> tuple[str, int]:
    tokens = lexical_token_spans(text)
    if not tokens or token_budget <= 0:
        return "", 0
    selected = tokens[:token_budget]
    return text[: selected[-1].end].strip(), len(selected)


def chunk_map(window: ContextWindow) -> dict[UUID, RetrievedChunk]:
    """Index evidence chunks by citation identifier."""
    return {result.chunk.chunk_id: result for result in window.chunks}
