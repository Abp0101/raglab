"""Deterministic validation and conversion of model-requested citations."""

import re

from raglab.core.schemas import Citation
from raglab.generation.context import ContextWindow, chunk_map
from raglab.generation.output import GroundedAnswer

WHITESPACE = re.compile(r"\s+")


def validate_citations(
    generated: GroundedAnswer,
    context: ContextWindow,
) -> tuple[tuple[Citation, ...], tuple[str, ...]]:
    """Accept only known chunk IDs and quotes present in their source chunk."""
    available = chunk_map(context)
    citations: list[Citation] = []
    warnings: list[str] = []
    seen: set[tuple[object, str]] = set()
    for requested in generated.citations:
        result = available.get(requested.chunk_id)
        if result is None:
            warnings.append(f"citation referenced unavailable chunk {requested.chunk_id}")
            continue
        normalized_quote = _normalize(requested.quoted_text)
        if normalized_quote not in _normalize(result.chunk.text):
            warnings.append(f"citation quote was not found in chunk {requested.chunk_id}")
            continue
        key = (requested.chunk_id, normalized_quote)
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            Citation(
                document_id=result.chunk.metadata.document_id,
                document_title=result.chunk.metadata.display_title,
                page_number=result.chunk.metadata.page_number,
                section_heading=result.chunk.metadata.section_heading,
                chunk_id=result.chunk.chunk_id,
                quoted_text=requested.quoted_text,
                relevance_score=result.relevance_score,
                reranker_score=result.reranker_score,
            )
        )
    return tuple(citations), tuple(warnings)


def _normalize(text: str) -> str:
    return WHITESPACE.sub(" ", text).strip()
