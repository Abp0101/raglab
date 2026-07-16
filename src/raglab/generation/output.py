"""Strict structured model output for grounded answers."""

from uuid import UUID

from pydantic import Field

from raglab.core.schemas import EvidenceStatus, RAGLabModel


class GeneratedCitation(RAGLabModel):
    """Citation requested from the model before deterministic validation."""

    chunk_id: UUID
    quoted_text: str = Field(min_length=1)


class GroundedAnswer(RAGLabModel):
    """Provider output contract validated before it reaches API consumers."""

    answer: str = Field(min_length=1)
    citations: tuple[GeneratedCitation, ...]
    evidence_status: EvidenceStatus
    confidence: float = Field(ge=0, le=1)
    warnings: tuple[str, ...] = ()
