"""Versioned evaluation dataset and result models."""

from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import Field

from raglab.core.schemas.common import RAGLabModel
from raglab.core.schemas.query import FrameworkName, RAGResponse


class EvaluationDifficulty(StrEnum):
    """Coarse benchmark difficulty grouping."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class EvaluationQuestion(RAGLabModel):
    """One versioned, independently scoreable benchmark item."""

    question_id: str = Field(min_length=1, max_length=100)
    dataset_version: str = Field(min_length=1, max_length=50)
    question: str = Field(min_length=1, max_length=4000)
    expected_answer: str | None = None
    expected_key_facts: tuple[str, ...] = ()
    relevant_document_ids: tuple[UUID, ...] = ()
    relevant_chunk_ids: tuple[UUID, ...] = ()
    expected_citation_chunk_ids: tuple[UUID, ...] = ()
    answerable: bool
    category: str = Field(min_length=1, max_length=100)
    difficulty: EvaluationDifficulty
    notes: str | None = None


class EvaluationMetricResult(RAGLabModel):
    """One deterministic or judge-based metric output."""

    name: str = Field(min_length=1, max_length=100)
    value: float
    passed: bool | None = None
    details: dict[str, str | int | float | bool] = Field(default_factory=dict)


class EvaluationResult(RAGLabModel):
    """Pipeline result and metrics for one evaluation question."""

    evaluation_id: UUID
    question_id: str = Field(min_length=1, max_length=100)
    dataset_version: str = Field(min_length=1, max_length=50)
    framework: FrameworkName
    response: RAGResponse | None = None
    retrieval_metrics: tuple[EvaluationMetricResult, ...] = ()
    answer_metrics: tuple[EvaluationMetricResult, ...] = ()
    created_at: datetime
    error: str | None = None
