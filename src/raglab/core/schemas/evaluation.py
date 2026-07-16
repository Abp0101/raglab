"""Versioned evaluation dataset and result models."""

from datetime import date, datetime
from enum import StrEnum
from typing import Self
from uuid import UUID

from pydantic import Field, model_validator

from raglab.core.schemas.common import RAGLabModel
from raglab.core.schemas.query import FrameworkName, RAGResponse, RetrievalMode


class EvaluationDatasetManifest(RAGLabModel):
    """Immutable metadata and integrity information for one dataset version."""

    name: str = Field(min_length=1, max_length=100)
    version: str = Field(min_length=1, max_length=50)
    description: str = Field(min_length=1, max_length=2000)
    collection_id: UUID
    published_on: date
    question_count: int = Field(ge=1)
    questions_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    domains: tuple[str, ...] = Field(min_length=1)
    license: str = Field(min_length=1, max_length=100)


class EvaluationDataset(RAGLabModel):
    """Validated manifest and unique benchmark questions loaded together."""

    manifest: EvaluationDatasetManifest
    questions: tuple["EvaluationQuestion", ...] = Field(min_length=1)


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

    @model_validator(mode="after")
    def annotations_match_answerability(self) -> Self:
        if self.answerable and not (self.relevant_chunk_ids or self.relevant_document_ids):
            raise ValueError("answerable questions require relevant chunk or document IDs")
        if not set(self.expected_citation_chunk_ids).issubset(self.relevant_chunk_ids):
            raise ValueError("expected citation IDs must be annotated as relevant chunks")
        return self


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


class EvaluationRunConfig(RAGLabModel):
    """Comparable query settings recorded with every benchmark run."""

    framework: FrameworkName
    retrieval_mode: RetrievalMode
    top_k: int = Field(ge=1, le=100)
    rerank: bool
    model: str = Field(min_length=1, max_length=255)
    concurrency: int = Field(default=1, ge=1, le=16)


class EvaluationRun(RAGLabModel):
    """Complete reproducibility record for one framework/configuration run."""

    run_id: UUID
    dataset_name: str = Field(min_length=1, max_length=100)
    dataset_version: str = Field(min_length=1, max_length=50)
    dataset_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    config: EvaluationRunConfig
    started_at: datetime
    completed_at: datetime
    results: tuple[EvaluationResult, ...]


class EvaluationMetricAggregate(RAGLabModel):
    """Aggregate over applicable question-level values only."""

    name: str = Field(min_length=1, max_length=100)
    mean: float
    minimum: float
    maximum: float
    sample_count: int = Field(ge=1)


class EvaluationReport(RAGLabModel):
    """Run plus deterministic aggregates rendered to JSON and Markdown."""

    run: EvaluationRun
    aggregates: tuple[EvaluationMetricAggregate, ...]
    successful_questions: int = Field(ge=0)
    failed_questions: int = Field(ge=0)
