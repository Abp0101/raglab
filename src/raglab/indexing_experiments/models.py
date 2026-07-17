"""Typed configuration and results for isolated native indexing experiments."""

from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal
from uuid import UUID

from pydantic import ConfigDict, Field, model_validator

from raglab.core.schemas import RAGLabModel


class IndexingFramework(StrEnum):
    """Frameworks with a meaningful native document indexing abstraction."""

    CUSTOM = "custom"
    LANGCHAIN = "langchain"
    LLAMAINDEX = "llamaindex"
    HAYSTACK = "haystack"


class IndexingBenchmarkQuery(RAGLabModel):
    """Natural-language retrieval query with one verbatim target passage."""

    query_id: str = Field(min_length=1, max_length=100)
    query: str = Field(min_length=1)
    relevant_passage: str = Field(min_length=1)


class IndexingBenchmarkCase(RAGLabModel):
    """One source and its annotated retrieval queries."""

    dataset_version: str = Field(min_length=1, max_length=50)
    case_id: str = Field(min_length=1, max_length=100)
    category: str = Field(min_length=1, max_length=100)
    text: str = Field(min_length=1)
    queries: tuple[IndexingBenchmarkQuery, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def targets_exist_and_query_ids_are_unique(self) -> "IndexingBenchmarkCase":
        if any(query.relevant_passage not in self.text for query in self.queries):
            raise ValueError("every relevant passage must occur verbatim in source text")
        query_ids = [query.query_id for query in self.queries]
        if len(query_ids) != len(set(query_ids)):
            raise ValueError("query IDs must be unique within an indexing case")
        return self


class IndexingExperimentControls(RAGLabModel):
    """Variables fixed across every native indexing path."""

    embedding_model: Literal["deterministic-hash-v1"] = "deterministic-hash-v1"
    embedding_dimensions: int = Field(default=128, ge=32, le=2048)
    chunk_size: int = Field(default=50, ge=16, le=2048)
    chunk_overlap: int = Field(default=8, ge=0)
    top_k: int = Field(default=1, ge=1, le=20)

    @model_validator(mode="after")
    def overlap_is_smaller_than_size(self) -> "IndexingExperimentControls":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk overlap must be smaller than chunk size")
        return self


class IndexingExperimentDefinition(RAGLabModel):
    """One declared framework-native indexing path."""

    framework: IndexingFramework
    strategy: str = Field(min_length=1, max_length=100)
    index_backend: str = Field(min_length=1, max_length=100)
    size_unit: Literal["lexical-tokens", "framework-tokens", "words"]


class IndexingExperimentPlan(RAGLabModel):
    """Versioned declaration that separates native experiments from the baseline."""

    benchmark: Literal["raglab-native-indexing"] = "raglab-native-indexing"
    version: str = Field(min_length=1, max_length=50)
    dataset_version: str = Field(min_length=1, max_length=50)
    dataset_path: Path
    controls: IndexingExperimentControls
    experiments: tuple[IndexingExperimentDefinition, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def frameworks_are_unique(self) -> "IndexingExperimentPlan":
        frameworks = [experiment.framework for experiment in self.experiments]
        if len(frameworks) != len(set(frameworks)):
            raise ValueError("indexing experiment frameworks must be unique")
        return self


class ExperimentChunk(RAGLabModel):
    """Minimal common observation retained from a framework-native index."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=False,
        validate_assignment=True,
    )

    chunk_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    token_count: int = Field(ge=1)

    @model_validator(mode="after")
    def end_follows_start(self) -> "ExperimentChunk":
        if self.end <= self.start:
            raise ValueError("chunk end must be greater than start")
        return self


class IndexingCaseResult(RAGLabModel):
    """Measurements for one dataset case through one native index."""

    case_id: str
    category: str
    framework: IndexingFramework
    strategy: str
    index_backend: str
    size_unit: str
    chunk_size: int = Field(gt=0)
    chunk_overlap: int = Field(ge=0)
    chunk_count: int = Field(ge=0)
    mean_chunk_characters: float = Field(ge=0)
    mean_chunk_tokens: float = Field(ge=0)
    redundancy_ratio: float = Field(ge=0)
    relevant_passage_containment: float = Field(ge=0, le=1)
    retrieval_recall_at_k: float = Field(ge=0, le=1)
    section_boundary_violations: int = Field(ge=0)
    indexing_ms: float = Field(ge=0)
    mean_query_ms: float = Field(ge=0)
    estimated_api_cost_usd: float = Field(default=0, ge=0, le=0)


class IndexingAggregate(RAGLabModel):
    """Mean measurements for one framework over all benchmark cases."""

    framework: IndexingFramework
    strategy: str
    index_backend: str
    case_count: int = Field(gt=0)
    mean_chunk_count: float = Field(ge=0)
    mean_chunk_tokens: float = Field(ge=0)
    mean_redundancy_ratio: float = Field(ge=0)
    mean_passage_containment: float = Field(ge=0, le=1)
    mean_retrieval_recall_at_k: float = Field(ge=0, le=1)
    mean_indexing_ms: float = Field(ge=0)
    mean_query_ms: float = Field(ge=0)
    estimated_api_cost_usd: float = Field(default=0, ge=0, le=0)


class IndexingExperimentRun(RAGLabModel):
    """Reproducible machine-readable native indexing experiment report."""

    run_id: UUID
    benchmark: str
    version: str
    dataset_version: str
    dataset_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    config_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    started_at: datetime
    completed_at: datetime
    controls: IndexingExperimentControls
    results: tuple[IndexingCaseResult, ...]
    aggregates: tuple[IndexingAggregate, ...]
