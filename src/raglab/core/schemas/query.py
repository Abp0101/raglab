"""Retrieval, generation, pipeline, citation, and response models."""

from datetime import date
from enum import StrEnum
from typing import Any, Self
from uuid import UUID

from pydantic import Field, computed_field, model_validator

from raglab.core.schemas.common import RAGLabModel
from raglab.core.schemas.documents import Chunk


class FrameworkName(StrEnum):
    """RAG implementations selectable through the shared API."""

    CUSTOM = "custom"
    LANGCHAIN = "langchain"
    LANGGRAPH = "langgraph"
    LLAMAINDEX = "llamaindex"
    HAYSTACK = "haystack"


class RetrievalMode(StrEnum):
    """Supported first-stage retrieval configurations."""

    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"


class EvidenceStatus(StrEnum):
    """Pipeline assessment of whether retrieved evidence supports an answer."""

    SUFFICIENT = "sufficient"
    INSUFFICIENT = "insufficient"
    CONFLICTING = "conflicting"


class MetadataFilter(RAGLabModel):
    """Portable filters that every retriever can translate where supported."""

    document_ids: tuple[UUID, ...] = ()
    authors: tuple[str, ...] = ()
    published_from: date | None = None
    published_to: date | None = None
    file_types: tuple[str, ...] = ()
    section_headings: tuple[str, ...] = ()
    attributes: dict[str, str | int | float | bool] = Field(default_factory=dict)

    @model_validator(mode="after")
    def publication_range_is_ordered(self) -> Self:
        if (
            self.published_from is not None
            and self.published_to is not None
            and self.published_from > self.published_to
        ):
            raise ValueError("published_from must not be after published_to")
        return self


class PipelineConfig(RAGLabModel):
    """Comparable defaults used to configure a RAG implementation."""

    retrieval_mode: RetrievalMode = RetrievalMode.HYBRID
    top_k: int = Field(default=5, ge=1, le=100)
    candidate_k: int = Field(default=20, ge=1, le=500)
    rerank: bool = True
    rerank_top_k: int = Field(default=5, ge=1, le=100)
    evidence_threshold: float = Field(default=0.5, ge=0, le=1)
    max_context_tokens: int = Field(default=6000, ge=256, le=128000)

    @model_validator(mode="after")
    def ranking_depths_are_consistent(self) -> Self:
        if self.candidate_k < self.top_k:
            raise ValueError("candidate_k must be greater than or equal to top_k")
        if self.rerank_top_k > self.candidate_k:
            raise ValueError("rerank_top_k must not exceed candidate_k")
        return self


class PipelineCapabilities(RAGLabModel):
    """Discoverable features implemented by a pipeline adapter."""

    ingestion: bool = True
    dense_retrieval: bool = True
    sparse_retrieval: bool = False
    hybrid_retrieval: bool = False
    reranking: bool = False
    metadata_filtering: bool = False
    streaming: bool = False
    agentic: bool = False


class QueryRequest(RAGLabModel):
    """Validated request accepted by any RAG pipeline."""

    query: str = Field(min_length=1, max_length=4000)
    framework: FrameworkName
    collection_id: UUID
    top_k: int = Field(default=5, ge=1, le=100)
    retrieval_mode: RetrievalMode = RetrievalMode.HYBRID
    rerank: bool = True
    metadata_filter: MetadataFilter | None = None
    model: str | None = Field(default=None, min_length=1, max_length=255)
    temperature: float = Field(default=0, ge=0, le=2)
    debug: bool = False


class RetrievalRequest(RAGLabModel):
    """Input shared by dense and sparse retriever implementations."""

    query: str = Field(min_length=1, max_length=4000)
    collection_id: UUID
    top_k: int = Field(default=20, ge=1, le=500)
    metadata_filter: MetadataFilter | None = None


class RetrievedChunk(RAGLabModel):
    """Chunk plus scores emitted throughout retrieval and reranking."""

    chunk: Chunk
    rank: int = Field(ge=1)
    relevance_score: float | None = None
    dense_score: float | None = None
    sparse_score: float | None = None
    fusion_score: float | None = None
    reranker_score: float | None = None


class Citation(RAGLabModel):
    """Evidence reference mapped to source metadata and quoted text."""

    document_id: UUID
    document_title: str = Field(min_length=1, max_length=500)
    page_number: int | None = Field(default=None, ge=1)
    section_heading: str | None = Field(default=None, max_length=500)
    chunk_id: UUID
    quoted_text: str = Field(min_length=1)
    relevance_score: float | None = None
    reranker_score: float | None = None


class LatencyMetrics(RAGLabModel):
    """Wall-clock durations for observable query stages."""

    total_ms: float = Field(ge=0)
    retrieval_ms: float = Field(default=0, ge=0)
    reranking_ms: float = Field(default=0, ge=0)
    generation_ms: float = Field(default=0, ge=0)


class UsageMetrics(RAGLabModel):
    """Provider usage and cost data when reported or estimable."""

    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    estimated_cost_usd: float | None = Field(default=None, ge=0)
    llm_calls: int = Field(default=0, ge=0)
    retrieval_iterations: int = Field(default=1, ge=0)

    @model_validator(mode="after")
    def total_matches_known_token_counts(self) -> Self:
        if (
            self.prompt_tokens is not None
            and self.completion_tokens is not None
            and self.total_tokens is not None
            and self.total_tokens != self.prompt_tokens + self.completion_tokens
        ):
            raise ValueError("total_tokens must equal prompt_tokens plus completion_tokens")
        return self


class GenerationRequest(RAGLabModel):
    """Provider-neutral prompt request."""

    system_prompt: str = Field(min_length=1)
    user_prompt: str = Field(min_length=1)
    model: str = Field(min_length=1, max_length=255)
    temperature: float = Field(default=0, ge=0, le=2)
    max_output_tokens: int | None = Field(default=None, ge=1)
    response_schema: dict[str, Any] | None = None


class GenerationResult(RAGLabModel):
    """Normalized provider output before conversion to a RAG response."""

    text: str = Field(min_length=1)
    model: str = Field(min_length=1, max_length=255)
    usage: UsageMetrics = Field(default_factory=UsageMetrics)
    raw_response_id: str | None = None


class RAGResponse(RAGLabModel):
    """Standard answer contract returned by all framework implementations."""

    answer: str = Field(min_length=1)
    citations: tuple[Citation, ...] = ()
    retrieved_chunks: tuple[RetrievedChunk, ...] = ()
    framework: FrameworkName
    model: str = Field(min_length=1, max_length=255)
    latency: LatencyMetrics
    usage: UsageMetrics = Field(default_factory=UsageMetrics)
    evidence_status: EvidenceStatus
    confidence: float | None = Field(default=None, ge=0, le=1)
    warnings: tuple[str, ...] = ()
    debug: dict[str, Any] | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def latency_ms(self) -> float:
        """Expose total latency using the public API field name."""
        return self.latency.total_ms

    @computed_field  # type: ignore[prop-decorator]
    @property
    def prompt_tokens(self) -> int | None:
        """Expose provider prompt usage at the response boundary."""
        return self.usage.prompt_tokens

    @computed_field  # type: ignore[prop-decorator]
    @property
    def completion_tokens(self) -> int | None:
        """Expose provider completion usage at the response boundary."""
        return self.usage.completion_tokens

    @computed_field  # type: ignore[prop-decorator]
    @property
    def estimated_cost(self) -> float | None:
        """Expose estimated USD cost using the public API field name."""
        return self.usage.estimated_cost_usd

    @computed_field  # type: ignore[prop-decorator]
    @property
    def evidence_sufficient(self) -> bool:
        """Preserve the simple boolean needed by API consumers."""
        return self.evidence_status is EvidenceStatus.SUFFICIENT
