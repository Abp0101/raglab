"""Dependency-injection contracts for shared RAG services and adapters."""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable
from uuid import UUID

from raglab.core.schemas import (
    Chunk,
    ChunkingConfig,
    Document,
    DocumentInput,
    DocumentStatus,
    Embedding,
    EvaluationMetricResult,
    EvaluationQuestion,
    GenerationRequest,
    GenerationResult,
    IngestionResult,
    ParsedDocument,
    PipelineCapabilities,
    PipelineConfig,
    QueryRequest,
    RAGResponse,
    RetrievalRequest,
    RetrievedChunk,
)


@runtime_checkable
class DocumentParser(Protocol):
    """Convert untrusted source bytes into normalized, page-aware text."""

    @property
    def name(self) -> str: ...

    async def parse(self, document: DocumentInput) -> ParsedDocument: ...


@runtime_checkable
class Chunker(Protocol):
    """Split parsed pages while retaining source provenance."""

    @property
    def name(self) -> str: ...

    def chunk(self, document: ParsedDocument, config: ChunkingConfig) -> Sequence[Chunk]: ...


@runtime_checkable
class DocumentRepository(Protocol):
    """Persist source records and chunks, and enforce collection-level deduplication."""

    async def find_by_hash(self, collection_id: UUID, content_hash: str) -> Document | None: ...

    async def save(self, document: Document, chunks: Sequence[Chunk]) -> None: ...

    async def set_status(self, document_id: UUID, status: DocumentStatus) -> None: ...

    async def delete(self, document_id: UUID) -> None: ...


@runtime_checkable
class ChunkRepository(Protocol):
    """Load context chunks from the relational source of truth."""

    async def get_by_ids(self, chunk_ids: Sequence[UUID]) -> Sequence[Chunk]: ...


@runtime_checkable
class VectorIndexer(Protocol):
    """Store dense chunk vectors and retrieval metadata."""

    async def upsert(self, chunks: Sequence[Chunk], embeddings: Sequence[Embedding]) -> None: ...

    async def delete(self, chunk_ids: Sequence[UUID]) -> None: ...


@runtime_checkable
class SparseIndexer(Protocol):
    """Store chunks in a lexical index such as BM25."""

    async def upsert(self, chunks: Sequence[Chunk]) -> None: ...

    async def delete(self, chunks: Sequence[Chunk]) -> None: ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Generate comparable document and query vectors."""

    @property
    def model_name(self) -> str: ...

    async def embed_chunks(self, chunks: Sequence[Chunk]) -> Sequence[Embedding]: ...

    async def embed_query(self, query: str) -> Sequence[float]: ...


@runtime_checkable
class DenseRetriever(Protocol):
    """Retrieve chunks using a dense query vector."""

    async def retrieve(
        self,
        request: RetrievalRequest,
        query_vector: Sequence[float],
    ) -> Sequence[RetrievedChunk]: ...


@runtime_checkable
class SparseRetriever(Protocol):
    """Retrieve chunks using lexical matching such as BM25."""

    async def retrieve(self, request: RetrievalRequest) -> Sequence[RetrievedChunk]: ...


@runtime_checkable
class Reranker(Protocol):
    """Rescore retrieved candidates against the original question."""

    async def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievedChunk],
        top_k: int,
    ) -> Sequence[RetrievedChunk]: ...


@runtime_checkable
class ContextExpander(Protocol):
    """Replace retrieval units with larger linked context where available."""

    async def expand(self, chunks: Sequence[RetrievedChunk]) -> Sequence[RetrievedChunk]: ...


@runtime_checkable
class LLMProvider(Protocol):
    """Generate text without exposing provider-specific SDK objects."""

    async def generate(self, request: GenerationRequest) -> GenerationResult: ...


@runtime_checkable
class RAGPipeline(Protocol):
    """Shared contract implemented by all five RAG approaches."""

    @property
    def config(self) -> PipelineConfig: ...

    @property
    def capabilities(self) -> PipelineCapabilities: ...

    async def ingest(self, documents: Sequence[DocumentInput]) -> Sequence[IngestionResult]: ...

    async def query(self, request: QueryRequest) -> RAGResponse: ...


@runtime_checkable
class EvaluationMetric(Protocol):
    """Score one question and optional pipeline response."""

    @property
    def name(self) -> str: ...

    async def evaluate(
        self,
        question: EvaluationQuestion,
        response: RAGResponse | None,
    ) -> EvaluationMetricResult: ...
