"""Dependency-injection contracts for shared RAG services and adapters."""

from collections.abc import Sequence
from datetime import timedelta
from typing import Protocol, runtime_checkable
from uuid import UUID

from raglab.core.schemas import (
    AuthPrincipal,
    Chunk,
    ChunkingConfig,
    Collection,
    CollectionCreate,
    CursorPage,
    Document,
    DocumentDeletionResult,
    DocumentInput,
    DocumentStatus,
    Embedding,
    EvaluationMetricResult,
    EvaluationQuestion,
    GenerationRequest,
    GenerationResult,
    IngestionJob,
    IngestionJobClaim,
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
class Authenticator(Protocol):
    """Resolve an opaque bearer credential into a safe caller principal."""

    def authenticate(self, credential: str | None) -> AuthPrincipal: ...


@runtime_checkable
class CatalogRepository(Protocol):
    """Manage collections and expose document metadata to the public API."""

    async def create_collection(self, request: CollectionCreate) -> Collection: ...

    async def list_collections(
        self,
        *,
        limit: int = 20,
        cursor: str | None = None,
    ) -> CursorPage[Collection]: ...

    async def get_collection(self, collection_id: UUID) -> Collection: ...

    async def list_documents(
        self,
        collection_id: UUID,
        *,
        limit: int = 20,
        cursor: str | None = None,
    ) -> CursorPage[Document]: ...

    async def get_document(self, document_id: UUID) -> Document: ...


@runtime_checkable
class IngestionJobRepository(Protocol):
    """Persist queued uploads and their lifecycle independently of API processes."""

    async def create(self, document: DocumentInput) -> IngestionJob: ...

    async def get(self, job_id: UUID) -> IngestionJob: ...

    async def list_for_collection(
        self,
        collection_id: UUID,
        *,
        limit: int = 20,
        cursor: str | None = None,
    ) -> CursorPage[IngestionJob]: ...

    async def claim_next(
        self,
        owner_id: UUID,
        lease_duration: timedelta,
    ) -> IngestionJobClaim | None: ...

    async def renew(
        self,
        job_id: UUID,
        owner_id: UUID,
        lease_duration: timedelta,
    ) -> bool: ...

    async def complete(
        self,
        job_id: UUID,
        owner_id: UUID,
        result: IngestionResult,
    ) -> bool: ...

    async def fail(
        self,
        job_id: UUID,
        owner_id: UUID,
        error_type: str,
        message: str,
    ) -> bool: ...

    async def release(self, job_id: UUID, owner_id: UUID) -> bool: ...


@runtime_checkable
class IngestionJobManager(Protocol):
    """Submit, inspect, recover, and stop background ingestion work."""

    async def start(self) -> None: ...

    async def submit(self, document: DocumentInput) -> IngestionJob: ...

    async def get(self, job_id: UUID) -> IngestionJob: ...

    async def list_for_collection(
        self,
        collection_id: UUID,
        *,
        limit: int = 20,
        cursor: str | None = None,
    ) -> CursorPage[IngestionJob]: ...

    async def close(self) -> None: ...


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

    async def mark_deleting(self, document_id: UUID) -> Document: ...

    async def delete(self, document_id: UUID) -> None: ...


@runtime_checkable
class ChunkRepository(Protocol):
    """Load context chunks from the relational source of truth."""

    async def get_by_ids(self, chunk_ids: Sequence[UUID]) -> Sequence[Chunk]: ...

    async def get_by_document(self, document_id: UUID) -> Sequence[Chunk]: ...


@runtime_checkable
class DocumentDeletionManager(Protocol):
    """Coordinate retry-safe document removal across shared stores."""

    async def delete(self, document_id: UUID) -> DocumentDeletionResult: ...


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
