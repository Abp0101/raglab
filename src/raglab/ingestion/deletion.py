"""Retry-safe document removal across relational, dense, and sparse stores."""

from uuid import UUID

from raglab.core.exceptions import ProviderUnavailableError, RAGLabError
from raglab.core.interfaces import (
    ChunkRepository,
    DocumentRepository,
    SparseIndexer,
    VectorIndexer,
)
from raglab.core.schemas import DocumentDeletionResult


class CoordinatedDocumentDeletionService:
    """Retain PostgreSQL source data until both external indexes are cleared."""

    def __init__(
        self,
        *,
        document_repository: DocumentRepository,
        chunk_repository: ChunkRepository,
        vector_indexer: VectorIndexer,
        sparse_indexer: SparseIndexer,
    ) -> None:
        self._documents = document_repository
        self._chunks = chunk_repository
        self._vectors = vector_indexer
        self._sparse = sparse_indexer

    async def delete(self, document_id: UUID) -> DocumentDeletionResult:
        """Delete one terminal document, preserving enough state for safe retries."""
        try:
            document = await self._documents.mark_deleting(document_id)
            chunks = tuple(await self._chunks.get_by_document(document_id))
            await self._vectors.delete([chunk.chunk_id for chunk in chunks])
            await self._sparse.delete(chunks)
            await self._documents.delete(document_id)
        except RAGLabError:
            raise
        except Exception as error:
            raise ProviderUnavailableError("document deletion could not complete") from error
        return DocumentDeletionResult(
            document_id=document.document_id,
            collection_id=document.collection_id,
            deleted_chunk_count=len(chunks),
        )
