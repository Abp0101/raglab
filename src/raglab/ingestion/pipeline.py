"""Document ingestion orchestration independent of storage implementations."""

import hashlib
import time

from raglab.core.interfaces import (
    Chunker,
    DocumentParser,
    DocumentRepository,
    EmbeddingProvider,
    SparseIndexer,
    VectorIndexer,
)
from raglab.core.schemas import (
    ChunkingConfig,
    DocumentInput,
    DocumentStatus,
    IngestionResult,
)
from raglab.ingestion.validation import PdfUploadValidator


class DocumentIngestionPipeline:
    """Coordinate validation, deduplication, parsing, chunking, and indexing."""

    def __init__(
        self,
        *,
        validator: PdfUploadValidator,
        parser: DocumentParser,
        chunker: Chunker,
        embedding_provider: EmbeddingProvider,
        document_repository: DocumentRepository,
        vector_indexer: VectorIndexer,
        sparse_indexer: SparseIndexer,
        chunking_config: ChunkingConfig,
    ) -> None:
        self._validator = validator
        self._parser = parser
        self._chunker = chunker
        self._embedding_provider = embedding_provider
        self._document_repository = document_repository
        self._vector_indexer = vector_indexer
        self._sparse_indexer = sparse_indexer
        self._chunking_config = chunking_config

    async def ingest(self, document_input: DocumentInput) -> IngestionResult:
        """Ingest one PDF and return a provider-neutral processing report."""
        started = time.perf_counter()
        self._validator.validate(document_input)
        content_hash = hashlib.sha256(document_input.content).hexdigest()
        duplicate = await self._document_repository.find_by_hash(
            document_input.collection_id, content_hash
        )
        if duplicate is not None:
            return IngestionResult(
                document_id=duplicate.document_id,
                collection_id=duplicate.collection_id,
                page_count=duplicate.page_count or 0,
                chunk_count=0,
                duration_ms=(time.perf_counter() - started) * 1000,
                parser=self._parser.name,
                chunking_strategy=self._chunker.name,
                embedding_model=self._embedding_provider.model_name,
                duplicate=True,
                warnings=("duplicate content was not re-indexed",),
            )

        parsed = await self._parser.parse(document_input)
        chunks = tuple(self._chunker.chunk(parsed, self._chunking_config))
        embeddings = tuple(await self._embedding_provider.embed_chunks(chunks))
        if len(embeddings) != len(chunks):
            raise ValueError("embedding provider must return one vector per chunk")

        await self._vector_indexer.upsert(chunks, embeddings)
        await self._sparse_indexer.upsert(chunks)
        ready_document = parsed.document.model_copy(update={"status": DocumentStatus.READY})
        await self._document_repository.save(ready_document, chunks)
        return IngestionResult(
            document_id=ready_document.document_id,
            collection_id=ready_document.collection_id,
            page_count=ready_document.page_count or 0,
            chunk_count=len(chunks),
            duration_ms=(time.perf_counter() - started) * 1000,
            parser=parsed.parser_name,
            chunking_strategy=self._chunker.name,
            embedding_model=self._embedding_provider.model_name,
            warnings=parsed.warnings,
        )
