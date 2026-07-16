"""LangChain-native document/splitter ingestion over RAGLab's shared stores."""

import hashlib
import time
from collections.abc import Iterator, Sequence
from contextlib import suppress
from uuid import UUID

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document as LangChainDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter

from raglab.chunking.base import active_heading, build_chunk
from raglab.chunking.tokenization import count_lexical_tokens
from raglab.core.exceptions import ProviderUnavailableError, RAGLabError
from raglab.core.interfaces import (
    DocumentParser,
    DocumentRepository,
    EmbeddingProvider,
    SparseIndexer,
    VectorIndexer,
)
from raglab.core.schemas import (
    Chunk,
    DocumentInput,
    DocumentStatus,
    IngestionResult,
    ParsedDocument,
)
from raglab.ingestion.validation import PdfUploadValidator


class ParsedPageLoader(BaseLoader):
    """Expose already validated parsed pages through LangChain's loader contract."""

    def __init__(self, document: ParsedDocument) -> None:
        self._document = document

    def lazy_load(self) -> Iterator[LangChainDocument]:
        for page in self._document.pages:
            yield LangChainDocument(
                page_content=page.text,
                metadata={"page_number": page.page_number},
            )


class LangChainIngestionPipeline:
    """Use LangChain Documents and splitting while retaining shared durable indexes."""

    name = "langchain-recursive-character"

    def __init__(
        self,
        *,
        validator: PdfUploadValidator,
        parser: DocumentParser,
        embedding_provider: EmbeddingProvider,
        document_repository: DocumentRepository,
        vector_indexer: VectorIndexer,
        sparse_indexer: SparseIndexer,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ) -> None:
        self._validator = validator
        self._parser = parser
        self._embedding_provider = embedding_provider
        self._documents = document_repository
        self._vector_indexer = vector_indexer
        self._sparse_indexer = sparse_indexer
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
        )

    async def ingest(self, document_input: DocumentInput) -> IngestionResult:
        started = time.perf_counter()
        self._validator.validate(document_input)
        content_hash = hashlib.sha256(document_input.content).hexdigest()
        duplicate = await self._documents.find_by_hash(
            document_input.collection_id,
            content_hash,
        )
        if duplicate is not None:
            return IngestionResult(
                document_id=duplicate.document_id,
                collection_id=duplicate.collection_id,
                page_count=duplicate.page_count or 0,
                chunk_count=0,
                duration_ms=(time.perf_counter() - started) * 1000,
                parser=self._parser.name,
                chunking_strategy=self.name,
                embedding_model=self._embedding_provider.model_name,
                duplicate=True,
                warnings=("duplicate content was not re-indexed",),
            )
        parsed = await self._parser.parse(document_input)
        langchain_pages = await ParsedPageLoader(parsed).aload()
        split_documents = await self._splitter.atransform_documents(langchain_pages)
        chunks = _to_chunks(parsed, split_documents)
        embeddings = tuple(await self._embedding_provider.embed_chunks(chunks))
        if len(embeddings) != len(chunks):
            raise ValueError("embedding provider must return one vector per chunk")
        processing = parsed.document.model_copy(update={"status": DocumentStatus.PROCESSING})
        await self._documents.save(processing, chunks)
        try:
            await self._vector_indexer.upsert(chunks, embeddings)
            await self._sparse_indexer.upsert(chunks)
            await self._documents.set_status(processing.document_id, DocumentStatus.READY)
        except Exception as error:
            await self._rollback(processing.document_id, chunks)
            if isinstance(error, RAGLabError):
                raise
            raise ProviderUnavailableError("LangChain document indexing failed") from error
        return IngestionResult(
            document_id=processing.document_id,
            collection_id=processing.collection_id,
            page_count=processing.page_count or 0,
            chunk_count=len(chunks),
            duration_ms=(time.perf_counter() - started) * 1000,
            parser=parsed.parser_name,
            chunking_strategy=self.name,
            embedding_model=self._embedding_provider.model_name,
            warnings=parsed.warnings,
        )

    async def _rollback(self, document_id: UUID, chunks: Sequence[Chunk]) -> None:
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        with suppress(Exception):
            await self._vector_indexer.delete(chunk_ids)
        with suppress(Exception):
            await self._sparse_indexer.delete(chunks)
        with suppress(Exception):
            await self._documents.delete(document_id)


def _to_chunks(
    parsed: ParsedDocument,
    documents: Sequence[LangChainDocument],
) -> tuple[Chunk, ...]:
    pages = {page.page_number: page for page in parsed.pages}
    chunks: list[Chunk] = []
    for document in documents:
        page_number = int(document.metadata["page_number"])
        page = pages[page_number]
        start = int(document.metadata["start_index"])
        end = start + len(document.page_content)
        chunk = build_chunk(
            parsed,
            page,
            start,
            end,
            len(chunks),
            namespace="langchain-recursive-character",
            token_count=count_lexical_tokens(document.page_content),
        )
        heading = active_heading(page, start)
        chunks.append(
            chunk.model_copy(
                update={
                    "text": document.page_content,
                    "metadata": chunk.metadata.model_copy(update={"section_heading": heading}),
                }
            )
        )
    return tuple(chunks)
