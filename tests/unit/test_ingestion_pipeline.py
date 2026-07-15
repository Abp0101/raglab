from collections.abc import Sequence
from uuid import UUID, uuid4

import pymupdf
import pytest

from raglab.chunking import RecursiveCharacterChunker
from raglab.core.schemas import (
    Chunk,
    ChunkingConfig,
    Document,
    DocumentInput,
    Embedding,
)
from raglab.ingestion.parsers import PyMuPDFParser
from raglab.ingestion.pipeline import DocumentIngestionPipeline
from raglab.ingestion.validation import PdfUploadValidator


def make_pdf() -> bytes:
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "METHODS", fontsize=18)
    page.insert_textbox(
        (72, 100, 500, 700),
        "An IMU measured rehabilitation motion at 100 Hz. " * 12,
        fontsize=11,
    )
    content = pdf.tobytes()
    pdf.close()
    return content


class FakeEmbeddingProvider:
    model_name = "test-embedding"

    def __init__(self) -> None:
        self.calls = 0

    async def embed_chunks(self, chunks: Sequence[Chunk]) -> Sequence[Embedding]:
        self.calls += 1
        return [
            Embedding(
                chunk_id=chunk.chunk_id, vector=(0.1, 0.2), model=self.model_name, dimensions=2
            )
            for chunk in chunks
        ]

    async def embed_query(self, query: str) -> Sequence[float]:
        return (0.1, 0.2)


class FakeDocumentRepository:
    def __init__(self, duplicate: Document | None = None) -> None:
        self.duplicate = duplicate
        self.saved: tuple[Document, Sequence[Chunk]] | None = None

    async def find_by_hash(self, collection_id: UUID, content_hash: str) -> Document | None:
        return self.duplicate

    async def save(self, document: Document, chunks: Sequence[Chunk]) -> None:
        self.saved = (document, chunks)


class FakeVectorIndexer:
    def __init__(self) -> None:
        self.upserted = 0

    async def upsert(self, chunks: Sequence[Chunk], embeddings: Sequence[Embedding]) -> None:
        assert len(chunks) == len(embeddings)
        self.upserted = len(chunks)


class FakeSparseIndexer:
    def __init__(self) -> None:
        self.upserted = 0

    async def upsert(self, chunks: Sequence[Chunk]) -> None:
        self.upserted = len(chunks)


def build_pipeline(
    repository: FakeDocumentRepository,
    embedding_provider: FakeEmbeddingProvider,
    vector_indexer: FakeVectorIndexer,
    sparse_indexer: FakeSparseIndexer,
) -> DocumentIngestionPipeline:
    validator = PdfUploadValidator(max_size_bytes=1_000_000)
    return DocumentIngestionPipeline(
        validator=validator,
        parser=PyMuPDFParser(validator, max_pages=10),
        chunker=RecursiveCharacterChunker(),
        embedding_provider=embedding_provider,
        document_repository=repository,
        vector_indexer=vector_indexer,
        sparse_indexer=sparse_indexer,
        chunking_config=ChunkingConfig(chunk_size=120, chunk_overlap=20),
    )


@pytest.mark.asyncio
async def test_pipeline_runs_each_ingestion_stage() -> None:
    repository = FakeDocumentRepository()
    embeddings = FakeEmbeddingProvider()
    vectors = FakeVectorIndexer()
    sparse = FakeSparseIndexer()
    pipeline = build_pipeline(repository, embeddings, vectors, sparse)

    result = await pipeline.ingest(
        DocumentInput(file_name="imu.pdf", content=make_pdf(), collection_id=uuid4())
    )

    assert result.duplicate is False
    assert result.chunk_count > 1
    assert result.parser == "pymupdf"
    assert embeddings.calls == 1
    assert vectors.upserted == result.chunk_count
    assert sparse.upserted == result.chunk_count
    assert repository.saved is not None
    assert repository.saved[0].status.value == "ready"


@pytest.mark.asyncio
async def test_pipeline_skips_processing_for_duplicate_content() -> None:
    content = make_pdf()
    collection_id = uuid4()
    validator = PdfUploadValidator(max_size_bytes=1_000_000)
    parsed = await PyMuPDFParser(validator, max_pages=10).parse(
        DocumentInput(file_name="imu.pdf", content=content, collection_id=collection_id)
    )
    repository = FakeDocumentRepository(duplicate=parsed.document)
    embeddings = FakeEmbeddingProvider()
    vectors = FakeVectorIndexer()
    sparse = FakeSparseIndexer()

    result = await build_pipeline(repository, embeddings, vectors, sparse).ingest(
        DocumentInput(file_name="copy.pdf", content=content, collection_id=collection_id)
    )

    assert result.duplicate is True
    assert result.document_id == parsed.document.document_id
    assert embeddings.calls == 0
    assert vectors.upserted == 0
    assert sparse.upserted == 0
    assert repository.saved is None
