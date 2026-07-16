from collections.abc import Sequence
from uuid import UUID, uuid4

import pymupdf
import pytest

from raglab.core.schemas import Chunk, Document, DocumentInput, DocumentStatus, Embedding
from raglab.ingestion.langchain_pipeline import LangChainIngestionPipeline
from raglab.ingestion.parsers import PyMuPDFParser
from raglab.ingestion.validation import PdfUploadValidator


def make_pdf() -> bytes:
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "METHODS", fontsize=18)
    page.insert_textbox(
        (72, 100, 500, 700),
        "An IMU measured rehabilitation motion at 100 Hz. " * 20,
        fontsize=11,
    )
    content = pdf.tobytes()
    pdf.close()
    return content


class FakeEmbeddings:
    model_name = "local-test"

    async def embed_chunks(self, chunks: Sequence[Chunk]) -> Sequence[Embedding]:
        return tuple(
            Embedding(
                chunk_id=chunk.chunk_id,
                vector=(0.1, 0.2),
                model=self.model_name,
                dimensions=2,
            )
            for chunk in chunks
        )

    async def embed_query(self, query: str) -> Sequence[float]:
        return (0.1, 0.2)


class FakeDocuments:
    def __init__(self, duplicate: Document | None = None) -> None:
        self.duplicate = duplicate
        self.saved: tuple[Document, Sequence[Chunk]] | None = None
        self.status: DocumentStatus | None = None

    async def find_by_hash(self, collection_id: UUID, content_hash: str) -> Document | None:
        return self.duplicate

    async def save(self, document: Document, chunks: Sequence[Chunk]) -> None:
        self.saved = (document, chunks)

    async def set_status(self, document_id: UUID, status: DocumentStatus) -> None:
        self.status = status

    async def delete(self, document_id: UUID) -> None:
        return None


class FakeVectorIndex:
    def __init__(self) -> None:
        self.chunks: Sequence[Chunk] = ()

    async def upsert(self, chunks: Sequence[Chunk], embeddings: Sequence[Embedding]) -> None:
        assert len(chunks) == len(embeddings)
        self.chunks = chunks

    async def delete(self, chunk_ids: Sequence[UUID]) -> None:
        return None


class FakeSparseIndex:
    async def upsert(self, chunks: Sequence[Chunk]) -> None:
        return None

    async def delete(self, chunks: Sequence[Chunk]) -> None:
        return None


@pytest.mark.asyncio
async def test_langchain_ingestion_uses_document_loader_and_recursive_splitter() -> None:
    validator = PdfUploadValidator(max_size_bytes=1_000_000)
    documents = FakeDocuments()
    vectors = FakeVectorIndex()
    pipeline = LangChainIngestionPipeline(
        validator=validator,
        parser=PyMuPDFParser(validator, max_pages=10),
        embedding_provider=FakeEmbeddings(),
        document_repository=documents,
        vector_indexer=vectors,
        sparse_indexer=FakeSparseIndex(),
        chunk_size=180,
        chunk_overlap=20,
    )

    result = await pipeline.ingest(
        DocumentInput(file_name="study.pdf", content=make_pdf(), collection_id=uuid4())
    )

    assert result.chunking_strategy == "langchain-recursive-character"
    assert result.chunk_count > 1
    assert documents.saved is not None
    assert documents.status is DocumentStatus.READY
    assert all(chunk.text_span is not None for chunk in vectors.chunks)
    assert all(chunk.metadata.section_heading == "METHODS" for chunk in vectors.chunks)
