from collections.abc import Sequence
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

from raglab.core.exceptions import ProviderUnavailableError
from raglab.core.schemas import Chunk, Document, DocumentMetadata, DocumentStatus
from raglab.ingestion.deletion import CoordinatedDocumentDeletionService


def make_document_and_chunk() -> tuple[Document, Chunk]:
    document = Document(
        document_id=uuid4(),
        collection_id=uuid4(),
        file_name="deletion.pdf",
        display_title="Deletion",
        uploaded_at=datetime.now(UTC),
        file_type="application/pdf",
        content_hash="d" * 64,
        status=DocumentStatus.READY,
    )
    chunk = Chunk(
        chunk_id=uuid4(),
        text="Locally indexed evidence.",
        metadata=DocumentMetadata(
            document_id=document.document_id,
            collection_id=document.collection_id,
            file_name=document.file_name,
            display_title=document.display_title,
            uploaded_at=document.uploaded_at,
            file_type=document.file_type,
            chunk_index=0,
            content_hash=document.content_hash,
        ),
    )
    return document, chunk


class RecordingDocuments:
    def __init__(self, document: Document, calls: list[str]) -> None:
        self.document = document
        self.calls = calls

    async def mark_deleting(self, document_id: UUID) -> Document:
        assert document_id == self.document.document_id
        self.calls.append("mark")
        return self.document.model_copy(update={"status": DocumentStatus.DELETING})

    async def delete(self, document_id: UUID) -> None:
        assert document_id == self.document.document_id
        self.calls.append("postgres")


class RecordingChunks:
    def __init__(self, chunks: Sequence[Chunk], calls: list[str]) -> None:
        self.chunks = chunks
        self.calls = calls

    async def get_by_document(self, document_id: UUID) -> Sequence[Chunk]:
        self.calls.append("load")
        return self.chunks


class RecordingVectors:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls
        self.ids: tuple[UUID, ...] = ()

    async def delete(self, chunk_ids: Sequence[UUID]) -> None:
        self.calls.append("qdrant")
        self.ids = tuple(chunk_ids)


class RecordingSparse:
    def __init__(self, calls: list[str], *, fail: bool = False) -> None:
        self.calls = calls
        self.fail = fail

    async def delete(self, chunks: Sequence[Chunk]) -> None:
        self.calls.append("redis")
        if self.fail:
            raise RuntimeError("redis unavailable")


def build_service(
    document: Document,
    chunks: Sequence[Chunk],
    calls: list[str],
    *,
    fail_sparse: bool = False,
) -> tuple[CoordinatedDocumentDeletionService, RecordingVectors]:
    vectors = RecordingVectors(calls)
    return (
        CoordinatedDocumentDeletionService(
            document_repository=RecordingDocuments(document, calls),  # type: ignore[arg-type]
            chunk_repository=RecordingChunks(chunks, calls),  # type: ignore[arg-type]
            vector_indexer=vectors,  # type: ignore[arg-type]
            sparse_indexer=RecordingSparse(calls, fail=fail_sparse),  # type: ignore[arg-type]
        ),
        vectors,
    )


@pytest.mark.asyncio
async def test_deletion_clears_indexes_before_postgres_source_data() -> None:
    document, chunk = make_document_and_chunk()
    calls: list[str] = []
    service, vectors = build_service(document, (chunk,), calls)

    result = await service.delete(document.document_id)

    assert calls == ["mark", "load", "qdrant", "redis", "postgres"]
    assert vectors.ids == (chunk.chunk_id,)
    assert result.document_id == document.document_id
    assert result.collection_id == document.collection_id
    assert result.deleted_chunk_count == 1


@pytest.mark.asyncio
async def test_deletion_with_no_chunks_still_removes_postgres_record() -> None:
    document, _ = make_document_and_chunk()
    calls: list[str] = []
    service, vectors = build_service(document, (), calls)

    result = await service.delete(document.document_id)

    assert calls == ["mark", "load", "qdrant", "redis", "postgres"]
    assert vectors.ids == ()
    assert result.deleted_chunk_count == 0


@pytest.mark.asyncio
async def test_external_failure_preserves_postgres_for_retry() -> None:
    document, chunk = make_document_and_chunk()
    calls: list[str] = []
    service, _ = build_service(document, (chunk,), calls, fail_sparse=True)

    with pytest.raises(ProviderUnavailableError, match="document deletion could not complete"):
        await service.delete(document.document_id)

    assert calls == ["mark", "load", "qdrant", "redis"]
