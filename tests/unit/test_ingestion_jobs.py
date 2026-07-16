import asyncio
from collections.abc import Sequence
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

from raglab.core.exceptions import DocumentValidationError, IngestionJobNotFoundError
from raglab.core.schemas import (
    DocumentInput,
    IngestionJob,
    IngestionJobError,
    IngestionJobStatus,
    IngestionResult,
    PipelineCapabilities,
    PipelineConfig,
    QueryRequest,
    RAGResponse,
)
from raglab.ingestion.jobs import BackgroundIngestionManager


class MemoryJobRepository:
    def __init__(self) -> None:
        self.documents: dict[UUID, DocumentInput] = {}
        self.jobs: dict[UUID, IngestionJob] = {}
        self.recoverable: tuple[UUID, ...] = ()

    async def create(self, document: DocumentInput) -> IngestionJob:
        now = datetime.now(UTC)
        job = IngestionJob(
            job_id=uuid4(),
            collection_id=document.collection_id,
            file_name=document.file_name,
            status=IngestionJobStatus.QUEUED,
            created_at=now,
            updated_at=now,
        )
        self.documents[job.job_id] = document
        self.jobs[job.job_id] = job
        return job

    async def get(self, job_id: UUID) -> IngestionJob:
        try:
            return self.jobs[job_id]
        except KeyError as error:
            raise IngestionJobNotFoundError from error

    async def list_recoverable(self) -> Sequence[UUID]:
        return self.recoverable

    async def claim(self, job_id: UUID) -> DocumentInput | None:
        job = self.jobs.get(job_id)
        if job is None or job.status is not IngestionJobStatus.QUEUED:
            return None
        self.jobs[job_id] = job.model_copy(update={"status": IngestionJobStatus.PROCESSING})
        return self.documents[job_id]

    async def complete(self, job_id: UUID, result: IngestionResult) -> None:
        self.jobs[job_id] = self.jobs[job_id].model_copy(
            update={"status": IngestionJobStatus.COMPLETED, "result": result}
        )

    async def fail(self, job_id: UUID, error_type: str, message: str) -> None:
        self.jobs[job_id] = self.jobs[job_id].model_copy(
            update={
                "status": IngestionJobStatus.FAILED,
                "error": IngestionJobError(type=error_type, message=message),
            }
        )

    async def requeue(self, job_id: UUID) -> None:
        self.jobs[job_id] = self.jobs[job_id].model_copy(
            update={"status": IngestionJobStatus.QUEUED}
        )


class StubPipeline:
    def __init__(self, *, fail: bool = False, block: bool = False) -> None:
        self.fail = fail
        self.block = block
        self.started = asyncio.Event()

    @property
    def config(self) -> PipelineConfig:
        return PipelineConfig()

    @property
    def capabilities(self) -> PipelineCapabilities:
        return PipelineCapabilities()

    async def ingest(self, documents: Sequence[DocumentInput]) -> Sequence[IngestionResult]:
        self.started.set()
        if self.block:
            await asyncio.Event().wait()
        if self.fail:
            raise DocumentValidationError("invalid test PDF")
        document = documents[0]
        return (
            IngestionResult(
                document_id=uuid4(),
                collection_id=document.collection_id,
                page_count=1,
                chunk_count=1,
                duration_ms=1,
                parser="stub",
                chunking_strategy="stub",
                embedding_model="local",
            ),
        )

    async def query(self, request: QueryRequest) -> RAGResponse:
        raise NotImplementedError


def document() -> DocumentInput:
    return DocumentInput(
        file_name="paper.pdf",
        content=b"%PDF-test",
        collection_id=uuid4(),
    )


@pytest.mark.asyncio
async def test_background_manager_completes_persisted_job() -> None:
    repository = MemoryJobRepository()
    manager = BackgroundIngestionManager(repository, StubPipeline())

    job = await manager.submit(document())
    for _ in range(20):
        if (await manager.get(job.job_id)).status is IngestionJobStatus.COMPLETED:
            break
        await asyncio.sleep(0)

    completed = await manager.get(job.job_id)
    assert completed.status is IngestionJobStatus.COMPLETED
    assert completed.result is not None
    await manager.close()


@pytest.mark.asyncio
async def test_background_manager_retains_safe_expected_failure() -> None:
    repository = MemoryJobRepository()
    manager = BackgroundIngestionManager(repository, StubPipeline(fail=True))

    job = await manager.submit(document())
    for _ in range(20):
        if (await manager.get(job.job_id)).status is IngestionJobStatus.FAILED:
            break
        await asyncio.sleep(0)

    failed = await manager.get(job.job_id)
    assert failed.error == IngestionJobError(type="DocumentValidation", message="invalid test PDF")
    await manager.close()


@pytest.mark.asyncio
async def test_background_manager_requeues_interrupted_job() -> None:
    repository = MemoryJobRepository()
    pipeline = StubPipeline(block=True)
    manager = BackgroundIngestionManager(repository, pipeline)

    job = await manager.submit(document())
    await pipeline.started.wait()
    await manager.close()

    assert (await repository.get(job.job_id)).status is IngestionJobStatus.QUEUED
