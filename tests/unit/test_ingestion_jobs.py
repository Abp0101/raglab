import asyncio
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest

from raglab.core.exceptions import DocumentValidationError, IngestionJobNotFoundError
from raglab.core.schemas import (
    CursorPage,
    DocumentInput,
    IngestionJob,
    IngestionJobClaim,
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
        self.owners: dict[UUID, UUID] = {}
        self.renew_count = 0
        self._lock = asyncio.Lock()

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

    async def list_for_collection(
        self,
        collection_id: UUID,
        *,
        limit: int = 20,
        cursor: str | None = None,
    ) -> CursorPage[IngestionJob]:
        if cursor is not None:
            raise AssertionError("cursor traversal is covered by pagination tests")
        items = [job for job in self.jobs.values() if job.collection_id == collection_id]
        return CursorPage(items=tuple(items[:limit]))

    async def claim_next(
        self,
        owner_id: UUID,
        lease_duration: timedelta,
    ) -> IngestionJobClaim | None:
        async with self._lock:
            now = datetime.now(UTC)
            eligible = [
                job
                for job in self.jobs.values()
                if job.status is IngestionJobStatus.QUEUED
                or (
                    job.status is IngestionJobStatus.PROCESSING
                    and (job.lease_expires_at is None or job.lease_expires_at <= now)
                )
            ]
            if not eligible:
                return None
            job = min(eligible, key=lambda item: (item.created_at, item.job_id))
            expires = now + lease_duration
            claimed = job.model_copy(
                update={
                    "status": IngestionJobStatus.PROCESSING,
                    "attempt_count": job.attempt_count + 1,
                    "lease_expires_at": expires,
                    "updated_at": now,
                }
            )
            self.jobs[job.job_id] = claimed
            self.owners[job.job_id] = owner_id
            return IngestionJobClaim(
                job_id=job.job_id,
                document=self.documents[job.job_id],
                attempt_count=claimed.attempt_count,
                lease_expires_at=expires,
            )

    async def renew(
        self,
        job_id: UUID,
        owner_id: UUID,
        lease_duration: timedelta,
    ) -> bool:
        async with self._lock:
            if not self._owns_live_lease(job_id, owner_id):
                return False
            self.renew_count += 1
            now = datetime.now(UTC)
            self.jobs[job_id] = self.jobs[job_id].model_copy(
                update={"lease_expires_at": now + lease_duration, "updated_at": now}
            )
            return True

    async def complete(
        self,
        job_id: UUID,
        owner_id: UUID,
        result: IngestionResult,
    ) -> bool:
        if not self._owns_live_lease(job_id, owner_id):
            return False
        self.jobs[job_id] = self.jobs[job_id].model_copy(
            update={
                "status": IngestionJobStatus.COMPLETED,
                "result": result,
                "lease_expires_at": None,
            }
        )
        self.owners.pop(job_id, None)
        return True

    async def fail(
        self,
        job_id: UUID,
        owner_id: UUID,
        error_type: str,
        message: str,
    ) -> bool:
        if not self._owns_live_lease(job_id, owner_id):
            return False
        self.jobs[job_id] = self.jobs[job_id].model_copy(
            update={
                "status": IngestionJobStatus.FAILED,
                "error": IngestionJobError(type=error_type, message=message),
                "lease_expires_at": None,
            }
        )
        self.owners.pop(job_id, None)
        return True

    async def release(self, job_id: UUID, owner_id: UUID) -> bool:
        if self.owners.get(job_id) != owner_id:
            return False
        self.jobs[job_id] = self.jobs[job_id].model_copy(
            update={
                "status": IngestionJobStatus.QUEUED,
                "lease_expires_at": None,
            }
        )
        self.owners.pop(job_id, None)
        return True

    def expire(self, job_id: UUID) -> None:
        self.jobs[job_id] = self.jobs[job_id].model_copy(
            update={"lease_expires_at": datetime.now(UTC) - timedelta(seconds=1)}
        )

    def _owns_live_lease(self, job_id: UUID, owner_id: UUID) -> bool:
        job = self.jobs[job_id]
        return (
            self.owners.get(job_id) == owner_id
            and job.status is IngestionJobStatus.PROCESSING
            and job.lease_expires_at is not None
            and job.lease_expires_at > datetime.now(UTC)
        )


class StubPipeline:
    def __init__(self, *, fail: bool = False, block: bool = False) -> None:
        self.fail = fail
        self.block = block
        self.started = asyncio.Event()
        self.finish = asyncio.Event()
        self.calls = 0

    @property
    def config(self) -> PipelineConfig:
        return PipelineConfig()

    @property
    def capabilities(self) -> PipelineCapabilities:
        return PipelineCapabilities()

    async def ingest(self, documents: Sequence[DocumentInput]) -> Sequence[IngestionResult]:
        self.calls += 1
        self.started.set()
        if self.block:
            await self.finish.wait()
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


async def wait_for_status(
    manager: BackgroundIngestionManager,
    job_id: UUID,
    status: IngestionJobStatus,
) -> IngestionJob:
    for _ in range(100):
        job = await manager.get(job_id)
        if job.status is status:
            return job
        await asyncio.sleep(0.001)
    raise AssertionError(f"job did not reach {status}")


@pytest.mark.asyncio
async def test_background_manager_completes_persisted_job() -> None:
    repository = MemoryJobRepository()
    manager = BackgroundIngestionManager(repository, StubPipeline(), poll_seconds=0.01)

    job = await manager.submit(document())
    completed = await wait_for_status(manager, job.job_id, IngestionJobStatus.COMPLETED)

    assert completed.result is not None
    assert completed.attempt_count == 1
    assert completed.lease_expires_at is None
    await manager.close()


@pytest.mark.asyncio
async def test_background_manager_retains_safe_expected_failure() -> None:
    repository = MemoryJobRepository()
    manager = BackgroundIngestionManager(
        repository,
        StubPipeline(fail=True),
        poll_seconds=0.01,
    )

    job = await manager.submit(document())
    failed = await wait_for_status(manager, job.job_id, IngestionJobStatus.FAILED)

    assert failed.error == IngestionJobError(
        type="DocumentValidation",
        message="invalid test PDF",
    )
    await manager.close()


@pytest.mark.asyncio
async def test_background_manager_releases_interrupted_job() -> None:
    repository = MemoryJobRepository()
    pipeline = StubPipeline(block=True)
    manager = BackgroundIngestionManager(repository, pipeline, poll_seconds=0.01)

    job = await manager.submit(document())
    await pipeline.started.wait()
    await manager.close()

    released = await repository.get(job.job_id)
    assert released.status is IngestionJobStatus.QUEUED
    assert released.lease_expires_at is None


@pytest.mark.asyncio
async def test_background_manager_renews_long_running_lease() -> None:
    repository = MemoryJobRepository()
    pipeline = StubPipeline(block=True)
    manager = BackgroundIngestionManager(repository, pipeline, poll_seconds=0.01)
    manager._heartbeat_seconds = 0.01

    job = await manager.submit(document())
    await pipeline.started.wait()
    for _ in range(100):
        if repository.renew_count:
            break
        await asyncio.sleep(0.002)
    pipeline.finish.set()
    await wait_for_status(manager, job.job_id, IngestionJobStatus.COMPLETED)

    assert repository.renew_count >= 1
    await manager.close()


@pytest.mark.asyncio
async def test_two_managers_process_one_job_only_once() -> None:
    repository = MemoryJobRepository()
    first_pipeline = StubPipeline()
    second_pipeline = StubPipeline()
    first = BackgroundIngestionManager(repository, first_pipeline, poll_seconds=0.01)
    second = BackgroundIngestionManager(repository, second_pipeline, poll_seconds=0.01)
    await second.start()

    job = await first.submit(document())
    await wait_for_status(first, job.job_id, IngestionJobStatus.COMPLETED)

    assert first_pipeline.calls + second_pipeline.calls == 1
    await first.close()
    await second.close()


@pytest.mark.asyncio
async def test_expired_lease_is_reclaimed_and_stale_owner_cannot_finish() -> None:
    repository = MemoryJobRepository()
    job = await repository.create(document())
    first_owner = uuid4()
    second_owner = uuid4()
    first = await repository.claim_next(first_owner, timedelta(seconds=30))
    assert first is not None
    repository.expire(job.job_id)

    second = await repository.claim_next(second_owner, timedelta(seconds=30))

    assert second is not None
    assert second.job_id == job.job_id
    assert second.attempt_count == 2
    result = IngestionResult(
        document_id=uuid4(),
        collection_id=job.collection_id,
        page_count=1,
        chunk_count=1,
        duration_ms=1,
        parser="stub",
        chunking_strategy="stub",
        embedding_model="local",
    )
    assert await repository.complete(job.job_id, first_owner, result) is False
    assert await repository.complete(job.job_id, second_owner, result) is True
