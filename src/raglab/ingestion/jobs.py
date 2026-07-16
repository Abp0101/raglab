"""Recoverable in-process runner backed by a persistent ingestion job repository."""

import asyncio
from uuid import UUID

from raglab.core.exceptions import RAGLabError
from raglab.core.interfaces import IngestionJobRepository, RAGPipeline
from raglab.core.schemas import DocumentInput, IngestionJob


class BackgroundIngestionManager:
    """Run durable jobs locally while retaining safe status across process restarts."""

    def __init__(
        self,
        repository: IngestionJobRepository,
        pipeline: RAGPipeline,
        *,
        max_concurrency: int = 1,
    ) -> None:
        self._repository = repository
        self._pipeline = pipeline
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._tasks: dict[UUID, asyncio.Task[None]] = {}
        self._closing = False

    async def start(self) -> None:
        """Resume queued uploads and jobs interrupted by the previous process."""
        for job_id in await self._repository.list_recoverable():
            self._schedule(job_id)

    async def submit(self, document: DocumentInput) -> IngestionJob:
        """Persist the upload before scheduling any background work."""
        if self._closing:
            raise RuntimeError("background ingestion manager is stopping")
        job = await self._repository.create(document)
        self._schedule(job.job_id)
        return job

    async def get(self, job_id: UUID) -> IngestionJob:
        return await self._repository.get(job_id)

    async def close(self) -> None:
        """Cancel active tasks; interrupted records are requeued for the next process."""
        self._closing = True
        tasks = tuple(self._tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()

    def _schedule(self, job_id: UUID) -> None:
        existing = self._tasks.get(job_id)
        if existing is not None and not existing.done():
            return
        task = asyncio.create_task(self._run(job_id), name=f"ingestion-{job_id}")
        self._tasks[job_id] = task
        task.add_done_callback(lambda _: self._tasks.pop(job_id, None))

    async def _run(self, job_id: UUID) -> None:
        async with self._semaphore:
            document = await self._repository.claim(job_id)
            if document is None:
                return
            try:
                results = await self._pipeline.ingest((document,))
            except asyncio.CancelledError:
                await self._repository.requeue(job_id)
                raise
            except RAGLabError as error:
                await self._repository.fail(job_id, _error_type(error), str(error))
            except Exception:
                await self._repository.fail(
                    job_id,
                    "Internal",
                    "background ingestion failed",
                )
            else:
                await self._repository.complete(job_id, results[0])


def _error_type(error: Exception) -> str:
    return type(error).__name__.removesuffix("Error")
