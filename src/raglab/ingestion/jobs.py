"""Distributed-safe ingestion workers backed by PostgreSQL job leases."""

import asyncio
import logging
from collections.abc import Sequence
from datetime import timedelta
from uuid import UUID, uuid4

from raglab.core.exceptions import RAGLabError
from raglab.core.interfaces import IngestionJobRepository, RAGPipeline
from raglab.core.schemas import (
    CursorPage,
    DocumentInput,
    IngestionJob,
    IngestionJobClaim,
    IngestionResult,
)

logger = logging.getLogger(__name__)


class BackgroundIngestionManager:
    """Poll and process owner-bound leases safely across API processes."""

    def __init__(
        self,
        repository: IngestionJobRepository,
        pipeline: RAGPipeline,
        *,
        max_concurrency: int = 1,
        lease_seconds: float = 60,
        poll_seconds: float = 1,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be positive")
        if lease_seconds < 3:
            raise ValueError("lease_seconds must be at least 3")
        if poll_seconds <= 0:
            raise ValueError("poll_seconds must be positive")
        self._repository = repository
        self._pipeline = pipeline
        self._max_concurrency = max_concurrency
        self._lease_duration = timedelta(seconds=lease_seconds)
        self._heartbeat_seconds = lease_seconds / 3
        self._poll_seconds = poll_seconds
        self._owner_id = uuid4()
        self._workers: set[asyncio.Task[None]] = set()
        self._wake = asyncio.Event()
        self._closing = False

    async def start(self) -> None:
        """Start local pollers; PostgreSQL arbitrates ownership across processes."""
        if self._closing:
            raise RuntimeError("background ingestion manager is stopping")
        if self._workers:
            return
        for index in range(self._max_concurrency):
            task = asyncio.create_task(
                self._worker(),
                name=f"ingestion-worker-{self._owner_id}-{index}",
            )
            self._workers.add(task)
            task.add_done_callback(self._workers.discard)

    async def submit(self, document: DocumentInput) -> IngestionJob:
        """Persist the upload and wake pollers without assigning local ownership."""
        if self._closing:
            raise RuntimeError("background ingestion manager is stopping")
        if not self._workers:
            await self.start()
        job = await self._repository.create(document)
        self._wake.set()
        return job

    async def get(self, job_id: UUID) -> IngestionJob:
        return await self._repository.get(job_id)

    async def list_for_collection(
        self,
        collection_id: UUID,
        *,
        limit: int = 20,
        cursor: str | None = None,
    ) -> CursorPage[IngestionJob]:
        return await self._repository.list_for_collection(
            collection_id,
            limit=limit,
            cursor=cursor,
        )

    async def close(self) -> None:
        """Stop pollers and release active leases for immediate reassignment."""
        self._closing = True
        self._wake.set()
        workers = tuple(self._workers)
        for worker in workers:
            worker.cancel()
        if workers:
            await asyncio.gather(*workers, return_exceptions=True)
        self._workers.clear()

    async def _worker(self) -> None:
        while not self._closing:
            self._wake.clear()
            try:
                claim = await self._repository.claim_next(
                    self._owner_id,
                    self._lease_duration,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("ingestion job claim failed")
                await asyncio.sleep(self._poll_seconds)
                continue
            if claim is not None:
                try:
                    await self._process(claim)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception(
                        "ingestion job processing coordination failed",
                        extra={"job_id": str(claim.job_id)},
                    )
                continue
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=self._poll_seconds)
            except TimeoutError:
                continue

    async def _process(self, claim: IngestionJobClaim) -> None:
        ingestion = asyncio.create_task(
            self._pipeline.ingest((claim.document,)),
            name=f"ingestion-run-{claim.job_id}",
        )
        heartbeat = asyncio.create_task(
            self._heartbeat(claim.job_id, ingestion),
            name=f"ingestion-heartbeat-{claim.job_id}",
        )
        try:
            results = await ingestion
        except asyncio.CancelledError:
            await _stop_task(heartbeat)
            await self._repository.release(claim.job_id, self._owner_id)
            if self._closing:
                raise
        except RAGLabError as error:
            await _stop_task(heartbeat)
            await self._repository.fail(
                claim.job_id,
                self._owner_id,
                _error_type(error),
                str(error),
            )
        except Exception:
            await _stop_task(heartbeat)
            await self._repository.fail(
                claim.job_id,
                self._owner_id,
                "Internal",
                "background ingestion failed",
            )
        else:
            await _stop_task(heartbeat)
            await self._repository.complete(
                claim.job_id,
                self._owner_id,
                results[0],
            )

    async def _heartbeat(
        self,
        job_id: UUID,
        ingestion: asyncio.Task[Sequence[IngestionResult]],
    ) -> None:
        while not ingestion.done():
            await asyncio.sleep(self._heartbeat_seconds)
            if ingestion.done():
                return
            try:
                renewed = await self._repository.renew(
                    job_id,
                    self._owner_id,
                    self._lease_duration,
                )
            except Exception:
                logger.exception("ingestion lease renewal failed", extra={"job_id": str(job_id)})
                ingestion.cancel()
                return
            if not renewed:
                logger.warning("ingestion lease lost", extra={"job_id": str(job_id)})
                ingestion.cancel()
                return


async def _stop_task(task: asyncio.Task[None]) -> None:
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


def _error_type(error: Exception) -> str:
    return type(error).__name__.removesuffix("Error")
