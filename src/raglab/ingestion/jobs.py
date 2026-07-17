"""Distributed-safe ingestion workers backed by PostgreSQL job leases."""

import asyncio
import logging
from collections.abc import Sequence
from datetime import timedelta
from uuid import UUID, uuid4

from raglab.core.exceptions import RAGLabError
from raglab.core.interfaces import IngestionJobRepository, RAGPipeline
from raglab.core.metrics import LocalMetrics
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
        metrics: LocalMetrics | None = None,
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
        self._metrics = metrics
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
        self._observe_job("queued")
        logger.info(
            "ingestion_job_queued",
            extra={"job_id": str(job.job_id), "outcome": "queued"},
        )
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
            except Exception as error:
                logger.error(
                    "ingestion_job_claim_failed",
                    extra={"error_type": type(error).__name__},
                )
                self._observe_error(type(error).__name__)
                await asyncio.sleep(self._poll_seconds)
                continue
            if claim is not None:
                self._observe_job("claimed")
                logger.info(
                    "ingestion_job_claimed",
                    extra={
                        "job_id": str(claim.job_id),
                        "outcome": "claimed",
                        "attempt_count": claim.attempt_count,
                    },
                )
                try:
                    await self._process(claim)
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    logger.error(
                        "ingestion_job_coordination_failed",
                        extra={
                            "job_id": str(claim.job_id),
                            "error_type": type(error).__name__,
                        },
                    )
                    self._observe_error(type(error).__name__)
                    self._observe_job("coordination_failed")
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
            released = await self._repository.release(claim.job_id, self._owner_id)
            outcome = "released" if released else "lease_lost"
            self._observe_job(outcome)
            logger.info(
                "ingestion_job_interrupted",
                extra={"job_id": str(claim.job_id), "outcome": outcome},
            )
            if self._closing:
                raise
        except RAGLabError as error:
            await _stop_task(heartbeat)
            failed = await self._repository.fail(
                claim.job_id,
                self._owner_id,
                _error_type(error),
                str(error),
            )
            self._observe_error(_error_type(error))
            outcome = "failed" if failed else "lease_lost"
            self._observe_job(outcome)
            logger.warning(
                "ingestion_job_failed",
                extra={
                    "job_id": str(claim.job_id),
                    "error_type": _error_type(error),
                    "outcome": outcome,
                },
            )
        except Exception as error:
            await _stop_task(heartbeat)
            error_type = type(error).__name__
            failed = await self._repository.fail(
                claim.job_id,
                self._owner_id,
                "Internal",
                "background ingestion failed",
            )
            self._observe_error(error_type)
            outcome = "failed" if failed else "lease_lost"
            self._observe_job(outcome)
            logger.error(
                "ingestion_job_failed",
                extra={
                    "job_id": str(claim.job_id),
                    "error_type": error_type,
                    "outcome": outcome,
                },
            )
        else:
            await _stop_task(heartbeat)
            completed = await self._repository.complete(
                claim.job_id,
                self._owner_id,
                results[0],
            )
            outcome = "completed" if completed else "lease_lost"
            self._observe_job(outcome)
            logger.info(
                "ingestion_job_finished",
                extra={"job_id": str(claim.job_id), "outcome": outcome},
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
            except Exception as error:
                logger.error(
                    "ingestion_lease_renewal_failed",
                    extra={"job_id": str(job_id), "error_type": type(error).__name__},
                )
                self._observe_error(type(error).__name__)
                ingestion.cancel()
                return
            if not renewed:
                logger.warning("ingestion lease lost", extra={"job_id": str(job_id)})
                ingestion.cancel()
                return

    def _observe_error(self, error_type: str) -> None:
        if self._metrics is not None:
            self._metrics.observe_error("ingestion_job", error_type)

    def _observe_job(self, outcome: str) -> None:
        if self._metrics is not None:
            self._metrics.observe_ingestion_job(outcome)


async def _stop_task(task: asyncio.Task[None]) -> None:
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


def _error_type(error: Exception) -> str:
    return type(error).__name__.removesuffix("Error")
