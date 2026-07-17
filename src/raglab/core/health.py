"""Infrastructure health contracts and concrete readiness checks."""

import asyncio
import logging
from collections.abc import Awaitable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from raglab.core.config import Settings
from raglab.core.metrics import LocalMetrics

logger = logging.getLogger(__name__)


class ReadinessProbe(Protocol):
    """Checks and closes resources used to determine API readiness."""

    async def check(self) -> Mapping[str, bool]: ...

    async def close(self) -> None: ...


@dataclass(slots=True)
class InfrastructureReadinessProbe:
    """Check required backing services without leaking connection details."""

    database: AsyncEngine
    qdrant: AsyncQdrantClient
    redis: Redis
    timeout_seconds: float = 2.0
    metrics: LocalMetrics | None = None

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        metrics: LocalMetrics | None = None,
    ) -> "InfrastructureReadinessProbe":
        return cls(
            database=create_async_engine(settings.postgres_dsn, pool_pre_ping=True),
            qdrant=AsyncQdrantClient(
                url=str(settings.qdrant_url),
                api_key=settings.qdrant_api_key,
                timeout=2,
            ),
            redis=Redis.from_url(str(settings.redis_dsn), decode_responses=True),
            metrics=metrics,
        )

    async def check(self) -> Mapping[str, bool]:
        results = await asyncio.gather(
            self._bounded("postgres", self._check_database()),
            self._bounded("qdrant", self._check_qdrant()),
            self._bounded("redis", self._check_redis()),
        )
        return dict(zip(("postgres", "qdrant", "redis"), results, strict=True))

    async def close(self) -> None:
        await self.database.dispose()
        await self.qdrant.close()
        await self.redis.aclose()

    async def _bounded(self, dependency: str, operation: Awaitable[Any]) -> bool:
        try:
            async with asyncio.timeout(self.timeout_seconds):
                await operation
        except Exception as error:  # Readiness must aggregate every dependency failure.
            error_type = type(error).__name__
            logger.warning(
                "readiness_check_failed",
                extra={"dependency": dependency, "error_type": error_type},
            )
            if self.metrics is not None:
                self.metrics.observe_error("readiness", error_type)
                self.metrics.set_dependency(dependency, False)
            return False
        if self.metrics is not None:
            self.metrics.set_dependency(dependency, True)
        return True

    async def _check_database(self) -> None:
        async with self.database.connect() as connection:
            await connection.execute(text("SELECT 1"))

    async def _check_qdrant(self) -> None:
        await self.qdrant.get_collections()

    async def _check_redis(self) -> None:
        await self.redis.ping()
