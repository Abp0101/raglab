"""Infrastructure health contracts and concrete readiness checks."""

import asyncio
from collections.abc import Awaitable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from raglab.core.config import Settings


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

    @classmethod
    def from_settings(cls, settings: Settings) -> "InfrastructureReadinessProbe":
        return cls(
            database=create_async_engine(settings.postgres_dsn, pool_pre_ping=True),
            qdrant=AsyncQdrantClient(
                url=str(settings.qdrant_url),
                api_key=settings.qdrant_api_key,
                timeout=2,
            ),
            redis=Redis.from_url(str(settings.redis_dsn), decode_responses=True),
        )

    async def check(self) -> Mapping[str, bool]:
        results = await asyncio.gather(
            self._bounded(self._check_database()),
            self._bounded(self._check_qdrant()),
            self._bounded(self._check_redis()),
        )
        return dict(zip(("postgres", "qdrant", "redis"), results, strict=True))

    async def close(self) -> None:
        await self.database.dispose()
        await self.qdrant.close()
        await self.redis.aclose()

    async def _bounded(self, operation: Awaitable[Any]) -> bool:
        try:
            async with asyncio.timeout(self.timeout_seconds):
                await operation
        except Exception:  # Readiness must aggregate failures from every dependency.
            return False
        return True

    async def _check_database(self) -> None:
        async with self.database.connect() as connection:
            await connection.execute(text("SELECT 1"))

    async def _check_qdrant(self) -> None:
        await self.qdrant.get_collections()

    async def _check_redis(self) -> None:
        await self.redis.ping()
