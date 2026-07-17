import pytest
from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis

from raglab.core.config import Settings
from raglab.core.health import InfrastructureReadinessProbe
from raglab.core.metrics import LocalMetrics
from raglab.database.session import create_engine

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_readiness_isolated_dependency_failure_without_hiding_healthy_services() -> None:
    settings = Settings(_env_file=None)
    metrics = LocalMetrics()
    probe = InfrastructureReadinessProbe(
        database=create_engine("postgresql+asyncpg://raglab:raglab@127.0.0.1:1/raglab"),
        qdrant=AsyncQdrantClient(
            url=str(settings.qdrant_url),
            check_compatibility=False,
        ),
        redis=Redis.from_url(str(settings.redis_dsn), decode_responses=True),
        timeout_seconds=0.5,
        metrics=metrics,
    )

    try:
        dependencies = await probe.check()
    finally:
        await probe.close()

    assert dependencies == {"postgres": False, "qdrant": True, "redis": True}
    rendered = metrics.render_prometheus()
    assert 'raglab_dependency_up{dependency="postgres"} 0' in rendered
    assert 'raglab_dependency_up{dependency="qdrant"} 1' in rendered
    assert 'raglab_dependency_up{dependency="redis"} 1' in rendered
    assert 'raglab_errors_total{operation="readiness",' in rendered
