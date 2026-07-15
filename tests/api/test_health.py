from collections.abc import Mapping

from apps.api.main import create_app
from fastapi.testclient import TestClient

from raglab.core.config import Settings


class StubReadinessProbe:
    def __init__(self, dependencies: Mapping[str, bool]) -> None:
        self.dependencies = dependencies
        self.closed = False

    async def check(self) -> Mapping[str, bool]:
        return self.dependencies

    async def close(self) -> None:
        self.closed = True


def make_client(probe: StubReadinessProbe) -> TestClient:
    settings = Settings(environment="test", _env_file=None)
    return TestClient(create_app(settings=settings, readiness_probe=probe))


def test_liveness_does_not_require_infrastructure() -> None:
    probe = StubReadinessProbe({"postgres": False, "qdrant": False, "redis": False})

    with make_client(probe) as client:
        response = client.get("/health/live", headers={"X-Request-ID": "test-request"})

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert response.headers["X-Request-ID"] == "test-request"
    assert probe.closed is True


def test_readiness_is_ok_when_dependencies_are_healthy() -> None:
    probe = StubReadinessProbe({"postgres": True, "qdrant": True, "redis": True})

    with make_client(probe) as client:
        response = client.get("/health/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "dependencies": {"postgres": True, "qdrant": True, "redis": True},
    }


def test_readiness_is_degraded_when_a_dependency_is_unavailable() -> None:
    probe = StubReadinessProbe({"postgres": True, "qdrant": False, "redis": True})

    with make_client(probe) as client:
        response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "degraded"
    assert response.json()["dependencies"]["qdrant"] is False
