from collections.abc import Mapping
from uuid import uuid4

import pytest
from apps.api.main import create_app
from apps.api.routes.query import _query_events
from apps.api.runtime import ApiServices
from fastapi import FastAPI
from fastapi.testclient import TestClient

from raglab.core.config import Settings
from raglab.core.exceptions import ProviderUnavailableError
from raglab.core.metrics import LocalMetrics
from raglab.core.schemas import FrameworkName, QueryRequest, RAGResponse
from raglab.pipelines import PipelineRegistry
from raglab.security import ApiKeyAuthenticator


class StubReadinessProbe:
    async def check(self) -> Mapping[str, bool]:
        return {"postgres": True, "qdrant": False, "redis": True}

    async def close(self) -> None:
        return None


class NoopJobManager:
    async def start(self) -> None:
        return None

    async def close(self) -> None:
        return None


class NoopDeletionManager:
    async def delete(self, document_id: object) -> object:
        raise AssertionError(f"unexpected deletion request for {document_id}")


def _app() -> FastAPI:
    services = ApiServices(
        catalog=None,  # type: ignore[arg-type]
        authenticator=ApiKeyAuthenticator(enabled=False, credentials=[]),
        pipelines=PipelineRegistry({}),
        ingestion_jobs=NoopJobManager(),  # type: ignore[arg-type]
        document_deletion=NoopDeletionManager(),  # type: ignore[arg-type]
        readiness_probe=StubReadinessProbe(),
    )
    application = create_app(
        Settings(environment="test", log_json=False, _env_file=None),
        services=services,
    )

    async def expected_failure() -> None:
        raise ProviderUnavailableError("local provider is unavailable")

    async def unexpected_failure() -> None:
        raise RuntimeError("private implementation detail")

    application.add_api_route("/test/expected-failure", expected_failure)
    application.add_api_route("/test/unexpected-failure", unexpected_failure)
    return application


def test_metrics_capture_degraded_readiness_and_safe_failures() -> None:
    with TestClient(_app(), raise_server_exceptions=False) as client:
        readiness = client.get("/health/ready")
        expected = client.get("/test/expected-failure")
        unexpected = client.get("/test/unexpected-failure")
        metrics = client.get("/metrics")

    assert readiness.status_code == 503
    assert expected.status_code == 503
    assert expected.headers["Retry-After"] == "5"
    assert expected.json() == {
        "error": {
            "type": "ProviderUnavailable",
            "message": "local provider is unavailable",
        }
    }
    assert unexpected.status_code == 500
    assert unexpected.json() == {
        "error": {"type": "Internal", "message": "request processing failed"}
    }
    assert "private implementation detail" not in unexpected.text
    assert metrics.status_code == 200
    assert metrics.headers["content-type"].startswith("text/plain")
    assert 'raglab_dependency_up{dependency="qdrant"} 0' in metrics.text
    assert (
        'raglab_errors_total{operation="/test/expected-failure",error_type="ProviderUnavailable"} 1'
    ) in metrics.text
    assert (
        'raglab_errors_total{operation="/test/unexpected-failure",error_type="Internal"} 1'
    ) in metrics.text
    assert 'route="/test/unexpected-failure",status_class="5xx"} 1' in metrics.text


def test_untrusted_request_id_is_replaced_before_logging_or_echoing() -> None:
    supplied = "x" * 129

    with TestClient(_app(), raise_server_exceptions=False) as client:
        response = client.get("/health/live", headers={"X-Request-ID": supplied})

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] != supplied
    assert len(response.headers["X-Request-ID"]) == 36


@pytest.mark.asyncio
async def test_stream_failure_uses_safe_event_and_records_error() -> None:
    metrics = LocalMetrics()
    request = QueryRequest(
        query="local question",
        collection_id=uuid4(),
        framework=FrameworkName.CUSTOM,
    )

    async def fail(_: QueryRequest) -> RAGResponse:
        raise RuntimeError("private stream implementation detail")

    events = [event async for event in _query_events(request, fail, metrics)]
    rendered = b"".join(events).decode()

    assert "event: query.accepted" in rendered
    assert "event: query.error" in rendered
    assert '"type":"Internal"' in rendered
    assert "private stream implementation detail" not in rendered
    assert (
        'raglab_errors_total{operation="query_stream",error_type="Internal"} 1'
        in metrics.render_prometheus()
    )
