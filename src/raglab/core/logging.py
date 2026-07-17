"""Structured application logging and request correlation."""

import json
import logging
import re
import time
from contextvars import ContextVar, Token
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from raglab.core.metrics import LocalMetrics

request_id_context: ContextVar[str | None] = ContextVar("request_id", default=None)
_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


class JsonFormatter(logging.Formatter):
    """Emit one machine-readable JSON object per log record."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if request_id := request_id_context.get():
            payload["request_id"] = request_id
        for field in (
            "method",
            "path",
            "route",
            "status_code",
            "status_class",
            "duration_ms",
            "operation",
            "error_type",
            "dependency",
            "job_id",
            "outcome",
            "attempt_count",
        ):
            if hasattr(record, field):
                payload[field] = getattr(record, field)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str, *, json_logs: bool = True) -> None:
    """Configure process logging with deterministic handlers."""
    handler = logging.StreamHandler()
    handler.setFormatter(
        JsonFormatter() if json_logs else logging.Formatter("%(levelname)s %(message)s")
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Attach a request ID and log request completion without body content."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        supplied_request_id = request.headers.get("X-Request-ID", "")
        request_id = (
            supplied_request_id
            if _REQUEST_ID_PATTERN.fullmatch(supplied_request_id)
            else str(uuid4())
        )
        token: Token[str | None] = request_id_context.set(request_id)
        started = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            duration_seconds = time.perf_counter() - started
            route = _route_template(request)
            logging.getLogger("raglab.request").info(
                "request_completed",
                extra={
                    "method": request.method,
                    "route": route,
                    "status_code": status_code,
                    "status_class": f"{status_code // 100}xx",
                    "duration_ms": round(duration_seconds * 1_000, 2),
                },
            )
            metrics = getattr(request.app.state, "metrics", None)
            if isinstance(metrics, LocalMetrics):
                metrics.observe_http(request.method, route, status_code, duration_seconds)
            request_id_context.reset(token)


def _route_template(request: Request) -> str:
    route = request.scope.get("route")
    path = getattr(route, "path", None)
    return path if isinstance(path, str) else "unmatched"
