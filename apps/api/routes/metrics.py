"""Local Prometheus-compatible metrics exposition."""

from typing import Annotated

from fastapi import APIRouter, Depends, Response

from apps.api.dependencies import get_metrics
from raglab.core.metrics import LocalMetrics

router = APIRouter(tags=["observability"])


@router.get(
    "/metrics",
    response_class=Response,
    responses={200: {"content": {"text/plain": {}}}},
)
async def metrics(registry: Annotated[LocalMetrics, Depends(get_metrics)]) -> Response:
    """Expose bounded process-local metrics without document or query content."""
    return Response(
        registry.render_prometheus(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
