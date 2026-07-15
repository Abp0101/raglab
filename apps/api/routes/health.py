"""Liveness and infrastructure readiness endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends, Response, status

from apps.api.dependencies import get_readiness_probe
from raglab.core.health import ReadinessProbe
from raglab.core.schemas import HealthResponse, ReadinessResponse

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/live", response_model=HealthResponse)
async def live() -> HealthResponse:
    """Report whether the API process can serve requests."""
    return HealthResponse(status="ok")


@router.get("/ready", response_model=ReadinessResponse)
async def ready(
    response: Response,
    probe: Annotated[ReadinessProbe, Depends(get_readiness_probe)],
) -> ReadinessResponse:
    """Report whether required infrastructure can accept work."""
    dependencies = await probe.check()
    is_ready = all(dependencies.values())
    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadinessResponse(
        status="ok" if is_ready else "degraded",
        dependencies=dependencies,
    )
