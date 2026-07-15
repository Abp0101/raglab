"""Foundation health endpoint models."""

from typing import Literal

from raglab.core.schemas.common import RAGLabModel


class HealthResponse(RAGLabModel):
    """Process liveness response."""

    status: Literal["ok"]


class ReadinessResponse(RAGLabModel):
    """Aggregate backing-service readiness response."""

    status: Literal["ok", "degraded"]
    dependencies: dict[str, bool]
