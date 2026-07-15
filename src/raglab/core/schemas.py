"""Shared API boundary models for foundation endpoints."""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class HealthResponse(BaseModel):
    """Process liveness response."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"]


class ReadinessResponse(BaseModel):
    """Aggregate backing-service readiness response."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "degraded"]
    dependencies: dict[str, bool]
