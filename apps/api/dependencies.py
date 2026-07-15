"""FastAPI dependencies shared by API routes."""

from typing import cast

from fastapi import Request

from raglab.core.health import ReadinessProbe


def get_readiness_probe(request: Request) -> ReadinessProbe:
    """Return the application-scoped infrastructure probe."""
    return cast(ReadinessProbe, request.app.state.readiness_probe)
