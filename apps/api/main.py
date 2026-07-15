"""FastAPI application factory and default application instance."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from apps.api.routes.health import router as health_router
from raglab.core.config import Settings, get_settings
from raglab.core.health import InfrastructureReadinessProbe, ReadinessProbe
from raglab.core.logging import RequestLoggingMiddleware, configure_logging


def create_app(
    settings: Settings | None = None,
    readiness_probe: ReadinessProbe | None = None,
) -> FastAPI:
    """Build an application with explicit, testable dependencies."""
    app_settings = settings or get_settings()
    configure_logging(app_settings.log_level, json_logs=app_settings.log_json)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        probe = readiness_probe or InfrastructureReadinessProbe.from_settings(app_settings)
        application.state.readiness_probe = probe
        yield
        await probe.close()

    application = FastAPI(
        title=app_settings.app_name,
        version="0.1.0",
        description="A shared API for benchmarking retrieval-augmented generation pipelines.",
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=app_settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.add_middleware(RequestLoggingMiddleware)
    application.include_router(health_router)
    return application


app = create_app()
