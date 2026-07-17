"""FastAPI application factory and default application instance."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from apps.api.errors import register_exception_handlers
from apps.api.routes.collections import router as collections_router
from apps.api.routes.documents import router as documents_router
from apps.api.routes.health import router as health_router
from apps.api.routes.pipelines import router as pipelines_router
from apps.api.routes.query import router as query_router
from apps.api.runtime import ApiServices, build_api_services
from raglab.core.config import Settings, get_settings
from raglab.core.health import ReadinessProbe
from raglab.core.logging import RequestLoggingMiddleware, configure_logging


def create_app(
    settings: Settings | None = None,
    readiness_probe: ReadinessProbe | None = None,
    services: ApiServices | None = None,
) -> FastAPI:
    """Build an application with explicit, testable dependencies."""
    app_settings = settings or get_settings()
    configure_logging(app_settings.log_level, json_logs=app_settings.log_json)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        runtime = services or build_api_services(app_settings)
        probe = readiness_probe or runtime.readiness_probe
        application.state.settings = app_settings
        application.state.readiness_probe = probe
        application.state.catalog_repository = runtime.catalog
        application.state.pipeline_registry = runtime.pipelines
        application.state.ingestion_job_manager = runtime.ingestion_jobs
        application.state.document_deletion_manager = runtime.document_deletion
        try:
            await runtime.ingestion_jobs.start()
            yield
        finally:
            await runtime.close()
            if readiness_probe is not None and readiness_probe is not runtime.readiness_probe:
                await readiness_probe.close()

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
    register_exception_handlers(application)
    application.include_router(health_router)
    application.include_router(collections_router)
    application.include_router(documents_router)
    application.include_router(pipelines_router)
    application.include_router(query_router)
    return application


app = create_app()
