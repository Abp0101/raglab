"""FastAPI dependencies shared by API routes."""

from typing import cast

from fastapi import Request

from raglab.core.config import Settings
from raglab.core.health import ReadinessProbe
from raglab.core.interfaces import (
    CatalogRepository,
    DocumentDeletionManager,
    IngestionJobManager,
)
from raglab.pipelines import PipelineRegistry


def get_readiness_probe(request: Request) -> ReadinessProbe:
    """Return the application-scoped infrastructure probe."""
    return cast(ReadinessProbe, request.app.state.readiness_probe)


def get_catalog_repository(request: Request) -> CatalogRepository:
    """Return the application-scoped collection and document catalog."""
    return cast(CatalogRepository, request.app.state.catalog_repository)


def get_pipeline_registry(request: Request) -> PipelineRegistry:
    """Return the application-scoped framework registry."""
    return cast(PipelineRegistry, request.app.state.pipeline_registry)


def get_app_settings(request: Request) -> Settings:
    """Return validated settings used to construct the running application."""
    return cast(Settings, request.app.state.settings)


def get_ingestion_job_manager(request: Request) -> IngestionJobManager:
    """Return the recoverable application-scoped background ingestion runner."""
    return cast(IngestionJobManager, request.app.state.ingestion_job_manager)


def get_document_deletion_manager(request: Request) -> DocumentDeletionManager:
    """Return the application-scoped coordinated deletion service."""
    return cast(DocumentDeletionManager, request.app.state.document_deletion_manager)
