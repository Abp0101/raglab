"""PDF ingestion and document metadata endpoints."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, UploadFile, status
from pydantic import HttpUrl

from apps.api.dependencies import (
    get_app_settings,
    get_catalog_repository,
    get_pipeline_registry,
)
from raglab.core.config import Settings
from raglab.core.exceptions import DocumentValidationError
from raglab.core.interfaces import CatalogRepository
from raglab.core.schemas import Document, DocumentInput, FrameworkName, IngestionResult
from raglab.pipelines import PipelineRegistry

router = APIRouter(tags=["documents"])


@router.post(
    "/collections/{collection_id}/documents",
    response_model=IngestionResult,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(
    collection_id: UUID,
    file: Annotated[UploadFile, File(description="Text-based PDF")],
    catalog: Annotated[CatalogRepository, Depends(get_catalog_repository)],
    pipelines: Annotated[PipelineRegistry, Depends(get_pipeline_registry)],
    settings: Annotated[Settings, Depends(get_app_settings)],
    display_title: Annotated[str | None, Form(max_length=500)] = None,
    source_url: Annotated[HttpUrl | None, Form()] = None,
) -> IngestionResult:
    """Validate and synchronously ingest one bounded PDF into all shared indexes."""
    await catalog.get_collection(collection_id)
    content = await file.read(settings.max_upload_size_mb * 1024 * 1024 + 1)
    if not content:
        raise DocumentValidationError("uploaded PDF is empty")
    document = DocumentInput(
        file_name=file.filename or "upload.pdf",
        content=content,
        collection_id=collection_id,
        display_title=display_title,
        source_url=source_url,
    )
    results = await pipelines.get(FrameworkName.CUSTOM).ingest((document,))
    return results[0]


@router.get("/collections/{collection_id}/documents", response_model=list[Document])
async def list_documents(
    collection_id: UUID,
    catalog: Annotated[CatalogRepository, Depends(get_catalog_repository)],
) -> list[Document]:
    """List source metadata without returning stored PDF bytes or chunk bodies."""
    return list(await catalog.list_documents(collection_id))


@router.get("/documents/{document_id}", response_model=Document)
async def get_document(
    document_id: UUID,
    catalog: Annotated[CatalogRepository, Depends(get_catalog_repository)],
) -> Document:
    """Return one document's ingestion metadata and lifecycle state."""
    return await catalog.get_document(document_id)
