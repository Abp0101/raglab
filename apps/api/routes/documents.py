"""PDF ingestion and document metadata endpoints."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile, status
from pydantic import HttpUrl

from apps.api.dependencies import (
    get_app_settings,
    get_catalog_repository,
    get_document_deletion_manager,
    get_ingestion_job_manager,
    get_pipeline_registry,
)
from apps.api.security import require_permission
from raglab.core.config import Settings
from raglab.core.exceptions import DocumentValidationError
from raglab.core.interfaces import (
    CatalogRepository,
    DocumentDeletionManager,
    IngestionJobManager,
)
from raglab.core.schemas import (
    CursorPage,
    Document,
    DocumentDeletionResult,
    DocumentInput,
    FrameworkName,
    IngestionJob,
    IngestionResult,
    Permission,
)
from raglab.ingestion.validation import PdfUploadValidator
from raglab.pipelines import PipelineRegistry

router = APIRouter(tags=["documents"])


@router.post(
    "/collections/{collection_id}/documents",
    response_model=IngestionResult,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_permission(Permission.INGESTION_WRITE))],
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
    document = await _document_input(
        collection_id,
        file,
        settings,
        display_title=display_title,
        source_url=source_url,
    )
    results = await pipelines.get(FrameworkName.CUSTOM).ingest((document,))
    return results[0]


@router.post(
    "/collections/{collection_id}/ingestion-jobs",
    response_model=IngestionJob,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(require_permission(Permission.INGESTION_WRITE))],
)
async def create_ingestion_job(
    collection_id: UUID,
    file: Annotated[UploadFile, File(description="Text-based PDF")],
    catalog: Annotated[CatalogRepository, Depends(get_catalog_repository)],
    jobs: Annotated[IngestionJobManager, Depends(get_ingestion_job_manager)],
    settings: Annotated[Settings, Depends(get_app_settings)],
    display_title: Annotated[str | None, Form(max_length=500)] = None,
    source_url: Annotated[HttpUrl | None, Form()] = None,
) -> IngestionJob:
    """Persist a bounded upload and return before local parsing and indexing complete."""
    await catalog.get_collection(collection_id)
    document = await _document_input(
        collection_id,
        file,
        settings,
        display_title=display_title,
        source_url=source_url,
    )
    return await jobs.submit(document)


@router.get(
    "/ingestion-jobs/{job_id}",
    response_model=IngestionJob,
    dependencies=[Depends(require_permission(Permission.CATALOG_READ))],
)
async def get_ingestion_job(
    job_id: UUID,
    jobs: Annotated[IngestionJobManager, Depends(get_ingestion_job_manager)],
) -> IngestionJob:
    """Poll durable queued, processing, completed, or failed ingestion state."""
    return await jobs.get(job_id)


@router.get(
    "/collections/{collection_id}/ingestion-jobs",
    response_model=CursorPage[IngestionJob],
    dependencies=[Depends(require_permission(Permission.CATALOG_READ))],
)
async def list_ingestion_jobs(
    collection_id: UUID,
    jobs: Annotated[IngestionJobManager, Depends(get_ingestion_job_manager)],
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    cursor: Annotated[str | None, Query(min_length=1, max_length=512)] = None,
) -> CursorPage[IngestionJob]:
    """List durable jobs without exposing retained source bytes or lease owners."""
    return await jobs.list_for_collection(collection_id, limit=limit, cursor=cursor)


@router.get(
    "/collections/{collection_id}/documents",
    response_model=CursorPage[Document],
    dependencies=[Depends(require_permission(Permission.CATALOG_READ))],
)
async def list_documents(
    collection_id: UUID,
    catalog: Annotated[CatalogRepository, Depends(get_catalog_repository)],
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    cursor: Annotated[str | None, Query(min_length=1, max_length=512)] = None,
) -> CursorPage[Document]:
    """List source metadata through a stable collection-scoped keyset."""
    return await catalog.list_documents(collection_id, limit=limit, cursor=cursor)


@router.get(
    "/documents/{document_id}",
    response_model=Document,
    dependencies=[Depends(require_permission(Permission.CATALOG_READ))],
)
async def get_document(
    document_id: UUID,
    catalog: Annotated[CatalogRepository, Depends(get_catalog_repository)],
) -> Document:
    """Return one document's ingestion metadata and lifecycle state."""
    return await catalog.get_document(document_id)


@router.delete(
    "/documents/{document_id}",
    response_model=DocumentDeletionResult,
    dependencies=[Depends(require_permission(Permission.DOCUMENT_DELETE))],
)
async def delete_document(
    document_id: UUID,
    deletion: Annotated[DocumentDeletionManager, Depends(get_document_deletion_manager)],
) -> DocumentDeletionResult:
    """Remove one terminal document from PostgreSQL, Qdrant, and Redis."""
    return await deletion.delete(document_id)


async def _document_input(
    collection_id: UUID,
    file: UploadFile,
    settings: Settings,
    *,
    display_title: str | None,
    source_url: HttpUrl | None,
) -> DocumentInput:
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
    PdfUploadValidator(max_size_bytes=settings.max_upload_size_mb * 1024 * 1024).validate(document)
    return document
