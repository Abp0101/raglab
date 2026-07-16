"""Shared RAG query endpoint."""

from typing import Annotated

from fastapi import APIRouter, Depends

from apps.api.dependencies import get_catalog_repository, get_pipeline_registry
from raglab.core.interfaces import CatalogRepository
from raglab.core.schemas import QueryRequest, RAGResponse
from raglab.pipelines import PipelineRegistry

router = APIRouter(tags=["query"])


@router.post("/query", response_model=RAGResponse)
async def query(
    request: QueryRequest,
    catalog: Annotated[CatalogRepository, Depends(get_catalog_repository)],
    registry: Annotated[PipelineRegistry, Depends(get_pipeline_registry)],
) -> RAGResponse:
    """Run a validated query through the selected registered framework."""
    await catalog.get_collection(request.collection_id)
    return await registry.get(request.framework).query(request)
