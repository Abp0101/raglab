"""Pipeline capability discovery endpoint."""

from typing import Annotated

from fastapi import APIRouter, Depends

from apps.api.dependencies import get_pipeline_registry
from raglab.core.schemas import PipelineSummary
from raglab.pipelines import PipelineRegistry

router = APIRouter(prefix="/pipelines", tags=["pipelines"])


@router.get("", response_model=list[PipelineSummary])
async def list_pipelines(
    registry: Annotated[PipelineRegistry, Depends(get_pipeline_registry)],
) -> list[PipelineSummary]:
    """Show implemented and planned frameworks with comparable capabilities."""
    return list(registry.summaries())
