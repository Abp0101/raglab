"""Pipeline capability discovery endpoint."""

from typing import Annotated

from fastapi import APIRouter, Depends

from apps.api.dependencies import get_pipeline_registry
from apps.api.security import require_permission
from raglab.core.schemas import Permission, PipelineSummary
from raglab.pipelines import PipelineRegistry

router = APIRouter(prefix="/pipelines", tags=["pipelines"])


@router.get(
    "",
    response_model=list[PipelineSummary],
    dependencies=[Depends(require_permission(Permission.CATALOG_READ))],
)
async def list_pipelines(
    registry: Annotated[PipelineRegistry, Depends(get_pipeline_registry)],
) -> list[PipelineSummary]:
    """Show implemented and planned frameworks with comparable capabilities."""
    return list(registry.summaries())
