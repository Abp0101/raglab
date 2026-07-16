"""Collection catalog endpoints."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, status

from apps.api.dependencies import get_catalog_repository
from raglab.core.interfaces import CatalogRepository
from raglab.core.schemas import Collection, CollectionCreate

router = APIRouter(prefix="/collections", tags=["collections"])


@router.post("", response_model=Collection, status_code=status.HTTP_201_CREATED)
async def create_collection(
    request: CollectionCreate,
    catalog: Annotated[CatalogRepository, Depends(get_catalog_repository)],
) -> Collection:
    """Create a logical corpus shared by every framework implementation."""
    return await catalog.create_collection(request)


@router.get("", response_model=list[Collection])
async def list_collections(
    catalog: Annotated[CatalogRepository, Depends(get_catalog_repository)],
) -> list[Collection]:
    """List collections with their current document counts."""
    return list(await catalog.list_collections())


@router.get("/{collection_id}", response_model=Collection)
async def get_collection(
    collection_id: UUID,
    catalog: Annotated[CatalogRepository, Depends(get_catalog_repository)],
) -> Collection:
    """Return one collection or a stable not-found error."""
    return await catalog.get_collection(collection_id)
