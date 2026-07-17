"""Caller identity inspection endpoint."""

from typing import Annotated

from fastapi import APIRouter, Depends

from apps.api.security import get_current_principal
from raglab.core.schemas import AuthPrincipal

router = APIRouter(prefix="/auth", tags=["authentication"])


@router.get("/me", response_model=AuthPrincipal)
async def get_me(
    principal: Annotated[AuthPrincipal, Depends(get_current_principal)],
) -> AuthPrincipal:
    """Return the authenticated subject, role, and effective permissions."""
    return principal
