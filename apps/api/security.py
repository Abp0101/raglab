"""FastAPI bearer authentication and permission dependencies."""

from collections.abc import Callable
from typing import Annotated

from fastapi import Depends, Request, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from raglab.core.exceptions import AuthorizationError
from raglab.core.interfaces import Authenticator
from raglab.core.schemas import AuthPrincipal, Permission

_bearer = HTTPBearer(auto_error=False, scheme_name="RAGLabApiKey")


def get_authenticator(request: Request) -> Authenticator:
    """Return the application-scoped credential authenticator."""
    return request.app.state.authenticator  # type: ignore[no-any-return]


def get_current_principal(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Security(_bearer)],
    authenticator: Annotated[Authenticator, Depends(get_authenticator)],
) -> AuthPrincipal:
    """Authenticate an optional bearer header according to runtime policy."""
    token = credentials.credentials if credentials is not None else None
    return authenticator.authenticate(token)


def require_permission(permission: Permission) -> Callable[..., AuthPrincipal]:
    """Build a dependency that enforces one role-derived permission."""

    def authorize(
        principal: Annotated[AuthPrincipal, Depends(get_current_principal)],
    ) -> AuthPrincipal:
        if permission not in principal.permissions:
            raise AuthorizationError(f"permission {permission.value} is required")
        return principal

    return authorize
