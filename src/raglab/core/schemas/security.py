"""Authentication principals and role-based authorization contracts."""

from enum import StrEnum

from raglab.core.schemas.common import RAGLabModel


class AuthRole(StrEnum):
    """Coarse API roles suitable for a local or small-team deployment."""

    VIEWER = "viewer"
    EDITOR = "editor"
    ADMIN = "admin"


class Permission(StrEnum):
    """Operations independently enforced at FastAPI route boundaries."""

    CATALOG_READ = "catalog:read"
    QUERY_EXECUTE = "query:execute"
    INGESTION_WRITE = "ingestion:write"
    DOCUMENT_DELETE = "document:delete"


class AuthPrincipal(RAGLabModel):
    """Authenticated caller identity exposed without credential material."""

    subject: str
    role: AuthRole
    permissions: tuple[Permission, ...]
