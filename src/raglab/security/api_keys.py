"""Local API-key authentication with role-derived permissions."""

import hashlib
import hmac
from dataclasses import dataclass
from typing import Final

from raglab.core.config import ApiKeyCredentialSettings
from raglab.core.exceptions import AuthenticationError
from raglab.core.schemas import AuthPrincipal, AuthRole, Permission

_ROLE_PERMISSIONS: Final[dict[AuthRole, frozenset[Permission]]] = {
    AuthRole.VIEWER: frozenset(
        {
            Permission.CATALOG_READ,
            Permission.QUERY_EXECUTE,
        }
    ),
    AuthRole.EDITOR: frozenset(
        {
            Permission.CATALOG_READ,
            Permission.QUERY_EXECUTE,
            Permission.INGESTION_WRITE,
        }
    ),
    AuthRole.ADMIN: frozenset(Permission),
}


@dataclass(frozen=True, slots=True)
class _StoredCredential:
    subject: str
    role: AuthRole
    digest: bytes


class ApiKeyAuthenticator:
    """Authenticate configured bearer keys without retaining their raw values."""

    def __init__(
        self,
        *,
        enabled: bool,
        credentials: list[ApiKeyCredentialSettings],
    ) -> None:
        self._enabled = enabled
        self._credentials = tuple(
            _StoredCredential(
                subject=credential.name,
                role=credential.role,
                digest=_digest(credential.key.get_secret_value()),
            )
            for credential in credentials
        )

    def authenticate(self, credential: str | None) -> AuthPrincipal:
        if not self._enabled:
            return _principal("local-development", AuthRole.ADMIN)
        if credential is None:
            raise AuthenticationError("a bearer API key is required")

        presented = _digest(credential)
        matched: _StoredCredential | None = None
        for candidate in self._credentials:
            if hmac.compare_digest(presented, candidate.digest):
                matched = candidate
        if matched is None:
            raise AuthenticationError("the bearer API key is invalid")
        return _principal(matched.subject, matched.role)


def permissions_for_role(role: AuthRole) -> frozenset[Permission]:
    """Return immutable permissions used by both authentication and tests."""
    return _ROLE_PERMISSIONS[role]


def _principal(subject: str, role: AuthRole) -> AuthPrincipal:
    permissions = tuple(sorted(permissions_for_role(role), key=lambda item: item.value))
    return AuthPrincipal(subject=subject, role=role, permissions=permissions)


def _digest(value: str) -> bytes:
    return hashlib.sha256(value.encode()).digest()
