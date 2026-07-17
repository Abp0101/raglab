import pytest

from raglab.core.config import ApiKeyCredentialSettings
from raglab.core.exceptions import AuthenticationError
from raglab.core.schemas import AuthRole, Permission
from raglab.security import ApiKeyAuthenticator

TEST_KEY = "local-test-key-with-at-least-32-characters"


def test_disabled_authentication_returns_local_admin_without_a_key() -> None:
    principal = ApiKeyAuthenticator(enabled=False, credentials=[]).authenticate(None)

    assert principal.subject == "local-development"
    assert principal.role is AuthRole.ADMIN
    assert set(principal.permissions) == set(Permission)


def test_configured_key_resolves_role_permissions_without_exposing_secret() -> None:
    config = ApiKeyCredentialSettings(name="portfolio-viewer", role=AuthRole.VIEWER, key=TEST_KEY)
    authenticator = ApiKeyAuthenticator(enabled=True, credentials=[config])

    principal = authenticator.authenticate(TEST_KEY)

    assert principal.subject == "portfolio-viewer"
    assert principal.role is AuthRole.VIEWER
    assert set(principal.permissions) == {
        Permission.CATALOG_READ,
        Permission.QUERY_EXECUTE,
    }
    assert TEST_KEY not in repr(config)
    assert TEST_KEY not in repr(authenticator.__dict__)


@pytest.mark.parametrize("credential", [None, "wrong-key"])
def test_missing_or_invalid_key_is_rejected(credential: str | None) -> None:
    config = ApiKeyCredentialSettings(name="portfolio-admin", role=AuthRole.ADMIN, key=TEST_KEY)
    authenticator = ApiKeyAuthenticator(enabled=True, credentials=[config])

    with pytest.raises(AuthenticationError):
        authenticator.authenticate(credential)
