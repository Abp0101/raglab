import pytest
from pydantic import ValidationError

from raglab.core.config import ApiKeyCredentialSettings, Settings
from raglab.core.schemas import AuthRole


def test_settings_have_local_defaults() -> None:
    settings = Settings(_env_file=None)

    assert settings.environment == "development"
    assert settings.postgres_dsn.startswith("postgresql+asyncpg://")
    assert str(settings.qdrant_url) == "http://localhost:6333/"
    assert str(settings.redis_dsn) == "redis://localhost:6379/0"
    assert settings.llm_provider == "ollama"
    assert settings.allow_paid_api_usage is False
    assert settings.ingestion_concurrency == 1
    assert settings.auth_enabled is False


def test_settings_load_prefixed_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGLAB_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("RAGLAB_AUTH_ENABLED", "true")
    monkeypatch.setenv(
        "RAGLAB_AUTH_API_KEYS",
        '[{"name":"environment-admin","role":"admin",'
        '"key":"environment-key-with-at-least-32-characters"}]',
    )

    settings = Settings(_env_file=None)

    assert settings.log_level == "DEBUG"
    assert settings.auth_enabled is True
    assert settings.auth_api_keys[0].name == "environment-admin"
    assert settings.auth_api_keys[0].key.get_secret_value().startswith("environment-key")


def test_production_authentication_fails_closed() -> None:
    with pytest.raises(ValidationError, match="authentication must be enabled"):
        Settings(environment="production", _env_file=None)

    with pytest.raises(ValidationError, match="at least one API key"):
        Settings(environment="production", auth_enabled=True, _env_file=None)


def test_api_key_configuration_rejects_weak_and_duplicate_secrets() -> None:
    with pytest.raises(ValidationError, match="between 32 and 256"):
        ApiKeyCredentialSettings(name="weak", role=AuthRole.ADMIN, key="short")

    duplicate = "duplicate-key-that-is-at-least-32-characters"
    with pytest.raises(ValidationError, match="API key values must be unique"):
        Settings(
            auth_enabled=True,
            auth_api_keys=[
                ApiKeyCredentialSettings(name="one", role=AuthRole.VIEWER, key=duplicate),
                ApiKeyCredentialSettings(name="two", role=AuthRole.ADMIN, key=duplicate),
            ],
            _env_file=None,
        )
