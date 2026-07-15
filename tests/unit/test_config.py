import pytest

from raglab.core.config import Settings


def test_settings_have_local_defaults() -> None:
    settings = Settings(_env_file=None)

    assert settings.environment == "development"
    assert settings.postgres_dsn.startswith("postgresql+asyncpg://")
    assert str(settings.qdrant_url) == "http://localhost:6333/"
    assert str(settings.redis_dsn) == "redis://localhost:6379/0"


def test_settings_load_prefixed_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGLAB_LOG_LEVEL", "DEBUG")

    settings = Settings(_env_file=None)

    assert settings.log_level == "DEBUG"
