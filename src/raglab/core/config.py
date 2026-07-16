"""Validated application configuration loaded from environment variables."""

from functools import lru_cache
from typing import Literal

from pydantic import AnyHttpUrl, Field, RedisDsn
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """RAGLab runtime settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="RAGLAB_",
        extra="ignore",
    )

    app_name: str = "RAGLab API"
    environment: Literal["development", "test", "staging", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_json: bool = True
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])
    max_upload_size_mb: int = Field(default=25, ge=1, le=250)
    max_pdf_pages: int = Field(default=500, ge=1, le=5000)
    ingestion_concurrency: int = Field(default=1, ge=1, le=8)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = Field(default=32, ge=1, le=512)
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_batch_size: int = Field(default=16, ge=1, le=256)
    qdrant_collection: str = "raglab_chunks"
    qdrant_timeout_seconds: int = Field(default=30, gt=0, le=300)
    bm25_key_prefix: str = "raglab:bm25"
    llm_provider: Literal["openai_compatible", "ollama"] = "ollama"
    llm_model: str = "qwen3:8b"
    allow_paid_api_usage: bool = False
    openai_base_url: AnyHttpUrl = AnyHttpUrl("https://api.openai.com/v1")
    openai_api_key: str | None = None
    openai_instruction_role: Literal["developer", "system"] = "developer"
    openai_structured_output_mode: Literal["json_schema", "json_object"] = "json_schema"
    openai_max_tokens_field: Literal["max_completion_tokens", "max_tokens"] = (
        "max_completion_tokens"
    )
    ollama_base_url: AnyHttpUrl = AnyHttpUrl("http://localhost:11434")
    llm_timeout_seconds: float = Field(default=120, gt=0, le=600)
    input_cost_per_million: float | None = Field(default=None, ge=0)
    output_cost_per_million: float | None = Field(default=None, ge=0)

    postgres_dsn: str = "postgresql+asyncpg://raglab:raglab@localhost:5432/raglab"
    qdrant_url: AnyHttpUrl = AnyHttpUrl("http://localhost:6333")
    qdrant_api_key: str | None = None
    redis_dsn: RedisDsn = RedisDsn("redis://localhost:6379/0")


@lru_cache
def get_settings() -> Settings:
    """Load and cache immutable process-level configuration."""
    return Settings()
