"""Configuration-driven LLM provider construction."""

from raglab.core.config import Settings
from raglab.core.interfaces import LLMProvider
from raglab.generation.providers.ollama import OllamaProvider
from raglab.generation.providers.openai_compatible import OpenAICompatibleProvider


def create_llm_provider(settings: Settings) -> LLMProvider:
    """Create the selected provider without exposing credentials to logs or models."""
    if settings.llm_provider == "ollama":
        return OllamaProvider(
            str(settings.ollama_base_url), timeout_seconds=settings.llm_timeout_seconds
        )
    return OpenAICompatibleProvider(
        str(settings.openai_base_url),
        api_key=settings.openai_api_key,
        timeout_seconds=settings.llm_timeout_seconds,
        input_cost_per_million=settings.input_cost_per_million,
        output_cost_per_million=settings.output_cost_per_million,
        instruction_role=settings.openai_instruction_role,
        structured_output_mode=settings.openai_structured_output_mode,
        max_tokens_field=settings.openai_max_tokens_field,
    )
