import json

import httpx
import pytest

from raglab.core.config import Settings
from raglab.core.exceptions import MalformedProviderResponseError, ProviderUnavailableError
from raglab.core.schemas import GenerationRequest
from raglab.generation.providers import (
    OllamaProvider,
    OpenAICompatibleProvider,
    create_llm_provider,
)


def request() -> GenerationRequest:
    return GenerationRequest(
        system_prompt="Use evidence only.",
        user_prompt="Question and evidence.",
        model="test-model",
        temperature=0,
        max_output_tokens=200,
        response_schema={"type": "object", "properties": {"answer": {"type": "string"}}},
    )


@pytest.mark.asyncio
async def test_openai_compatible_provider_maps_structured_request_usage_and_cost() -> None:
    async def handler(http_request: httpx.Request) -> httpx.Response:
        payload = json.loads(http_request.content)
        assert http_request.url.path == "/v1/chat/completions"
        assert http_request.headers["Authorization"] == "Bearer secret"
        assert payload["messages"][0]["role"] == "developer"
        assert payload["response_format"]["type"] == "json_schema"
        assert payload["max_completion_tokens"] == 200
        return httpx.Response(
            200,
            json={
                "id": "chat-1",
                "model": "resolved-model",
                "choices": [{"message": {"content": '{"answer":"grounded"}'}}],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "total_tokens": 120,
                },
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAICompatibleProvider(
        "https://provider.test/v1",
        api_key="secret",
        input_cost_per_million=2,
        output_cost_per_million=10,
        client=client,
    )

    result = await provider.generate(request())

    assert result.text == '{"answer":"grounded"}'
    assert result.model == "resolved-model"
    assert result.usage.total_tokens == 120
    assert result.usage.estimated_cost_usd == 0.0004
    assert result.raw_response_id == "chat-1"
    await client.aclose()


@pytest.mark.asyncio
async def test_ollama_provider_uses_native_schema_and_usage_fields() -> None:
    async def handler(http_request: httpx.Request) -> httpx.Response:
        payload = json.loads(http_request.content)
        assert http_request.url.path == "/api/chat"
        assert payload["stream"] is False
        assert payload["format"]["type"] == "object"
        assert payload["options"] == {"temperature": 0.0, "num_predict": 200}
        return httpx.Response(
            200,
            json={
                "model": "qwen3:8b",
                "message": {"role": "assistant", "content": '{"answer":"local"}'},
                "prompt_eval_count": 80,
                "eval_count": 15,
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OllamaProvider("http://ollama.test", client=client)

    result = await provider.generate(request())

    assert result.text == '{"answer":"local"}'
    assert result.usage.prompt_tokens == 80
    assert result.usage.completion_tokens == 15
    assert result.usage.estimated_cost_usd == 0
    await client.aclose()


@pytest.mark.asyncio
async def test_provider_converts_http_failure_to_safe_error() -> None:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(503, json={"error": "private"}))
    )
    provider = OpenAICompatibleProvider("https://provider.test/v1", client=client)

    with pytest.raises(ProviderUnavailableError, match="generation request failed"):
        await provider.generate(request())
    await client.aclose()


@pytest.mark.asyncio
async def test_provider_rejects_missing_assistant_content() -> None:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"choices": []}))
    )
    provider = OpenAICompatibleProvider("https://provider.test/v1", client=client)

    with pytest.raises(MalformedProviderResponseError, match="assistant text"):
        await provider.generate(request())
    await client.aclose()


def test_provider_factory_selects_configured_adapter() -> None:
    ollama = create_llm_provider(Settings(llm_provider="ollama", _env_file=None))
    compatible = create_llm_provider(Settings(llm_provider="openai_compatible", _env_file=None))

    assert isinstance(ollama, OllamaProvider)
    assert isinstance(compatible, OpenAICompatibleProvider)
