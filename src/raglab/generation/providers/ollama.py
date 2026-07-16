"""Native Ollama chat adapter with JSON-schema structured output."""

from typing import Any

import httpx

from raglab.core.exceptions import MalformedProviderResponseError, ProviderUnavailableError
from raglab.core.schemas import GenerationRequest, GenerationResult, UsageMetrics


class OllamaProvider:
    """Generate local structured output through Ollama's native chat API."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        *,
        timeout_seconds: float = 120,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds
        self._client = client

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        body: dict[str, Any] = {
            "model": request.model,
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_prompt},
            ],
            "stream": False,
            "options": {"temperature": request.temperature},
        }
        if request.max_output_tokens is not None:
            body["options"]["num_predict"] = request.max_output_tokens
        if request.response_schema is not None:
            body["format"] = request.response_schema
        try:
            if self._client is not None:
                response = await self._client.post(f"{self._base_url}/api/chat", json=body)
            else:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.post(f"{self._base_url}/api/chat", json=body)
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise ProviderUnavailableError("Ollama generation request failed") from error
        try:
            payload = response.json()
            text = payload["message"]["content"]
        except (KeyError, TypeError, ValueError) as error:
            raise MalformedProviderResponseError(
                "Ollama response did not contain assistant text"
            ) from error
        if not isinstance(text, str) or not text.strip():
            raise MalformedProviderResponseError("provider returned empty assistant text")
        prompt_tokens = _optional_int(payload.get("prompt_eval_count"))
        completion_tokens = _optional_int(payload.get("eval_count"))
        total_tokens = (
            prompt_tokens + completion_tokens
            if prompt_tokens is not None and completion_tokens is not None
            else None
        )
        return GenerationResult(
            text=text,
            model=str(payload.get("model") or request.model),
            usage=UsageMetrics(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                estimated_cost_usd=0,
                llm_calls=1,
            ),
        )


def _optional_int(value: object) -> int | None:
    return int(str(value)) if value is not None else None
