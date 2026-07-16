"""Chat Completions adapter for OpenAI and compatible APIs."""

from typing import Any, Literal

import httpx

from raglab.core.exceptions import MalformedProviderResponseError, ProviderUnavailableError
from raglab.core.schemas import GenerationRequest, GenerationResult, UsageMetrics


class OpenAICompatibleProvider:
    """Generate structured output through a configurable Chat Completions endpoint."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        timeout_seconds: float = 120,
        input_cost_per_million: float | None = None,
        output_cost_per_million: float | None = None,
        instruction_role: Literal["developer", "system"] = "developer",
        structured_output_mode: Literal["json_schema", "json_object"] = "json_schema",
        max_tokens_field: Literal["max_completion_tokens", "max_tokens"] = (
            "max_completion_tokens"
        ),
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout_seconds
        self._input_cost = input_cost_per_million
        self._output_cost = output_cost_per_million
        self._instruction_role = instruction_role
        self._structured_output_mode = structured_output_mode
        self._max_tokens_field = max_tokens_field
        self._client = client

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        body: dict[str, Any] = {
            "model": request.model,
            "messages": [
                {"role": self._instruction_role, "content": request.system_prompt},
                {"role": "user", "content": request.user_prompt},
            ],
            "temperature": request.temperature,
        }
        if request.max_output_tokens is not None:
            body[self._max_tokens_field] = request.max_output_tokens
        if request.response_schema is not None:
            body["response_format"] = (
                {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "raglab_grounded_answer",
                        "strict": True,
                        "schema": request.response_schema,
                    },
                }
                if self._structured_output_mode == "json_schema"
                else {"type": "json_object"}
            )
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        try:
            if self._client is not None:
                response = await self._client.post(
                    f"{self._base_url}/chat/completions", json=body, headers=headers
                )
            else:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.post(
                        f"{self._base_url}/chat/completions", json=body, headers=headers
                    )
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise ProviderUnavailableError("OpenAI-compatible generation request failed") from error
        try:
            payload = response.json()
            text = payload["choices"][0]["message"]["content"]
            model = str(payload.get("model") or request.model)
            usage_payload = payload.get("usage") or {}
            prompt_tokens = _optional_int(usage_payload.get("prompt_tokens"))
            completion_tokens = _optional_int(usage_payload.get("completion_tokens"))
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise MalformedProviderResponseError(
                "OpenAI-compatible response did not contain assistant text"
            ) from error
        if not isinstance(text, str) or not text.strip():
            raise MalformedProviderResponseError("provider returned empty assistant text")
        total_tokens = (
            prompt_tokens + completion_tokens
            if prompt_tokens is not None and completion_tokens is not None
            else _optional_int(usage_payload.get("total_tokens"))
        )
        return GenerationResult(
            text=text,
            model=model,
            usage=UsageMetrics(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                estimated_cost_usd=_cost(
                    prompt_tokens,
                    completion_tokens,
                    self._input_cost,
                    self._output_cost,
                ),
                llm_calls=1,
            ),
            raw_response_id=(str(payload["id"]) if payload.get("id") else None),
        )


def _cost(
    prompt_tokens: int | None,
    completion_tokens: int | None,
    input_rate: float | None,
    output_rate: float | None,
) -> float | None:
    if None in (prompt_tokens, completion_tokens, input_rate, output_rate):
        return None
    return (prompt_tokens * input_rate + completion_tokens * output_rate) / 1_000_000  # type: ignore[operator]


def _optional_int(value: object) -> int | None:
    return int(str(value)) if value is not None else None
