"""Verify Ollama structured output and shared usage mapping with an installed model."""

import argparse
import asyncio

from raglab.core.schemas import GenerationRequest
from raglab.generation.output import GroundedAnswer
from raglab.generation.providers import OllamaProvider


async def run(model: str) -> None:
    provider = OllamaProvider(timeout_seconds=180)
    result = await provider.generate(
        GenerationRequest(
            system_prompt="Return only valid JSON matching the schema. Use the supplied fact only.",
            user_prompt="Fact: the IMU sampled at 100 Hz. State the sampling rate.",
            model=model,
            temperature=0,
            max_output_tokens=128,
            response_schema=GroundedAnswer.model_json_schema(),
        )
    )
    GroundedAnswer.model_validate_json(result.text)
    print(
        f"model={result.model} prompt_tokens={result.usage.prompt_tokens} "
        f"completion_tokens={result.usage.completion_tokens}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="An already-installed Ollama model")
    arguments = parser.parse_args()
    asyncio.run(run(arguments.model))


if __name__ == "__main__":
    main()
