"""Run deterministic metrics against a local Ollama-backed RAG pipeline."""

import argparse
import asyncio
from pathlib import Path

from apps.api.runtime import build_api_services

from raglab.core.config import Settings
from raglab.core.schemas import EvaluationRunConfig, FrameworkName, RetrievalMode
from raglab.evaluation import EvaluationRunner, build_report, load_dataset, write_report


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Already installed local Ollama model")
    parser.add_argument(
        "--framework",
        choices=[FrameworkName.CUSTOM.value, FrameworkName.LANGCHAIN.value],
        default=FrameworkName.CUSTOM.value,
    )
    parser.add_argument("--dataset", type=Path, default=Path("datasets/evaluation/v1"))
    parser.add_argument("--output", type=Path, default=Path("reports/generated"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--retrieval-mode",
        choices=[mode.value for mode in RetrievalMode],
        default=RetrievalMode.HYBRID.value,
    )
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--concurrency", type=int, default=1)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    settings = Settings(
        llm_provider="ollama",
        llm_model=args.model,
        allow_paid_api_usage=False,
    )
    services = build_api_services(settings)
    try:
        framework = FrameworkName(args.framework)
        pipeline = services.pipelines.get(framework)
        run = await EvaluationRunner(pipeline.query).run(
            dataset,
            EvaluationRunConfig(
                framework=framework,
                retrieval_mode=RetrievalMode(args.retrieval_mode),
                top_k=args.top_k,
                rerank=not args.no_rerank,
                model=args.model,
                concurrency=args.concurrency,
            ),
        )
        for result in run.results:
            if result.response is not None and result.response.usage.estimated_cost_usd not in (
                None,
                0,
            ):
                raise RuntimeError("local evaluation must report exactly zero API cost")
        report = build_report(run)
        json_path, markdown_path = write_report(report, args.output)
        print(
            f"run_id={run.run_id} successful={report.successful_questions} "
            f"failed={report.failed_questions} cost=0.0"
        )
        print(f"json={json_path}")
        print(f"markdown={markdown_path}")
        if report.failed_questions:
            raise RuntimeError("one or more evaluation questions failed to execute")
    finally:
        await services.close()


if __name__ == "__main__":
    asyncio.run(main())
