"""Run a controlled local comparison over the same dataset and query settings."""

import argparse
import asyncio
import json
from pathlib import Path
from uuid import uuid4

from apps.api.runtime import build_api_services

from raglab.core.config import Settings
from raglab.core.schemas import (
    EvaluationReport,
    EvaluationRunConfig,
    FrameworkName,
    RetrievalMode,
)
from raglab.evaluation import EvaluationRunner, build_report, load_dataset

FRAMEWORKS = (FrameworkName.CUSTOM, FrameworkName.LANGCHAIN)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Already installed local Ollama model")
    parser.add_argument("--dataset", type=Path, default=Path("datasets/evaluation/v1"))
    parser.add_argument("--output", type=Path, default=Path("reports/generated"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-rerank", action="store_true")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    settings = Settings(
        llm_provider="ollama",
        llm_model=args.model,
        allow_paid_api_usage=False,
    )
    services = build_api_services(settings)
    try:
        reports = []
        for framework in FRAMEWORKS:
            pipeline = services.pipelines.get(framework)
            run = await EvaluationRunner(pipeline.query).run(
                dataset,
                EvaluationRunConfig(
                    framework=framework,
                    retrieval_mode=RetrievalMode.HYBRID,
                    top_k=args.top_k,
                    rerank=not args.no_rerank,
                    model=args.model,
                    concurrency=1,
                ),
            )
            report = build_report(run)
            _assert_valid_local_run(report)
            reports.append(report)
        comparison_id = uuid4()
        args.output.mkdir(parents=True, exist_ok=True)
        json_path = args.output / f"framework-comparison-{comparison_id}.json"
        markdown_path = args.output / f"framework-comparison-{comparison_id}.md"
        json_path.write_text(
            json.dumps(
                {
                    "comparison_id": str(comparison_id),
                    "dataset_sha256": dataset.manifest.questions_sha256,
                    "reports": [report.model_dump(mode="json") for report in reports],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        markdown_path.write_text(_markdown(comparison_id, reports), encoding="utf-8")
        print(f"comparison_id={comparison_id} cost=0.0")
        print(f"json={json_path}")
        print(f"markdown={markdown_path}")
    finally:
        await services.close()


def _assert_valid_local_run(report: EvaluationReport) -> None:
    if report.failed_questions:
        raise RuntimeError("one or more comparison questions failed")
    for result in report.run.results:
        if result.response is not None and result.response.usage.estimated_cost_usd not in (
            None,
            0,
        ):
            raise RuntimeError("local comparison must report zero API cost")


def _markdown(comparison_id: object, reports: list[EvaluationReport]) -> str:
    metric_names = sorted({aggregate.name for report in reports for aggregate in report.aggregates})
    aggregate_maps = [
        {aggregate.name: aggregate.mean for aggregate in report.aggregates} for report in reports
    ]
    lines = [
        "# Controlled framework comparison",
        "",
        "This table reports measurements from one controlled local run; it does not prove that "
        "one framework is generally superior.",
        "",
        f"- Comparison ID: `{comparison_id}`",
        f"- Dataset SHA-256: `{reports[0].run.dataset_sha256}`",
        f"- Model: `{reports[0].run.config.model}`",
        f"- Top K: `{reports[0].run.config.top_k}`",
        f"- Reranking: `{str(reports[0].run.config.rerank).lower()}`",
        "- Paid API cost: `$0.00`",
        "",
        "| Metric | Custom | LangChain |",
        "| --- | ---: | ---: |",
    ]
    lines.extend(
        f"| {name} | {aggregate_maps[0].get(name, 0):.4f} | {aggregate_maps[1].get(name, 0):.4f} |"
        for name in metric_names
    )
    lines.extend(
        [
            "",
            "Both implementations use the same PostgreSQL/Qdrant/Redis corpus, local embedding "
            "model, hybrid retrieval service, reranker, grounding schema, citation validator, "
            "question order, and Ollama model. The changing variable is orchestration and the "
            "model adapter (direct native Ollama HTTP versus LangChain ChatOllama).",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    asyncio.run(main())
