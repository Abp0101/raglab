"""Stable JSON and human-readable Markdown evaluation reports."""

import json
import re
from collections import defaultdict
from pathlib import Path

from raglab.core.schemas import (
    EvaluationMetricAggregate,
    EvaluationMetricResult,
    EvaluationReport,
    EvaluationRun,
)


def build_report(run: EvaluationRun) -> EvaluationReport:
    """Aggregate only applicable metrics from successful questions."""
    values: dict[str, list[float]] = defaultdict(list)
    for result in run.results:
        if result.error is not None:
            continue
        for metric in (*result.retrieval_metrics, *result.answer_metrics):
            if _applicable(metric):
                values[metric.name].append(metric.value)
    aggregates = tuple(
        EvaluationMetricAggregate(
            name=name,
            mean=sum(metric_values) / len(metric_values),
            minimum=min(metric_values),
            maximum=max(metric_values),
            sample_count=len(metric_values),
        )
        for name, metric_values in sorted(values.items())
    )
    failures = sum(result.error is not None for result in run.results)
    return EvaluationReport(
        run=run,
        aggregates=aggregates,
        successful_questions=len(run.results) - failures,
        failed_questions=failures,
    )


def write_report(report: EvaluationReport, output_directory: Path) -> tuple[Path, Path]:
    """Write machine-readable and reviewable artifacts using the run UUID."""
    output_directory.mkdir(parents=True, exist_ok=True)
    stem = (
        f"{_safe_component(report.run.dataset_name)}-"
        f"{_safe_component(report.run.dataset_version)}-{report.run.run_id}"
    )
    json_path = output_directory / f"{stem}.json"
    markdown_path = output_directory / f"{stem}.md"
    json_path.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def _markdown(report: EvaluationReport) -> str:
    run = report.run
    lines = [
        f"# Evaluation report: {run.dataset_name} {run.dataset_version}",
        "",
        "This report contains deterministic measurements, not a claim that one framework is best.",
        "",
        "## Reproducibility",
        "",
        f"- Run ID: `{run.run_id}`",
        f"- Dataset SHA-256: `{run.dataset_sha256}`",
        f"- Framework: `{run.config.framework.value}`",
        f"- Retrieval: `{run.config.retrieval_mode.value}`",
        f"- Top K: `{run.config.top_k}`",
        f"- Reranking: `{str(run.config.rerank).lower()}`",
        f"- Model: `{run.config.model}`",
        f"- Started: `{run.started_at.isoformat()}`",
        f"- Completed: `{run.completed_at.isoformat()}`",
        "",
        "## Aggregate metrics",
        "",
        "| Metric | Mean | Min | Max | N |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    lines.extend(
        f"| {metric.name} | {metric.mean:.4f} | {metric.minimum:.4f} | "
        f"{metric.maximum:.4f} | {metric.sample_count} |"
        for metric in report.aggregates
    )
    lines.extend(
        [
            "",
            "## Question results",
            "",
            "| Question | Status | Evidence | Latency ms |",
            "| --- | --- | --- | ---: |",
        ]
    )
    for result in run.results:
        status = f"error: {result.error}" if result.error else "ok"
        evidence = result.response.evidence_status.value if result.response else "-"
        latency = f"{result.response.latency.total_ms:.2f}" if result.response else "-"
        lines.append(f"| {result.question_id} | {status} | {evidence} | {latency} |")
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- Retrieval relevance depends on the dataset's annotated chunk or document IDs.",
            "- Key-fact coverage is normalized lexical containment, not semantic entailment.",
            "- Citation metrics verify expected source selection, not claim-level logical support.",
            "- Latency is environment-specific and requires controlled conditions for comparison.",
            "- No paid or remote LLM judge is used.",
            "",
        ]
    )
    return "\n".join(lines)


def _applicable(metric: EvaluationMetricResult) -> bool:
    return metric.details.get("applicable", True) is not False


def _safe_component(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-.")
    return normalized or "evaluation"
