"""Machine-readable and Markdown reports for native indexing experiments."""

import json
from pathlib import Path

from raglab.indexing_experiments.models import IndexingExperimentRun


def write_indexing_report(
    run: IndexingExperimentRun,
    json_path: Path,
    markdown_path: Path,
) -> None:
    """Write stable artifacts without declaring a universally best framework."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(run.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(_markdown(run), encoding="utf-8")


def _markdown(run: IndexingExperimentRun) -> str:
    controls = run.controls
    lines = [
        "# Framework-native indexing experiment",
        "",
        "This experiment intentionally changes indexing implementations and is separate from "
        "RAGLab's canonical fair-comparison baseline. It does not establish that one framework "
        "is generally superior.",
        "",
        "## Reproducibility",
        "",
        f"- Run ID: `{run.run_id}`",
        f"- Benchmark version: `{run.version}`",
        f"- Dataset version: `{run.dataset_version}`",
        f"- Dataset SHA-256: `{run.dataset_sha256}`",
        f"- Configuration SHA-256: `{run.config_sha256}`",
        f"- Embedding control: `{controls.embedding_model}` ({controls.embedding_dimensions}d)",
        f"- Chunk target / overlap: `{controls.chunk_size}` / `{controls.chunk_overlap}`",
        f"- Retrieval cutoff: `{controls.top_k}`",
        "- Paid API cost: `$0.00`",
        "",
        "## Aggregate observations",
        "",
        "| Framework | Native strategy | Native index | Chunks | Tokens/chunk | Redundancy | "
        f"Containment | Recall@{controls.top_k} | Index ms | Query ms |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    lines.extend(
        f"| {aggregate.framework.value} | {aggregate.strategy} | {aggregate.index_backend} | "
        f"{aggregate.mean_chunk_count:.2f} | {aggregate.mean_chunk_tokens:.2f} | "
        f"{aggregate.mean_redundancy_ratio:.3f} | "
        f"{aggregate.mean_passage_containment:.3f} | "
        f"{aggregate.mean_retrieval_recall_at_k:.3f} | "
        f"{aggregate.mean_indexing_ms:.2f} | {aggregate.mean_query_ms:.2f} |"
        for aggregate in run.aggregates
    )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- The deterministic hashing embedding is a control, not a production semantic model.",
            "- Split-size units follow each native framework and are reported rather than hidden.",
            "- Annotated natural-language queries isolate indexing and retrieval; "
            "they do not measure generation.",
            "- In-memory indexes remove network variance but do not represent production scale.",
            "- Latency is environment-specific; compare only runs made under controlled "
            "conditions.",
            "- LangGraph is excluded because it orchestrates retrieval and does not own indexing.",
            "",
        ]
    )
    return "\n".join(lines)
