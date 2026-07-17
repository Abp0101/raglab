import json
from pathlib import Path

import pytest

from raglab.indexing_experiments import (
    IndexingExperimentError,
    IndexingFramework,
    load_plan,
    load_plan_cases,
    run_indexing_experiments,
    write_indexing_report,
)
from raglab.indexing_experiments.embeddings import deterministic_hash_embedding


def test_native_indexing_plan_runs_every_isolated_framework(tmp_path: Path) -> None:
    plan, config_sha256 = load_plan(Path("configs/indexing_experiments/v1.json"))
    cases, dataset_sha256 = load_plan_cases(plan)

    run = run_indexing_experiments(
        plan,
        cases,
        dataset_sha256=dataset_sha256,
        config_sha256=config_sha256,
    )
    json_path = tmp_path / "native-indexing.json"
    markdown_path = tmp_path / "native-indexing.md"
    write_indexing_report(run, json_path, markdown_path)

    assert len(run.results) == len(cases) * len(IndexingFramework)
    assert {result.framework for result in run.results} == set(IndexingFramework)
    assert {aggregate.framework for aggregate in run.aggregates} == set(IndexingFramework)
    assert all(result.chunk_count > 0 for result in run.results)
    assert all(result.estimated_api_cost_usd == 0 for result in run.results)
    assert all(0 <= result.retrieval_recall_at_k <= 1 for result in run.results)
    assert len({result.index_backend for result in run.results}) == len(IndexingFramework)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["dataset_sha256"] == dataset_sha256
    assert payload["config_sha256"] == config_sha256
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "separate from RAGLab's canonical fair-comparison baseline" in markdown
    assert "Paid API cost: `$0.00`" in markdown
    assert "LangGraph is excluded" in markdown


def test_plan_declaration_must_match_native_adapter() -> None:
    plan, config_sha256 = load_plan(Path("configs/indexing_experiments/v1.json"))
    cases, dataset_sha256 = load_plan_cases(plan)
    first = plan.experiments[0].model_copy(update={"strategy": "misleading-strategy"})
    invalid = plan.model_copy(update={"experiments": (first, *plan.experiments[1:])})

    with pytest.raises(IndexingExperimentError, match="declared experiment does not match"):
        run_indexing_experiments(
            invalid,
            cases,
            dataset_sha256=dataset_sha256,
            config_sha256=config_sha256,
        )


def test_hash_embedding_is_deterministic_normalized_and_local() -> None:
    first = deterministic_hash_embedding("IMU sampled at 100 Hz", 64)
    second = deterministic_hash_embedding("IMU sampled at 100 Hz", 64)

    assert first == second
    assert sum(value * value for value in first) == pytest.approx(1)
    assert deterministic_hash_embedding("", 64) == [0.0] * 64
    with pytest.raises(ValueError, match="dimensions must be positive"):
        deterministic_hash_embedding("invalid", 0)
