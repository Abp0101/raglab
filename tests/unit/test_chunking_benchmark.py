import json
from pathlib import Path

from raglab.chunking.benchmark import load_cases, run_benchmark, write_results
from raglab.core.schemas import ChunkingStrategy


def test_benchmark_runs_every_strategy_and_writes_json(tmp_path: Path) -> None:
    dataset = Path("datasets/evaluation/chunking_benchmark_v1.jsonl")

    cases = load_cases(dataset)
    results = run_benchmark(cases)
    output = tmp_path / "chunking.json"
    write_results(output, results)

    assert len(results) == len(cases) * len(ChunkingStrategy)
    assert {result.strategy for result in results} == set(ChunkingStrategy)
    assert {result.strategy: result.size_unit for result in results[: len(ChunkingStrategy)]} == {
        ChunkingStrategy.FIXED_TOKEN: "tokens",
        ChunkingStrategy.RECURSIVE_CHARACTER: "characters",
        ChunkingStrategy.SECTION_AWARE: "characters",
        ChunkingStrategy.PARENT_CHILD: "characters",
    }
    assert all(0 <= result.relevant_passage_containment <= 1 for result in results)
    assert all(
        result.section_boundary_violations == 0
        for result in results
        if result.strategy is ChunkingStrategy.SECTION_AWARE
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["benchmark"] == "raglab-chunking-v1"
    assert len(payload["results"]) == len(results)


def test_benchmark_rejects_missing_relevant_passage(tmp_path: Path) -> None:
    dataset = tmp_path / "invalid.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "case_id": "invalid",
                "dataset_version": "1.0.0",
                "category": "test",
                "text": "Available evidence.",
                "relevant_passages": ["Missing evidence."],
            }
        ),
        encoding="utf-8",
    )

    try:
        load_cases(dataset)
    except ValueError as error:
        assert "line 1" in str(error)
    else:
        raise AssertionError("invalid benchmark data was accepted")
