import json
from pathlib import Path

import pytest

from raglab.evaluation.datasets import EvaluationDatasetError, load_dataset


def test_committed_dataset_has_valid_checksum_and_unique_questions() -> None:
    dataset = load_dataset(Path("datasets/evaluation/v1"))

    assert dataset.manifest.version == "1.0.0"
    assert len(dataset.questions) == 7
    assert len({question.question_id for question in dataset.questions}) == 7
    assert any(not question.answerable for question in dataset.questions)


def test_dataset_loader_rejects_checksum_mismatch(tmp_path: Path) -> None:
    source = Path("datasets/evaluation/v1")
    (tmp_path / "manifest.json").write_bytes((source / "manifest.json").read_bytes())
    question = json.loads((source / "questions.jsonl").read_text().splitlines()[0])
    question["question"] = "Tampered question"
    (tmp_path / "questions.jsonl").write_text(json.dumps(question) + "\n")

    with pytest.raises(EvaluationDatasetError, match="checksum"):
        load_dataset(tmp_path)
