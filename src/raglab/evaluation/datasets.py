"""Strict loading and integrity validation for versioned JSONL datasets."""

import hashlib
import json
from pathlib import Path

from pydantic import ValidationError

from raglab.core.schemas import (
    EvaluationDataset,
    EvaluationDatasetManifest,
    EvaluationQuestion,
)


class EvaluationDatasetError(ValueError):
    """A dataset is malformed, inconsistent, or fails its checksum."""


def load_dataset(directory: Path) -> EvaluationDataset:
    """Load one immutable dataset directory and verify its question bytes."""
    manifest_path = directory / "manifest.json"
    questions_path = directory / "questions.jsonl"
    try:
        manifest = EvaluationDatasetManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
        question_bytes = questions_path.read_bytes()
    except (OSError, ValidationError) as error:
        raise EvaluationDatasetError("dataset manifest or questions could not be loaded") from error
    checksum = hashlib.sha256(question_bytes).hexdigest()
    if checksum != manifest.questions_sha256:
        raise EvaluationDatasetError("questions checksum does not match the manifest")
    questions = _parse_questions(question_bytes)
    if len(questions) != manifest.question_count:
        raise EvaluationDatasetError("question count does not match the manifest")
    if any(question.dataset_version != manifest.version for question in questions):
        raise EvaluationDatasetError("question dataset versions do not match the manifest")
    question_ids = [question.question_id for question in questions]
    if len(set(question_ids)) != len(question_ids):
        raise EvaluationDatasetError("question IDs must be unique")
    return EvaluationDataset(manifest=manifest, questions=questions)


def _parse_questions(content: bytes) -> tuple[EvaluationQuestion, ...]:
    questions: list[EvaluationQuestion] = []
    try:
        lines = content.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise EvaluationDatasetError("questions JSONL must be valid UTF-8") from error
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            questions.append(EvaluationQuestion.model_validate(json.loads(line)))
        except (json.JSONDecodeError, ValidationError) as error:
            raise EvaluationDatasetError(f"invalid question on JSONL line {line_number}") from error
    if not questions:
        raise EvaluationDatasetError("dataset contains no questions")
    return tuple(questions)
