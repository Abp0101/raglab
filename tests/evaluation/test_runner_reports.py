import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from raglab.core.schemas import (
    EvaluationDataset,
    EvaluationDatasetManifest,
    EvaluationDifficulty,
    EvaluationQuestion,
    EvaluationRunConfig,
    EvidenceStatus,
    FrameworkName,
    LatencyMetrics,
    QueryRequest,
    RAGResponse,
    RetrievalMode,
    UsageMetrics,
)
from raglab.evaluation.reports import build_report, write_report
from raglab.evaluation.runner import EvaluationRunner


@pytest.mark.asyncio
async def test_runner_preserves_dataset_order_and_report_writes_both_formats(
    tmp_path: Path,
) -> None:
    collection_id = uuid4()
    questions = tuple(
        EvaluationQuestion(
            question_id=f"q{index}",
            dataset_version="1",
            question=f"Question {index}",
            expected_key_facts=(f"fact {index}",),
            relevant_document_ids=(uuid4(),),
            answerable=True,
            category="test",
            difficulty=EvaluationDifficulty.EASY,
        )
        for index in range(3)
    )
    dataset = EvaluationDataset(
        manifest=EvaluationDatasetManifest(
            name="test-dataset",
            version="1",
            description="Test dataset",
            collection_id=collection_id,
            published_on=datetime.now(UTC).date(),
            question_count=3,
            questions_sha256="a" * 64,
            domains=("test",),
            license="CC0",
        ),
        questions=questions,
    )

    async def query(request: QueryRequest) -> RAGResponse:
        text = request.query
        index = int(text.rsplit(" ", 1)[1])
        return RAGResponse(
            answer=f"fact {index}",
            framework=FrameworkName.CUSTOM,
            model="local",
            latency=LatencyMetrics(total_ms=float(3 - index)),
            usage=UsageMetrics(estimated_cost_usd=0),
            evidence_status=EvidenceStatus.SUFFICIENT,
        )

    run = await EvaluationRunner(query).run(
        dataset,
        EvaluationRunConfig(
            framework=FrameworkName.CUSTOM,
            retrieval_mode=RetrievalMode.HYBRID,
            top_k=5,
            rerank=True,
            model="local",
            concurrency=3,
        ),
    )
    report = build_report(run)
    json_path, markdown_path = write_report(report, tmp_path)

    assert [result.question_id for result in run.results] == ["q0", "q1", "q2"]
    assert report.successful_questions == 3
    assert json.loads(json_path.read_text())["run"]["dataset_name"] == "test-dataset"
    assert "not a claim that one framework is best" in markdown_path.read_text()
