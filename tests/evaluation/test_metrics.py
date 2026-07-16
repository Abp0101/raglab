from uuid import uuid4

import pytest

from raglab.core.schemas import (
    Citation,
    EvaluationDifficulty,
    EvaluationQuestion,
    EvidenceStatus,
    FrameworkName,
    LatencyMetrics,
    RAGResponse,
    RetrievedChunk,
    UsageMetrics,
)
from raglab.evaluation.metrics import evaluate_response
from tests.unit.retrieval_fixtures import make_chunk


def test_deterministic_metrics_score_rank_citations_refusal_and_facts() -> None:
    relevant_id = uuid4()
    relevant = make_chunk(
        "The IMU sampled at 100 Hz.",
        chunk_id=relevant_id,
        document_id=uuid4(),
    )
    irrelevant = make_chunk("Battery life was six hours.")
    question = EvaluationQuestion(
        question_id="q1",
        dataset_version="1",
        question="What was the sampling rate?",
        expected_key_facts=("100 Hz",),
        relevant_chunk_ids=(relevant_id,),
        expected_citation_chunk_ids=(relevant_id,),
        answerable=True,
        category="factoid",
        difficulty=EvaluationDifficulty.EASY,
    )
    response = RAGResponse(
        answer="The sampling rate was 100 Hz.",
        citations=(
            Citation(
                document_id=relevant.metadata.document_id,
                document_title=relevant.metadata.display_title,
                page_number=1,
                chunk_id=relevant_id,
                quoted_text="100 Hz",
            ),
        ),
        retrieved_chunks=(
            RetrievedChunk(chunk=irrelevant, rank=1),
            RetrievedChunk(chunk=relevant, rank=2),
        ),
        framework=FrameworkName.CUSTOM,
        model="local",
        latency=LatencyMetrics(total_ms=12),
        usage=UsageMetrics(estimated_cost_usd=0),
        evidence_status=EvidenceStatus.SUFFICIENT,
    )

    retrieval, answer = evaluate_response(question, response)
    retrieval_values = {metric.name: metric.value for metric in retrieval}
    answer_values = {metric.name: metric.value for metric in answer}

    assert retrieval_values["retrieval_precision"] == 0.5
    assert retrieval_values["retrieval_recall"] == 1
    assert retrieval_values["mrr"] == 0.5
    assert retrieval_values["ndcg"] == pytest.approx(1 / 1.5849625007)
    assert answer_values["citation_precision"] == 1
    assert answer_values["citation_recall"] == 1
    assert answer_values["refusal_accuracy"] == 1
    assert answer_values["key_fact_coverage"] == 1
    assert answer_values["estimated_cost_usd"] == 0


def test_unanswerable_question_scores_correct_refusal_without_fake_relevance_metrics() -> None:
    question = EvaluationQuestion(
        question_id="q2",
        dataset_version="1",
        question="Which protocol?",
        answerable=False,
        category="unanswerable",
        difficulty=EvaluationDifficulty.MEDIUM,
    )
    response = RAGResponse(
        answer="The documents do not contain sufficient evidence.",
        framework=FrameworkName.CUSTOM,
        model="local",
        latency=LatencyMetrics(total_ms=2),
        evidence_status=EvidenceStatus.INSUFFICIENT,
    )

    retrieval, answer = evaluate_response(question, response)

    assert all(metric.details["applicable"] is False for metric in retrieval)
    assert next(metric for metric in answer if metric.name == "refusal_accuracy").value == 1
