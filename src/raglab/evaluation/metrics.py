"""Deterministic retrieval, citation, refusal, and lexical answer metrics."""

import math
import re
from collections.abc import Sequence
from uuid import UUID

from raglab.core.schemas import (
    EvaluationMetricResult,
    EvaluationQuestion,
    EvidenceStatus,
    RAGResponse,
    RetrievedChunk,
)

NORMALIZE_PATTERN = re.compile(r"[^a-z0-9]+")


def evaluate_response(
    question: EvaluationQuestion,
    response: RAGResponse,
) -> tuple[tuple[EvaluationMetricResult, ...], tuple[EvaluationMetricResult, ...]]:
    """Return retrieval and answer groups without any LLM-as-a-judge call."""
    retrieval = _retrieval_metrics(question, response.retrieved_chunks)
    answers = (
        _citation_precision(question, response),
        _citation_recall(question, response),
        _refusal_accuracy(question, response),
        _key_fact_coverage(question, response),
        EvaluationMetricResult(name="latency_ms", value=response.latency.total_ms),
        EvaluationMetricResult(
            name="estimated_cost_usd",
            value=response.usage.estimated_cost_usd or 0,
        ),
    )
    return retrieval, answers


def _retrieval_metrics(
    question: EvaluationQuestion,
    retrieved: Sequence[RetrievedChunk],
) -> tuple[EvaluationMetricResult, ...]:
    relevant = _relevant_ids(question)
    if not relevant:
        return tuple(
            _not_applicable(name)
            for name in ("retrieval_precision", "retrieval_recall", "mrr", "ndcg")
        )
    returned = [_result_id(question, result) for result in retrieved]
    seen: set[UUID] = set()
    hits: list[bool] = []
    for identifier in returned:
        hits.append(identifier in relevant and identifier not in seen)
        seen.add(identifier)
    hit_count = sum(hits)
    precision = hit_count / len(returned) if returned else 0
    recall = hit_count / len(relevant)
    first_hit = next((rank for rank, hit in enumerate(hits, start=1) if hit), None)
    reciprocal_rank = 1 / first_hit if first_hit else 0
    dcg = sum((1 / math.log2(rank + 1)) for rank, hit in enumerate(hits, start=1) if hit)
    ideal_hits = min(len(relevant), len(returned))
    ideal_dcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    ndcg = dcg / ideal_dcg if ideal_dcg else 0
    details = {"relevant_count": len(relevant), "retrieved_count": len(returned)}
    return (
        EvaluationMetricResult(name="retrieval_precision", value=precision, details=details),
        EvaluationMetricResult(name="retrieval_recall", value=recall, details=details),
        EvaluationMetricResult(name="mrr", value=reciprocal_rank, details=details),
        EvaluationMetricResult(name="ndcg", value=ndcg, details=details),
    )


def _citation_precision(
    question: EvaluationQuestion, response: RAGResponse
) -> EvaluationMetricResult:
    expected = set(question.expected_citation_chunk_ids or question.relevant_chunk_ids)
    if not expected:
        return _not_applicable("citation_precision")
    cited = {citation.chunk_id for citation in response.citations}
    value = len(cited & expected) / len(cited) if cited else 0
    return EvaluationMetricResult(
        name="citation_precision",
        value=value,
        details={"expected_count": len(expected), "cited_count": len(cited)},
    )


def _citation_recall(question: EvaluationQuestion, response: RAGResponse) -> EvaluationMetricResult:
    expected = set(question.expected_citation_chunk_ids or question.relevant_chunk_ids)
    if not expected:
        return _not_applicable("citation_recall")
    cited = {citation.chunk_id for citation in response.citations}
    return EvaluationMetricResult(
        name="citation_recall",
        value=len(cited & expected) / len(expected),
        details={"expected_count": len(expected), "cited_count": len(cited)},
    )


def _refusal_accuracy(
    question: EvaluationQuestion, response: RAGResponse
) -> EvaluationMetricResult:
    refused = response.evidence_status is EvidenceStatus.INSUFFICIENT
    correct = refused is (not question.answerable)
    return EvaluationMetricResult(
        name="refusal_accuracy",
        value=float(correct),
        passed=correct,
        details={"answerable": question.answerable, "refused": refused},
    )


def _key_fact_coverage(
    question: EvaluationQuestion, response: RAGResponse
) -> EvaluationMetricResult:
    if not question.expected_key_facts:
        return _not_applicable("key_fact_coverage")
    answer = _normalize(response.answer)
    covered = sum(_normalize(fact) in answer for fact in question.expected_key_facts)
    return EvaluationMetricResult(
        name="key_fact_coverage",
        value=covered / len(question.expected_key_facts),
        details={"expected_count": len(question.expected_key_facts), "covered_count": covered},
    )


def _relevant_ids(question: EvaluationQuestion) -> set[UUID]:
    return set(question.relevant_chunk_ids or question.relevant_document_ids)


def _result_id(question: EvaluationQuestion, result: RetrievedChunk) -> UUID:
    if question.relevant_chunk_ids:
        return result.chunk.chunk_id
    return result.chunk.metadata.document_id


def _not_applicable(name: str) -> EvaluationMetricResult:
    return EvaluationMetricResult(
        name=name,
        value=0,
        passed=None,
        details={"applicable": False},
    )


def _normalize(value: str) -> str:
    return " ".join(NORMALIZE_PATTERN.sub(" ", value.casefold()).split())
