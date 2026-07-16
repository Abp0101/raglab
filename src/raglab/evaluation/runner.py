"""Bounded, framework-neutral evaluation execution."""

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from uuid import uuid4

from raglab.core.schemas import (
    EvaluationDataset,
    EvaluationResult,
    EvaluationRun,
    EvaluationRunConfig,
    QueryRequest,
    RAGResponse,
)
from raglab.evaluation.metrics import evaluate_response


class EvaluationRunner:
    """Run every question with a fixed query configuration and bounded concurrency."""

    def __init__(
        self,
        query: Callable[[QueryRequest], Awaitable[RAGResponse]],
    ) -> None:
        self._query = query

    async def run(
        self,
        dataset: EvaluationDataset,
        config: EvaluationRunConfig,
    ) -> EvaluationRun:
        started_at = datetime.now(UTC)
        semaphore = asyncio.Semaphore(config.concurrency)

        async def evaluate(question_index: int) -> tuple[int, EvaluationResult]:
            question = dataset.questions[question_index]
            async with semaphore:
                try:
                    response = await self._query(
                        QueryRequest(
                            query=question.question,
                            framework=config.framework,
                            collection_id=dataset.manifest.collection_id,
                            top_k=config.top_k,
                            retrieval_mode=config.retrieval_mode,
                            rerank=config.rerank,
                            model=config.model,
                        )
                    )
                    retrieval, answers = evaluate_response(question, response)
                    result = EvaluationResult(
                        evaluation_id=uuid4(),
                        question_id=question.question_id,
                        dataset_version=question.dataset_version,
                        framework=config.framework,
                        response=response,
                        retrieval_metrics=retrieval,
                        answer_metrics=answers,
                        created_at=datetime.now(UTC),
                    )
                except Exception as error:
                    result = EvaluationResult(
                        evaluation_id=uuid4(),
                        question_id=question.question_id,
                        dataset_version=question.dataset_version,
                        framework=config.framework,
                        created_at=datetime.now(UTC),
                        error=type(error).__name__,
                    )
                return question_index, result

        indexed_results = await asyncio.gather(
            *(evaluate(index) for index in range(len(dataset.questions)))
        )
        ordered = tuple(result for _, result in sorted(indexed_results))
        return EvaluationRun(
            run_id=uuid4(),
            dataset_name=dataset.manifest.name,
            dataset_version=dataset.manifest.version,
            dataset_sha256=dataset.manifest.questions_sha256,
            config=config,
            started_at=started_at,
            completed_at=datetime.now(UTC),
            results=ordered,
        )
