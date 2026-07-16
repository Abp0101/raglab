import json
from collections.abc import Sequence
from uuid import uuid4

import pytest

from raglab.core.exceptions import MalformedProviderResponseError
from raglab.core.schemas import (
    DocumentInput,
    EvidenceStatus,
    FrameworkName,
    GenerationRequest,
    GenerationResult,
    PipelineConfig,
    QueryRequest,
    RetrievalOptions,
    RetrievalRequest,
    RetrievedChunk,
    UsageMetrics,
)
from raglab.pipelines.custom_rag import REFUSAL, CustomRAGPipeline
from tests.unit.retrieval_fixtures import make_chunk


class FakeIngestion:
    async def ingest(self, document: DocumentInput) -> object:
        raise AssertionError("ingestion is not used by query tests")


class FakeRetrieval:
    def __init__(self, results: Sequence[RetrievedChunk]) -> None:
        self.results = results

    async def retrieve(
        self, request: RetrievalRequest, options: RetrievalOptions
    ) -> Sequence[RetrievedChunk]:
        return self.results


class FakeLLM:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        self.calls += 1
        assert request.response_schema is not None
        return GenerationResult(
            text=self.text,
            model=request.model,
            usage=UsageMetrics(
                prompt_tokens=50, completion_tokens=10, total_tokens=60, llm_calls=1
            ),
        )


def pipeline(results: Sequence[RetrievedChunk], llm: FakeLLM) -> CustomRAGPipeline:
    return CustomRAGPipeline(
        ingestion=FakeIngestion(),  # type: ignore[arg-type]
        retrieval=FakeRetrieval(results),  # type: ignore[arg-type]
        llm=llm,
        default_model="test-model",
        config=PipelineConfig(candidate_k=5, top_k=3, max_context_tokens=1000),
    )


def query() -> QueryRequest:
    return QueryRequest(
        query="What was the sampling rate?",
        framework=FrameworkName.CUSTOM,
        collection_id=uuid4(),
        rerank=False,
        debug=True,
    )


@pytest.mark.asyncio
async def test_pipeline_returns_grounded_answer_with_validated_citation_and_usage() -> None:
    chunk = make_chunk("The rehabilitation IMU sampled motion at 100 Hz.")
    retrieved = RetrievedChunk(chunk=chunk, rank=1, dense_score=0.9, relevance_score=0.9)
    llm = FakeLLM(
        json.dumps(
            {
                "answer": "The IMU sampled motion at 100 Hz.",
                "citations": [
                    {"chunk_id": str(chunk.chunk_id), "quoted_text": "sampled motion at 100 Hz"}
                ],
                "evidence_status": "sufficient",
                "confidence": 0.91,
                "warnings": [],
            }
        )
    )

    response = await pipeline([retrieved], llm).query(query())

    assert response.answer.endswith("100 Hz.")
    assert response.evidence_sufficient is True
    assert response.citations[0].chunk_id == chunk.chunk_id
    assert response.prompt_tokens == 50
    assert response.debug is not None and response.debug["context_estimated_tokens"] > 0


@pytest.mark.asyncio
async def test_pipeline_refuses_without_retrieved_evidence_and_skips_llm() -> None:
    llm = FakeLLM("not used")

    response = await pipeline([], llm).query(query())

    assert response.answer == REFUSAL
    assert response.evidence_status is EvidenceStatus.INSUFFICIENT
    assert response.citations == ()
    assert llm.calls == 0


@pytest.mark.asyncio
async def test_pipeline_rejects_supported_answer_without_valid_citations() -> None:
    chunk = make_chunk("Evidence says 100 Hz.")
    llm = FakeLLM(
        json.dumps(
            {
                "answer": "It was 200 Hz.",
                "citations": [
                    {"chunk_id": str(chunk.chunk_id), "quoted_text": "Evidence says 200 Hz"}
                ],
                "evidence_status": "sufficient",
                "confidence": 0.8,
                "warnings": [],
            }
        )
    )

    response = await pipeline([RetrievedChunk(chunk=chunk, rank=1)], llm).query(query())

    assert response.answer == REFUSAL
    assert response.evidence_sufficient is False
    assert "no valid citations" in response.warnings[-1]


@pytest.mark.asyncio
async def test_pipeline_rejects_malformed_model_json() -> None:
    chunk = make_chunk("Evidence says 100 Hz.")

    with pytest.raises(MalformedProviderResponseError, match="grounded answer schema"):
        await pipeline([RetrievedChunk(chunk=chunk, rank=1)], FakeLLM("not-json")).query(query())
