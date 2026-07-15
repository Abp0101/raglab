from collections.abc import Sequence
from uuid import UUID, uuid4

import pytest

from raglab.core.interfaces import RAGPipeline
from raglab.core.schemas import (
    DocumentInput,
    EvidenceStatus,
    FrameworkName,
    IngestionResult,
    LatencyMetrics,
    PipelineCapabilities,
    PipelineConfig,
    QueryRequest,
    RAGResponse,
    UsageMetrics,
)


class ContractPipeline:
    @property
    def config(self) -> PipelineConfig:
        return PipelineConfig()

    @property
    def capabilities(self) -> PipelineCapabilities:
        return PipelineCapabilities()

    async def ingest(self, documents: Sequence[DocumentInput]) -> Sequence[IngestionResult]:
        return [
            IngestionResult(
                document_id=uuid4(),
                collection_id=document.collection_id,
                page_count=1,
                chunk_count=1,
                duration_ms=1,
                parser="contract-parser",
                chunking_strategy="recursive",
                embedding_model="contract-embedding",
            )
            for document in documents
        ]

    async def query(self, request: QueryRequest) -> RAGResponse:
        return RAGResponse(
            answer="The available evidence is insufficient.",
            framework=request.framework,
            model=request.model or "contract-model",
            latency=LatencyMetrics(total_ms=2, retrieval_ms=1, generation_ms=1),
            usage=UsageMetrics(
                prompt_tokens=20,
                completion_tokens=8,
                total_tokens=28,
                estimated_cost_usd=0.001,
                llm_calls=1,
            ),
            evidence_status=EvidenceStatus.INSUFFICIENT,
            warnings=("No relevant evidence passed the threshold.",),
        )


@pytest.mark.asyncio
async def test_pipeline_implements_shared_ingestion_and_query_contract() -> None:
    pipeline = ContractPipeline()
    collection_id = uuid4()

    assert isinstance(pipeline, RAGPipeline)
    ingestion = await pipeline.ingest(
        [DocumentInput(file_name="study.pdf", content=b"%PDF-1.7", collection_id=collection_id)]
    )
    response = await pipeline.query(
        QueryRequest(
            query="What was measured?",
            framework=FrameworkName.CUSTOM,
            collection_id=collection_id,
        )
    )

    assert ingestion[0].collection_id == collection_id
    assert response.evidence_sufficient is False
    assert response.citations == ()
    assert response.model_dump(mode="json") | {
        "latency_ms": 2.0,
        "prompt_tokens": 20,
        "completion_tokens": 8,
        "estimated_cost": 0.001,
        "evidence_sufficient": False,
    } == response.model_dump(mode="json")


def test_pipeline_contract_is_structural() -> None:
    pipeline: RAGPipeline = ContractPipeline()
    collection_id: UUID = uuid4()

    assert pipeline.config.top_k == 5
    assert collection_id.version == 4
