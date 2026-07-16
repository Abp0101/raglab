from collections.abc import Sequence
from uuid import uuid4

import pytest
from llama_index.core.callbacks import CBEventType, EventPayload
from llama_index.core.llms import ChatResponse
from llama_index.core.prompts import PromptTemplate
from llama_index.core.schema import TextNode

from raglab.core.interfaces import RAGPipeline
from raglab.core.schemas import (
    DocumentInput,
    EvidenceStatus,
    FrameworkName,
    PipelineConfig,
    QueryRequest,
    RetrievalOptions,
    RetrievalRequest,
    RetrievedChunk,
    UsageMetrics,
)
from raglab.generation.llamaindex_ollama import OllamaUsageCapture
from raglab.generation.output import GeneratedCitation, GroundedAnswer
from raglab.pipelines.llamaindex_rag import (
    LlamaIndexRAGPipeline,
    SharedStoreLlamaIndexRetriever,
)
from raglab.retrieval.service import RetrievalService
from tests.unit.retrieval_fixtures import make_chunk


class FakeIngestion:
    async def ingest(self, document: DocumentInput) -> object:
        raise AssertionError("ingestion is not used in query tests")


class FakeEmbeddings:
    model_name = "local-test"

    async def embed_chunks(self, chunks: Sequence[object]) -> Sequence[object]:
        return ()

    async def embed_query(self, query: str) -> Sequence[float]:
        return (0.1, 0.2)


class FakeDenseRetriever:
    def __init__(self, results: Sequence[RetrievedChunk]) -> None:
        self.results = results

    async def retrieve(
        self,
        request: RetrievalRequest,
        query_vector: Sequence[float],
    ) -> Sequence[RetrievedChunk]:
        return self.results


class FakeSparseRetriever:
    async def retrieve(self, request: RetrievalRequest) -> Sequence[RetrievedChunk]:
        return ()


def retrieval(results: Sequence[RetrievedChunk]) -> RetrievalService:
    return RetrievalService(
        embedding_provider=FakeEmbeddings(),  # type: ignore[arg-type]
        dense_retriever=FakeDenseRetriever(results),
        sparse_retriever=FakeSparseRetriever(),
    )


def query() -> QueryRequest:
    return QueryRequest(
        query="What was the sampling rate?",
        framework=FrameworkName.LLAMAINDEX,
        collection_id=uuid4(),
        retrieval_mode="dense",
        rerank=False,
        debug=True,
    )


class FakePredictor:
    def __init__(self, answer: GroundedAnswer) -> None:
        self.answer = answer
        self.formatted_prompt = ""

    async def predict(
        self,
        prompt: PromptTemplate,
        **prompt_args: str,
    ) -> tuple[GroundedAnswer, UsageMetrics, str]:
        self.formatted_prompt = prompt.format(**prompt_args)
        return (
            self.answer,
            UsageMetrics(
                prompt_tokens=40,
                completion_tokens=12,
                total_tokens=52,
                estimated_cost_usd=0,
                llm_calls=1,
            ),
            "local-test",
        )


def pipeline(
    results: Sequence[RetrievedChunk],
    answer: GroundedAnswer,
) -> tuple[LlamaIndexRAGPipeline, FakePredictor]:
    predictor = FakePredictor(answer)
    adapter = LlamaIndexRAGPipeline(
        ingestion=FakeIngestion(),  # type: ignore[arg-type]
        retrieval=retrieval(results),
        model_factory=lambda model, temperature: predictor,
        default_model="local-test",
        config=PipelineConfig(candidate_k=5, top_k=3, max_context_tokens=1000),
    )
    return adapter, predictor


@pytest.mark.asyncio
async def test_llamaindex_pipeline_uses_native_nodes_prompt_and_shared_contract() -> None:
    chunk = make_chunk("The rehabilitation IMU sampled motion at 100 Hz.")
    result = RetrievedChunk(
        chunk=chunk,
        rank=1,
        relevance_score=0.8,
        reranker_score=0.95,
    )
    generated = GroundedAnswer(
        answer="The IMU sampled motion at 100 Hz.",
        citations=(
            GeneratedCitation(chunk_id=chunk.chunk_id, quoted_text="sampled motion at 100 Hz"),
        ),
        evidence_status=EvidenceStatus.SUFFICIENT,
        confidence=0.9,
    )
    adapter, predictor = pipeline((result,), generated)

    response = await adapter.query(query())

    assert isinstance(adapter, RAGPipeline)
    assert response.framework is FrameworkName.LLAMAINDEX
    assert response.citations[0].chunk_id == chunk.chunk_id
    assert response.usage.total_tokens == 52
    assert response.usage.estimated_cost_usd == 0
    assert "What was the sampling rate?" in predictor.formatted_prompt
    assert response.debug is not None
    assert response.debug["orchestration"][0] == "BaseRetriever.aretrieve"
    assert response.debug["llamaindex_node_ids"] == [str(chunk.chunk_id)]


@pytest.mark.asyncio
async def test_llamaindex_retriever_maps_scores_and_serialized_provenance() -> None:
    chunk = make_chunk("Evidence says 100 Hz.")
    result = RetrievedChunk(chunk=chunk, rank=1, relevance_score=0.7, reranker_score=0.91)
    request = RetrievalRequest(
        query="sampling", collection_id=chunk.metadata.collection_id, top_k=5
    )
    retriever = SharedStoreLlamaIndexRetriever(
        service=retrieval((result,)),
        request=request,
        options=RetrievalOptions(mode="dense", candidate_k=5, top_k=5, rerank=False),
    )

    nodes = await retriever.aretrieve("sampling")

    assert isinstance(nodes[0].node, TextNode)
    assert nodes[0].node.node_id == str(chunk.chunk_id)
    assert nodes[0].score == 0.91
    restored = RetrievedChunk.model_validate_json(nodes[0].node.metadata["raglab_retrieved_chunk"])
    assert restored == result


@pytest.mark.asyncio
async def test_llamaindex_pipeline_refuses_without_evidence_and_skips_model() -> None:
    generated = GroundedAnswer(
        answer="unused",
        citations=(),
        evidence_status=EvidenceStatus.INSUFFICIENT,
        confidence=0,
    )
    adapter, predictor = pipeline((), generated)

    response = await adapter.query(query())

    assert response.evidence_status is EvidenceStatus.INSUFFICIENT
    assert response.usage.llm_calls == 0
    assert response.usage.estimated_cost_usd == 0
    assert predictor.formatted_prompt == ""


@pytest.mark.asyncio
async def test_llamaindex_pipeline_rejects_invalid_citation() -> None:
    chunk = make_chunk("Evidence says 100 Hz.")
    generated = GroundedAnswer(
        answer="It was 200 Hz.",
        citations=(GeneratedCitation(chunk_id=chunk.chunk_id, quoted_text="200 Hz"),),
        evidence_status=EvidenceStatus.SUFFICIENT,
        confidence=0.5,
    )
    adapter, _ = pipeline((RetrievedChunk(chunk=chunk, rank=1),), generated)

    response = await adapter.query(query())

    assert response.evidence_status is EvidenceStatus.INSUFFICIENT
    assert response.citations == ()
    assert response.warnings[-1] == "answer was rejected because it had no valid citations"


def test_llamaindex_ollama_usage_capture_normalizes_local_tokens() -> None:
    capture = OllamaUsageCapture("requested-model")
    response = ChatResponse(
        message={"role": "assistant", "content": "{}"},
        raw={
            "model": "resolved-model",
            "usage": {
                "prompt_tokens": 17,
                "completion_tokens": 5,
                "total_tokens": 22,
            },
        },
    )

    capture.on_event_end(CBEventType.LLM, {EventPayload.RESPONSE: response})
    usage = capture.usage()

    assert capture.model == "resolved-model"
    assert usage.prompt_tokens == 17
    assert usage.completion_tokens == 5
    assert usage.total_tokens == 22
    assert usage.estimated_cost_usd == 0
    assert usage.llm_calls == 1
