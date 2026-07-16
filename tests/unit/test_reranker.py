import numpy as np
import pytest

from raglab.core.schemas import RetrievedChunk
from raglab.reranking import CrossEncoderReranker
from tests.unit.retrieval_fixtures import make_chunk


class FakeCrossEncoder:
    def predict(self, sentences: list[tuple[str, str]], **kwargs: object) -> np.ndarray:
        assert kwargs["show_progress_bar"] is False
        return np.array([0.1 if "weak" in text else 0.9 for _, text in sentences])


@pytest.mark.asyncio
async def test_cross_encoder_reorders_and_preserves_first_stage_scores() -> None:
    candidates = (
        RetrievedChunk(chunk=make_chunk("weak evidence"), rank=1, dense_score=0.95),
        RetrievedChunk(chunk=make_chunk("strong evidence"), rank=2, sparse_score=2.0),
    )
    reranker = CrossEncoderReranker("test", model=FakeCrossEncoder())

    results = await reranker.rerank("question", candidates, top_k=2)

    assert results[0].chunk.text == "strong evidence"
    assert results[0].reranker_score == 0.9
    assert results[0].sparse_score == 2.0
    assert results[1].dense_score == 0.95


@pytest.mark.asyncio
async def test_cross_encoder_skips_model_for_empty_candidates() -> None:
    reranker = CrossEncoderReranker("not-loaded")
    assert await reranker.rerank("question", (), top_k=5) == ()


@pytest.mark.live_model
@pytest.mark.asyncio
async def test_default_cross_encoder_ranks_relevant_evidence_first() -> None:
    reranker = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
    candidates = (
        RetrievedChunk(chunk=make_chunk("The sky appears blue."), rank=1),
        RetrievedChunk(
            chunk=make_chunk("The rehabilitation IMU sampled motion at 100 Hz."), rank=2
        ),
    )

    results = await reranker.rerank("What was the IMU sampling rate?", candidates, top_k=2)

    assert results[0].chunk.text.endswith("100 Hz.")
