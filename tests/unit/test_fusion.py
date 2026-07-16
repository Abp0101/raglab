from raglab.core.schemas import RetrievedChunk
from raglab.retrieval.fusion import reciprocal_rank_fusion
from tests.unit.retrieval_fixtures import make_chunk


def test_rrf_rewards_chunks_present_in_both_rankings_and_preserves_scores() -> None:
    shared = make_chunk("shared")
    dense_only = make_chunk("dense")
    sparse_only = make_chunk("sparse")
    dense = (
        RetrievedChunk(chunk=dense_only, rank=1, dense_score=0.9, relevance_score=0.9),
        RetrievedChunk(chunk=shared, rank=2, dense_score=0.8, relevance_score=0.8),
    )
    sparse = (
        RetrievedChunk(chunk=shared, rank=1, sparse_score=3.2, relevance_score=3.2),
        RetrievedChunk(chunk=sparse_only, rank=2, sparse_score=2.1, relevance_score=2.1),
    )

    fused = reciprocal_rank_fusion((dense, sparse), rrf_k=60)

    assert fused[0].chunk.chunk_id == shared.chunk_id
    assert fused[0].dense_score == 0.8
    assert fused[0].sparse_score == 3.2
    assert fused[0].fusion_score == 1 / 62 + 1 / 61
    assert [result.rank for result in fused] == [1, 2, 3]


def test_rrf_rejects_invalid_constant() -> None:
    try:
        reciprocal_rank_fusion((), rrf_k=0)
    except ValueError as error:
        assert "positive" in str(error)
    else:
        raise AssertionError("invalid RRF constant was accepted")
