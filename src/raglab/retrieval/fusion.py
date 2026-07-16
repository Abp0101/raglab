"""Rank fusion that preserves backend-native score provenance."""

from collections import defaultdict
from collections.abc import Sequence

from raglab.core.schemas import RetrievedChunk


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[RetrievedChunk]],
    *,
    rrf_k: int = 60,
    limit: int | None = None,
) -> tuple[RetrievedChunk, ...]:
    """Fuse rankings using RRF without normalizing incomparable native scores."""
    if rrf_k < 1:
        raise ValueError("rrf_k must be positive")
    scores: dict[object, float] = defaultdict(float)
    candidates: dict[object, RetrievedChunk] = {}
    for ranking in rankings:
        for position, result in enumerate(ranking, start=1):
            chunk_id = result.chunk.chunk_id
            scores[chunk_id] += 1 / (rrf_k + position)
            existing = candidates.get(chunk_id)
            if existing is None:
                candidates[chunk_id] = result
            else:
                candidates[chunk_id] = existing.model_copy(
                    update={
                        "dense_score": (
                            existing.dense_score
                            if existing.dense_score is not None
                            else result.dense_score
                        ),
                        "sparse_score": (
                            existing.sparse_score
                            if existing.sparse_score is not None
                            else result.sparse_score
                        ),
                    }
                )
    ordered_ids = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], str(chunk_id)))
    if limit is not None:
        ordered_ids = ordered_ids[:limit]
    return tuple(
        candidates[chunk_id].model_copy(
            update={
                "rank": rank,
                "relevance_score": scores[chunk_id],
                "fusion_score": scores[chunk_id],
            }
        )
        for rank, chunk_id in enumerate(ordered_ids, start=1)
    )
