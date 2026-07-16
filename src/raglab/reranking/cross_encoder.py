"""Lazy asynchronous cross-encoder reranking."""

import asyncio
from collections.abc import Sequence
from typing import Any, Protocol, cast

from raglab.core.schemas import RetrievedChunk


class PairScorer(Protocol):
    """Narrow surface used from Sentence Transformers CrossEncoder."""

    def predict(self, sentences: Sequence[tuple[str, str]], **kwargs: Any) -> Any: ...


class CrossEncoderReranker:
    """Score query/chunk pairs locally without blocking the event loop."""

    def __init__(
        self,
        model_name: str,
        *,
        batch_size: int = 16,
        model: PairScorer | None = None,
    ) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._model = model
        self._load_lock = asyncio.Lock()

    async def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievedChunk],
        top_k: int,
    ) -> Sequence[RetrievedChunk]:
        if not candidates:
            return ()
        model = await self._get_model()
        raw = await asyncio.to_thread(
            model.predict,
            [(query, candidate.chunk.text) for candidate in candidates],
            batch_size=self._batch_size,
            show_progress_bar=False,
        )
        scores = cast(list[float], raw.tolist())
        scored = sorted(
            zip(candidates, scores, strict=True),
            key=lambda item: (-float(item[1]), str(item[0].chunk.chunk_id)),
        )[:top_k]
        return tuple(
            candidate.model_copy(
                update={
                    "rank": rank,
                    "relevance_score": float(score),
                    "reranker_score": float(score),
                }
            )
            for rank, (candidate, score) in enumerate(scored, start=1)
        )

    async def _get_model(self) -> PairScorer:
        if self._model is not None:
            return self._model
        async with self._load_lock:
            if self._model is None:
                from sentence_transformers import CrossEncoder

                self._model = await asyncio.to_thread(CrossEncoder, self._model_name)
        return cast(PairScorer, self._model)
