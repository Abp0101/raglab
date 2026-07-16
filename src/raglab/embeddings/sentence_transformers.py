"""Lazy asynchronous Sentence Transformers embedding provider."""

import asyncio
from collections.abc import Sequence
from typing import Any, Protocol, cast

from raglab.core.schemas import Chunk, Embedding


class SentenceEncoder(Protocol):
    """Narrow typed surface used from the third-party encoder."""

    def encode(self, sentences: Sequence[str], **kwargs: Any) -> Any: ...


class SentenceTransformerEmbeddingProvider:
    """Generate normalized local embeddings without blocking the event loop."""

    def __init__(
        self,
        model_name: str,
        *,
        batch_size: int = 32,
        model: SentenceEncoder | None = None,
    ) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._model = model
        self._load_lock = asyncio.Lock()

    @property
    def model_name(self) -> str:
        return self._model_name

    async def embed_chunks(self, chunks: Sequence[Chunk]) -> Sequence[Embedding]:
        if not chunks:
            return ()
        vectors = await self._encode([chunk.text for chunk in chunks])
        return tuple(
            Embedding(
                chunk_id=chunk.chunk_id,
                vector=tuple(vector),
                model=self.model_name,
                dimensions=len(vector),
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        )

    async def embed_query(self, query: str) -> Sequence[float]:
        vectors = await self._encode([query])
        return tuple(vectors[0])

    async def _encode(self, texts: Sequence[str]) -> list[list[float]]:
        model = await self._get_model()
        raw = await asyncio.to_thread(
            model.encode,
            list(texts),
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return cast(list[list[float]], raw.tolist())

    async def _get_model(self) -> SentenceEncoder:
        if self._model is not None:
            return self._model
        async with self._load_lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                self._model = await asyncio.to_thread(SentenceTransformer, self.model_name)
        return cast(SentenceEncoder, self._model)
