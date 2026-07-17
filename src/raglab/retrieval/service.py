"""Framework-free dense, sparse, hybrid, reranked retrieval orchestration."""

import asyncio
from collections.abc import Sequence

from raglab.core.exceptions import ProviderUnavailableError, RAGLabError
from raglab.core.interfaces import (
    ContextExpander,
    DenseRetriever,
    EmbeddingProvider,
    Reranker,
    SparseRetriever,
)
from raglab.core.schemas import (
    RetrievalMode,
    RetrievalOptions,
    RetrievalRequest,
    RetrievedChunk,
)
from raglab.retrieval.fusion import reciprocal_rank_fusion


class RetrievalService:
    """Execute comparable retrieval configurations behind one boundary."""

    def __init__(
        self,
        *,
        embedding_provider: EmbeddingProvider,
        dense_retriever: DenseRetriever,
        sparse_retriever: SparseRetriever,
        reranker: Reranker | None = None,
        context_expander: ContextExpander | None = None,
    ) -> None:
        self._embedding_provider = embedding_provider
        self._dense_retriever = dense_retriever
        self._sparse_retriever = sparse_retriever
        self._reranker = reranker
        self._context_expander = context_expander

    async def retrieve(
        self,
        request: RetrievalRequest,
        options: RetrievalOptions,
    ) -> Sequence[RetrievedChunk]:
        if options.rerank and self._reranker is None:
            raise ValueError("reranking was requested but no reranker is configured")
        try:
            return await self._retrieve(request, options)
        except RAGLabError:
            raise
        except Exception as error:
            raise ProviderUnavailableError("retrieval provider request failed") from error

    async def _retrieve(
        self,
        request: RetrievalRequest,
        options: RetrievalOptions,
    ) -> Sequence[RetrievedChunk]:
        candidate_request = request.model_copy(update={"top_k": options.candidate_k})
        candidates = await self._first_stage(candidate_request, options)
        if options.rerank:
            reranker = self._reranker
            if reranker is None:
                raise RuntimeError("validated reranker configuration changed during retrieval")
            candidates = await reranker.rerank(request.query, candidates, options.candidate_k)
        if options.expand_parents and self._context_expander is not None:
            candidates = await self._context_expander.expand(candidates)
        return tuple(
            result.model_copy(update={"rank": rank})
            for rank, result in enumerate(candidates[: options.top_k], start=1)
        )

    async def _first_stage(
        self,
        request: RetrievalRequest,
        options: RetrievalOptions,
    ) -> Sequence[RetrievedChunk]:
        if options.mode is RetrievalMode.SPARSE:
            return await self._sparse_retriever.retrieve(request)
        query_vector = await self._embedding_provider.embed_query(request.query)
        if options.mode is RetrievalMode.DENSE:
            return await self._dense_retriever.retrieve(request, query_vector)
        dense, sparse = await asyncio.gather(
            self._dense_retriever.retrieve(request, query_vector),
            self._sparse_retriever.retrieve(request),
        )
        return reciprocal_rank_fusion(
            (dense, sparse), rrf_k=options.rrf_k, limit=options.candidate_k
        )
