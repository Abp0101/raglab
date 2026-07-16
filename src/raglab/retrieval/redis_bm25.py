"""Redis-backed lexical token persistence for BM25 retrieval."""

import json
import math
import re
from collections import Counter
from collections.abc import Awaitable, Sequence
from typing import Any, cast

from redis.asyncio import Redis

from raglab.core.schemas import Chunk, MetadataFilter, RetrievalRequest, RetrievedChunk
from raglab.retrieval.serialization import chunk_to_payload, payload_to_chunk

TOKEN_PATTERN = re.compile(r"(?u)\b\w\w+\b")


def tokenize(text: str) -> tuple[str, ...]:
    """Apply deterministic case-folded lexical tokenization."""
    return tuple(TOKEN_PATTERN.findall(text.casefold()))


class RedisBM25Indexer:
    """Persist tokenized chunks by logical collection and source document."""

    def __init__(self, client: Redis, *, key_prefix: str = "raglab:bm25") -> None:
        self._client = client
        self._key_prefix = key_prefix.rstrip(":")

    async def upsert(self, chunks: Sequence[Chunk]) -> None:
        if not chunks:
            return
        pipeline = self._client.pipeline(transaction=True)
        for chunk in chunks:
            payload = json.dumps(
                {
                    "tokens": tokenize(chunk.text),
                    "chunk": chunk_to_payload(chunk),
                },
                separators=(",", ":"),
            )
            pipeline.hset(
                self._collection_key(str(chunk.metadata.collection_id)),
                str(chunk.chunk_id),
                payload,
            )
            pipeline.sadd(self._document_key(str(chunk.metadata.document_id)), str(chunk.chunk_id))
        await pipeline.execute()

    async def delete(self, chunks: Sequence[Chunk]) -> None:
        if not chunks:
            return
        pipeline = self._client.pipeline(transaction=True)
        for chunk in chunks:
            pipeline.hdel(
                self._collection_key(str(chunk.metadata.collection_id)), str(chunk.chunk_id)
            )
            pipeline.srem(self._document_key(str(chunk.metadata.document_id)), str(chunk.chunk_id))
        await pipeline.execute()

    def _collection_key(self, collection_id: str) -> str:
        return f"{self._key_prefix}:collection:{collection_id}:chunks"

    def _document_key(self, document_id: str) -> str:
        return f"{self._key_prefix}:document:{document_id}:chunks"


class RedisBM25Retriever:
    """Score collection chunks with deterministic Okapi BM25."""

    def __init__(
        self,
        client: Redis,
        *,
        key_prefix: str = "raglab:bm25",
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self._client = client
        self._key_prefix = key_prefix.rstrip(":")
        self._k1 = k1
        self._b = b

    async def retrieve(self, request: RetrievalRequest) -> Sequence[RetrievedChunk]:
        raw_documents = await cast(
            Awaitable[dict[str | bytes, str | bytes]],
            self._client.hgetall(f"{self._key_prefix}:collection:{request.collection_id}:chunks"),
        )
        documents = [_decode_entry(value) for value in raw_documents.values()]
        filtered = [
            entry
            for entry in documents
            if _matches_filter(payload_to_chunk(entry["chunk"]), request.metadata_filter)
        ]
        query_terms = tuple(dict.fromkeys(tokenize(request.query)))
        if not filtered or not query_terms:
            return ()
        average_length = sum(len(entry["tokens"]) for entry in filtered) / len(filtered)
        document_frequency = {
            term: sum(term in set(entry["tokens"]) for entry in filtered) for term in query_terms
        }
        scored: list[tuple[float, Chunk]] = []
        for entry in filtered:
            tokens = entry["tokens"]
            frequencies = Counter(tokens)
            score = sum(
                _bm25_term_score(
                    term_frequency=frequencies[term],
                    document_frequency=document_frequency[term],
                    document_length=len(tokens),
                    average_length=average_length,
                    document_count=len(filtered),
                    k1=self._k1,
                    b=self._b,
                )
                for term in query_terms
                if frequencies[term] > 0
            )
            if score > 0:
                scored.append((score, payload_to_chunk(entry["chunk"])))
        scored.sort(key=lambda item: (-item[0], str(item[1].chunk_id)))
        return tuple(
            RetrievedChunk(
                chunk=chunk,
                rank=rank,
                relevance_score=score,
                sparse_score=score,
            )
            for rank, (score, chunk) in enumerate(scored[: request.top_k], start=1)
        )


def _decode_entry(value: str | bytes) -> dict[str, Any]:
    raw = value.decode() if isinstance(value, bytes) else value
    return json.loads(raw)  # type: ignore[no-any-return]


def _bm25_term_score(
    *,
    term_frequency: int,
    document_frequency: int,
    document_length: int,
    average_length: float,
    document_count: int,
    k1: float,
    b: float,
) -> float:
    inverse_document_frequency = math.log(
        1 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5)
    )
    normalization = term_frequency + k1 * (1 - b + b * document_length / average_length)
    return inverse_document_frequency * term_frequency * (k1 + 1) / normalization


def _matches_filter(chunk: Chunk, metadata_filter: MetadataFilter | None) -> bool:
    if metadata_filter is None:
        return True
    metadata = chunk.metadata
    if metadata_filter.document_ids and metadata.document_id not in metadata_filter.document_ids:
        return False
    if metadata_filter.authors and not set(metadata.authors).intersection(metadata_filter.authors):
        return False
    if metadata_filter.file_types and metadata.file_type not in metadata_filter.file_types:
        return False
    if (
        metadata_filter.section_headings
        and metadata.section_heading not in metadata_filter.section_headings
    ):
        return False
    if metadata_filter.published_from and (
        metadata.publication_date is None
        or metadata.publication_date < metadata_filter.published_from
    ):
        return False
    if metadata_filter.published_to and (
        metadata.publication_date is None
        or metadata.publication_date > metadata_filter.published_to
    ):
        return False
    values: dict[str, object] = {
        "file_name": metadata.file_name,
        "display_title": metadata.display_title,
        "page_number": metadata.page_number,
        "chunk_index": metadata.chunk_index,
        "content_hash": metadata.content_hash,
    }
    return all(values[key] == value for key, value in metadata_filter.attributes.items())
