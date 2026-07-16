"""Redis-backed lexical token persistence for BM25 retrieval."""

import json
import re
from collections.abc import Sequence

from redis.asyncio import Redis

from raglab.core.schemas import Chunk

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
                    "document_id": str(chunk.metadata.document_id),
                    "tokens": tokenize(chunk.text),
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
