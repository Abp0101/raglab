import json
from datetime import date
from uuid import uuid4

import pytest

from raglab.core.schemas import MetadataFilter, RetrievalRequest
from raglab.retrieval.redis_bm25 import RedisBM25Retriever, tokenize
from raglab.retrieval.serialization import chunk_to_payload
from tests.unit.retrieval_fixtures import make_chunk


class FakeRedis:
    def __init__(self, values: dict[str, str]) -> None:
        self.values = values

    async def hgetall(self, key: str) -> dict[str, str]:
        return self.values


def entry(chunk_text: str, chunk: object) -> str:
    return json.dumps({"tokens": tokenize(chunk_text), "chunk": chunk_to_payload(chunk)})  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_bm25_ranks_rare_matching_terms_and_applies_filters() -> None:
    collection_id = uuid4()
    calibration = make_chunk(
        "calibration calibration sensor protocol",
        collection_id=collection_id,
        section_heading="METHODS",
    )
    generic = make_chunk(
        "sensor protocol overview",
        collection_id=collection_id,
        section_heading="BACKGROUND",
        chunk_index=1,
    )
    redis = FakeRedis(
        {
            str(calibration.chunk_id): entry(calibration.text, calibration),
            str(generic.chunk_id): entry(generic.text, generic),
        }
    )
    retriever = RedisBM25Retriever(redis)  # type: ignore[arg-type]

    results = await retriever.retrieve(
        RetrievalRequest(
            query="calibration sensor",
            collection_id=collection_id,
            metadata_filter=MetadataFilter(
                section_headings=("METHODS",), published_to=date(2025, 12, 31)
            ),
        )
    )

    assert [result.chunk.chunk_id for result in results] == [calibration.chunk_id]
    assert results[0].sparse_score is not None and results[0].sparse_score > 0


@pytest.mark.asyncio
async def test_bm25_returns_empty_when_query_has_no_matching_terms() -> None:
    collection_id = uuid4()
    chunk = make_chunk("wearable sensor", collection_id=collection_id)
    retriever = RedisBM25Retriever(
        FakeRedis({str(chunk.chunk_id): entry(chunk.text, chunk)})  # type: ignore[arg-type]
    )

    assert (
        await retriever.retrieve(
            RetrievalRequest(query="unrelated vocabulary", collection_id=collection_id)
        )
        == ()
    )
