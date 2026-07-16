from datetime import UTC, datetime
from uuid import uuid4

import pytest

from raglab.core.schemas import Chunk, DocumentMetadata
from raglab.retrieval.redis_bm25 import RedisBM25Indexer, tokenize


class FakePipeline:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...]]] = []

    def hset(self, *args: object) -> "FakePipeline":
        self.calls.append(("hset", args))
        return self

    def sadd(self, *args: object) -> "FakePipeline":
        self.calls.append(("sadd", args))
        return self

    def hdel(self, *args: object) -> "FakePipeline":
        self.calls.append(("hdel", args))
        return self

    def srem(self, *args: object) -> "FakePipeline":
        self.calls.append(("srem", args))
        return self

    async def execute(self) -> list[bool]:
        return [True] * len(self.calls)


class FakeRedis:
    def __init__(self) -> None:
        self.last_pipeline: FakePipeline | None = None

    def pipeline(self, *, transaction: bool) -> FakePipeline:
        assert transaction is True
        self.last_pipeline = FakePipeline()
        return self.last_pipeline


def make_chunk() -> Chunk:
    document_id = uuid4()
    return Chunk(
        chunk_id=uuid4(),
        text="IMUs measured knee-motion at 100 Hz.",
        metadata=DocumentMetadata(
            document_id=document_id,
            collection_id=uuid4(),
            file_name="imu.pdf",
            display_title="IMU",
            uploaded_at=datetime.now(UTC),
            file_type="application/pdf",
            chunk_index=0,
            content_hash="d" * 64,
        ),
    )


def test_tokenizer_is_deterministic_and_case_folded() -> None:
    assert tokenize("IMUs, imuS; 100 Hz") == ("imus", "imus", "100", "hz")


@pytest.mark.asyncio
async def test_indexer_persists_and_deletes_collection_tokens() -> None:
    redis = FakeRedis()
    indexer = RedisBM25Indexer(redis, key_prefix="test")  # type: ignore[arg-type]
    chunk = make_chunk()

    await indexer.upsert([chunk])
    assert redis.last_pipeline is not None
    assert [call[0] for call in redis.last_pipeline.calls] == ["hset", "sadd"]

    await indexer.delete([chunk])
    assert redis.last_pipeline is not None
    assert [call[0] for call in redis.last_pipeline.calls] == ["hdel", "srem"]
