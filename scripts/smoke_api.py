"""Exercise the complete local API path and remove the temporary corpus afterward."""

import argparse
import asyncio
import json
import time
from contextlib import suppress
from typing import Any, cast
from uuid import UUID

import pymupdf
from apps.api.main import create_app
from fastapi.testclient import TestClient
from qdrant_client import AsyncQdrantClient, models
from redis.asyncio import Redis
from sqlalchemy import delete

from raglab.core.config import Settings
from raglab.database.models import CollectionRecord
from raglab.database.session import create_engine, create_session_factory


def _pdf_bytes() -> bytes:
    pdf = pymupdf.open()  # type: ignore[no-untyped-call]
    page = pdf.new_page()
    page.insert_text((72, 72), "RESULTS", fontsize=18)
    page.insert_text(
        (72, 110),
        "The wearable rehabilitation prototype sampled IMU motion at 100 Hz.",
        fontsize=11,
    )
    content = cast(bytes, pdf.tobytes())  # type: ignore[no-untyped-call]
    pdf.close()  # type: ignore[no-untyped-call]
    return content


async def _cleanup(settings: Settings, collection_id: UUID, document_id: UUID | None) -> None:
    qdrant = AsyncQdrantClient(
        url=str(settings.qdrant_url),
        api_key=settings.qdrant_api_key,
        check_compatibility=False,
    )
    redis = Redis.from_url(str(settings.redis_dsn), decode_responses=True)
    engine = create_engine(settings.postgres_dsn)
    sessions = create_session_factory(engine)
    try:
        with suppress(Exception):
            if await qdrant.collection_exists(settings.qdrant_collection):
                await qdrant.delete(
                    collection_name=settings.qdrant_collection,
                    points_selector=models.FilterSelector(
                        filter=models.Filter(
                            must=[
                                models.FieldCondition(
                                    key="collection_id",
                                    match=models.MatchValue(value=str(collection_id)),
                                )
                            ]
                        )
                    ),
                    wait=True,
                )
        with suppress(Exception):
            keys = [f"{settings.bm25_key_prefix}:collection:{collection_id}:chunks"]
            if document_id is not None:
                keys.append(f"{settings.bm25_key_prefix}:document:{document_id}:chunks")
            await redis.delete(*keys)
        with suppress(Exception):
            async with sessions() as session, session.begin():
                await session.execute(
                    delete(CollectionRecord).where(CollectionRecord.id == collection_id)
                )
    finally:
        await qdrant.close()
        await redis.aclose()
        await engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="An already installed local Ollama model")
    args = parser.parse_args()
    settings = Settings(
        llm_provider="ollama",
        llm_model=args.model,
        allow_paid_api_usage=False,
        _env_file=None,
    )
    collection_id: UUID | None = None
    document_id: UUID | None = None
    try:
        with TestClient(create_app(settings)) as client:
            collection_response = client.post(
                "/collections", json={"name": "Temporary local API smoke test"}
            )
            collection_response.raise_for_status()
            collection_id = UUID(collection_response.json()["collection_id"])

            ingestion_response = client.post(
                f"/collections/{collection_id}/ingestion-jobs",
                files={"file": ("wearable-study.pdf", _pdf_bytes(), "application/pdf")},
            )
            ingestion_response.raise_for_status()
            job_id = ingestion_response.json()["job_id"]
            deadline = time.monotonic() + 300
            while time.monotonic() < deadline:
                job_response = client.get(f"/ingestion-jobs/{job_id}")
                job_response.raise_for_status()
                job = job_response.json()
                if job["status"] == "completed":
                    document_id = UUID(job["result"]["document_id"])
                    break
                if job["status"] == "failed":
                    raise RuntimeError(f"background ingestion failed: {job['error']}")
                time.sleep(0.25)
            else:
                raise TimeoutError("background ingestion did not finish within 300 seconds")

            query_response = client.post(
                "/query/stream",
                json={
                    "query": "At what rate did the prototype sample IMU motion?",
                    "framework": "custom",
                    "collection_id": str(collection_id),
                    "model": args.model,
                },
            )
            query_response.raise_for_status()
            answer = _sse_result(query_response.text)
            if answer["estimated_cost"] != 0.0:
                raise RuntimeError("local Ollama smoke test must report zero API cost")
            print(
                f"document_id={document_id} status={answer['evidence_status']} "
                f"citations={len(answer['citations'])} cost={answer['estimated_cost']}"
            )
    finally:
        if collection_id is not None:
            asyncio.run(_cleanup(settings, collection_id, document_id))


def _sse_result(body: str) -> dict[str, Any]:
    event = ""
    for line in body.splitlines():
        if line.startswith("event: "):
            event = line.removeprefix("event: ")
        elif event == "query.result" and line.startswith("data: "):
            return cast(dict[str, Any], json.loads(line.removeprefix("data: ")))
    raise RuntimeError("stream did not contain a query.result event")


if __name__ == "__main__":
    main()
