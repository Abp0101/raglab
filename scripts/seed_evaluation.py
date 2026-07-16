"""Idempotently ingest the committed evaluation PDFs into local backing services."""

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import UUID

from apps.api.runtime import build_api_services
from sqlalchemy import select

from raglab.core.config import Settings
from raglab.core.schemas import DocumentInput, FrameworkName
from raglab.database.models import CollectionRecord
from raglab.database.session import create_engine, create_session_factory


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("datasets/evaluation/v1"),
    )
    args = parser.parse_args()
    corpus = cast(
        dict[str, Any],
        json.loads((args.dataset / "corpus.json").read_text(encoding="utf-8")),
    )
    collection_id = UUID(corpus["collection_id"])
    settings = Settings(
        llm_provider="ollama",
        allow_paid_api_usage=False,
    )
    await _ensure_collection(settings, collection_id, corpus["collection_name"])
    services = build_api_services(settings)
    try:
        pipeline = services.pipelines.get(FrameworkName.CUSTOM)
        for document in corpus["documents"]:
            path = args.dataset / document["file_name"]
            result = (
                await pipeline.ingest(
                    (
                        DocumentInput(
                            file_name=path.name,
                            content=path.read_bytes(),
                            collection_id=collection_id,
                            display_title=document["display_title"],
                        ),
                    )
                )
            )[0]
            print(
                f"file={path.name} document_id={result.document_id} "
                f"chunks={result.chunk_count} duplicate={result.duplicate}"
            )
    finally:
        await services.close()


async def _ensure_collection(settings: Settings, collection_id: UUID, name: str) -> None:
    engine = create_engine(settings.postgres_dsn)
    sessions = create_session_factory(engine)
    try:
        async with sessions() as session, session.begin():
            existing = await session.scalar(
                select(CollectionRecord).where(CollectionRecord.id == collection_id)
            )
            if existing is None:
                now = datetime.now(UTC)
                session.add(
                    CollectionRecord(
                        id=collection_id,
                        name=name,
                        description="Versioned local evaluation corpus",
                        created_at=now,
                        updated_at=now,
                    )
                )
            elif existing.name != name:
                raise ValueError("evaluation collection ID is already used by another collection")
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
