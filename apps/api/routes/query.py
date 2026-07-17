"""Shared RAG query endpoint."""

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Callable, Coroutine, Mapping
from typing import Annotated, Any

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from apps.api.dependencies import get_catalog_repository, get_metrics, get_pipeline_registry
from apps.api.errors import public_error_payload
from apps.api.security import require_permission
from raglab.core.exceptions import RAGLabError
from raglab.core.interfaces import CatalogRepository
from raglab.core.metrics import LocalMetrics
from raglab.core.schemas import Permission, QueryRequest, RAGResponse
from raglab.pipelines import PipelineRegistry

router = APIRouter(tags=["query"])
logger = logging.getLogger(__name__)


@router.post(
    "/query",
    response_model=RAGResponse,
    dependencies=[Depends(require_permission(Permission.QUERY_EXECUTE))],
)
async def query(
    request: QueryRequest,
    catalog: Annotated[CatalogRepository, Depends(get_catalog_repository)],
    registry: Annotated[PipelineRegistry, Depends(get_pipeline_registry)],
) -> RAGResponse:
    """Run a validated query through the selected registered framework."""
    await catalog.get_collection(request.collection_id)
    return await registry.get(request.framework).query(request)


@router.post(
    "/query/stream",
    response_class=StreamingResponse,
    responses={
        200: {
            "content": {"text/event-stream": {}},
            "description": "Lifecycle events followed by one validated RAG response.",
        }
    },
    dependencies=[Depends(require_permission(Permission.QUERY_EXECUTE))],
)
async def stream_query(
    request: QueryRequest,
    catalog: Annotated[CatalogRepository, Depends(get_catalog_repository)],
    registry: Annotated[PipelineRegistry, Depends(get_pipeline_registry)],
    metrics: Annotated[LocalMetrics, Depends(get_metrics)],
) -> StreamingResponse:
    """Stream progress and one citation-validated result using Server-Sent Events."""
    await catalog.get_collection(request.collection_id)
    pipeline = registry.get(request.framework)
    return StreamingResponse(
        _query_events(request, pipeline.query, metrics),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


async def _query_events(
    request: QueryRequest,
    run: Callable[[QueryRequest], Coroutine[Any, Any, RAGResponse]],
    metrics: LocalMetrics | None = None,
) -> AsyncIterator[bytes]:
    task: asyncio.Task[RAGResponse] = asyncio.create_task(run(request))
    yield _sse("query.accepted", {"framework": request.framework.value})
    try:
        while True:
            try:
                response = await asyncio.wait_for(asyncio.shield(task), timeout=10)
            except TimeoutError:
                yield _sse("query.heartbeat", {"status": "processing"})
                continue
            yield _sse("query.result", response.model_dump(mode="json"))
            return
    except RAGLabError as error:
        error_type = type(error).__name__.removesuffix("Error")
        if metrics is not None:
            metrics.observe_error("query_stream", error_type)
        logger.warning("stream_query_failed", extra={"error_type": error_type})
        yield _sse("query.error", public_error_payload(error))
    except asyncio.CancelledError:
        task.cancel()
        raise
    except Exception as error:
        if metrics is not None:
            metrics.observe_error("query_stream", "Internal")
        logger.error(
            "unexpected_stream_query_failure",
            extra={"error_type": type(error).__name__},
        )
        yield _sse("query.error", public_error_payload(RuntimeError()))
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


def _sse(event: str, data: Mapping[str, object]) -> bytes:
    payload = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n".encode()
