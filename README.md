# RAGLab

RAGLab is a portfolio-grade platform for implementing and fairly benchmarking retrieval-augmented generation pipelines across custom Python, LangChain, LangGraph, LlamaIndex, and Haystack implementations.

The current repository contains the project foundation and **Phase 2 shared contracts**. It intentionally does not yet contain document ingestion or framework integrations.

## Foundation features

- Python 3.12 package with strict Ruff, mypy, and pytest configuration
- FastAPI application factory and OpenAPI documentation
- Typed, environment-driven Pydantic settings
- Async SQLAlchemy foundation with Alembic migration configuration
- JSON structured request logs with request IDs
- Separate liveness and dependency-aware readiness endpoints
- PostgreSQL, Qdrant, and Redis development services
- GitHub Actions quality pipeline
- Framework-independent models for documents, chunks, embeddings, retrieval, citations, evaluation, latency, and usage
- Async protocols for parsers, chunkers, providers, retrievers, rerankers, pipelines, and evaluation metrics
- A shared pipeline contract test that future implementations must pass

## Quick start

Prerequisites: Python 3.12, Docker with Compose, and GNU Make.

```bash
cp .env.example .env
make install
make infra-up
make run
```

Then open:

- API documentation: <http://localhost:8000/docs>
- Liveness: <http://localhost:8000/health/live>
- Readiness: <http://localhost:8000/health/ready>

`/health/live` confirms the API process is responding. `/health/ready` returns HTTP 200 only when PostgreSQL, Qdrant, and Redis respond; otherwise it returns HTTP 503 with a per-service status map.

## Development commands

```bash
make format       # format and apply safe lint fixes
make lint         # verify formatting and lint rules
make typecheck    # run strict mypy checks
make test         # run tests with branch coverage
make check        # run all local quality gates
make infra-down   # stop local backing services
alembic upgrade head  # apply database migrations
```

## Configuration

All runtime variables use the `RAGLAB_` prefix. Copy `.env.example` for local development; never commit `.env` or provider credentials.

| Variable | Purpose | Local default |
| --- | --- | --- |
| `RAGLAB_ENVIRONMENT` | Runtime environment | `development` |
| `RAGLAB_LOG_LEVEL` | Python log threshold | `INFO` |
| `RAGLAB_LOG_JSON` | Emit JSON logs | `true` |
| `RAGLAB_CORS_ORIGINS` | JSON array of allowed web origins | `["http://localhost:3000"]` |
| `RAGLAB_POSTGRES_DSN` | Async SQLAlchemy connection URL | Local Compose PostgreSQL |
| `RAGLAB_QDRANT_URL` | Qdrant HTTP endpoint | `http://localhost:6333` |
| `RAGLAB_QDRANT_API_KEY` | Optional Qdrant credential | Empty |
| `RAGLAB_REDIS_DSN` | Async Redis connection URL | `redis://localhost:6379/0` |

## Current architecture

```text
HTTP client
    │
    ▼
FastAPI ── request ID + structured logging
    │
    └── /health/ready ──┬── PostgreSQL
                        ├── Qdrant
                        └── Redis
```

Application code is split between the deployable API in `apps/api` and reusable, framework-independent code in `src/raglab`. RAG implementations depend on the models in `raglab.core.schemas` and protocols in `raglab.core.interfaces`, rather than importing types from another framework adapter.

The shared `RAGPipeline` boundary is intentionally small:

```python
class RAGPipeline(Protocol):
    async def ingest(
        self, documents: Sequence[DocumentInput]
    ) -> Sequence[IngestionResult]: ...

    async def query(self, request: QueryRequest) -> RAGResponse: ...
```

Every response carries a normalized framework name, evidence status, citations, retrieved chunks, stage latency, provider usage, warnings, and optional debug data. This supports fair evaluation without forcing each framework to use the same internal architecture.

## Roadmap

1. PDF ingestion and configurable chunking
2. Framework-free hybrid RAG baseline
3. LangChain, LangGraph, LlamaIndex, and Haystack adapters
4. Evaluation harness and reproducible benchmark reports
5. Next.js inspection and evaluation UI

Benchmark tables will be added only after the evaluation dataset and pipelines have been run. No performance claims are made at this stage.
