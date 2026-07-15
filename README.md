# RAGLab

RAGLab is a portfolio-grade platform for implementing and fairly benchmarking retrieval-augmented generation pipelines across custom Python, LangChain, LangGraph, LlamaIndex, and Haystack implementations.

The current repository contains the project foundation, shared contracts, and the first **Phase 3 document-ingestion milestone**. It intentionally does not yet contain question answering or framework integrations.

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
- Secure in-memory validation and parsing for text-based PDFs with PyMuPDF
- Page-preserving extraction, text normalization, PDF metadata, section headings, and content hashes
- Deterministic recursive character chunks with overlap, source offsets, and metadata propagation
- Injection-ready ingestion orchestration for duplicate checks, embeddings, dense indexing, sparse indexing, and persistence

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
| `RAGLAB_MAX_UPLOAD_SIZE_MB` | Maximum accepted PDF size | `25` |
| `RAGLAB_MAX_PDF_PAGES` | Maximum pages processed per PDF | `500` |
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

### PDF ingestion behavior

The current ingestion service accepts text-based PDFs only. It rejects path-bearing file names, non-PDF extensions, signature mismatches, oversized uploads, encrypted files, excessive page counts, malformed PDFs, and documents without extractable text. Files are parsed directly from bytes; the parser does not construct a filesystem path from the upload name.

PDF metadata and page numbers are retained. Section headings are detected conservatively from font-size and textual signals, so heading metadata is useful but not guaranteed. Scanned PDFs and OCR are explicitly out of scope for this milestone.

Extracted document text remains untrusted data. Future generation prompts must delimit it as evidence and explicitly prohibit following instructions found inside a document; parsing and file validation alone cannot prevent prompt injection.

Storage remains dependency-injected: concrete PostgreSQL, Qdrant, local embedding, and BM25 adapters are the next ingestion milestone. Tests use in-memory contract fakes, not hidden production substitutes.

## Roadmap

1. PostgreSQL, Qdrant, local embedding, and BM25 ingestion adapters
2. Complete configurable chunking strategy suite and benchmark
3. Framework-free hybrid RAG baseline
4. LangChain, LangGraph, LlamaIndex, and Haystack adapters
5. Evaluation harness and reproducible benchmark reports
6. Next.js inspection and evaluation UI

Benchmark tables will be added only after the evaluation dataset and pipelines have been run. No performance claims are made at this stage.
