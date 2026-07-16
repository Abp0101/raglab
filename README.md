# RAGLab

RAGLab is a portfolio-grade platform for implementing and fairly benchmarking retrieval-augmented generation pipelines across custom Python, LangChain, LangGraph, LlamaIndex, and Haystack implementations.

The current repository contains shared contracts, persistent ingestion, chunking, retrieval, executable **custom and LangChain RAG pipelines**, a typed API, and a deterministic local evaluation harness.

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
- PostgreSQL document/chunk persistence with collection-scoped duplicate constraints and Alembic migrations
- Lazy local Sentence Transformers embeddings, a shared Qdrant vector collection, and Redis-backed BM25 tokens
- Compensating cleanup when multi-store indexing fails, plus marked real-service integration tests
- Fixed-token, recursive-character, section-aware, and parent-child chunking with deterministic provenance
- A versioned structural chunking benchmark and generated-report workflow
- Dense Qdrant, exact BM25, and hybrid Reciprocal Rank Fusion retrieval with shared filters
- Optional local cross-encoder reranking and deduplicated parent-context expansion
- OpenAI-compatible and Ollama generation providers with structured usage and optional cost data
- Grounded structured answers, deterministic citation checks, prompt-injection boundaries, and refusal rules
- Collection creation, bounded PDF upload, document listing, pipeline discovery, and shared query endpoints
- Durable background-ingestion jobs with bounded local concurrency and restart recovery
- Server-Sent Event query progress with citation-validated terminal answers
- Stable safe-error envelopes for validation, missing resources, unavailable frameworks, and providers
- A runtime guard that disables metered OpenAI-compatible generation unless explicitly opted in
- A checksum-verified synthetic biomedical/technical dataset with reproducible document/chunk IDs
- Deterministic retrieval, citation, refusal, key-fact, latency, and zero-cost evaluation reports
- A native LangChain adapter using `BaseRetriever`, `ChatPromptTemplate`, Runnable composition, structured `ChatOllama`, and LangChain document splitting
- A controlled Custom-versus-LangChain comparison runner that rejects paid cost or failed questions

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
make test-integration # test PostgreSQL, Qdrant, and Redis adapters
make test-live-model  # download/load and verify the default embedding model
make benchmark-chunking # compare chunk structure without claiming a winner
make build-evaluation-dataset # rebuild byte-stable synthetic PDFs and annotations
make seed-evaluation # idempotently ingest the evaluation corpus locally
make evaluate RAGLAB_LLM_MODEL=llama3.2:latest # evaluate Custom (default)
make evaluate RAGLAB_LLM_MODEL=llama3.2:latest RAGLAB_FRAMEWORK=langchain # evaluate LangChain
make compare-frameworks RAGLAB_LLM_MODEL=llama3.2:latest # compare both local pipelines
make smoke-ollama RAGLAB_LLM_MODEL=llama3.2:latest # verify local structured generation
make smoke-api RAGLAB_LLM_MODEL=llama3.2:latest # exercise the complete local API path
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
| `RAGLAB_INGESTION_CONCURRENCY` | Maximum local background ingestions | `1` |
| `RAGLAB_EMBEDDING_MODEL` | Local Sentence Transformers model | `sentence-transformers/all-MiniLM-L6-v2` |
| `RAGLAB_EMBEDDING_BATCH_SIZE` | Local embedding batch size | `32` |
| `RAGLAB_RERANKER_MODEL` | Local cross-encoder reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| `RAGLAB_RERANKER_BATCH_SIZE` | Reranker batch size | `16` |
| `RAGLAB_QDRANT_COLLECTION` | Shared dense-vector collection | `raglab_chunks` |
| `RAGLAB_QDRANT_TIMEOUT_SECONDS` | Local vector-store request timeout | `30` |
| `RAGLAB_BM25_KEY_PREFIX` | Redis namespace for sparse tokens | `raglab:bm25` |
| `RAGLAB_LLM_PROVIDER` | `ollama` or `openai_compatible` | `ollama` |
| `RAGLAB_LLM_MODEL` | Model passed to the selected provider | `qwen3:8b` |
| `RAGLAB_ALLOW_PAID_API_USAGE` | Explicit safety opt-in for a metered provider | `false` |
| `RAGLAB_OPENAI_BASE_URL` | OpenAI-compatible API root | `https://api.openai.com/v1` |
| `RAGLAB_OPENAI_API_KEY` | Optional bearer credential | Empty |
| `RAGLAB_OPENAI_INSTRUCTION_ROLE` | Compatible instruction role | `developer` |
| `RAGLAB_OPENAI_STRUCTURED_OUTPUT_MODE` | `json_schema` or `json_object` | `json_schema` |
| `RAGLAB_OPENAI_MAX_TOKENS_FIELD` | Compatible output-limit field | `max_completion_tokens` |
| `RAGLAB_OLLAMA_BASE_URL` | Local Ollama root | `http://localhost:11434` |
| `RAGLAB_LLM_TIMEOUT_SECONDS` | Generation request timeout | `120` |
| `RAGLAB_INPUT_COST_PER_MILLION` | Optional provider input rate | Empty |
| `RAGLAB_OUTPUT_COST_PER_MILLION` | Optional provider output rate | Empty |
| `RAGLAB_POSTGRES_DSN` | Async SQLAlchemy connection URL | Local Compose PostgreSQL |
| `RAGLAB_QDRANT_URL` | Qdrant HTTP endpoint | `http://localhost:6333` |
| `RAGLAB_QDRANT_API_KEY` | Optional Qdrant credential | Empty |
| `RAGLAB_REDIS_DSN` | Async Redis connection URL | `redis://localhost:6379/0` |

## Current architecture

```text
PDF bytes ── validation ── PyMuPDF ── configurable chunks ── local embeddings
                                              │                  │
                                              ▼                  ▼
                                         PostgreSQL           Qdrant
                                              │
                                              └── tokenization ── Redis/BM25

HTTP client ── FastAPI ── request ID + structured logging
              ├── collection + document catalog ── PostgreSQL
              ├── /query ── pipeline registry ── custom or LangChain local RAG
              └── /health/ready checks all three stores
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

Extracted document text remains untrusted data. Generation prompts delimit it as evidence and explicitly prohibit following instructions found inside a document; parsing and file validation alone cannot prevent prompt injection.

Storage remains dependency-injected. PostgreSQL is the metadata and chunk source of truth, Qdrant stores normalized dense vectors and filterable payloads, and Redis persists tokenized chunks for BM25 scoring. A shared Qdrant collection is partitioned logically by `collection_id`, avoiding one physical collection per user collection. Ingestion first records `processing` state, then indexes dense and sparse data, marks the document `ready`, and performs best-effort compensation across stores when indexing fails.

Normal tests use contract fakes and Qdrant local mode. `make test-integration` requires the Compose services and validates all three real backing stores. `make test-live-model` may download the configured Hugging Face model on first use.

### Chunking strategies

Ingestion can select fixed lexical-token windows, recursive character boundaries, detected sections, or linked parent-child units through the shared `ChunkingConfig`. All strategies retain page, heading, collection, source offsets, and deterministic IDs. Strategy details, size-unit differences, and benchmark interpretation are documented in [`docs/chunking.md`](docs/chunking.md).

`make benchmark-chunking` compares all strategies against versioned synthetic technical and biomedical boundary cases. It reports structural measurements only; retrieval and answer-quality evaluation remains necessary before choosing a strategy.

### Retrieval baseline

The framework-free retrieval service supports dense, sparse, and hybrid modes with explicit native and fused score provenance. It applies portable metadata filters consistently, optionally reranks candidates using a local cross-encoder, and expands linked children to larger relational parent context. The design, formulas, filtering semantics, and current BM25 scaling boundary are documented in [`docs/retrieval.md`](docs/retrieval.md).

### Grounded generation

The custom and LangChain pipelines build bounded untrusted context, request strict structured output, validate exact citation quotes, and replace unsupported answers with an insufficient-evidence refusal. Their native boundaries and controlled comparison method are documented in [`docs/framework-comparison.md`](docs/framework-comparison.md). OpenAI-compatible and Ollama provider behavior, prompt-injection defenses, cost configuration, and limitations are documented in [`docs/generation.md`](docs/generation.md).

### HTTP API

FastAPI exposes collection creation and listing, bounded multipart PDF ingestion, durable background jobs, document metadata, pipeline capability discovery, synchronous querying, and safe SSE query progress. Endpoint examples, job semantics, event names, and status mappings are documented in [`docs/api.md`](docs/api.md).

### Zero paid API policy

The supported default path is fully local: Ollama generation, Sentence Transformers embeddings, cross-encoder reranking, PostgreSQL, Qdrant, and Redis. `RAGLAB_ALLOW_PAID_API_USAGE=false` blocks construction of the OpenAI-compatible adapter even if someone changes the provider name. Its implementation remains for portfolio completeness and is tested only with mocked HTTP; RAGLab development, tests, demos, and evaluation must not invoke metered APIs.

### Evaluation harness

The first versioned dataset is a small synthetic harness-validation corpus covering wearable sensors, rehabilitation safety, conflicting evidence, and refusal. The loader verifies its checksum and annotations before a run. Reports record the full configuration and compute deterministic retrieval, citation, refusal, lexical key-fact, latency, and cost measurements. Metric formulas, reproducibility rules, limitations, and the testing strategy are documented in [`docs/evaluation-methodology.md`](docs/evaluation-methodology.md).

The first measured custom baseline is recorded in [`reports/baselines/custom-hybrid-reranked-llama3.2-v1.md`](reports/baselines/custom-hybrid-reranked-llama3.2-v1.md). The first controlled two-pipeline run is recorded in [`reports/baselines/custom-vs-langchain-llama3.2-v1.md`](reports/baselines/custom-vs-langchain-llama3.2-v1.md). Both retain observed misses and are not presented as framework rankings. `make compare-frameworks` reproduces a local report over the same dataset.

## Roadmap

1. LangGraph adapter with explicit retrieval/generation state transitions
2. Distributed job leases, pagination, deletion, and authentication
3. LlamaIndex and Haystack adapters
4. Observability and failure-path integration hardening
5. Next.js inspection and evaluation UI

Cross-framework reports use the exact same versioned dataset and declared configuration. They are measurements of those runs, not framework-superiority claims.
