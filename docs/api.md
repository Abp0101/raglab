# HTTP API

RAGLab exposes one framework-neutral FastAPI surface. The current endpoints are synchronous: ingestion returns after PostgreSQL, Qdrant, and Redis have been updated, and querying returns after local generation completes.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health/live` | Process liveness |
| `GET` | `/health/ready` | PostgreSQL, Qdrant, and Redis readiness |
| `POST` | `/collections` | Create a shared logical corpus |
| `GET` | `/collections` | List collections and document counts |
| `GET` | `/collections/{collection_id}` | Fetch one collection |
| `POST` | `/collections/{collection_id}/documents` | Upload and ingest a text PDF |
| `POST` | `/collections/{collection_id}/ingestion-jobs` | Persist a PDF and enqueue local ingestion |
| `GET` | `/ingestion-jobs/{job_id}` | Poll queued, processing, completed, or failed state |
| `GET` | `/collections/{collection_id}/documents` | List document metadata |
| `GET` | `/documents/{document_id}` | Fetch one document record |
| `GET` | `/pipelines` | Discover implemented and planned frameworks |
| `POST` | `/query` | Run the selected shared RAG contract |
| `POST` | `/query/stream` | Stream safe lifecycle events and one validated result over SSE |

Interactive request and response schemas are available at `/docs` when the API is running.

With the local services migrated and an Ollama model already installed, `make smoke-api RAGLAB_LLM_MODEL=llama3.2:latest` exercises collection creation, durable background ingestion and polling, retrieval, reranking, safe SSE delivery, local generation, citation validation, and temporary-data cleanup through the real application service graph.

## Minimal local flow

```bash
curl -X POST http://localhost:8000/collections \
  -H 'Content-Type: application/json' \
  -d '{"name":"Biomedical papers"}'

curl -X POST http://localhost:8000/collections/REPLACE_WITH_ID/documents \
  -F 'file=@paper.pdf;type=application/pdf' \
  -F 'display_title=Example paper'

curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{
    "query":"What does the paper report?",
    "framework":"custom",
    "collection_id":"REPLACE_WITH_ID"
  }'
```

`custom`, `langchain`, `langgraph`, `llamaindex`, and `haystack` are executable. `/pipelines` marks all five target frameworks available. LangGraph reports `agentic: true`; selecting an unregistered implementation returns HTTP 501.

## Background ingestion

The background endpoint stores the bounded upload in PostgreSQL before returning HTTP 202. A concurrency-limited local runner claims queued work and records a terminal `completed` or `failed` result. Source bytes are cleared from the job record at either terminal state. On graceful shutdown, active work is moved back to `queued`; startup resumes queued and interrupted jobs.

Each API process runs a bounded number of pollers and competes for jobs through PostgreSQL `FOR UPDATE SKIP LOCKED`. A claim records an opaque worker owner, increments the attempt count, and receives an expiring lease. Active work renews its lease every third of the configured ownership window. Graceful shutdown releases owned work immediately; a crashed worker's job becomes claimable after expiry. Completion and failure updates require the current, unexpired owner, preventing a stale worker from overwriting a newer attempt.

The queue provides at-least-once execution, not exactly-once execution. A crash after indexing but before job completion can cause a retry; deterministic document IDs, duplicate checks, unique constraints, and ingestion compensation limit the effect, but PostgreSQL, Qdrant, and Redis do not share one transaction. Public job responses expose `attempt_count` and `lease_expires_at` for operational diagnosis without revealing worker identity.

## Safe query streaming

`POST /query/stream` uses Server-Sent Events with these event names:

- `query.accepted` — the request passed synchronous collection/framework validation;
- `query.heartbeat` — local retrieval, reranking, or generation is still running;
- `query.result` — one complete shared `RAGResponse` after citation and refusal validation;
- `query.error` — a safe error envelope after the stream has opened.

RAGLab deliberately does not stream raw model tokens. Structured output, citation checking, and insufficient-evidence refusal happen before answer text reaches the client; streaming unvalidated tokens would bypass that safety boundary. Client disconnects cancel the in-flight query task.

## Upload and failure behavior

Uploads are read only up to the configured size limit plus one byte, then pass through signature, extension, encryption, page-count, and extractable-text validation. Uploaded bytes are not stored as filesystem paths.

Expected failures use a stable envelope:

```json
{
  "error": {
    "type": "CollectionNotFound",
    "message": "collection ... does not exist"
  }
}
```

Validation failures return 422, missing resources 404, unavailable framework adapters 501, local provider outages 503, and malformed provider responses 502. Provider response bodies, credentials, and internal tracebacks are not included.

## Current boundary

Pagination, authentication, and deletion with coordinated multi-store cleanup remain deliberately deferred. The current SSE contract streams lifecycle state and a validated terminal answer rather than unsafe raw model tokens.
