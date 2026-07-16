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
| `GET` | `/collections/{collection_id}/documents` | List document metadata |
| `GET` | `/documents/{document_id}` | Fetch one document record |
| `GET` | `/pipelines` | Discover implemented and planned frameworks |
| `POST` | `/query` | Run the selected shared RAG contract |

Interactive request and response schemas are available at `/docs` when the API is running.

With the local services migrated and an Ollama model already installed, `make smoke-api RAGLAB_LLM_MODEL=llama3.2:latest` exercises collection creation, PDF ingestion, retrieval, reranking, local generation, citation validation, and temporary-data cleanup through the real application service graph.

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

Only `custom` is currently executable. `/pipelines` returns all five target framework names and marks unimplemented adapters unavailable. Selecting an unavailable implementation returns HTTP 501.

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

Streaming, background ingestion jobs, cancellation, pagination, authentication, and deletion with coordinated multi-store cleanup are deliberately deferred. The API does not present a fake streaming interface over a non-streaming provider; native streaming will be added as a separate capability with explicit event contracts.
