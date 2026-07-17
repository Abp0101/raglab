# Local observability and failure operations

RAGLab exposes enough operational evidence to diagnose its local API and workers without sending telemetry to a hosted service. Structured logs go to standard error, bounded metrics are held in process memory, and `GET /metrics` renders Prometheus-compatible text. The path has no paid API dependency and contains no document text, query text, credentials, collection IDs, document IDs, or raw request URLs.

## Signal flow

```text
HTTP request ── request-ID validation ── route handler ── safe response
      │                                  │
      ├── duration/status by route ──────┤
      └── structured completion log      └── sanitized error type/count

readiness probe ── PostgreSQL/Qdrant/Redis result ── dependency gauge + safe log
background job ── claimed/completed/failed/released ── outcome counter + safe log

GET /metrics ── process-local snapshot only; no outbound exporter
```

Request metrics use FastAPI route templates such as `/documents/{document_id}` rather than raw URLs. Error labels contain a bounded operation name and exception class. This avoids unbounded time series and prevents identifiers or user-controlled content from becoming metric labels.

## Exported metrics

| Metric | Type | Meaning |
| --- | --- | --- |
| `raglab_http_requests_total` | Counter | Completed requests by method, route template, and status class |
| `raglab_http_request_duration_seconds` | Histogram | Request duration using fixed local buckets |
| `raglab_errors_total` | Counter | Sanitized failures by operation and error type |
| `raglab_ingestion_jobs_total` | Counter | Queued, claimed, completed, failed, released, lease-lost, or coordination-failed jobs |
| `raglab_dependency_up` | Gauge | Latest readiness result for PostgreSQL, Qdrant, and Redis |

Inspect the current process directly:

```bash
curl --fail-with-body http://localhost:8000/metrics
```

Counters reset at process restart. With multiple API workers, each worker owns a separate registry; scrape every worker or introduce a dedicated Prometheus deployment before interpreting aggregates.

## Failure contracts

| Failure | HTTP/SSE behavior | Operational behavior |
| --- | --- | --- |
| Invalid input or domain conflict | Existing typed 4xx envelope | Counted by sanitized error type |
| Local provider unavailable | HTTP 503 with `Retry-After: 5` | No provider details exposed; error counted |
| Malformed provider response | HTTP 502 | Safe typed message; error counted |
| Unexpected route exception | HTTP 500 with `Internal` / `request processing failed` | Concrete exception class is logged and the public message is redacted |
| Failure after SSE headers open | `query.error` event with the same safe envelope | Counted as `query_stream`; task is cancelled and joined |
| Readiness dependency failure | HTTP 503 with every dependency's boolean state | Healthy dependencies remain distinguishable from the failed dependency |
| Background ingestion failure | Durable failed job with safe error; stale owners cannot commit | Outcome and safe error type counted; leases remain reclaimable |

The API validates caller-supplied `X-Request-ID` values against a short safe character set and replaces invalid or oversized values. Logs include correlation IDs and operational identifiers where useful, but never request bodies or model prompts.

## Runbook

1. Check `/health/live` to confirm the process is responsive.
2. Check `/health/ready` to isolate PostgreSQL, Qdrant, or Redis degradation.
3. Check `/metrics` for recent 5xx classes, error types, job outcomes, and latency changes.
4. Filter JSON logs by `request_id` or `job_id` for the affected operation.
5. Retry 503 responses only after the named local dependency is healthy; respect `Retry-After`.
6. For failed ingestion, inspect the durable job error and resubmit after resolving the dependency. Expired leases are reclaimed automatically.

## Scale-up boundary

The current design optimizes for one developer or a small local deployment. It deliberately avoids Prometheus, Grafana, OpenTelemetry collectors, remote log storage, and alerting infrastructure. Revisit it when there are multiple hosts, long-lived production SLOs, audit retention requirements, or on-call response. At that point, retain the bounded metric names and safe error contracts while adding a local/self-hosted Prometheus and Grafana stack, trace sampling, retention limits, and alerts tied to explicit availability and latency objectives.
