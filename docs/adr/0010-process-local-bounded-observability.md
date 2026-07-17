# ADR 0010: Use process-local bounded observability

- Status: Accepted
- Date: 2026-07-17

## Context

RAGLab already emitted structured request logs, propagated request IDs, exposed liveness/readiness, and persisted background job failures. It did not provide a scrapeable metric contract, unexpected exceptions relied on framework defaults, readiness failures were not observable beyond the immediate response, and storage failures during retrieval could escape as untyped HTTP 500 responses.

The project must remain runnable with no paid API, hosted telemetry, credential, or additional service. Signals must not capture document content, questions, prompts, credentials, or high-cardinality resource identifiers.

## Decision

Use one dependency-free `LocalMetrics` registry per API process. Request middleware records method, route template, status class, and duration. Readiness probes publish per-dependency gauges. Background ingestion publishes fixed outcome counters. Safe exception handlers and SSE handling publish sanitized error types.

Expose the registry at public `GET /metrics` for local Prometheus-compatible scraping. The endpoint publishes operational aggregates only. Continue emitting JSON logs to standard error and use validated request IDs for correlation. Translate unexpected route failures to a fixed redacted HTTP 500 envelope, add a retry hint to typed provider-unavailable 503 responses, and translate retrieval adapter failures to `ProviderUnavailableError`.

No outbound exporter, hosted collector, tracing SDK, or analytics client is constructed.

## Consequences

- Local failures can be correlated across request logs, readiness, API errors, and durable job outcomes.
- Metrics add no service, network, credential, or API-cost requirement.
- Route templates and fixed outcomes bound metric cardinality.
- Counters reset on restart and are not aggregated across workers.
- Public local metrics expose request volume and route names, so the API should not be placed directly on the public internet.
- Metrics and logs provide debugging evidence, not durable audit storage or alert delivery.

## Alternatives considered

- Add Prometheus and Grafana to Docker Compose. Rejected for this milestone because it adds persistent operational services before the project has multi-process retention or alerting requirements.
- Add OpenTelemetry with a collector. Deferred until distributed traces span multiple deployable services; today it would add configuration and SDK surface without a trace backend.
- Use a hosted observability platform. Rejected because it violates the local-only, zero-cost constraint and introduces credentials and outbound telemetry.
- Log every failure only. Rejected because logs alone do not provide bounded rates, latency distributions, or current dependency state.

## Revisit when

RAGLab runs across multiple hosts or workers, defines production SLOs, needs durable audit retention, or has an on-call rotation. Preserve the safe labels and public failure envelopes while adding self-hosted collection, retention, dashboards, alert thresholds, and sampled traces.
