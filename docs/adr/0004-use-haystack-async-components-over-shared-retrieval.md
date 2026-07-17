# ADR 0004: Use Haystack async components over shared retrieval

- Status: Accepted
- Date: 2026-07-17
- Deciders: RAGLab maintainers

## Context

The fifth adapter must demonstrate meaningful Haystack integration while preserving the controlled comparison used by the other frameworks. A separate Haystack document store or embedding pipeline would duplicate the corpus and make retrieval changes dominate the experiment. The project also has a hard zero-paid-API and local-only requirement; Haystack telemetry is enabled by default upstream unless explicitly disabled.

## Decision

Build a request-scoped Haystack `AsyncPipeline`. An async custom retriever calls RAGLab's canonical retrieval service and maps each result to a scored Haystack `Document`, retaining the complete typed result as serialized metadata. A context component restores those records and applies the shared evidence budget. `ChatPromptBuilder` renders role-aware messages, and a safe generation component calls the official local `OllamaChatGenerator` with the shared `GroundedAnswer` JSON schema.

The generation component receives an explicit evidence signal and does not call Ollama when the context is empty. After generation, RAGLab applies the same exact-quote citation validator and refusal rules as every other adapter.

Set `HAYSTACK_TELEMETRY_ENABLED=false` before Haystack imports and clear the telemetry singleton if Haystack was already imported. Do not construct any remote generator, tracer, or managed Haystack service. Normalize local Ollama token metadata with an explicit estimated cost of zero.

## Options considered

### Use a Haystack document store and retriever

This would exercise more of the framework, but duplicate storage and indexing would invalidate the current controlled comparison. It belongs in a separately configured indexing experiment.

### Call only `OllamaChatGenerator`

This would be a thin model wrapper and would not demonstrate native evidence mapping or orchestration.

### Use a synchronous Pipeline

The canonical retrieval service is asynchronous. A synchronous graph would require thread or event-loop bridging and weaken cancellation behavior.

### Async components over the canonical data plane

Selected because it uses Haystack's current component, document, prompt, and orchestration abstractions while holding retrieval inputs constant.

## Consequences

- Retrieval metrics, chunk IDs, filters, and reranking remain directly comparable across all five pipelines.
- Haystack-specific scheduling, prompt rendering, structured-output behavior, and generation latency remain visible.
- Empty-evidence requests retain the zero-model-call safety behavior.
- The adapter does not evaluate Haystack document stores, embedders, retrievers, agents, managed services, or remote tracing.
- The upstream Haystack core dependency transitively includes clients for remote services, but RAGLab neither configures nor invokes them.
- Request-scoped graph construction is simple and isolates model settings. At higher throughput, revisit compiled graph reuse or a bounded cache keyed by model and temperature after measuring construction overhead and concurrency safety.
- Framework-specific indexing remains a future experiment with its own benchmark declaration.
