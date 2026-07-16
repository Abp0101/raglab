# ADR 0003: Map canonical retrieval into LlamaIndex nodes

- Status: Accepted
- Date: 2026-07-16
- Deciders: RAGLab maintainers

## Context

The LlamaIndex adapter must be a genuine framework integration while remaining comparable with the existing Custom, LangChain, and LangGraph pipelines. Creating a separate `VectorStoreIndex` would duplicate the corpus and allow chunking, embedding, filtering, and scoring differences to dominate the comparison.

LlamaIndex's older declarative QueryPipeline API is feature-frozen and deprecated in favor of workflows. RAGLab already has a bounded workflow implementation in LangGraph, so this milestone should focus on LlamaIndex's retrieval, node, prompt, and structured-model abstractions rather than add another orchestration graph.

## Decision

Implement an async LlamaIndex `BaseRetriever` that calls the canonical retrieval service and maps every `RetrievedChunk` into a `TextNode` wrapped by `NodeWithScore`. The complete RAGLab retrieval record is serialized into excluded node metadata so identifiers and score provenance survive the framework boundary without leaking into the model prompt.

Use LlamaIndex `PromptTemplate` and its local Ollama integration's `astructured_predict` method with the shared `GroundedAnswer` Pydantic schema. Capture Ollama token counts through a request-scoped LlamaIndex callback handler and normalize them into `UsageMetrics` with an explicit zero estimated cost.

Keep context limits, exact citation validation, refusal rules, ingestion, stores, embeddings, and reranking shared.

## Options considered

### Build a separate LlamaIndex `VectorStoreIndex`

This would showcase more built-in indexing but confound the controlled query comparison and duplicate operational state. It can be added later as a separately named experiment.

### Use the deprecated QueryPipeline API

This provides declarative composition but starts new work on a feature-frozen interface. It was rejected in favor of current component APIs.

### Use only the Ollama integration

This would be too thin: the implementation would not exercise LlamaIndex retrieval nodes or score handling.

### Native retriever/nodes/prompt/structured model over the canonical data plane

Selected because it uses meaningful current LlamaIndex abstractions while holding retrieval inputs constant.

## Consequences

- Retrieval metrics and chunk IDs remain directly comparable across frameworks.
- LlamaIndex-specific prompt formatting and structured-output behavior remain visible in answer metrics, latency, and token usage.
- The adapter does not measure LlamaIndex's vector store, ingestion pipeline, response synthesizers, workflows, or managed cloud services.
- LlamaIndex instrumentation is local only; no remote tracing or LlamaCloud dependency is configured.
- Framework-specific indexing should be evaluated later under an explicitly different benchmark configuration.
