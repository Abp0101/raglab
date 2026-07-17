# ADR 0009: Isolate framework-native indexing experiments

## Status

Accepted

## Context

RAGLab's main framework comparison intentionally shares chunks, embeddings, indexes, retrieval, and reranking. Replacing those controls with each framework's preferred indexing path would make answer differences impossible to attribute to orchestration alone. The project still needs to demonstrate native indexing knowledge and measure the trade-offs those abstractions introduce.

The initial assumptions are a small versioned synthetic corpus, natural-language questions with exact annotated target passages, single-process execution, in-memory indexes, no persistence requirement, no production embedding download, and structural plus retrieval measurements rather than generation quality. The experiment must run offline with zero paid API cost and must not mutate the API corpus.

## Decision

Create a second, explicitly isolated benchmark with one checked-in configuration and four native adapters:

```text
canonical pipeline comparison ── shared production data plane ── orchestration conclusions

native indexing experiment ── isolated per-framework memory stores ── indexing observations
```

- Custom uses RAGLab fixed-token chunks and a minimal cosine index.
- LangChain uses `RecursiveCharacterTextSplitter` and `InMemoryVectorStore`.
- LlamaIndex uses `SentenceSplitter` and `VectorStoreIndex`.
- Haystack uses `DocumentSplitter` and `InMemoryDocumentStore` with its embedding retriever.
- LangGraph is excluded because it does not own a distinct indexing abstraction.
- Every adapter receives the same deterministic hashing embedding dimensions, target size, overlap, cases, annotated queries, and top-K cutoff.
- Framework-native size units and index names are declared rather than described as equivalent.
- Dataset and configuration byte hashes are embedded in JSON and Markdown reports.
- Generated artifacts report measurements and limitations without selecting a winner.

## Alternatives considered

### Replace the canonical shared indexes

This would make the five-pipeline benchmark confound indexing, retrieval, and orchestration. It was rejected because the existing controlled results would no longer be interpretable.

### Use the production Sentence Transformers model and Qdrant

This is closer to deployment but introduces model download and service variance while still forcing frameworks through one vector store. It is better suited to a later persistent-index experiment after the deterministic harness is stable.

### Use each framework's default embedding provider

Defaults can trigger remote or metered services and would change both the embedding model and index. This violates the zero-paid-API constraint and prevents attribution.

### Include LangGraph as another index

LangGraph can orchestrate an indexing workflow but has no separate document index or splitter to compare. Including it under the shared Custom index would duplicate a result under a misleading label.

## Consequences

- Native framework indexing is executable without weakening the canonical fair comparison.
- Reports expose splitter units, backends, checksums, containment, Recall@K, and local latency.
- Deterministic hashing improves reproducibility but cannot measure semantic embedding quality.
- In-memory indexes do not exercise persistence, network failure, concurrent writers, or large-corpus scaling.
- Import-time Haystack telemetry remains forcibly disabled.

At larger scale, revisit persistent per-framework namespaces, a real locally cached embedding model, repeated trials with confidence intervals, memory and index-size metrics, cold-start separation, concurrency, update/delete behavior, and failure recovery. Any such plan must keep dataset, model, hardware, query order, and cost policy explicit and must remain separate from the canonical orchestration benchmark.
