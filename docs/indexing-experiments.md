# Framework-native indexing experiments

RAGLab uses two deliberately different comparison modes. The canonical pipeline benchmark fixes PostgreSQL, Qdrant, Redis, chunk IDs, embeddings, retrieval, and reranking so orchestration differences remain interpretable. The native indexing experiment changes splitters and indexes on purpose, records those changes explicitly, and never mixes its data with the API corpus.

## Experiment boundary

```text
Versioned synthetic cases + annotated natural-language queries
                       │
                       ├── Custom fixed-token ── RAGLab in-memory cosine index
                       ├── LangChain recursive splitter ── InMemoryVectorStore
                       ├── LlamaIndex SentenceSplitter ── VectorStoreIndex
                       └── Haystack DocumentSplitter ── InMemoryDocumentStore
                                                   │
                          deterministic-hash-v1 embeddings + Recall@K
```

LangGraph is not included because it orchestrates retrieval and generation but does not provide a distinct document indexing abstraction. Its behavior remains covered by the canonical five-pipeline comparison.

The versioned declaration is [`configs/indexing_experiments/v1.json`](../configs/indexing_experiments/v1.json). It fixes:

- the exact dataset path and semantic version;
- 128-dimensional `deterministic-hash-v1` embeddings;
- a target chunk size of 50 and overlap of 8;
- exact relevant passages as queries;
- retrieval at one result;
- four separately named native index backends.

The hashing embedding applies normalized lexical feature hashing using only Python and SHA-256. It makes the run deterministic, fast, offline, and free; it is not presented as a production semantic embedding model.

## Run locally

```bash
make benchmark-native-indexing
```

The command requires only the installed Python dependencies. It does not require PostgreSQL, Qdrant, Redis, Ollama, a downloaded Sentence Transformers model, or an API key. Haystack telemetry is disabled before native components are imported. JSON and Markdown reports are written to the ignored `reports/generated/native-indexing-v1.*` paths.

Every report records the byte-level configuration and dataset SHA-256 hashes, run ID, timestamps, controls, per-case measurements, aggregates, and zero estimated API cost.

The first observed local result is committed as [`reports/baselines/native-indexing-deterministic-v1.md`](../reports/baselines/native-indexing-deterministic-v1.md). It retains the observed Recall@1 differences and explicitly avoids a framework ranking.

## Metrics

| Metric | Meaning |
| --- | --- |
| Chunk count and mean size | Native segmentation shape after applying the declared target |
| Redundancy ratio | Total indexed characters divided by source characters |
| Passage containment | Fraction of annotated passages preserved wholly inside a chunk |
| Recall@1 | Fraction of annotated queries retrieving a chunk containing the target passage |
| Section violations | Chunks crossing a detected heading offset |
| Index and query latency | Local wall-clock observations for controlled runs |

Size units are not falsely normalized: Custom and LangChain use RAGLab lexical tokens, LlamaIndex uses its configured framework tokenizer, and Haystack uses words. Actual lexical token counts are still reported for every emitted chunk.

## Interpretation limits

This small synthetic test validates the harness and exposes indexing trade-offs; it is not a framework leaderboard. The annotated questions still have strong lexical overlap with a small corpus, in-memory stores exclude network and persistence behavior, and latency varies by machine and warm-up state. A production experiment should add a larger real-document corpus, harder semantic questions, repeated warm and cold runs, confidence intervals, memory measurements, and isolated persistent namespaces in Qdrant or framework-specific stores.

The architectural decision and scale-up criteria are recorded in [ADR 0009](adr/0009-isolate-framework-native-indexing-experiments.md).
