# Framework-native indexing baseline

This is one observed local run of the isolated native indexing experiment. It validates the benchmark and records trade-offs; it is not a framework leaderboard.

## Reproducibility

- Run ID: `d55ecc7d-9e3c-4995-8d37-6376ef86f81f`
- Benchmark version: `1.0.0`
- Dataset version: `1.0.0`
- Dataset SHA-256: `6f918c29ee9451f4c3c26d14d502a45b36d52ddccd151ba242bc2f0d63f520c5`
- Configuration SHA-256: `a7aa4649d5cb9e728ea03f653d72c167c82a473b45accff87324fc8c611526d0`
- Embedding control: `deterministic-hash-v1` (128 dimensions)
- Chunk target / overlap: `50` / `8`
- Retrieval cutoff: `1`
- Paid API cost: `$0.00`

## Aggregate observations

| Framework | Native strategy | Native index | Chunks | Tokens/chunk | Redundancy | Containment | Recall@1 | Index ms | Query ms |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Custom | Fixed token | RAGLab in-memory cosine | 2.33 | 43.67 | 1.122 | 1.000 | 0.833 | 0.18 | 0.02 |
| LangChain | Recursive character with token budget | `InMemoryVectorStore` | 2.67 | 34.50 | 0.994 | 1.000 | 0.667 | 0.16 | 0.05 |
| LlamaIndex | `SentenceSplitter` | `VectorStoreIndex` | 2.00 | 44.67 | 0.998 | 1.000 | 0.833 | 15.99 | 0.58 |
| Haystack | Word `DocumentSplitter` | `InMemoryDocumentStore` | 2.00 | 49.33 | 1.100 | 1.000 | 0.833 | 0.39 | 0.05 |

All four paths preserved every annotated passage inside at least one chunk. Recall@1 differed despite the shared hashing embedding, demonstrating why native indexing cannot be silently substituted into the canonical orchestration comparison. The latency values include framework object construction and warm-up on one machine and should not be generalized.

## Limits

- The small synthetic corpus and lexically related questions validate the harness only.
- Deterministic hashing is not a production semantic embedding model.
- In-memory stores exclude persistence, network, update, and concurrency behavior.
- Latency needs repeated controlled trials and confidence intervals before comparison.
- LangGraph is excluded because it does not own a separate indexing abstraction.
