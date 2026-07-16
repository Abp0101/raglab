# Custom versus LangChain local baseline

This is one controlled harness run, not evidence that either framework is generally superior.

- Run date: 2026-07-16
- Comparison ID: `8cce1e41-37e3-4f8f-88db-00dd6f9d7122`
- Dataset SHA-256: `9987baf611926beb99ba2cffd86a2d0bfa0e3091b6fb3d67e189628d79f5ae7f`
- Model: local `llama3.2:latest`
- Retrieval: hybrid, top K 5, local cross-encoder reranking enabled
- Successful questions: 7 of 7 for each pipeline
- Paid API cost: `$0.00`

| Metric | Custom | LangChain |
| --- | ---: | ---: |
| Citation precision | 0.8056 | 0.8333 |
| Citation recall | 1.0000 | 0.8333 |
| Key-fact coverage | 0.4167 | 0.4167 |
| Mean latency (ms) | 4930.4721 | 2238.1737 |
| MRR | 1.0000 | 1.0000 |
| NDCG | 1.0000 | 1.0000 |
| Refusal accuracy | 0.8571 | 0.8571 |
| Retrieval precision | 0.2333 | 0.2333 |
| Retrieval recall | 1.0000 | 1.0000 |
| Estimated API cost (USD) | 0.0000 | 0.0000 |

Both implementations queried the same PostgreSQL/Qdrant/Redis corpus through the same retrieval service, local embedding model, reranker, grounding schema, exact citation validator, question order, and Ollama model. The changing implementation boundary was orchestration and model integration: direct native Ollama HTTP for Custom and LangChain `BaseRetriever`, prompt/Runnable composition, and `ChatOllama` for LangChain.

The retrieval metrics match because retrieval was deliberately held constant. Generation-level citation behavior differed in this run. Latency includes local model execution and warm-state effects; one sequential run is not a performance benchmark. The shared refusal miss and low lexical key-fact score remain visible rather than being converted into a framework-ranking claim.
