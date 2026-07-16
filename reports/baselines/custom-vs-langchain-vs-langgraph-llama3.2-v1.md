# Custom versus LangChain versus LangGraph local baseline

This is one controlled harness run, not evidence that any framework is generally superior.

- Run date: 2026-07-16
- Comparison ID: `5da76f5a-ad5d-40d7-92c9-213ec24368e8`
- Dataset SHA-256: `9987baf611926beb99ba2cffd86a2d0bfa0e3091b6fb3d67e189628d79f5ae7f`
- Model: local `llama3.2:latest`
- Retrieval: hybrid, top K 5, local cross-encoder reranking enabled
- Successful questions: 7 of 7 for every pipeline
- Paid API cost: `$0.00`

| Metric | Custom | LangChain | LangGraph |
| --- | ---: | ---: | ---: |
| Citation precision | 0.8056 | 0.8333 | 0.8333 |
| Citation recall | 1.0000 | 0.8333 | 0.8333 |
| Key-fact coverage | 0.4167 | 0.4167 | 0.4167 |
| Mean latency (ms) | 4788.0987 | 2230.6685 | 2228.1717 |
| Mean local LLM calls | 1.0000 | 1.0000 | 1.0000 |
| MRR | 1.0000 | 1.0000 | 1.0000 |
| NDCG | 1.0000 | 1.0000 | 1.0000 |
| Refusal accuracy | 0.8571 | 0.8571 | 0.8571 |
| Retrieval precision | 0.2333 | 0.2333 | 0.2333 |
| Retrieval recall | 1.0000 | 1.0000 | 1.0000 |
| Estimated API cost (USD) | 0.0000 | 0.0000 | 0.0000 |

All three implementations queried the same PostgreSQL/Qdrant/Redis corpus through the same retrieval service, local embedding model, reranker, grounding schema, exact citation validator, question order, and Ollama model. Custom uses direct native Ollama HTTP, LangChain uses `BaseRetriever` plus prompt/Runnable composition, and LangGraph uses a compiled `StateGraph` with named retrieval, context, generation, validation, finalization, and refusal nodes.

LangGraph's citation-repair route was not triggered in this run: every response used one local model call. Retrieval metrics match because retrieval was deliberately held constant. Generation-level results for LangChain and LangGraph also matched in this run because they share the local `ChatOllama` structured-output boundary and no repair was necessary.

Latency includes local model execution and warm-state effects; one sequential run is not a performance benchmark. The shared refusal miss and low lexical key-fact score remain visible rather than being converted into a framework-ranking claim.
