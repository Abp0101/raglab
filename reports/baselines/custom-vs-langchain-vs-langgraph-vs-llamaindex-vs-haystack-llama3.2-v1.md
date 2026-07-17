# Five-framework local baseline

This is one controlled harness run, not evidence that any framework is generally superior.

- Run date: 2026-07-17
- Comparison ID: `dbbda712-5569-45a7-969c-afb9ea5dc3c6`
- Dataset SHA-256: `9987baf611926beb99ba2cffd86a2d0bfa0e3091b6fb3d67e189628d79f5ae7f`
- Model: local `llama3.2:latest`
- Retrieval: hybrid, top K 5, local cross-encoder reranking enabled
- Successful questions: 7 of 7 for every pipeline
- Paid API cost: `$0.00`

| Metric | Custom | LangChain | LangGraph | LlamaIndex | Haystack |
| --- | ---: | ---: | ---: | ---: | ---: |
| Citation precision | 0.8056 | 0.8333 | 1.0000 | 1.0000 | 0.8333 |
| Citation recall | 1.0000 | 0.8333 | 1.0000 | 1.0000 | 0.8333 |
| Key-fact coverage | 0.4167 | 0.4167 | 0.4167 | 0.4167 | 0.4167 |
| Mean latency (ms) | 2726.7392 | 1111.8233 | 1332.2198 | 1891.7938 | 1508.1083 |
| Mean local LLM calls | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| MRR | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| NDCG | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Refusal accuracy | 0.8571 | 0.7143 | 1.0000 | 0.8571 | 0.7143 |
| Retrieval precision | 0.2333 | 0.2333 | 0.2333 | 0.2333 | 0.2333 |
| Retrieval recall | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Estimated API cost (USD) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

All implementations queried the same PostgreSQL/Qdrant/Redis corpus through the same retrieval service, local embedding model, reranker, grounding schema, citation validator, question order, and Ollama model. Custom uses direct Ollama HTTP, LangChain uses its retriever and Runnable abstractions, LangGraph uses a bounded state graph, LlamaIndex maps retrieval into `TextNode`/`NodeWithScore` objects, and Haystack executes native async components over scored Haystack `Document` objects.

All five implementations made one local model call per question in this run, so LangGraph's repair route was not triggered. Retrieval measurements match by design. The generation adapters produced different evidence-status and citation decisions, illustrating why per-question artifacts and repeated runs matter.

Latency includes local model execution, framework prompt formatting, request-scoped graph construction, and warm-state effects. This single sequential run is not a performance benchmark. The shared low lexical key-fact score and the LangChain/Haystack refusal misses remain visible rather than being converted into a ranking claim.
