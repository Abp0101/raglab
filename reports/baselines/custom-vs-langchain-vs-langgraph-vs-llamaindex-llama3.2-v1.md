# Four-framework local baseline

This is one controlled harness run, not evidence that any framework is generally superior.

- Run date: 2026-07-16
- Comparison ID: `448cae59-1777-42a5-8525-4188025f0003`
- Dataset SHA-256: `9987baf611926beb99ba2cffd86a2d0bfa0e3091b6fb3d67e189628d79f5ae7f`
- Model: local `llama3.2:latest`
- Retrieval: hybrid, top K 5, local cross-encoder reranking enabled
- Successful questions: 7 of 7 for every pipeline
- Paid API cost: `$0.00`

| Metric | Custom | LangChain | LangGraph | LlamaIndex |
| --- | ---: | ---: | ---: | ---: |
| Citation precision | 0.8056 | 0.8333 | 0.8333 | 0.8333 |
| Citation recall | 1.0000 | 0.8333 | 0.8333 | 0.8333 |
| Key-fact coverage | 0.4167 | 0.4167 | 0.4167 | 0.4167 |
| Mean latency (ms) | 4801.1221 | 2239.4798 | 1914.5994 | 3408.3316 |
| Mean local LLM calls | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| MRR | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| NDCG | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Refusal accuracy | 0.8571 | 0.8571 | 0.8571 | 0.8571 |
| Retrieval precision | 0.2333 | 0.2333 | 0.2333 | 0.2333 |
| Retrieval recall | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Estimated API cost (USD) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

All implementations queried the same PostgreSQL/Qdrant/Redis corpus through the same retrieval service, local embedding model, reranker, grounding schema, citation validator, question order, and Ollama model. Custom uses direct Ollama HTTP, LangChain uses its retriever and Runnable abstractions, LangGraph uses a bounded state graph, and LlamaIndex maps retrieval into `TextNode`/`NodeWithScore` objects before structured prediction.

All four implementations made one local model call per question in this run, so LangGraph's repair route was not triggered. Retrieval measurements match by design. The generation adapters produced different evidence-status classifications on some questions despite equal aggregate scores, illustrating why per-question artifacts and repeated runs matter.

Latency includes local model execution, framework prompt formatting, and warm-state effects. This single sequential run is not a performance benchmark. The shared refusal miss and low lexical key-fact score remain visible rather than being converted into a ranking claim.
