# Custom, LangChain, and LangGraph pipelines

RAGLab currently provides three executable implementations behind the same `RAGPipeline` contract. All run locally and return the same typed `RAGResponse`.

| Concern | Custom | LangChain | LangGraph |
| --- | --- | --- | --- |
| Retrieval boundary | Direct service call | `BaseRetriever.ainvoke` | `retrieve` graph node |
| Orchestration | Explicit Python | Prompt/Runnable chain | Compiled `StateGraph` |
| Model integration | Native Ollama HTTP | Structured `ChatOllama` | Structured `ChatOllama` graph node |
| Conditional control | Deterministic early refusal | Deterministic early refusal | Evidence routes plus one citation-repair loop |
| Final safety | Shared context and citation validation | Same shared safeguards | Same safeguards with bounded repair |
| Paid API use | Disabled by default | No paid provider path | No paid provider path |

The LangChain ingestion adapter also uses `BaseLoader`, LangChain `Document` objects, and `RecursiveCharacterTextSplitter`, then maps the result into RAGLab's deterministic chunk and storage contracts.

## Controlled comparison

Run all executable pipelines over the same versioned dataset and settings:

```bash
make compare-frameworks RAGLAB_LLM_MODEL=llama3.2:latest
```

The runner fixes hybrid retrieval, question order, concurrency, model, corpus, embedding model, stores, reranker, grounding schema, and citation rules. It writes JSON and Markdown artifacts under `reports/generated/` and rejects failed questions or any non-zero estimated API cost.

This experiment measures complete query behavior, not isolated framework overhead. LangGraph may make a second local model call after invalid citations, which is recorded in usage. The experiment does not establish a universal framework ranking, isolate model nondeterminism, or compare framework-specific vector stores. The shared data-plane rationale is recorded in [ADR 0001](adr/0001-shared-canonical-retrieval-for-framework-adapters.md), and the bounded graph decision in [ADR 0002](adr/0002-bounded-langgraph-citation-repair.md).

## Tracing

LangSmith tracing is not configured. RAGLab's default is a local-only, zero-paid-API workflow, so the repository does not require a LangSmith account, key, or remote trace upload. Local stage latency remains available; LangGraph debug responses also include the node trace and repair count. Remote tracing should be introduced only as a separate, explicit policy decision.

## Upstream concepts

The implementation follows LangChain's documented [retriever interface](https://docs.langchain.com/oss/python/integrations/retrievers/index), [model and structured-output interface](https://docs.langchain.com/oss/python/langchain/models), and local [ChatOllama integration](https://docs.langchain.com/oss/python/integrations/chat/ollama). The graph follows the official LangGraph [Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api) model of shared state, nodes, edges, compilation, and conditional routing.
