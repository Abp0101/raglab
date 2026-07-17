# Custom, LangChain, LangGraph, LlamaIndex, and Haystack pipelines

RAGLab provides five executable implementations behind the same `RAGPipeline` contract. All run locally and return the same typed `RAGResponse`.

| Concern | Custom | LangChain | LangGraph | LlamaIndex | Haystack |
| --- | --- | --- | --- | --- | --- |
| Retrieval boundary | Direct service | `BaseRetriever.ainvoke` | Graph node | `BaseRetriever.aretrieve` | Async component |
| Native evidence type | RAGLab chunk | LangChain `Document` | Graph state | `TextNode`/`NodeWithScore` | Haystack `Document` |
| Orchestration | Explicit Python | Runnable chain | `StateGraph` | Component composition | `AsyncPipeline` |
| Model integration | Native Ollama HTTP | `ChatOllama` | `ChatOllama` node | LlamaIndex Ollama | `OllamaChatGenerator` |
| Conditional control | Early refusal | Early refusal | Evidence routes and repair | Early refusal | Safe generator component |
| Final safety | Shared validators | Shared validators | Shared validators and repair | Shared validators | Shared validators |
| Paid API use | Disabled | No paid path | No paid path | No paid path | No paid path; telemetry off |

The LangChain ingestion adapter also uses `BaseLoader`, LangChain `Document` objects, and `RecursiveCharacterTextSplitter`, then maps the result into RAGLab's deterministic chunk and storage contracts.

## Controlled comparison

Run all executable pipelines over the same versioned dataset and settings:

```bash
make compare-frameworks RAGLAB_LLM_MODEL=llama3.2:latest
```

The runner fixes hybrid retrieval, question order, concurrency, model, corpus, embedding model, stores, reranker, grounding schema, and citation rules. It writes JSON and Markdown artifacts under `reports/generated/` and rejects failed questions or any non-zero estimated API cost.

This experiment measures complete query behavior, not isolated framework overhead. LangGraph may make a second local model call after invalid citations, which is recorded in usage. The experiment does not establish a universal framework ranking, isolate model nondeterminism, or compare framework-specific vector stores. The shared data-plane rationale is recorded in [ADR 0001](adr/0001-shared-canonical-retrieval-for-framework-adapters.md), the bounded graph decision in [ADR 0002](adr/0002-bounded-langgraph-citation-repair.md), the LlamaIndex mapping in [ADR 0003](adr/0003-map-canonical-retrieval-to-llamaindex-nodes.md), and the Haystack graph in [ADR 0004](adr/0004-use-haystack-async-components-over-shared-retrieval.md).

## Tracing

Remote framework tracing and telemetry are not configured. RAGLab's local-only, zero-paid-API workflow does not require LangSmith, LlamaCloud, Haystack telemetry, an account key, or remote trace upload. The Haystack environment flag is set to false before imports, and RAGLab nulls the telemetry singleton if another dependency imported Haystack first. Local stage latency remains available; debug responses expose native orchestration and node identifiers where useful. Remote tracing should be introduced only as a separate, explicit policy decision.

## Upstream concepts

The implementation follows LangChain's documented [retriever interface](https://docs.langchain.com/oss/python/integrations/retrievers/index), [model and structured-output interface](https://docs.langchain.com/oss/python/langchain/models), and local [ChatOllama integration](https://docs.langchain.com/oss/python/integrations/chat/ollama). The graph follows the official LangGraph [Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api). The LlamaIndex adapter follows its [retriever APIs](https://developers.llamaindex.ai/python/framework-api-reference/retrievers/) and local [Ollama integration](https://developers.llamaindex.ai/python/framework/integrations/llm/ollama/). The Haystack adapter follows its [custom component](https://docs.haystack.deepset.ai/docs/custom-components), [AsyncPipeline](https://docs.haystack.deepset.ai/docs/asyncpipeline), [ChatPromptBuilder](https://docs.haystack.deepset.ai/docs/chatpromptbuilder), and local [OllamaChatGenerator](https://docs.haystack.deepset.ai/docs/ollamachatgenerator) interfaces.
