# Custom and LangChain pipelines

RAGLab currently provides two executable implementations behind the same `RAGPipeline` contract. Both run locally and return the same typed `RAGResponse`.

| Concern | Custom | LangChain |
| --- | --- | --- |
| Retrieval boundary | Direct `RetrievalService` call | Native `BaseRetriever.ainvoke` |
| Prompt construction | RAGLab message schema | `ChatPromptTemplate` |
| Model integration | Native Ollama HTTP provider | `ChatOllama` |
| Structured output | JSON-schema request plus Pydantic validation | `with_structured_output` plus Pydantic validation |
| Final safety | Shared context limits, exact citation validation, and refusal | Same shared safeguards |
| Paid API use | Disabled by default | No paid provider path exists |

The LangChain ingestion adapter also uses `BaseLoader`, LangChain `Document` objects, and `RecursiveCharacterTextSplitter`, then maps the result into RAGLab's deterministic chunk and storage contracts.

## Controlled comparison

Run both pipelines over the same versioned dataset and settings:

```bash
make compare-frameworks RAGLAB_LLM_MODEL=llama3.2:latest
```

The runner fixes hybrid retrieval, question order, concurrency, model, corpus, embedding model, stores, reranker, grounding schema, and citation rules. It writes JSON and Markdown artifacts under `reports/generated/` and rejects failed questions or any non-zero estimated API cost.

This experiment measures the complete query behavior of two orchestration/model adapters. It does not establish a universal framework ranking, isolate model nondeterminism, or compare LangChain-specific vector-store implementations. The rationale is recorded in [ADR 0001](adr/0001-shared-canonical-retrieval-for-framework-adapters.md).

## Tracing

LangSmith tracing is not configured. RAGLab's default is a local-only, zero-paid-API workflow, so the repository does not require a LangSmith account, key, or remote trace upload. Local stage latency and optional response debug metadata remain available. Remote tracing should be introduced only as a separate, explicit policy decision.

## Upstream concepts

The implementation follows LangChain's documented [retriever interface](https://docs.langchain.com/oss/python/integrations/retrievers/index), [model and structured-output interface](https://docs.langchain.com/oss/python/langchain/models), and local [ChatOllama integration](https://docs.langchain.com/oss/python/integrations/chat/ollama).
