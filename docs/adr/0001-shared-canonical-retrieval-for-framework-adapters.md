# ADR 0001: Share canonical retrieval across framework adapters

- Status: Accepted
- Date: 2026-07-16
- Deciders: RAGLab maintainers

## Context

RAGLab compares RAG orchestration frameworks through one typed API and one versioned evaluation dataset. A comparison becomes difficult to interpret if each framework silently changes the corpus, chunk identifiers, vector store, sparse index, filtering rules, reranker, or context safety checks. The project must also remain runnable without metered APIs.

LangChain should still be represented by real framework primitives rather than a label placed around the custom pipeline.

## Decision

RAGLab uses PostgreSQL, Qdrant, Redis/BM25, local embeddings, the retrieval service, reranker, context builder, citation validator, and response schema as canonical shared infrastructure.

The LangChain adapter uses native framework boundaries where orchestration differs:

- `BaseLoader`, LangChain `Document`, and `RecursiveCharacterTextSplitter` in its ingestion adapter;
- `BaseRetriever` to expose the canonical retrieval service;
- `ChatPromptTemplate` and the Runnable composition boundary for prompting;
- `ChatOllama.with_structured_output` for local schema-constrained generation.

The controlled evaluation corpus is ingested once into the canonical stores. Custom-versus-LangChain reports therefore vary orchestration and the model adapter, while holding retrieval inputs and safeguards constant.

## Options considered

### Independent stores and ingestion for every framework

This maximizes framework autonomy, but chunk drift and store-specific scoring would confound an orchestration comparison. It also duplicates operational infrastructure and makes exact citation comparison harder.

### Wrap `CustomRAGPipeline` and change only the framework label

This preserves perfect comparability but does not exercise LangChain primitives, so it would not be a genuine framework implementation.

### Shared canonical data plane with native orchestration adapters

This is the selected approach. It keeps the experimental variable narrow while still testing meaningful native framework composition.

## Consequences

- Cross-framework reports can state exactly which variable changed.
- Chunk and citation identifiers remain comparable.
- Storage, retrieval, and safety fixes apply consistently to every adapter.
- The comparison does not measure framework-specific vector stores or retrievers; those require a separately declared experiment.
- LangChain ingestion can be tested independently, but the controlled query comparison uses the canonical pre-ingested corpus.
- Ollama remains the only model path in the LangChain adapter, and reported API cost is exactly zero.
