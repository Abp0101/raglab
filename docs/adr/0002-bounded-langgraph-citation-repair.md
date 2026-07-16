# ADR 0002: Bound the LangGraph citation-repair loop

- Status: Accepted
- Date: 2026-07-16
- Deciders: RAGLab maintainers

## Context

The LangGraph implementation should demonstrate stateful, conditional RAG orchestration rather than reproducing a linear chain under a different framework label. It must also remain safe, predictable, locally runnable, and comparable with the Custom and LangChain implementations.

Unbounded tool use, open-ended retrieval loops, and LLM-as-a-judge grading would make latency and behavior difficult to control. They would also add no reliable safety guarantee for the current small evaluation corpus.

## Decision

Use one request-scoped `StateGraph` with these nodes and routes:

```text
START → retrieve ── no evidence ───────────────→ refuse → END
                 └─ evidence → build_context
                                  ├─ empty ────→ refuse → END
                                  └─ context → generate → validate
                                                          ├─ valid → finalize → END
                                                          ├─ insufficient → refuse → END
                                                          └─ invalid citation → generate once
                                                                                 └→ validate
```

The repair route is limited to one additional local model call. It tells the model to use exact quotes and available chunk IDs or report insufficient evidence. A second invalid result is deterministically refused. The graph has a recursion limit and does not use a checkpointer.

## Options considered

### Linear graph without conditional routes

Simple, but it would provide little value beyond the LangChain Runnable pipeline and would not demonstrate graph state or routing.

### Open-ended agent with retrieval tools

Flexible, but difficult to benchmark and vulnerable to runaway work. The current canonical retrieval service already exposes the desired dense, sparse, hybrid, filtering, and reranking behavior.

### LLM evidence grader before generation

Common in agentic RAG examples, but it adds another nondeterministic model call and effectively introduces an LLM judge into the answer path. Deterministic absence and citation checks are clearer for this milestone.

### One bounded citation-repair route

Selected because it exercises meaningful conditional state transitions while preserving deterministic termination and zero paid API cost.

## Consequences

- LangGraph reports may include a second generation call when citations fail validation, so latency and token counts are part of the framework behavior rather than pure orchestration overhead.
- Debug responses expose the node trace and repair count without uploading remote traces.
- The graph is request-scoped and cannot resume after process failure. A checkpointer should be reconsidered when distributed execution and human review are implemented.
- Retrieval remains canonical and executes once per query, keeping retrieval metrics comparable.
