# Grounded generation and refusal behavior

Phase 5B completes the framework-free RAG path from retrieval to a validated answer. RAGLab supports an OpenAI-compatible Chat Completions endpoint and Ollama's native chat endpoint behind the same `LLMProvider` contract.

## Provider boundary

The OpenAI-compatible adapter uses `POST /chat/completions`, structured `response_format`, provider-reported token usage, and optional user-configured per-million-token rates. It does not hard-code model pricing because prices change and compatible providers differ. Current OpenAI defaults use a `developer` instruction message, JSON Schema structured output, and `max_completion_tokens`; each can be changed for compatible servers through environment settings.

RAGLab's project policy is zero paid API usage. The OpenAI-compatible adapter is disabled by default with `RAGLAB_ALLOW_PAID_API_USAGE=false`; selecting it without the explicit opt-in fails during service construction. It remains in the codebase as a provider-boundary demonstration and is covered using mocked HTTP only. The normal runtime, demos, and future evaluation use Ollama.

The Ollama adapter uses `POST /api/chat`, disables streaming for structured validation, passes the Pydantic JSON Schema through `format`, and maps `prompt_eval_count` and `eval_count` into shared usage metrics. Local generation reports estimated API cost as zero; hardware and electricity costs are outside the current estimator.

Pull the default model and verify it with:

```bash
ollama pull qwen3:8b
make smoke-ollama
```

To use a model already installed locally, override the Make variable, for example `make smoke-ollama RAGLAB_LLM_MODEL=llama3.2:latest`. The command performs one small structured-output request and validates the returned JSON locally.

Relevant references:

- [OpenAI Chat Completions reference](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)
- [Ollama chat API](https://docs.ollama.com/api/chat)
- [Ollama structured outputs](https://docs.ollama.com/capabilities/structured-outputs)

## Grounding contract

The generator receives a strict JSON schema containing:

- answer text;
- citation requests with chunk UUIDs and exact quotes;
- `sufficient`, `insufficient`, or `conflicting` evidence status;
- confidence from 0 to 1;
- warnings.

Pydantic validates the model response. RAGLab then independently checks every requested chunk ID against the context and verifies that each quote occurs in the visible evidence after whitespace normalization. A supposedly supported answer with no valid citation is replaced by a standard insufficient-evidence refusal. If retrieval or context construction produces no evidence, the pipeline refuses without calling the LLM.

Citation validation proves that a quoted span exists; it does not prove that every sentence in an answer logically follows from that span. Claim-level entailment evaluation is part of the evaluation milestone.

## Prompt-injection boundary

Evidence is serialized as JSON inside an explicit `untrusted_evidence_json` delimiter. The higher-priority instruction states that evidence is data, never instructions, and forbids following commands, role changes, policies, or URLs embedded in documents. Only the text actually included within the context budget can be cited.

This reduces direct document prompt injection but cannot guarantee model behavior. Deterministic citation validation, refusal rules, provider isolation, output schemas, logging redaction, and later adversarial evaluation provide defense in depth.

## Context budget

The current context builder uses RAGLab's deterministic lexical tokenizer so it remains provider-independent. This is an estimate, not the selected model's exact tokenizer. Provider-specific token counting and reserved output budgets should be added before supporting very tight context windows.

## Failure behavior

HTTP failures become `ProviderUnavailableError` without returning provider bodies or credentials. Missing assistant content and schema-invalid JSON become `MalformedProviderResponseError`. API routes will translate these typed failures into safe responses in the API milestone.
