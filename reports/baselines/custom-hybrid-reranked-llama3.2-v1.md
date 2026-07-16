# Custom RAG local baseline — evaluation dataset v1

This is the first measured RAGLab baseline. It validates the harness and records current behavior; it is not evidence that this configuration is best and is not a clinical-performance result.

## Run configuration

| Field | Value |
| --- | --- |
| Dataset | `raglab-synthetic-biomedical-technical` `1.0.0` |
| Dataset SHA-256 | `9987baf611926beb99ba2cffd86a2d0bfa0e3091b6fb3d67e189628d79f5ae7f` |
| Run ID | `41ccbc7b-ac60-4151-b346-5b1ee27044f8` |
| Date | 2026-07-16 |
| Framework | Custom Python |
| Retrieval | Hybrid dense + BM25 with Reciprocal Rank Fusion |
| Reranking | Local cross-encoder enabled |
| Top K | 5 |
| Generation | Local Ollama `llama3.2:latest` |
| Paid API cost | `$0.00` |
| Successful questions | 7 / 7 |

## Deterministic measurements

| Metric | Mean | Min | Max | N |
| --- | ---: | ---: | ---: | ---: |
| Retrieval precision | 0.2333 | 0.2000 | 0.4000 | 6 |
| Retrieval recall | 1.0000 | 1.0000 | 1.0000 | 6 |
| MRR | 1.0000 | 1.0000 | 1.0000 | 6 |
| nDCG | 1.0000 | 1.0000 | 1.0000 | 6 |
| Citation precision | 0.8056 | 0.3333 | 1.0000 | 6 |
| Citation recall | 1.0000 | 1.0000 | 1.0000 | 6 |
| Refusal accuracy | 0.8571 | 0.0000 | 1.0000 | 7 |
| Lexical key-fact coverage | 0.4167 | 0.0000 | 1.0000 | 6 |
| Latency | 5055.98 ms | 2724.41 ms | 13328.37 ms | 7 |

## Result interpretation

The annotated relevant chunk appeared first for all six answerable questions and all annotated relevant chunks appeared in the top five. Low retrieval precision is expected at `top_k=5` in a corpus containing only five chunks; it shows that returning a fixed depth includes irrelevant context.

All expected citation targets were present, but citation precision fell below 1 because the model sometimes cited additional retrieved chunks. The pipeline identified the two different IMU rates as conflicting evidence. It failed the unanswerable wireless-protocol case by returning `sufficient`, leaving refusal accuracy at 6/7.

Lexical key-fact coverage is deliberately strict and does not treat paraphrases or number words as equivalent. It therefore underestimates semantically correct paraphrases and must not be read as answer correctness.

## Follow-up hypotheses

- Reduce context noise with a smaller final top K and compare citation precision and refusal accuracy.
- Add an explicit evidence-sufficiency threshold before generation.
- Evaluate whether stricter unanswerable prompt examples improve refusal without harming answerable cases.
- Add local claim-level entailment or human review before treating answer quality as measured.
- Repeat the exact dataset/configuration for each framework before comparing framework overhead.

The raw UUID-named JSON/Markdown artifacts remain ignored under `reports/generated`; this curated baseline retains the reproducibility fields and the findings worth reviewing in Git history.
