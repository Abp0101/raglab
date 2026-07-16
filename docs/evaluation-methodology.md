# Evaluation methodology

RAGLab evaluates every framework through the same `QueryRequest` and `RAGResponse` contracts. The initial harness uses deterministic annotations and metrics only. It does not call a paid API or an LLM judge, and it does not claim that the current custom pipeline or any future framework is best.

## Versioned dataset

`datasets/evaluation/v1` is a small synthetic biomedical-engineering and wearable-sensor corpus released as CC0-1.0. It exists to validate the evaluation machinery and exercise:

- single-source fact retrieval;
- multi-fact answers;
- calibration and safety guidance;
- conflicting sampling-rate evidence across documents;
- refusal when the corpus contains no answer.

The corpus is synthetic and must not be used to assess clinical safety, diagnostic performance, or real-world generalization. `manifest.json` records the version, fixed collection UUID, domains, question count, and SHA-256 of the exact `questions.jsonl` bytes. Loading fails on checksum, count, version, validation, or duplicate-ID errors.

The committed PDFs and their IDs are rebuilt by `scripts/build_evaluation_dataset.py`. PDF output is deterministic, document IDs are UUIDv5 values derived from collection ID and content SHA-256, and chunk IDs are derived from document ID, strategy, page, and offsets. Running the builder twice must produce identical file hashes.

## Annotations

Each JSONL question records:

- dataset version, category, and difficulty;
- whether the corpus contains enough evidence;
- normalized key facts expected in an answer;
- relevant document and chunk UUIDs;
- expected citation chunk UUIDs.

Chunk annotations are tied to the default recursive-character configuration. A chunking experiment must build a separately versioned annotation set rather than silently reusing incompatible IDs.

## Deterministic metrics

For an ordered retrieved list and a binary set of annotated relevant items:

- **retrieval precision** = relevant returned items / returned items;
- **retrieval recall** = relevant returned items / annotated relevant items;
- **MRR** = reciprocal rank of the first relevant result;
- **nDCG** = binary discounted cumulative gain divided by the ideal gain at the returned depth;
- **citation precision** = expected unique citations / all unique citations;
- **citation recall** = expected unique citations returned / expected citations;
- **refusal accuracy** = 1 when answerable questions are answered and unanswerable questions are refused;
- **key-fact coverage** = normalized expected fact strings found in the answer / expected facts.

Latency and estimated API cost are recorded as observations. Metrics without applicable annotations are marked `applicable=false` and excluded from aggregates instead of being treated as zeros.

## What these metrics do not prove

Lexical key-fact coverage is not semantic correctness or entailment. Citation UUID accuracy does not prove that every answer claim follows from the cited text. Binary relevance does not express graded usefulness, and latency is hardware- and load-dependent. The initial dataset is intentionally too small for statistical performance claims.

Future additions may include local NLI checks, human ratings, confidence intervals, adversarial prompt-injection cases, and optional RAGAS/DeepEval adapters. Any LLM-judge metric must remain optional, separately labeled, reproducible, and local by default.

## Reproducible local run

Prerequisites are the migrated Compose services and an already installed Ollama model.

```bash
make build-evaluation-dataset
make seed-evaluation
make evaluate RAGLAB_LLM_MODEL=llama3.2:latest
```

`seed-evaluation` idempotently creates the fixed collection and ingests the three committed PDFs. `evaluate` forcibly selects Ollama and disables paid-provider opt-in, runs with bounded concurrency, and writes UUID-named JSON and Markdown files under the ignored `reports/generated` directory. The runner aborts if any successful response reports nonzero API cost.

For a fair comparison, change one declared variable at a time—framework, retrieval mode, top K, reranking, model, or chunking dataset version—and retain the complete report configuration and dataset checksum.

The first curated result is [`reports/baselines/custom-hybrid-reranked-llama3.2-v1.md`](../reports/baselines/custom-hybrid-reranked-llama3.2-v1.md). It records weaknesses as well as successful metrics and should be treated as a harness baseline, not a leaderboard.

## Testing strategy

The evaluation layer follows the project testing pyramid:

- unit tests cover formula edge cases, applicability, refusal scoring, runner ordering, and report aggregation;
- dataset tests verify the committed checksum and reject tampering;
- integration tests exercise local storage and ingestion independently;
- the manual evaluation command covers the full local retrieval/reranking/Ollama path.

Current gaps are statistical confidence intervals, larger real-document annotations, human adjudication, load testing, and cross-framework runs. These are recorded as future work rather than hidden behind a single aggregate score.
