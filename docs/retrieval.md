# Retrieval baseline

Phase 5A implements retrieval without an LLM or framework wrapper. This keeps retrieval quality measurable before answer generation can hide weak evidence selection.

## Modes

| Mode | Execution |
| --- | --- |
| `dense` | Embed the query locally and search Qdrant with cosine similarity |
| `sparse` | Tokenize the query and score the collection with Okapi BM25 |
| `hybrid` | Run dense and BM25 retrieval, then combine ranks with Reciprocal Rank Fusion |

Dense and sparse native scores are not normalized together. Cosine similarity and BM25 have different meanings and scales. Hybrid results therefore preserve `dense_score` and `sparse_score` while adding a separate `fusion_score` computed as `Σ 1 / (rrf_k + rank)`.

The optional cross-encoder reranker scores query/chunk pairs after first-stage retrieval. Its raw score is retained as `reranker_score`; it does not overwrite dense, sparse, or fusion provenance.

## Metadata filtering

Both dense and sparse retrieval support the same portable filters:

- document IDs;
- authors;
- publication-date range;
- file types;
- section headings;
- selected exact attributes: file name, display title, page number, chunk index, and content hash.

Qdrant translates these into server-side filters. The BM25 baseline applies them before corpus statistics and scoring, so filtered BM25 scores describe the filtered corpus. Unsupported arbitrary attributes are rejected at request validation rather than ignored.

## Parent-child expansion

Parent-child ingestion stores all chunks in PostgreSQL but indexes only child retrieval units in Qdrant and Redis. After optional reranking, linked children can be replaced with their larger PostgreSQL parent. Siblings resolving to the same parent are deduplicated, and the strongest child's score provenance is retained.

## Current scaling boundary

The baseline BM25 retriever reads the tokenized logical collection from Redis and computes exact scores in Python. This is deterministic and transparent for the initial benchmark corpus, but it is not intended for millions of chunks. Later scale testing may justify Redis Search, Qdrant sparse vectors, or a dedicated search engine; such a change must preserve the shared retrieval contract and benchmark comparability.

Generation, citation validation, evidence sufficiency, and refusal behavior are intentionally deferred to Phase 5B.
