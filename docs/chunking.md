# Chunking strategies and benchmark

RAGLab keeps chunking framework-independent so every RAG implementation can ingest the same source units. Each chunk retains its document and collection IDs, page number, closest detected heading, deterministic chunk ID, chunk position, optional parent ID, and exact character offsets into normalized page text.

## Implemented strategies

| Strategy | Size unit | Intended trade-off |
| --- | --- | --- |
| `fixed-token` | Lexical tokens | Predictable token windows and overlap, but no semantic boundary preference |
| `recursive-character` | Characters | Prefers paragraphs, lines, sentences, then spaces while respecting a hard character limit |
| `section-aware` | Characters | Never crosses a detected heading boundary; recursively splits long sections |
| `parent-child` | Characters | Emits small child chunks linked to larger context parents for later retrieval expansion |

The fixed-token strategy uses RAGLab's deterministic lexical tokenizer (`word` and punctuation tokens), not a provider-specific model tokenizer. This makes benchmark runs stable and offline, but its counts will differ from an LLM tokenizer. Context-window enforcement during generation must use the selected LLM's tokenizer.

Parent-child output contains both parents and children. Children carry `parent_chunk_id`; benchmark retrieval measurements exclude referenced parent records and treat them as context-expansion units.

## Run the benchmark

```bash
make benchmark-chunking
```

The command reads `datasets/evaluation/chunking_benchmark_v1.jsonl` and writes ignored output to `reports/generated/chunking-benchmark.json`. Every case carries an explicit semantic dataset version. The input contains synthetic technical and biomedical text plus verbatim passages that should ideally remain within a retrieval chunk.

Measurements include:

- emitted and retrieval chunk counts;
- mean character and lexical-token size;
- overlap redundancy ratio;
- relevant-passage containment;
- chunks crossing detected section boundaries;
- parent and linked-child counts.

These structural measurements do not establish answer quality. A strategy should be selected only after retrieval and generation evaluation on representative documents and questions. RAGLab therefore does not commit a winner or a benchmark-results table at this stage.
