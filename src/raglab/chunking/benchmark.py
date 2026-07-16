"""Deterministic structural benchmark for chunking strategies."""

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from typing import Literal
from uuid import NAMESPACE_URL, uuid5

from pydantic import Field, model_validator

from raglab.chunking.registry import create_chunker
from raglab.chunking.tokenization import count_lexical_tokens
from raglab.core.schemas import (
    Chunk,
    ChunkingConfig,
    ChunkingStrategy,
    Document,
    DocumentPage,
    DocumentStatus,
    ParsedDocument,
    RAGLabModel,
)
from raglab.ingestion.metadata import detect_section_headings


class ChunkBenchmarkCase(RAGLabModel):
    """One versioned source text with passages that should remain retrievable."""

    dataset_version: str = Field(min_length=1, max_length=50)
    case_id: str = Field(min_length=1, max_length=100)
    category: str = Field(min_length=1, max_length=100)
    text: str = Field(min_length=1)
    relevant_passages: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def passages_exist_in_source(self) -> "ChunkBenchmarkCase":
        missing = [passage for passage in self.relevant_passages if passage not in self.text]
        if missing:
            raise ValueError("every relevant passage must occur verbatim in source text")
        return self


class ChunkBenchmarkResult(RAGLabModel):
    """Structural measurements for one case and strategy."""

    dataset_version: str
    case_id: str
    category: str
    strategy: ChunkingStrategy
    size_unit: Literal["tokens", "characters"]
    chunk_size: int
    chunk_overlap: int
    emitted_chunk_count: int = Field(ge=0)
    retrieval_chunk_count: int = Field(ge=0)
    parent_chunk_count: int = Field(ge=0)
    linked_child_count: int = Field(ge=0)
    mean_chunk_characters: float = Field(ge=0)
    mean_chunk_tokens: float = Field(ge=0)
    redundancy_ratio: float = Field(ge=0)
    relevant_passage_containment: float = Field(ge=0, le=1)
    section_boundary_violations: int = Field(ge=0)


DEFAULT_CONFIGS: Mapping[ChunkingStrategy, ChunkingConfig] = {
    ChunkingStrategy.FIXED_TOKEN: ChunkingConfig(
        strategy=ChunkingStrategy.FIXED_TOKEN,
        chunk_size=48,
        chunk_overlap=8,
    ),
    ChunkingStrategy.RECURSIVE_CHARACTER: ChunkingConfig(
        strategy=ChunkingStrategy.RECURSIVE_CHARACTER,
        chunk_size=220,
        chunk_overlap=32,
    ),
    ChunkingStrategy.SECTION_AWARE: ChunkingConfig(
        strategy=ChunkingStrategy.SECTION_AWARE,
        chunk_size=220,
        chunk_overlap=32,
    ),
    ChunkingStrategy.PARENT_CHILD: ChunkingConfig(
        strategy=ChunkingStrategy.PARENT_CHILD,
        chunk_size=160,
        chunk_overlap=24,
        parent_chunk_size=360,
        parent_chunk_overlap=48,
    ),
}


def load_cases(path: Path) -> tuple[ChunkBenchmarkCase, ...]:
    """Load a JSON Lines benchmark dataset with line-aware validation."""
    cases: list[ChunkBenchmarkCase] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            cases.append(ChunkBenchmarkCase.model_validate_json(line))
        except ValueError as error:
            raise ValueError(f"invalid benchmark case on line {line_number}") from error
    if not cases:
        raise ValueError("benchmark dataset contains no cases")
    return tuple(cases)


def run_benchmark(
    cases: Iterable[ChunkBenchmarkCase],
    configs: Mapping[ChunkingStrategy, ChunkingConfig] = DEFAULT_CONFIGS,
) -> tuple[ChunkBenchmarkResult, ...]:
    """Run every configured strategy against every benchmark case."""
    results: list[ChunkBenchmarkResult] = []
    for case in cases:
        document = _parsed_document(case)
        for strategy, config in configs.items():
            chunks = tuple(create_chunker(strategy).chunk(document, config))
            results.append(_measure(case, document.pages[0], chunks, config))
    return tuple(results)


def write_results(path: Path, results: Sequence[ChunkBenchmarkResult]) -> None:
    """Write machine-readable results without claiming a preferred strategy."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark": "raglab-chunking-v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "results": [result.model_dump(mode="json") for result in results],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _parsed_document(case: ChunkBenchmarkCase) -> ParsedDocument:
    document_id = uuid5(NAMESPACE_URL, f"raglab:chunk-benchmark:{case.case_id}")
    collection_id = uuid5(NAMESPACE_URL, "raglab:chunk-benchmark")
    return ParsedDocument(
        document=Document(
            document_id=document_id,
            collection_id=collection_id,
            file_name=f"{case.case_id}.txt",
            display_title=case.case_id,
            uploaded_at=datetime(2026, 1, 1, tzinfo=UTC),
            file_type="text/plain",
            content_hash=hashlib.sha256(case.text.encode()).hexdigest(),
            page_count=1,
            status=DocumentStatus.PROCESSING,
        ),
        pages=(
            DocumentPage(
                page_number=1,
                text=case.text,
                section_headings=detect_section_headings(case.text),
            ),
        ),
        parser_name="chunk-benchmark",
    )


def _measure(
    case: ChunkBenchmarkCase,
    page: DocumentPage,
    chunks: Sequence[Chunk],
    config: ChunkingConfig,
) -> ChunkBenchmarkResult:
    parent_ids = {
        chunk.metadata.parent_chunk_id
        for chunk in chunks
        if chunk.metadata.parent_chunk_id is not None
    }
    retrieval_chunks = tuple(chunk for chunk in chunks if chunk.chunk_id not in parent_ids)
    linked_children = sum(chunk.metadata.parent_chunk_id is not None for chunk in chunks)
    lengths = [len(chunk.text) for chunk in retrieval_chunks]
    token_counts = [
        chunk.token_count or count_lexical_tokens(chunk.text) for chunk in retrieval_chunks
    ]
    relevant_spans = [
        (case.text.index(passage), case.text.index(passage) + len(passage))
        for passage in case.relevant_passages
    ]
    contained = sum(
        any(_contains(chunk, start, end) for chunk in retrieval_chunks)
        for start, end in relevant_spans
    )
    heading_offsets = [heading.start for heading in page.section_headings]
    violations = sum(
        any(
            chunk.text_span is not None
            and chunk.text_span.start < heading_offset < chunk.text_span.end
            for heading_offset in heading_offsets
        )
        for chunk in retrieval_chunks
    )
    return ChunkBenchmarkResult(
        case_id=case.case_id,
        dataset_version=case.dataset_version,
        category=case.category,
        strategy=config.strategy,
        size_unit=("tokens" if config.strategy is ChunkingStrategy.FIXED_TOKEN else "characters"),
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        emitted_chunk_count=len(chunks),
        retrieval_chunk_count=len(retrieval_chunks),
        parent_chunk_count=len(parent_ids),
        linked_child_count=linked_children,
        mean_chunk_characters=fmean(lengths) if lengths else 0,
        mean_chunk_tokens=fmean(token_counts) if token_counts else 0,
        redundancy_ratio=sum(lengths) / len(case.text) if case.text else 0,
        relevant_passage_containment=contained / len(relevant_spans),
        section_boundary_violations=violations,
    )


def _contains(chunk: Chunk, start: int, end: int) -> bool:
    span = chunk.text_span
    return span is not None and span.start <= start and span.end >= end
