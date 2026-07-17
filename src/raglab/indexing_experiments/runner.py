"""Reproducible runner for isolated framework-native indexing experiments."""

import hashlib
import json
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from uuid import uuid4

from pydantic import ValidationError

from raglab.indexing_experiments.adapters import (
    DEFAULT_ADAPTER_FACTORIES,
    NativeIndexAdapter,
)
from raglab.indexing_experiments.models import (
    ExperimentChunk,
    IndexingAggregate,
    IndexingBenchmarkCase,
    IndexingCaseResult,
    IndexingExperimentDefinition,
    IndexingExperimentPlan,
    IndexingExperimentRun,
    IndexingFramework,
)
from raglab.ingestion.metadata import detect_section_headings

AdapterFactory = Callable[[], NativeIndexAdapter]


class IndexingExperimentError(ValueError):
    """An experiment declaration or native adapter violated benchmark controls."""


def load_plan(path: Path) -> tuple[IndexingExperimentPlan, str]:
    """Load a strict plan and return the checksum of its exact bytes."""
    try:
        content = path.read_bytes()
        plan = IndexingExperimentPlan.model_validate_json(content)
    except (OSError, ValidationError) as error:
        raise IndexingExperimentError("indexing experiment plan could not be loaded") from error
    return plan, hashlib.sha256(content).hexdigest()


def load_plan_cases(
    plan: IndexingExperimentPlan,
    *,
    project_root: Path = Path("."),
) -> tuple[tuple[IndexingBenchmarkCase, ...], str]:
    """Load the plan's immutable dataset and verify its declared semantic version."""
    path = project_root / plan.dataset_path
    try:
        content = path.read_bytes()
    except OSError as error:
        raise IndexingExperimentError("indexing experiment dataset could not be loaded") from error
    cases = _parse_cases(content)
    versions = {case.dataset_version for case in cases}
    if versions != {plan.dataset_version}:
        raise IndexingExperimentError("dataset versions do not match the experiment plan")
    return cases, hashlib.sha256(content).hexdigest()


def run_indexing_experiments(
    plan: IndexingExperimentPlan,
    cases: Sequence[IndexingBenchmarkCase],
    *,
    dataset_sha256: str,
    config_sha256: str,
    adapter_factories: Mapping[IndexingFramework, AdapterFactory] = DEFAULT_ADAPTER_FACTORIES,
) -> IndexingExperimentRun:
    """Build and query every declared native index under the same controls."""
    started = datetime.now(UTC)
    results: list[IndexingCaseResult] = []
    for definition in plan.experiments:
        try:
            adapter = adapter_factories[definition.framework]()
        except KeyError as error:
            raise IndexingExperimentError(
                f"no native adapter is registered for {definition.framework.value}"
            ) from error
        _validate_adapter_declaration(adapter, definition)
        for case in cases:
            indexing_started = time.perf_counter()
            index = adapter.build(case, plan.controls)
            indexing_ms = (time.perf_counter() - indexing_started) * 1000
            _validate_index(index.chunks, case)
            results.append(
                _measure_case(
                    case,
                    adapter,
                    index.chunks,
                    index.search,
                    plan,
                    indexing_ms,
                )
            )
    result_tuple = tuple(results)
    return IndexingExperimentRun(
        run_id=uuid4(),
        benchmark=plan.benchmark,
        version=plan.version,
        dataset_version=plan.dataset_version,
        dataset_sha256=dataset_sha256,
        config_sha256=config_sha256,
        started_at=started,
        completed_at=datetime.now(UTC),
        controls=plan.controls,
        results=result_tuple,
        aggregates=_aggregate(result_tuple),
    )


def _measure_case(
    case: IndexingBenchmarkCase,
    adapter: NativeIndexAdapter,
    chunks: Sequence[ExperimentChunk],
    search: Callable[[str, int], tuple[str, ...]],
    plan: IndexingExperimentPlan,
    indexing_ms: float,
) -> IndexingCaseResult:
    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    contained = sum(
        any(query.relevant_passage in chunk.text for chunk in chunks) for query in case.queries
    )
    retrieved = 0
    query_durations: list[float] = []
    for query in case.queries:
        query_started = time.perf_counter()
        result_ids = search(query.query, plan.controls.top_k)
        query_durations.append((time.perf_counter() - query_started) * 1000)
        if any(
            result_id in chunk_by_id and query.relevant_passage in chunk_by_id[result_id].text
            for result_id in result_ids
        ):
            retrieved += 1
    heading_offsets = [heading.start for heading in detect_section_headings(case.text)]
    violations = sum(
        any(chunk.start < heading_offset < chunk.end for heading_offset in heading_offsets)
        for chunk in chunks
    )
    return IndexingCaseResult(
        case_id=case.case_id,
        category=case.category,
        framework=adapter.framework,
        strategy=adapter.strategy,
        index_backend=adapter.index_backend,
        size_unit=adapter.size_unit,
        chunk_size=plan.controls.chunk_size,
        chunk_overlap=plan.controls.chunk_overlap,
        chunk_count=len(chunks),
        mean_chunk_characters=fmean(len(chunk.text) for chunk in chunks) if chunks else 0,
        mean_chunk_tokens=fmean(chunk.token_count for chunk in chunks) if chunks else 0,
        redundancy_ratio=(sum(len(chunk.text) for chunk in chunks) / len(case.text)),
        relevant_passage_containment=contained / len(case.queries),
        retrieval_recall_at_k=retrieved / len(case.queries),
        section_boundary_violations=violations,
        indexing_ms=indexing_ms,
        mean_query_ms=fmean(query_durations),
    )


def _aggregate(results: Sequence[IndexingCaseResult]) -> tuple[IndexingAggregate, ...]:
    grouped: dict[IndexingFramework, list[IndexingCaseResult]] = defaultdict(list)
    for result in results:
        grouped[result.framework].append(result)
    return tuple(
        IndexingAggregate(
            framework=framework,
            strategy=framework_results[0].strategy,
            index_backend=framework_results[0].index_backend,
            case_count=len(framework_results),
            mean_chunk_count=fmean(result.chunk_count for result in framework_results),
            mean_chunk_tokens=fmean(result.mean_chunk_tokens for result in framework_results),
            mean_redundancy_ratio=fmean(result.redundancy_ratio for result in framework_results),
            mean_passage_containment=fmean(
                result.relevant_passage_containment for result in framework_results
            ),
            mean_retrieval_recall_at_k=fmean(
                result.retrieval_recall_at_k for result in framework_results
            ),
            mean_indexing_ms=fmean(result.indexing_ms for result in framework_results),
            mean_query_ms=fmean(result.mean_query_ms for result in framework_results),
        )
        for framework, framework_results in grouped.items()
    )


def _validate_adapter_declaration(
    adapter: NativeIndexAdapter,
    definition: IndexingExperimentDefinition,
) -> None:
    declared = (
        definition.framework,
        definition.strategy,
        definition.index_backend,
        definition.size_unit,
    )
    implemented = (
        adapter.framework,
        adapter.strategy,
        adapter.index_backend,
        adapter.size_unit,
    )
    if declared != implemented:
        raise IndexingExperimentError(
            f"declared experiment does not match {adapter.framework.value} adapter"
        )


def _validate_index(chunks: Sequence[ExperimentChunk], case: IndexingBenchmarkCase) -> None:
    if not chunks:
        raise IndexingExperimentError(f"native index emitted no chunks for {case.case_id}")
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise IndexingExperimentError("native index emitted duplicate chunk IDs")
    if any(chunk.end > len(case.text) for chunk in chunks):
        raise IndexingExperimentError("native chunk offsets exceed source text")
    if any(case.text[chunk.start : chunk.end] != chunk.text for chunk in chunks):
        raise IndexingExperimentError("native chunk offsets do not preserve exact source text")


def _parse_cases(content: bytes) -> tuple[IndexingBenchmarkCase, ...]:
    try:
        lines = content.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise IndexingExperimentError("indexing experiment dataset must be UTF-8") from error
    cases: list[IndexingBenchmarkCase] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            cases.append(IndexingBenchmarkCase.model_validate(json.loads(line)))
        except (json.JSONDecodeError, ValidationError) as error:
            raise IndexingExperimentError(
                f"invalid indexing experiment case on line {line_number}"
            ) from error
    if not cases:
        raise IndexingExperimentError("indexing experiment dataset contains no cases")
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise IndexingExperimentError("indexing experiment case IDs must be unique")
    return tuple(cases)
