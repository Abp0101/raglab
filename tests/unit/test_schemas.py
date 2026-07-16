from datetime import UTC, date, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from raglab.core.schemas import (
    Chunk,
    ChunkingConfig,
    DocumentMetadata,
    Embedding,
    FrameworkName,
    MetadataFilter,
    PipelineConfig,
    QueryRequest,
    RetrievalOptions,
    TextSpan,
    UsageMetrics,
)


def make_metadata() -> DocumentMetadata:
    return DocumentMetadata(
        document_id=uuid4(),
        collection_id=uuid4(),
        file_name="imu-study.pdf",
        display_title="Wearable IMU Study",
        authors=("A. Engineer",),
        uploaded_at=datetime.now(UTC),
        publication_date=date(2025, 1, 2),
        file_type="application/pdf",
        page_number=4,
        section_heading="Methods",
        chunk_index=2,
        content_hash="a" * 64,
    )


def test_chunk_preserves_traceable_metadata() -> None:
    chunk = Chunk(
        chunk_id=uuid4(),
        text="The IMU was sampled at 100 Hz.",
        metadata=make_metadata(),
        token_count=9,
        text_span=TextSpan(start=10, end=42),
    )

    assert chunk.metadata.page_number == 4
    assert chunk.metadata.section_heading == "Methods"
    assert chunk.metadata.collection_id is not None


def test_embedding_rejects_dimension_mismatch() -> None:
    with pytest.raises(ValidationError, match="dimensions must match"):
        Embedding(chunk_id=uuid4(), vector=(0.1, 0.2), model="local/model", dimensions=3)


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"chunk_size": 100, "chunk_overlap": 100}, "overlap"),
        ({"chunk_size": 31}, "greater than or equal"),
    ],
)
def test_chunking_config_rejects_invalid_ranges(config: dict[str, int], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        ChunkingConfig(**config)


def test_query_request_strips_whitespace_and_rejects_unknown_fields() -> None:
    request = QueryRequest(
        query="  What sampling rate was used?  ",
        framework=FrameworkName.CUSTOM,
        collection_id=uuid4(),
    )

    assert request.query == "What sampling rate was used?"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        QueryRequest(
            query="Question",
            framework=FrameworkName.CUSTOM,
            collection_id=uuid4(),
            unsupported=True,  # type: ignore[call-arg]
        )


def test_pipeline_config_requires_enough_candidates() -> None:
    with pytest.raises(ValidationError, match="candidate_k"):
        PipelineConfig(top_k=10, candidate_k=5)


def test_metadata_filter_rejects_reversed_date_range() -> None:
    with pytest.raises(ValidationError, match="published_from"):
        MetadataFilter(published_from=date(2025, 2, 1), published_to=date(2025, 1, 1))


def test_metadata_filter_rejects_unsupported_attribute() -> None:
    with pytest.raises(ValidationError, match="unsupported metadata attributes"):
        MetadataFilter(attributes={"secret_field": "value"})


def test_retrieval_options_require_enough_candidates() -> None:
    with pytest.raises(ValidationError, match="candidate_k"):
        RetrievalOptions(candidate_k=3, top_k=5)


def test_usage_requires_consistent_total_tokens() -> None:
    with pytest.raises(ValidationError, match="total_tokens"):
        UsageMetrics(prompt_tokens=10, completion_tokens=5, total_tokens=20)
