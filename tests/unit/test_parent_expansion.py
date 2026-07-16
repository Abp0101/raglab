from collections.abc import Sequence
from uuid import UUID

import pytest

from raglab.core.schemas import RetrievedChunk
from raglab.retrieval.parent_expansion import ParentChildContextExpander
from tests.unit.retrieval_fixtures import make_chunk


class FakeChunkRepository:
    def __init__(self, chunks: Sequence[object]) -> None:
        self.chunks = chunks

    async def get_by_ids(self, chunk_ids: Sequence[UUID]) -> Sequence[object]:
        return [chunk for chunk in self.chunks if chunk.chunk_id in chunk_ids]  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_expander_replaces_siblings_with_one_parent_and_preserves_best_score() -> None:
    parent = make_chunk("large parent context")
    first_child = make_chunk("child one", parent_chunk_id=parent.chunk_id)
    second_child = make_chunk("child two", parent_chunk_id=parent.chunk_id)
    standalone = make_chunk("standalone")
    expander = ParentChildContextExpander(FakeChunkRepository([parent]))  # type: ignore[arg-type]

    expanded = await expander.expand(
        (
            RetrievedChunk(chunk=first_child, rank=1, fusion_score=0.04),
            RetrievedChunk(chunk=second_child, rank=2, fusion_score=0.03),
            RetrievedChunk(chunk=standalone, rank=3, fusion_score=0.02),
        )
    )

    assert [result.chunk.chunk_id for result in expanded] == [parent.chunk_id, standalone.chunk_id]
    assert expanded[0].fusion_score == 0.04
    assert [result.rank for result in expanded] == [1, 2]
