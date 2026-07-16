"""Parent context expansion backed by the relational chunk source of truth."""

from collections.abc import Sequence

from raglab.core.interfaces import ChunkRepository
from raglab.core.schemas import RetrievedChunk


class ParentChildContextExpander:
    """Replace linked children with unique parent context chunks."""

    def __init__(self, repository: ChunkRepository) -> None:
        self._repository = repository

    async def expand(self, chunks: Sequence[RetrievedChunk]) -> Sequence[RetrievedChunk]:
        parent_ids = tuple(
            dict.fromkeys(
                chunk.chunk.metadata.parent_chunk_id
                for chunk in chunks
                if chunk.chunk.metadata.parent_chunk_id is not None
            )
        )
        parents = {chunk.chunk_id: chunk for chunk in await self._repository.get_by_ids(parent_ids)}
        expanded: list[RetrievedChunk] = []
        seen: set[object] = set()
        for result in chunks:
            parent_id = result.chunk.metadata.parent_chunk_id
            replacement = parents.get(parent_id) if parent_id is not None else None
            final_chunk = replacement or result.chunk
            if final_chunk.chunk_id in seen:
                continue
            seen.add(final_chunk.chunk_id)
            expanded.append(
                result.model_copy(update={"chunk": final_chunk, "rank": len(expanded) + 1})
            )
        return tuple(expanded)
