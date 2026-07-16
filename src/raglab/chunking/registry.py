"""Explicit chunking strategy selection without global mutable state."""

from raglab.chunking.fixed import FixedTokenChunker
from raglab.chunking.parent_child import ParentChildChunker
from raglab.chunking.recursive import RecursiveCharacterChunker
from raglab.chunking.section_aware import SectionAwareChunker
from raglab.core.interfaces import Chunker
from raglab.core.schemas import ChunkingStrategy


def create_chunker(strategy: ChunkingStrategy) -> Chunker:
    """Create the selected stateless chunking implementation."""
    chunkers: dict[ChunkingStrategy, Chunker] = {
        ChunkingStrategy.FIXED_TOKEN: FixedTokenChunker(),
        ChunkingStrategy.RECURSIVE_CHARACTER: RecursiveCharacterChunker(),
        ChunkingStrategy.SECTION_AWARE: SectionAwareChunker(),
        ChunkingStrategy.PARENT_CHILD: ParentChildChunker(),
    }
    return chunkers[strategy]
