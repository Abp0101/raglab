"""Deterministic, local feature hashing used only as an experiment control."""

import hashlib
import math
from collections.abc import Sequence

from raglab.chunking.tokenization import lexical_token_spans


def deterministic_hash_embedding(text: str, dimensions: int) -> list[float]:
    """Create a normalized non-negative hashing vector without models or network calls."""
    if dimensions <= 0:
        raise ValueError("embedding dimensions must be positive")
    vector = [0.0] * dimensions
    for token in lexical_token_spans(text.casefold()):
        digest = hashlib.sha256(token.text.encode()).digest()
        index = int.from_bytes(digest[:8], "big") % dimensions
        vector[index] += 1
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Compare already normalized experiment vectors."""
    if len(left) != len(right):
        raise ValueError("embedding dimensions must match")
    return sum(a * b for a, b in zip(left, right, strict=True))
