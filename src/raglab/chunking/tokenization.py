"""Deterministic token spans for framework-independent fixed chunking."""

import re
from dataclasses import dataclass

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@dataclass(frozen=True, slots=True)
class TokenSpan:
    """One lexical token and its half-open character offsets."""

    text: str
    start: int
    end: int


def lexical_token_spans(text: str) -> tuple[TokenSpan, ...]:
    """Tokenize words and punctuation while retaining exact source offsets."""
    return tuple(
        TokenSpan(match.group(), match.start(), match.end())
        for match in TOKEN_PATTERN.finditer(text)
    )


def count_lexical_tokens(text: str) -> int:
    """Count tokens using the fixed chunker's documented tokenizer."""
    return sum(1 for _ in TOKEN_PATTERN.finditer(text))
