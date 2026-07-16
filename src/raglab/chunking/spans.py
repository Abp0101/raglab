"""Reusable character-boundary splitting algorithms."""

DEFAULT_SEPARATORS = ("\n\n", "\n", ". ", " ")


def recursive_spans(
    text: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
    offset: int = 0,
) -> list[tuple[int, int]]:
    """Return overlapping spans preferring natural boundaries."""
    spans: list[tuple[int, int]] = []
    start = 0
    while start < len(text):
        maximum_end = min(start + chunk_size, len(text))
        end = maximum_end
        if maximum_end < len(text):
            minimum_boundary = start + max(chunk_size // 2, 1)
            for separator in DEFAULT_SEPARATORS:
                boundary = text.rfind(separator, minimum_boundary, maximum_end)
                if boundary >= minimum_boundary:
                    end = boundary + (1 if separator == ". " else len(separator))
                    break
        spans.append((offset + start, offset + end))
        if end >= len(text):
            break
        start = max(end - chunk_overlap, start + 1)
    return spans
