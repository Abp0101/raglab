"""Text normalization and conservative section-heading detection."""

import re
from collections.abc import Iterable

from raglab.core.schemas import SectionHeading

SOFT_HYPHEN = "\N{SOFT HYPHEN}"
LINE_WHITESPACE = re.compile(r"[^\S\n]+")
EXCESS_NEWLINES = re.compile(r"\n{3,}")
HYPHENATED_LINE_BREAK = re.compile(r"(?<=\w)-\s*\n\s*(?=\w)")
NUMBERED_HEADING = re.compile(r"^(?:\d+(?:\.\d+)*[.)]?|[A-Z][.)])\s+\S+")


def normalize_page_text(text: str) -> str:
    """Clean common PDF extraction artifacts while preserving paragraphs."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace(SOFT_HYPHEN, "")
    text = HYPHENATED_LINE_BREAK.sub("", text)
    lines = [LINE_WHITESPACE.sub(" ", line).strip() for line in text.splitlines()]
    return EXCESS_NEWLINES.sub("\n\n", "\n".join(lines)).strip()


def detect_section_headings(
    text: str,
    font_heading_candidates: Iterable[str] = (),
) -> tuple[SectionHeading, ...]:
    """Locate high-confidence font or textual headings in normalized text."""
    candidates = {normalize_page_text(candidate) for candidate in font_heading_candidates}
    for line in text.splitlines():
        stripped = line.strip()
        word_count = len(stripped.split())
        looks_uppercase = stripped.isupper() and any(character.isalpha() for character in stripped)
        if 1 <= word_count <= 14 and (looks_uppercase or NUMBERED_HEADING.match(stripped)):
            candidates.add(stripped)

    headings: list[SectionHeading] = []
    search_from = 0
    for candidate in sorted(candidates, key=lambda item: text.find(item)):
        if not candidate or len(candidate) > 500:
            continue
        start = text.find(candidate, search_from)
        if start < 0:
            start = text.find(candidate)
        if start >= 0 and all(existing.start != start for existing in headings):
            headings.append(SectionHeading(text=candidate, start=start))
            search_from = start + len(candidate)
    return tuple(sorted(headings, key=lambda heading: heading.start))
