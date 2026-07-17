"""Base behavior shared by external and internal boundary models."""

from pydantic import BaseModel, ConfigDict


class RAGLabModel(BaseModel):
    """Reject unknown fields and validate mutations at all typed boundaries."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )


class CursorPage[PageItemT](RAGLabModel):
    """Stable keyset page returned by growing collection endpoints."""

    items: tuple[PageItemT, ...] = ()
    next_cursor: str | None = None
