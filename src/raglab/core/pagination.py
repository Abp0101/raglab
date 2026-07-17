"""Opaque, endpoint-scoped keyset cursor encoding and validation."""

import base64
import binascii
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID

from raglab.core.exceptions import InvalidCursorError
from raglab.core.schemas.common import RAGLabModel

MAX_CURSOR_LENGTH = 512


class CursorKind(StrEnum):
    """Resource ordering encoded into a cursor."""

    COLLECTIONS = "collections"
    DOCUMENTS = "documents"
    INGESTION_JOBS = "ingestion_jobs"


class _CursorPayload(RAGLabModel):
    version: int
    kind: CursorKind
    scope: UUID | None
    ordered_at: datetime
    item_id: UUID


@dataclass(frozen=True, slots=True)
class CursorPosition:
    """Validated database keyset position."""

    ordered_at: datetime
    item_id: UUID


def encode_cursor(
    *,
    kind: CursorKind,
    scope: UUID | None,
    ordered_at: datetime,
    item_id: UUID,
) -> str:
    """Encode a canonical versioned cursor without exposing it as an API schema."""
    if ordered_at.tzinfo is None or ordered_at.utcoffset() is None:
        raise ValueError("cursor timestamps must include a timezone")
    payload = _CursorPayload(
        version=1,
        kind=kind,
        scope=scope,
        ordered_at=ordered_at.astimezone(UTC),
        item_id=item_id,
    )
    raw = json.dumps(
        payload.model_dump(mode="json"),
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def decode_cursor(
    cursor: str | None,
    *,
    kind: CursorKind,
    scope: UUID | None,
) -> CursorPosition | None:
    """Validate cursor syntax, version, resource kind, and collection scope."""
    if cursor is None:
        return None
    try:
        if not cursor or len(cursor) > MAX_CURSOR_LENGTH:
            raise ValueError
        encoded = cursor.encode("ascii")
        padding = b"=" * (-len(encoded) % 4)
        raw = base64.b64decode(encoded + padding, altchars=b"-_", validate=True)
        payload = _CursorPayload.model_validate(json.loads(raw.decode("utf-8")))
        if payload.version != 1 or payload.kind is not kind or payload.scope != scope:
            raise ValueError
        if payload.ordered_at.tzinfo is None or payload.ordered_at.utcoffset() is None:
            raise ValueError
    except (
        UnicodeError,
        ValueError,
        binascii.Error,
    ) as error:
        raise InvalidCursorError("cursor is invalid for this resource") from error
    return CursorPosition(
        ordered_at=payload.ordered_at.astimezone(UTC),
        item_id=payload.item_id,
    )
