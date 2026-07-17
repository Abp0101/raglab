from datetime import UTC, datetime
from uuid import uuid4

import pytest

from raglab.core.exceptions import InvalidCursorError
from raglab.core.pagination import CursorKind, decode_cursor, encode_cursor


def test_cursor_round_trip_preserves_keyset_position() -> None:
    scope = uuid4()
    item_id = uuid4()
    ordered_at = datetime(2026, 7, 17, 12, 30, tzinfo=UTC)
    cursor = encode_cursor(
        kind=CursorKind.DOCUMENTS,
        scope=scope,
        ordered_at=ordered_at,
        item_id=item_id,
    )

    position = decode_cursor(cursor, kind=CursorKind.DOCUMENTS, scope=scope)

    assert position is not None
    assert position.ordered_at == ordered_at
    assert position.item_id == item_id


@pytest.mark.parametrize("cursor", ["", "%%%", "bm90LWpzb24", "x" * 513])
def test_malformed_cursor_is_rejected(cursor: str) -> None:
    with pytest.raises(InvalidCursorError, match="invalid for this resource"):
        decode_cursor(cursor, kind=CursorKind.COLLECTIONS, scope=None)


def test_cursor_cannot_cross_resource_or_collection_scope() -> None:
    first_collection = uuid4()
    cursor = encode_cursor(
        kind=CursorKind.INGESTION_JOBS,
        scope=first_collection,
        ordered_at=datetime.now(UTC),
        item_id=uuid4(),
    )

    with pytest.raises(InvalidCursorError):
        decode_cursor(
            cursor,
            kind=CursorKind.INGESTION_JOBS,
            scope=uuid4(),
        )
    with pytest.raises(InvalidCursorError):
        decode_cursor(
            cursor,
            kind=CursorKind.DOCUMENTS,
            scope=first_collection,
        )


def test_cursor_requires_timezone_aware_ordering() -> None:
    with pytest.raises(ValueError, match="timezone"):
        encode_cursor(
            kind=CursorKind.COLLECTIONS,
            scope=None,
            ordered_at=datetime(2026, 7, 17),
            item_id=uuid4(),
        )
