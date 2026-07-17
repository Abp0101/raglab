"""Add keyset pagination indexes.

Revision ID: 20260717_0004
Revises: 20260717_0003
Create Date: 2026-07-17
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20260717_0004"
down_revision: str | None = "20260717_0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Index each public keyset ordering path."""
    op.create_index(
        "ix_collections_created_id",
        "collections",
        ["created_at", "id"],
    )
    op.create_index(
        "ix_documents_collection_uploaded_id",
        "documents",
        ["collection_id", "uploaded_at", "id"],
    )
    op.create_index(
        "ix_ingestion_jobs_collection_created_id",
        "ingestion_jobs",
        ["collection_id", "created_at", "id"],
    )


def downgrade() -> None:
    """Remove public keyset ordering indexes."""
    op.drop_index(
        "ix_ingestion_jobs_collection_created_id",
        table_name="ingestion_jobs",
    )
    op.drop_index(
        "ix_documents_collection_uploaded_id",
        table_name="documents",
    )
    op.drop_index("ix_collections_created_id", table_name="collections")
