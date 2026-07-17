"""Add distributed ingestion job leases.

Revision ID: 20260717_0003
Revises: 20260716_0002
Create Date: 2026-07-17
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260717_0003"
down_revision: str | None = "20260716_0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add owner-bound expiring leases and claim-attempt observability."""
    op.add_column(
        "ingestion_jobs",
        sa.Column("attempt_count", sa.Integer(), server_default="0", nullable=False),
    )
    op.add_column(
        "ingestion_jobs",
        sa.Column("lease_owner", sa.Uuid(), nullable=True),
    )
    op.add_column(
        "ingestion_jobs",
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_ingestion_jobs_claimable",
        "ingestion_jobs",
        ["status", "lease_expires_at", "created_at"],
    )


def downgrade() -> None:
    """Remove distributed lease state."""
    op.drop_index("ix_ingestion_jobs_claimable", table_name="ingestion_jobs")
    op.drop_column("ingestion_jobs", "lease_expires_at")
    op.drop_column("ingestion_jobs", "lease_owner")
    op.drop_column("ingestion_jobs", "attempt_count")
