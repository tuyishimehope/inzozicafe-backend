"""v3_add_document_content_hash

Revision ID: e4b1c2d3f5a6
Revises: d2ce7b9c9035
Create Date: 2026-11-08 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "e4b1c2d3f5a6"
down_revision: Union[str, Sequence[str], None] = "d2ce7b9c9035"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Existing rows get "" so the column can be NOT NULL. "" never equals a real SHA-256,
    # so the next index run re-indexes those files once and fills in their real hash.
    op.add_column(
        "document",
        sa.Column("content_hash", sa.String(length=64), nullable=False, server_default=""),
    )
    op.alter_column("document", "content_hash", server_default=None)


def downgrade() -> None:
    op.drop_column("document", "content_hash")
