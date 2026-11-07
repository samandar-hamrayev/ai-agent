"""track token usage and cost per run and step

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-07
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TOKEN_COLUMNS = ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens")


def upgrade() -> None:
    with op.batch_alter_table("agent_runs") as batch:
        for name in _TOKEN_COLUMNS:
            batch.add_column(sa.Column(name, sa.Integer(), nullable=False, server_default="0"))
        batch.add_column(sa.Column("cost_usd", sa.Numeric(12, 6), nullable=False, server_default="0"))
    with op.batch_alter_table("agent_steps") as batch:
        batch.add_column(sa.Column("usage", sa.JSON(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("agent_steps") as batch:
        batch.drop_column("usage")
    with op.batch_alter_table("agent_runs") as batch:
        batch.drop_column("cost_usd")
        for name in reversed(_TOKEN_COLUMNS):
            batch.drop_column(name)
