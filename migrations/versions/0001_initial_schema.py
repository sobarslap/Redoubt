"""initial AegisMem pgvector schema

Revision ID: 0001_initial
Revises:
Create Date: production P6/P8

Applies the same schema `PgVectorMemoryStore` creates on first connect, reusing
its DDL as the single source of truth so the migration can never drift from the
store. Downgrade drops the tables (and the vector extension is left in place).
"""

from __future__ import annotations

from alembic import op

from aegismem.memory.pgstore import _DDL

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None

_DIM = 384  # bge-small-en-v1.5, matching PgVectorMemoryStore default


def upgrade() -> None:
    for statement in _DDL.format(dim=_DIM).split(";"):
        stmt = statement.strip()
        if stmt:
            op.execute(stmt)


def downgrade() -> None:
    for table in ("provenance_edges", "transitions", "review_queue", "memories"):
        op.execute(f"DROP TABLE IF EXISTS {table}")
