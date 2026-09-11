"""Alembic migration environment (Production Phase P6/P8).

Runs raw-SQL migrations against the pgvector backend. The DSN is read from the
environment (`AEGISMEM_PG_DSN` / `DATABASE_URL`) — never committed. `PgVectorMemoryStore`
creates the schema on first connect for dev; these migrations are the versioned,
reviewable path for production changes and rollbacks.
"""

from __future__ import annotations

import os

from alembic import context
from sqlalchemy import engine_from_config, pool

config = context.config

_dsn = os.environ.get("AEGISMEM_PG_DSN") or os.environ.get("DATABASE_URL")
if _dsn:
    config.set_main_option("sqlalchemy.url", _dsn)


def run_migrations_offline() -> None:
    context.configure(url=config.get_main_option("sqlalchemy.url"), literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
