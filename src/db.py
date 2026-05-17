"""PostgreSQL connection pool and schema initialisation."""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import psycopg2
from psycopg2 import pool as pg_pool
from psycopg2.extras import RealDictCursor

_pool: pg_pool.ThreadedConnectionPool | None = None

_DSN_DEFAULT = (
    "host=localhost port=5432 dbname=audit_db user=audituser password=auditpass"
)


def _dsn() -> str:
    return os.getenv("DATABASE_URL", _DSN_DEFAULT)


def get_pool() -> pg_pool.ThreadedConnectionPool:
    global _pool
    if _pool is None:
        _pool = pg_pool.ThreadedConnectionPool(minconn=1, maxconn=10, dsn=_dsn())
    return _pool


@contextmanager
def get_conn() -> Generator[psycopg2.extensions.connection, None, None]:
    """Yield a connection from the pool; auto-commit or roll back on error."""
    conn = get_pool().getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        get_pool().putconn(conn)


def init_db(schema_path: str | None = None) -> None:
    """Create all tables if they don't exist yet (idempotent).

    Skips silently if the schema has already been applied by a superuser.
    Tables are created with IF NOT EXISTS so re-running is always safe.
    """
    if schema_path is None:
        schema_path = str(Path(__file__).parent.parent / "schema.sql")
    sql = Path(schema_path).read_text(encoding="utf-8")
    with get_conn() as conn:
        with conn.cursor() as cur:
            try:
                cur.execute(sql)
            except Exception as exc:
                # Tables already exist and user lacks DDL rights — that's fine
                conn.rollback()
                if "already exists" not in str(exc) and "InsufficientPrivilege" not in type(exc).__name__:
                    raise


def close_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.closeall()
        _pool = None
