from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from urllib.parse import quote

from dotenv import load_dotenv

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schema.sql"
load_dotenv(SCHEMA_PATH.parent / ".env")


def build_database_url(env: dict[str, str] | None = None) -> str | None:
    """Build a PostgreSQL connection URL from DATABASE_URL or PG* variables."""
    env = env or os.environ

    database_url = env.get("DATABASE_URL")
    if database_url:
        return database_url

    host = env.get("PGHOST")
    if not host:
        return None

    port = env.get("PGPORT", "5432")
    user = env.get("PGUSER", "postgres")
    password = env.get("PGPASSWORD", "")
    database = env.get("PGDATABASE", "postgres")

    auth = quote(user, safe="")
    if password:
        auth = f"{auth}:{quote(password, safe='')}"

    return f"postgresql://{auth}@{host}:{port}/{quote(database, safe='')}"


def is_database_configured(env: dict[str, str] | None = None) -> bool:
    return build_database_url(env=env) is not None


def _import_psycopg():
    try:
        import psycopg  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only when dependency missing
        raise RuntimeError(
            "psycopg is required for PostgreSQL support. Install requirements.txt first."
        ) from exc
    return psycopg


def get_conn(database_url: str | None = None, **connect_kwargs: Any):
    psycopg = _import_psycopg()
    dsn = database_url or build_database_url()
    if not dsn:
        raise RuntimeError(
            "PostgreSQL is not configured. Set DATABASE_URL or PGHOST/PGPORT/PGUSER/PGPASSWORD/PGDATABASE."
        )
    return psycopg.connect(dsn, **connect_kwargs)


def init_db(conn=None, schema_path: str | Path | None = None) -> None:
    """Initialize the database schema from schema.sql."""
    owns_conn = conn is None
    connection = conn or get_conn()
    schema_file = Path(schema_path) if schema_path else SCHEMA_PATH
    schema_sql = schema_file.read_text(encoding="utf-8")

    try:
        with connection.cursor() as cur:
            cur.execute(schema_sql)
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        if owns_conn:
            connection.close()
