from src.db import build_database_url


def test_build_database_url_prefers_database_url(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@localhost:5432/audit")
    monkeypatch.setenv("PGHOST", "ignored-host")

    assert build_database_url() == "postgresql://user:pass@localhost:5432/audit"


def test_build_database_url_from_pg_env(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("PGHOST", "db.example.com")
    monkeypatch.setenv("PGPORT", "5433")
    monkeypatch.setenv("PGUSER", "audit_user")
    monkeypatch.setenv("PGPASSWORD", "secret")
    monkeypatch.setenv("PGDATABASE", "gatekeeper")

    assert build_database_url() == "postgresql://audit_user:secret@db.example.com:5433/gatekeeper"


def test_build_database_url_returns_none_when_not_configured(monkeypatch):
    for key in ["DATABASE_URL", "PGHOST", "PGPORT", "PGUSER", "PGPASSWORD", "PGDATABASE"]:
        monkeypatch.delenv(key, raising=False)

    assert build_database_url() is None
