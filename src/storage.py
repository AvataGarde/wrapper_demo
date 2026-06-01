from __future__ import annotations
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.db import get_conn, init_db, is_database_configured
from src.schema import StandardResponse

OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"
DB_ENABLED_ENV = "FALSE"
DB_SCHEMA_INITIALIZED = False


STRUCTURED_ANSWER_FIELDS = (
    "answer",
    "source_selection_justification",
    "location",
    "copyright_subject_matter",
    "social_media_use",
    "fair_dealing",
    "licensing",
)


RUN_FIELD_ORDER = (
    "query_id",
    "provider",
    "model",
    "attempt_no",
    "timestamp",
    "latency_ms",
    "prompt_tokens",
    "completion_tokens",
    "total_cost",
    "error",
)


QUERY_FIELD_ORDER = ("id", "topic", "framing", "text")


SOURCE_FIELD_ORDER = (
    "rank",
    "url",
    "resolved_url",
    "title",
    "snippet",
    "published_date",
)


DEFAULT_SAVE_MODE = "file"  # set to "both" to re-enable DB saving


def _import_pandas():
    import pandas as pd

    return pd


def _normalize_timestamp(timestamp: str) -> str:
    if not timestamp:
        return datetime.now(timezone.utc).isoformat()
    if timestamp.endswith("Z"):
        return timestamp[:-1] + "+00:00"
    return timestamp


def _split_reasoning_steps(reasoning_steps: str | None) -> list[dict[str, Any]]:
    if not reasoning_steps:
        return []
    parts = [part.strip() for part in re.split(r"\n\s*\n", reasoning_steps) if part.strip()]
    if len(parts) <= 1:
        parts = [line.strip() for line in reasoning_steps.splitlines() if line.strip()]
    return [{"step_no": idx, "content": part} for idx, part in enumerate(parts, start=1)]


def build_db_records(response: StandardResponse) -> dict[str, Any]:
    query = {
        "id": response.prompt_id,
        "topic": response.prompt_id.rsplit("-", 1)[0],
        "framing": response.framing or "unknown",
        "text": response.prompt_text,
    }
    run = {
        "query_id": response.prompt_id,
        "provider": response.provider,
        "model": response.model,
        "attempt_no": response.attempt_no,
        "timestamp": _normalize_timestamp(response.timestamp),
        "latency_ms": response.latency_ms,
        "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
        "completion_tokens": response.usage.completion_tokens if response.usage else None,
        "total_cost": response.usage.total_cost if response.usage else None,
        "error": response.error,
    }
    answer = {field: getattr(response, field) for field in STRUCTURED_ANSWER_FIELDS}

    def _sr_to_dict(sr, idx: int) -> dict:
        return {
            "rank": idx,
            "url": sr.url,
            "resolved_url": sr.resolved_url,
            "title": sr.title,
            "snippet": sr.snippet,
            "published_date": sr.published_date,
        }

    searched_sources = [_sr_to_dict(sr, idx) for idx, sr in enumerate(response.searched_sources, start=1)]
    cited_sources = [_sr_to_dict(sr, idx) for idx, sr in enumerate(response.cited_sources, start=1)]
    return {
        "query": query,
        "run": run,
        "answer": answer,
        "reasoning_steps": _split_reasoning_steps(response.reasoning_steps),
        "searched_sources": searched_sources,
        "cited_sources": cited_sources,
        "raw_response": response.raw,
    }


def _ensure_db_schema(conn) -> None:
    global DB_SCHEMA_INITIALIZED
    if not DB_SCHEMA_INITIALIZED:
        init_db(conn=conn)
        DB_SCHEMA_INITIALIZED = True


def save_to_db(response: StandardResponse, conn=None) -> int:
    records = build_db_records(response)
    owns_conn = conn is None
    connection = conn or get_conn()
    try:
        _ensure_db_schema(connection)
        with connection.cursor() as cur:
            query = records["query"]
            cur.execute(
                """
                INSERT INTO queries (id, topic, framing, text)
                VALUES (%(id)s, %(topic)s, %(framing)s, %(text)s)
                ON CONFLICT (id) DO UPDATE
                SET topic = EXCLUDED.topic,
                    framing = EXCLUDED.framing,
                    text = EXCLUDED.text
                """,
                query,
            )

            run_payload = records["run"]
            cur.execute(
                """
                INSERT INTO runs (
                    query_id, provider, model, attempt_no, timestamp,
                    latency_ms, prompt_tokens, completion_tokens, total_cost, error
                )
                VALUES (
                    %(query_id)s, %(provider)s, %(model)s, %(attempt_no)s, %(timestamp)s,
                    %(latency_ms)s, %(prompt_tokens)s, %(completion_tokens)s, %(total_cost)s, %(error)s
                )
                ON CONFLICT (query_id, provider, attempt_no) DO UPDATE
                SET model = EXCLUDED.model,
                    timestamp = EXCLUDED.timestamp,
                    latency_ms = EXCLUDED.latency_ms,
                    prompt_tokens = EXCLUDED.prompt_tokens,
                    completion_tokens = EXCLUDED.completion_tokens,
                    total_cost = EXCLUDED.total_cost,
                    error = EXCLUDED.error
                RETURNING id
                """,
                run_payload,
            )
            run_id = cur.fetchone()[0]

            cur.execute(
                """
                INSERT INTO raw_responses (run_id, payload)
                VALUES (%s, %s::jsonb)
                ON CONFLICT (run_id) DO UPDATE
                SET payload = EXCLUDED.payload
                """,
                (run_id, json.dumps(records["raw_response"] or {})),
            )

            answer_payload = dict(records["answer"])
            answer_payload["run_id"] = run_id
            cur.execute(
                """
                INSERT INTO answers (
                    run_id, answer, source_selection_justification, location,
                    copyright_subject_matter, social_media_use, fair_dealing, licensing
                )
                VALUES (
                    %(run_id)s, %(answer)s, %(source_selection_justification)s, %(location)s,
                    %(copyright_subject_matter)s, %(social_media_use)s, %(fair_dealing)s, %(licensing)s
                )
                ON CONFLICT (run_id) DO UPDATE
                SET answer = EXCLUDED.answer,
                    source_selection_justification = EXCLUDED.source_selection_justification,
                    location = EXCLUDED.location,
                    copyright_subject_matter = EXCLUDED.copyright_subject_matter,
                    social_media_use = EXCLUDED.social_media_use,
                    fair_dealing = EXCLUDED.fair_dealing,
                    licensing = EXCLUDED.licensing
                """,
                answer_payload,
            )

            cur.execute("DELETE FROM reasoning_steps WHERE run_id = %s", (run_id,))
            for step in records["reasoning_steps"]:
                cur.execute(
                    "INSERT INTO reasoning_steps (run_id, step_no, content) VALUES (%s, %s, %s)",
                    (run_id, step["step_no"], step["content"]),
                )

            cur.execute("DELETE FROM searched_sources WHERE run_id = %s", (run_id,))
            for src in records["searched_sources"]:
                cur.execute(
                    """
                    INSERT INTO searched_sources (run_id, url, resolved_url, title, snippet, published_date, rank)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (run_id, src["url"], src["resolved_url"], src["title"],
                     src["snippet"], src["published_date"], src["rank"]),
                )

            cur.execute("DELETE FROM cited_sources WHERE run_id = %s", (run_id,))
            for src in records["cited_sources"]:
                cur.execute(
                    """
                    INSERT INTO cited_sources (run_id, url, resolved_url, title, snippet, published_date, rank)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (run_id, src["url"], src["resolved_url"], src["title"],
                     src["snippet"], src["published_date"], src["rank"]),
                )
        connection.commit()
        return run_id
    except Exception:
        connection.rollback()
        raise
    finally:
        if owns_conn:
            connection.close()


def _db_mode_enabled() -> bool:
    import os

    mode = os.environ.get(DB_ENABLED_ENV, DEFAULT_SAVE_MODE).strip().lower()
    if mode in {"0", "false", "off", "file", "files"}:
        return False
    return is_database_configured()


def save_with_optional_db(response: StandardResponse) -> tuple[str, int | None]:
    path = save(response)
    run_id = save_to_db(response) if _db_mode_enabled() else None
    return path, run_id


def _date_dir(response: StandardResponse) -> Path:
    # Parse date from timestamp or use today
    try:
        dt = datetime.fromisoformat(response.timestamp)
        date_str = dt.strftime("%Y-%m-%d")
    except Exception:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return OUTPUTS_DIR / date_str


def save(response: StandardResponse) -> str:
    date_dir = _date_dir(response)
    date_dir.mkdir(parents=True, exist_ok=True)

    filename = (
        f"{response.prompt_id}__{response.provider}__attempt{response.attempt_no}.json"
    )
    path = date_dir / filename
    path.write_text(response.model_dump_json(indent=2), encoding="utf-8")
    return str(path)


def load(path: str) -> StandardResponse:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return StandardResponse.model_validate(data)


def list_runs(
    date: str | None = None,
    prompt_id: str | None = None,
    provider: str | None = None,
) -> list[str]:
    """Return sorted list of JSON file paths matching the given filters."""
    if date:
        search_dirs = [OUTPUTS_DIR / date]
    else:
        search_dirs = sorted(OUTPUTS_DIR.iterdir()) if OUTPUTS_DIR.exists() else []

    results: list[str] = []
    for d in search_dirs:
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.json")):
            name = f.stem  # e.g. iran-uk-war-neutral__perplexity__attempt1
            if prompt_id and not name.startswith(f"{prompt_id}__"):
                continue
            if provider and f"__{provider}__" not in name:
                continue
            results.append(str(f))
    return results


def _format_search_results(response: StandardResponse) -> str:
    lines = []
    for sr in response.searched_sources:
        display_url = sr.resolved_url or sr.url
        parts = [display_url]
        if sr.title:
            parts.append(sr.title)
        if sr.snippet:
            parts.append(sr.snippet[:120])
        lines.append(" | ".join(parts))
    return "\n".join(lines)


def _to_row(r: StandardResponse) -> dict:
    return {
        "timestamp":                    r.timestamp,
        "provider":                     r.provider,
        "model":                        r.model,
        "prompt_id":                    r.prompt_id,
        "framing":                      r.framing,
        "attempt_no":                   r.attempt_no,
        "prompt_text":                  r.prompt_text,
        "answer":                       r.answer,
        "source_selection_justification": r.source_selection_justification,
        "location":                     r.location,
        "copyright_subject_matter":     r.copyright_subject_matter,
        "social_media_use":             r.social_media_use,
        "fair_dealing":                 r.fair_dealing,
        "licensing":                    r.licensing,
        "search_results":               _format_search_results(r),
        "n_sources":                    len(r.searched_sources),
        "reasoning_steps":              r.reasoning_steps,
        "usage_prompt_tokens":          r.usage.prompt_tokens if r.usage else None,
        "usage_completion_tokens":      r.usage.completion_tokens if r.usage else None,
        "usage_total_tokens":           r.usage.total_tokens if r.usage else None,
        "usage_total_cost":             r.usage.total_cost if r.usage else None,
        "latency_ms":                   r.latency_ms,
        "error":                        r.error,
    }


def export_excel(
    paths: list[str] | None = None,
    date: str | None = None,
    provider: str | None = None,
    out_path: str | None = None,
    topic: str | None = None,
    providers: list[str] | None = None,
) -> str:
    """Export matching runs to an Excel file (one row per run, no raw field).

    Args:
        paths:     explicit list of JSON file paths to include (overrides other filters)
        date:      filter by date string e.g. "2026-05-16"
        provider:  filter by provider name (for list_runs filtering)
        out_path:  output .xlsx path; auto-generated if not given
        topic:     topic name to include in the auto-generated filename
        providers: list of provider names to include in the auto-generated filename
    """
    if paths is None:
        paths = list_runs(date=date, provider=provider)
    if not paths:
        raise ValueError("No runs found matching the given filters.")

    rows = [_to_row(load(p)) for p in paths]
    pd = _import_pandas()
    df = pd.DataFrame(rows)

    if out_path is None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        parts = ["audit_export"]
        if topic:
            parts.append(topic)
        if providers:
            parts.append("+".join(sorted(providers)))
        parts.append(today)
        out_path = str(OUTPUTS_DIR / f"{'_'.join(parts)}.xlsx")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Audit")

        # Auto-size columns and wrap long text cells
        ws = writer.sheets["Audit"]
        wrap = __import__("openpyxl").styles.Alignment(wrap_text=True, vertical="top")
        for col in ws.columns:
            max_len = 0
            col_letter = col[0].column_letter
            for cell in col:
                cell.alignment = wrap
                if cell.value:
                    max_len = max(max_len, min(len(str(cell.value).split("\n")[0]), 80))
            ws.column_dimensions[col_letter].width = max(12, max_len + 2)

        # Freeze header row
        ws.freeze_panes = "A2"

    return out_path


def next_attempt_no(prompt_id: str, provider: str) -> int:
    existing = list_runs(prompt_id=prompt_id, provider=provider)
    if not existing:
        return 1
    # Extract attempt numbers from filenames
    nums = []
    for path in existing:
        m = re.search(r"attempt(\d+)\.json$", path)
        if m:
            nums.append(int(m.group(1)))
    return max(nums) + 1 if nums else 1
