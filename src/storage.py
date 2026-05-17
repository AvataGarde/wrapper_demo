from __future__ import annotations
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from src.schema import StandardResponse

OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"


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
    for sr in response.search_results:
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
        "n_sources":                    len(r.search_results),
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


def save_to_db(response: StandardResponse) -> int | None:
    """Persist a StandardResponse to PostgreSQL. Returns the run.id or None on failure."""
    try:
        from src.db import get_conn  # lazy import so file-only usage still works
    except ImportError:
        return None

    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                # ── queries (upsert) ──────────────────────────────────────────
                cur.execute(
                    """
                    INSERT INTO queries (id, topic, framing, text)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (id) DO NOTHING
                    """,
                    (
                        response.prompt_id,
                        _topic_from_id(response.prompt_id),
                        response.framing or "",
                        response.prompt_text,
                    ),
                )

                # ── runs ──────────────────────────────────────────────────────
                ts = _parse_ts(response.timestamp)
                cur.execute(
                    """
                    INSERT INTO runs
                        (query_id, provider, model, attempt_no, timestamp,
                         latency_ms, prompt_tokens, completion_tokens,
                         total_tokens, total_cost, error)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (query_id, provider, attempt_no) DO UPDATE
                        SET timestamp=EXCLUDED.timestamp,
                            latency_ms=EXCLUDED.latency_ms,
                            total_cost=EXCLUDED.total_cost,
                            error=EXCLUDED.error
                    RETURNING id
                    """,
                    (
                        response.prompt_id,
                        response.provider,
                        response.model,
                        response.attempt_no,
                        ts,
                        response.latency_ms,
                        response.usage.prompt_tokens if response.usage else None,
                        response.usage.completion_tokens if response.usage else None,
                        response.usage.total_tokens if response.usage else None,
                        response.usage.total_cost if response.usage else None,
                        response.error,
                    ),
                )
                run_id: int = cur.fetchone()[0]

                # ── raw_responses ─────────────────────────────────────────────
                cur.execute(
                    """
                    INSERT INTO raw_responses (run_id, payload)
                    VALUES (%s, %s)
                    ON CONFLICT (run_id) DO UPDATE SET payload=EXCLUDED.payload
                    """,
                    (run_id, json.dumps(response.raw)),
                )

                # ── answers ───────────────────────────────────────────────────
                cur.execute(
                    """
                    INSERT INTO answers
                        (run_id, answer, source_selection_justification,
                         location, copyright_subject_matter, social_media_use,
                         fair_dealing, licensing)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (run_id) DO UPDATE
                        SET answer=EXCLUDED.answer,
                            source_selection_justification=EXCLUDED.source_selection_justification,
                            location=EXCLUDED.location,
                            copyright_subject_matter=EXCLUDED.copyright_subject_matter,
                            social_media_use=EXCLUDED.social_media_use,
                            fair_dealing=EXCLUDED.fair_dealing,
                            licensing=EXCLUDED.licensing
                    """,
                    (
                        run_id,
                        response.answer,
                        response.source_selection_justification,
                        response.location,
                        response.copyright_subject_matter,
                        response.social_media_use,
                        response.fair_dealing,
                        response.licensing,
                    ),
                )

                # ── reasoning_steps ───────────────────────────────────────────
                if response.reasoning_steps:
                    cur.execute(
                        "DELETE FROM reasoning_steps WHERE run_id=%s", (run_id,)
                    )
                    for i, line in enumerate(response.reasoning_steps.splitlines()):
                        if line.strip():
                            cur.execute(
                                "INSERT INTO reasoning_steps (run_id, step_no, content) VALUES (%s,%s,%s)",
                                (run_id, i, line.strip()),
                            )

                # ── citations ─────────────────────────────────────────────────
                cur.execute("DELETE FROM citations WHERE run_id=%s", (run_id,))
                for rank, sr in enumerate(response.search_results):
                    domain = _extract_domain(sr.resolved_url or sr.url)
                    cur.execute(
                        """
                        INSERT INTO citations
                            (run_id, rank, url, resolved_url, title,
                             snippet, published_date, domain)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                        """,
                        (
                            run_id,
                            rank,
                            sr.url,
                            sr.resolved_url,
                            sr.title,
                            sr.snippet,
                            sr.published_date,
                            domain,
                        ),
                    )

        return run_id
    except Exception as exc:
        import traceback
        print(f"[db] save_to_db failed: {exc}\n{traceback.format_exc()}")
        return None


# ── helpers ───────────────────────────────────────────────────────────────────

def _topic_from_id(prompt_id: str) -> str:
    """Best-effort: strip the trailing framing suffix to get the topic."""
    for suffix in ("-neutral", "-left", "-right", "-outlets-national", "-outlets-local"):
        if prompt_id.endswith(suffix):
            return prompt_id[: -len(suffix)]
    return prompt_id


def _extract_domain(url: str | None) -> str | None:
    if not url:
        return None
    try:
        return urlparse(url).netloc.removeprefix("www.")
    except Exception:
        return None


def _parse_ts(ts: str):
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        return datetime.now(timezone.utc)


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
