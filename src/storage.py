from __future__ import annotations
import json
import re
from datetime import datetime, timezone
from pathlib import Path

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
