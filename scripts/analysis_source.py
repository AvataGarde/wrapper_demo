#!/usr/bin/env python3
"""Source-category analysis across one or more output dates.
Usage:
  python scripts/analysis_source.py
  python scripts/analysis_source.py --date 2026-05-27
  python scripts/analysis_source.py --date 2026-05-27 --out outputs/analysis_2026-05-27
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.domains import categorize, extract_domain, get_metadata, list_unknown_domains
from src.storage import list_runs, load
from src.wrapper import PROVIDERS

PROVIDER_ORDER = list(PROVIDERS.keys())


# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze source categories across providers")
    p.add_argument("--date", help="Only analyze outputs/<date>/*.json (default: all dates)")
    p.add_argument("--out", help="Output directory (default: outputs/analysis_<date>)")
    p.add_argument(
        "--source-pool",
        choices=["searched", "cited", "both"],
        default="both",
        help="Which source pool to analyze (default: both — produces both views)",
    )
    return p.parse_args()


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _source_url(sr) -> str:
    """Prefer resolved_url (Gemini) over url (redirect)."""
    return sr.resolved_url or sr.url or ""


def _iter_responses(date: str | None = None):
    for path in list_runs(date=date):
        try:
            yield load(path)
        except Exception as exc:
            print(f"WARN: failed to load {path}: {exc}", file=sys.stderr)


def _write_csv(path: Path, header: list[str], rows: Iterable[Iterable]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _ordered_providers(seen: set[str]) -> list[str]:
    """Canonical ordering: known first, unknowns last."""
    return [p for p in PROVIDER_ORDER if p in seen] + sorted(seen - set(PROVIDER_ORDER))


# -----------------------------------------------------------------------------
# Analyses
# -----------------------------------------------------------------------------
def analyze_categories(responses: list, source_pool: str) -> dict[str, Counter]:
    """provider → Counter[category]."""
    cat_counts: dict[str, Counter] = defaultdict(Counter)
    for r in responses:
        if r.error:
            continue
        sources = r.searched_sources if source_pool == "searched" else r.cited_sources
        for sr in sources:
            cat = categorize(_source_url(sr))
            cat_counts[r.provider][cat] += 1
    return cat_counts


def analyze_domain_frequency(
    responses: list, source_pool: str
) -> tuple[dict[tuple[str, str], int], dict[str, dict]]:
    """(provider, domain) → count, plus domain → metadata dict."""
    counts: dict[tuple[str, str], int] = defaultdict(int)
    metadata: dict[str, dict] = {}
    for r in responses:
        if r.error:
            continue
        sources = r.searched_sources if source_pool == "searched" else r.cited_sources
        for sr in sources:
            url = _source_url(sr)
            domain = extract_domain(url)
            if not domain:
                continue
            counts[(r.provider, domain)] += 1
            if domain not in metadata:
                metadata[domain] = get_metadata(url)
    return counts, metadata


def analyze_leaning_by_framing(responses: list) -> dict[tuple[str, str], int]:
    """(leaning, framing) → count."""
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for r in responses:
        if r.error or not r.framing:
            continue
        for sr in r.searched_sources:
            meta = get_metadata(_source_url(sr))
            leaning = meta.get("leaning") or "n/a"
            counts[(leaning, r.framing)] += 1
    return counts


def analyze_citation_selectivity(responses: list) -> list[dict]:
    """Per-run cited/searched ratio."""
    rows: list[dict] = []
    for r in responses:
        if r.error:
            continue
        ns = len(r.searched_sources)
        nc = len(r.cited_sources)
        rows.append({
            "provider": r.provider,
            "prompt_id": r.prompt_id,
            "framing": r.framing or "",
            "attempt_no": r.attempt_no,
            "searched": ns,
            "cited": nc,
            "ratio": round(nc / ns, 4) if ns else 0.0,
        })
    return rows


# -----------------------------------------------------------------------------
# CSV writers
# -----------------------------------------------------------------------------
def write_category_csv(out: Path, cat_counts: dict[str, Counter], pool: str) -> None:
    providers = _ordered_providers(set(cat_counts.keys()))
    cats = sorted(
        {c for counter in cat_counts.values() for c in counter},
        key=lambda c: -sum(cat_counts[p].get(c, 0) for p in providers),
    )
    rows = [
        [c] + [cat_counts[p].get(c, 0) for p in providers] +
        [sum(cat_counts[p].get(c, 0) for p in providers)]
        for c in cats
    ]
    _write_csv(out / f"category_by_provider__{pool}.csv",
               ["category"] + providers + ["total"], rows)


def write_domain_csv(
    out: Path,
    counts: dict[tuple[str, str], int],
    metadata: dict[str, dict],
    pool: str,
) -> None:
    providers = _ordered_providers({p for (p, _) in counts.keys()})
    domains = sorted(
        {d for (_, d) in counts.keys()},
        key=lambda d: -sum(counts.get((p, d), 0) for p in providers),
    )
    header = ["domain", "category", "country", "leaning", "outlet_type"] + providers + ["total"]
    rows = []
    for d in domains:
        meta = metadata.get(d, {})
        per_provider = [counts.get((p, d), 0) for p in providers]
        rows.append([
            d,
            meta.get("category", "other"),
            meta.get("country", ""),
            meta.get("leaning", ""),
            meta.get("outlet_type", ""),
        ] + per_provider + [sum(per_provider)])
    _write_csv(out / f"domain_by_provider__{pool}.csv", header, rows)


def write_leaning_csv(out: Path, leaning_counts: dict) -> None:
    if not leaning_counts:
        return
    framings = sorted({f for (_, f) in leaning_counts})
    leaning_order = ["left", "centre-left", "centre", "centre-right", "right", "n/a"]
    leanings = [l for l in leaning_order if any((l, f) in leaning_counts for f in framings)]
    # Catch any leanings not in the canonical order
    leanings += sorted(
        {l for (l, _) in leaning_counts if l not in leanings}
    )
    rows = []
    for l in leanings:
        per_fr = [leaning_counts.get((l, f), 0) for f in framings]
        rows.append([l] + per_fr + [sum(per_fr)])
    _write_csv(out / "leaning_by_framing.csv",
               ["leaning"] + framings + ["total"], rows)


def write_selectivity_csv(out: Path, rows: list[dict]) -> None:
    if not rows:
        return
    header = list(rows[0].keys())
    _write_csv(out / "citation_selectivity.csv",
               header, [[r[k] for k in header] for r in rows])


def write_unknown_csv(out: Path, responses: list) -> None:
    urls = []
    for r in responses:
        if r.error:
            continue
        for sr in r.searched_sources + r.cited_sources:
            urls.append(_source_url(sr))
    unknown = list_unknown_domains(urls)
    _write_csv(out / "unknown_domains.csv", ["domain", "count"], unknown)


# -----------------------------------------------------------------------------
# Markdown report
# -----------------------------------------------------------------------------
def write_markdown_report(
    out: Path,
    responses: list,
    cat_searched: dict,
    cat_cited: dict,
    leaning_counts: dict,
    selectivity_rows: list,
    date_label: str,
) -> None:
    lines = [f"# Source-Category Analysis — {date_label}\n"]
    lines.append(f"Generated from `{len(responses)}` runs.\n")
    
    # § Coverage
    ok = Counter(r.provider for r in responses if not r.error)
    fail = Counter(r.provider for r in responses if r.error)
    lines.append("## 1. Coverage\n")
    lines.append("| Provider | OK | Failed |")
    lines.append("|---|---:|---:|")
    for p in _ordered_providers(set(ok) | set(fail)):
        lines.append(f"| {p} | {ok.get(p, 0)} | {fail.get(p, 0)} |")
    lines.append("")
    
    # § Category × provider
    for pool, counts in [("searched", cat_searched), ("cited", cat_cited)]:
        if not counts:
            continue
        providers = _ordered_providers(set(counts.keys()))
        cats = sorted(
            {c for counter in counts.values() for c in counter},
            key=lambda c: -sum(counts[p].get(c, 0) for p in providers),
        )
        lines.append(f"\n## 2. Category distribution — {pool} sources\n")
        lines.append("| Category | " + " | ".join(providers) + " | Total |")
        lines.append("|---" + "|---:" * (len(providers) + 1) + "|")
        for c in cats:
            vals = [counts[p].get(c, 0) for p in providers]
            cell = lambda v: str(v) if v else "·"
            lines.append(
                f"| `{c}` | " + " | ".join(cell(v) for v in vals)
                + f" | **{sum(vals)}** |"
            )
        totals = [sum(counts[p].get(c, 0) for c in cats) for p in providers]
        lines.append(
            "| **TOTAL** | " + " | ".join(f"**{t}**" for t in totals)
            + f" | **{sum(totals)}** |"
        )
        lines.append("")
    
    # § Citation selectivity
    if selectivity_rows:
        lines.append("\n## 3. Citation selectivity (cited / searched)\n")
        by_provider = defaultdict(list)
        for r in selectivity_rows:
            by_provider[r["provider"]].append(r["ratio"])
        lines.append("| Provider | N runs | Mean | Min | Max |")
        lines.append("|---|---:|---:|---:|---:|")
        for p in _ordered_providers(set(by_provider)):
            ratios = by_provider[p]
            lines.append(
                f"| {p} | {len(ratios)} | "
                f"{sum(ratios)/len(ratios):.1%} | "
                f"{min(ratios):.1%} | {max(ratios):.1%} |"
            )
        lines.append("")
    
    # § Leaning × framing
    if leaning_counts and any(l != "n/a" for (l, _) in leaning_counts):
        framings = sorted({f for (_, f) in leaning_counts})
        leaning_order = ["left", "centre-left", "centre", "centre-right", "right", "n/a"]
        leanings = [l for l in leaning_order if any((l, f) in leaning_counts for f in framings)]
        lines.append("\n## 4. Political leaning × prompt framing\n")
        lines.append("| Leaning | " + " | ".join(framings) + " |")
        lines.append("|---" + "|---:" * len(framings) + "|")
        for l in leanings:
            row = [str(leaning_counts.get((l, f), 0)) for f in framings]
            lines.append(f"| {l} | " + " | ".join(row) + " |")
        lines.append("")
    
    # § Unknown domains
    urls = []
    for r in responses:
        if r.error:
            continue
        for sr in r.searched_sources + r.cited_sources:
            urls.append(_source_url(sr))
    unknown = list_unknown_domains(urls)
    if unknown:
        lines.append("\n## 5. Unknown domains (top 20)\n")
        lines.append("These fell through to `other`. Consider adding to `domains.yaml`.\n")
        lines.append("| Domain | Count |")
        lines.append("|---|---:|")
        for d, n in unknown[:20]:
            lines.append(f"| `{d}` | {n} |")
        if len(unknown) > 20:
            lines.append(f"\n*({len(unknown) - 20} more — see `unknown_domains.csv`.)*")
        lines.append("")
    
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    args = parse_args()
    
    responses = list(_iter_responses(date=args.date))
    if not responses:
        msg = f"No runs found{' for date ' + args.date if args.date else ''}."
        print(msg, file=sys.stderr)
        sys.exit(1)
    
    date_label = args.date or "all dates"
    out = Path(args.out) if args.out else (
        Path(__file__).parent.parent / "outputs" / f"analysis_{args.date or 'all'}"
    )
    out.mkdir(parents=True, exist_ok=True)
    
    print(f"Analyzing {len(responses)} runs → {out}")
    
    cat_searched = analyze_categories(responses, "searched")
    cat_cited = analyze_categories(responses, "cited")
    
    if args.source_pool in ("searched", "both"):
        write_category_csv(out, cat_searched, "searched")
        dc_s, meta_s = analyze_domain_frequency(responses, "searched")
        write_domain_csv(out, dc_s, meta_s, "searched")
    
    if args.source_pool in ("cited", "both"):
        write_category_csv(out, cat_cited, "cited")
        dc_c, meta_c = analyze_domain_frequency(responses, "cited")
        write_domain_csv(out, dc_c, meta_c, "cited")
    
    leaning_counts = analyze_leaning_by_framing(responses)
    write_leaning_csv(out, leaning_counts)
    
    sel_rows = analyze_citation_selectivity(responses)
    write_selectivity_csv(out, sel_rows)
    
    write_unknown_csv(out, responses)
    
    write_markdown_report(
        out, responses,
        cat_searched, cat_cited,
        leaning_counts, sel_rows, date_label,
    )
    
    print(f"Done. See {out}/report.md")


if __name__ == "__main__":
    main()