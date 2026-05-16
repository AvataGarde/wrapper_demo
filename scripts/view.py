#!/usr/bin/env python3
"""Browse a saved run JSON file.

Usage:
  python scripts/view.py outputs/2026-05-15/uk-elections-2026-neutral__perplexity__attempt1.json
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage import load


def truncate(text: str | None, n: int) -> str:
    if not text:
        return "(empty)"
    return text[:n] + ("..." if len(text) > n else "")


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/view.py <path-to-json>")
        sys.exit(1)

    path = sys.argv[1]
    r = load(path)

    print("=" * 70)
    print(f"  PROVIDER  : {r.provider}")
    print(f"  MODEL     : {r.model}")
    print(f"  PROMPT ID : {r.prompt_id}")
    print(f"  FRAMING   : {r.framing}")
    print(f"  TIMESTAMP : {r.timestamp}")
    latency = f"{r.latency_ms} ms" if r.latency_ms is not None else "unknown"
    print(f"  LATENCY   : {latency}")
    if r.usage:
        print(f"  TOKENS    : in={r.usage.prompt_tokens} out={r.usage.completion_tokens} total={r.usage.total_tokens}")
        if r.usage.total_cost:
            print(f"  COST      : ${r.usage.total_cost:.6f}")
    if r.error:
        print(f"  ERROR     : {r.error}")
    print("=" * 70)

    print("\nPROMPT:")
    print(f"  {r.prompt_text}\n")

    print("ANSWER (first 500 chars):")
    print(truncate(r.answer, 500))
    print()

    # Structured source metadata fields
    fields = [
        ("SOURCE JUSTIFICATION", r.source_selection_justification),
        ("LOCATION",             r.location),
        ("COPYRIGHT",            r.copyright_subject_matter),
        ("SOCIAL MEDIA",         r.social_media_use),
        ("FAIR DEALING",         r.fair_dealing),
        ("LICENSING",            r.licensing),
    ]
    for label, value in fields:
        if value:
            print(f"{label}:")
            print(f"  {truncate(value, 200)}")
    print()

    print(f"SEARCH RESULTS ({len(r.search_results)}):")
    for i, sr in enumerate(r.search_results, 1):
        print(f"  [{i}] {sr.url}")
        if sr.title:
            print(f"       title   : {sr.title}")
        if sr.snippet:
            print(f"       snippet : {truncate(sr.snippet, 120)}")
        if sr.published_date:
            print(f"       date    : {sr.published_date}")
    if not r.search_results:
        print("  (none)")
    print()

    if r.reasoning_steps:
        print("REASONING STEPS (first 300 chars):")
        print(truncate(r.reasoning_steps, 300))
        print()

    raw_keys = list(r.raw.keys()) if r.raw else []
    print(f"RAW KEYS: {raw_keys}")
    print("=" * 70)


if __name__ == "__main__":
    main()
