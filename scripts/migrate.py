#!/usr/bin/env python3
"""Migrate existing output JSON files into PostgreSQL."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db import get_conn, init_db
from src.storage import list_runs, load, save_to_db


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest saved JSON runs into PostgreSQL")
    parser.add_argument(
        "paths",
        nargs="*",
        help="Optional explicit JSON paths to ingest. Defaults to all outputs/**/*.json",
    )
    parser.add_argument("--date", help="Only ingest files under outputs/<date>")
    parser.add_argument("--provider", help="Only ingest files for one provider")
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on the first ingestion error instead of continuing",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> list[str]:
    if args.paths:
        return [str(Path(path).resolve()) for path in args.paths]
    return list_runs(date=args.date, provider=args.provider)


def main() -> int:
    args = parse_args()
    paths = resolve_paths(args)
    if not paths:
        print("No JSON files found to ingest.")
        return 1

    ok = 0
    failed = 0
    conn = get_conn()
    try:
        init_db(conn=conn)
        for path in paths:
            try:
                response = load(path)
                run_id = save_to_db(response, conn=conn)
                ok += 1
                print(f"OK   {path} -> run_id={run_id}")
            except Exception as exc:
                failed += 1
                print(f"FAIL {path}: {exc}")
                if args.fail_fast:
                    raise
    finally:
        conn.close()

    print(f"Done. ingested={ok} failed={failed} total={len(paths)}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
