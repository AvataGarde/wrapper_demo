"""One-off migration: load all existing JSON output files into PostgreSQL."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db import init_db
from src.storage import list_runs, load, save_to_db


def main() -> None:
    print("Initialising database schema…")
    init_db()

    paths = list_runs()
    if not paths:
        print("No JSON output files found — nothing to migrate.")
        return

    print(f"Found {len(paths)} run(s) to migrate.\n")

    ok = fail = 0
    for path in paths:
        try:
            response = load(path)
            run_id = save_to_db(response)
            if run_id:
                print(f"  ✓  {Path(path).name}  →  run_id={run_id}")
                ok += 1
            else:
                print(f"  ✗  {Path(path).name}  (save_to_db returned None)")
                fail += 1
        except Exception as exc:
            print(f"  ✗  {Path(path).name}  ({exc})")
            fail += 1

    print(f"\nDone. {ok} migrated, {fail} failed.")


if __name__ == "__main__":
    main()
