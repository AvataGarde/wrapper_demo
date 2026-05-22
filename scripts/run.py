#!/usr/bin/env python3
"""Main run script for LLM Audit Demo.

Usage examples:
  python scripts/run.py --prompt iran-uk-war-neutral
  python scripts/run.py --prompt uk-elections-2026-neutral --provider perplexity
  python scripts/run.py --topic uk-elections-2026 --provider gemini --repeat 3 --export
  python scripts/run.py --all --provider perplexity --repeat 3 --export
"""
import argparse
import sys
import time
from pathlib import Path
from collections import defaultdict

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.prompts import load_prompts, filter_prompts, get_prompt
from src.wrapper import PROVIDERS, run_one
from src.storage import export_excel, next_attempt_no, save_with_optional_db


def main():
    parser = argparse.ArgumentParser(description="Run LLM audit queries")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prompt", metavar="PROMPT_ID", help="Run a single prompt ID")
    group.add_argument("--topic", metavar="TOPIC", help="Run all prompts for a topic")
    group.add_argument("--all", action="store_true", help="Run all prompts × all providers")

    parser.add_argument("--provider", nargs="+", metavar="PROVIDER", choices=list(PROVIDERS.keys()), help="Which LLM providers to use (default: all)")
    parser.add_argument("--repeat", type=int, default=1, metavar="N", help="Number of times to sample each prompt (default: 1)")
    parser.add_argument("--export", action="store_true", help="Export results to Excel after running")
    args = parser.parse_args()

    if args.all:
        prompts = load_prompts()
    elif args.topic:
        prompts = filter_prompts(topic=args.topic)
        if not prompts:
            print(f"No prompts found for topic '{args.topic}'")
            sys.exit(1)
    else:
        prompts = [get_prompt(args.prompt)]

    # Determine providers to run
    providers = args.provider or list(PROVIDERS.keys()) # if not specified, use all providers

    total = len(prompts) * len(providers) * args.repeat
    idx = 0
    stats: dict[str, dict] = defaultdict(lambda: {"ok": 0, "fail": 0, "tokens": 0, "cost": 0.0, "has_cost": False}) # record ok/fail counts for each provider
    wall_start = time.perf_counter()
    saved_paths: list[str] = []

    for _ in range(args.repeat):
        for prompt in prompts:
            for provider in providers:
                idx += 1 # total number of runs
                attempt_no = next_attempt_no(prompt["id"], provider) # next attempt number for this prompt and provider
                print(f"[{idx}/{total}] {prompt['id']} × {provider} ... ", end="", flush=True) # print progress

                result = run_one(prompt, provider, attempt_no=attempt_no) # run the prompt with the provider
                path, run_id = save_with_optional_db(result) # save the result (and DB if configured)
                saved_paths.append(path)

                db_suffix = f" [db run_id={run_id}]" if run_id is not None else ""
                if result.error:
                    stats[provider]["fail"] += 1
                    print(f"FAIL ({result.error[:80]}) -> {path}{db_suffix}")
                else:
                    stats[provider]["ok"] += 1 # increment ok count for this provider
                    latency = f"{result.latency_ms / 1000:.1f}s" if result.latency_ms else "?s"
                    ncit = len(result.search_results)
                    print(f"OK ({latency}, {ncit} sources) -> {path}{db_suffix}")
                if result.usage:
                    stats[provider]["tokens"] += result.usage.total_tokens or 0
                    if result.usage.total_cost is not None and result.usage.total_cost > 0:
                        stats[provider]["cost"] += result.usage.total_cost
                        stats[provider]["has_cost"] = True

    wall_elapsed = time.perf_counter() - wall_start
    print(f"\n{'='*60}")
    print(f"Done in {wall_elapsed:.1f}s | {total} calls")
    print(f"{'Provider':<15} {'OK':>5} {'FAIL':>5} {'Tokens':>10} {'Cost':>10}")
    print(f"{'-'*45}")
    total_ok = total_fail = total_tokens = 0
    total_cost = 0.0
    any_cost = False
    for p in providers:
        ok = stats[p].get("ok", 0)
        fail = stats[p].get("fail", 0)
        tokens = stats[p].get("tokens", 0)
        cost = stats[p].get("cost", 0.0)
        has_cost = stats[p].get("has_cost", False)
        total_ok += ok
        total_fail += fail
        total_tokens += tokens
        total_cost += cost
        if has_cost:
            any_cost = True
        cost_str = f"${cost:.4f}" if has_cost else "N/A"
        print(f"{p:<15} {ok:>5} {fail:>5} {tokens:>10,} {cost_str:>10}")
    print(f"{'-'*45}")
    total_cost_str = f"${total_cost:.4f}" if any_cost else "N/A"
    print(f"{'TOTAL':<15} {total_ok:>5} {total_fail:>5} {total_tokens:>10,} {total_cost_str:>10}")

    if args.export and saved_paths:
        topic = args.topic or ("all" if args.all else args.prompt)
        excel_path = export_excel(paths=saved_paths, topic=topic, providers=providers)
        print(f"\nExcel exported → {excel_path}")


if __name__ == "__main__":
    main()
