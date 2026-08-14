#!/usr/bin/env python3
"""
Run_history_ablation.py -- model-call stage for the 2 x 5 history ablation.

Dry-run mode is the default. Use `--live` for real provider calls. The run is
resumable: completed (task, condition, seed) records are flushed to disk and
skipped on later runs.

Examples:
  python 02_run_history_ablation.py
  python 02_run_history_ablation.py --live --task-limit 5
  python 02_run_history_ablation.py --live --model gpt-4o --n-repeat 3
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

from llm_providers import call_model, make_client, default_model, out_path_for

DEFAULT_PROMPTS = "history_ablation_prompts.jsonl"
DEFAULT_OUT_LIVE = "history_ablation_responses.jsonl"
DEFAULT_OUT_DRYRUN = "history_ablation_responses_DRYRUN.jsonl"

NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--live", action="store_true", help="Make real API calls")
    p.add_argument("--provider", choices=["openai", "deepseek", "anthropic"],
                   default="openai", help="which model provider to call")
    p.add_argument("--prompts", default=DEFAULT_PROMPTS)
    p.add_argument("--output", default=None, help="Override output path")
    p.add_argument("--model", default=None,
                   help="model string; default is the provider's default model")
    p.add_argument("--n-repeat", type=int, default=3)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=16)
    p.add_argument("--max-retries", type=int, default=5)
    p.add_argument("--backoff-base", type=float, default=2.0)
    p.add_argument(
        "--task-limit",
        type=int,
        default=None,
        help="Run the first N unique tasks, preserving all 10 conditions per task",
    )
    return p.parse_args()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {lineno} of {path}: {exc}") from exc
    return records


def parse_lift(text: str | None) -> float | None:
    if text is None:
        return None
    m = NUM_RE.search(text.replace(",", ""))
    if not m:
        return None
    try:
        return float(m.group())
    except ValueError:
        return None


def record_key(r: dict[str, Any]) -> tuple[int, int, str, int]:
    return (
        int(r["store_nbr"]),
        int(r["item_nbr"]),
        str(r["condition"]),
        int(r["seed"]),
    )


def load_done(path: str | Path) -> set[tuple[int, int, str, int]]:
    done: set[tuple[int, int, str, int]] = set()
    if not os.path.exists(path):
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                done.add(record_key(json.loads(line)))
            except Exception:
                continue
    return done


def mock_response(prompt: str, seed: int) -> str:
    rng = random.Random(hash((prompt, seed)) & 0xFFFFFFFF)
    value = round(rng.uniform(-0.3, 1.5), 2)
    u = rng.random()
    if u < 0.03:
        return "no numeric estimate"
    if u < 0.13:
        return f"approximately {value}"
    return str(value)


def filter_task_limit(
    prompts: list[dict[str, Any]], task_limit: int | None
) -> list[dict[str, Any]]:
    if task_limit is None:
        return prompts
    if task_limit <= 0:
        raise ValueError("--task-limit must be positive")
    chosen: list[str] = []
    chosen_set: set[str] = set()
    for p in prompts:
        task_id = str(p["task_id"])
        if task_id not in chosen_set and len(chosen) < task_limit:
            chosen.append(task_id)
            chosen_set.add(task_id)
    return [p for p in prompts if str(p["task_id"]) in chosen_set]


def main() -> None:
    args = parse_args()
    if args.n_repeat <= 0:
        raise SystemExit("--n-repeat must be positive")
    if not os.path.exists(args.prompts):
        raise SystemExit(
            f"Missing {args.prompts}. Run Make_history_ablation_prompts.py first."
        )

    prompts = filter_task_limit(read_jsonl(args.prompts), args.task_limit)
    if not prompts:
        raise SystemExit("No prompts selected.")

    provider = args.provider
    model = args.model or default_model(provider)
    # output: openai keeps the bare names; deepseek/anthropic get a suffix so
    # the GPT-4o history_ablation_responses.jsonl is never touched.
    base_live = out_path_for(DEFAULT_OUT_LIVE, provider)
    base_dry = out_path_for(DEFAULT_OUT_DRYRUN, provider)
    out_path = args.output or (base_live if args.live else base_dry)
    unique_tasks = len({p["task_id"] for p in prompts})
    total_possible = len(prompts) * args.n_repeat

    print(f"Loaded {len(prompts)} prompts from {unique_tasks} tasks")
    print(f"Mode: {'LIVE API calls' if args.live else 'DRY RUN'}")
    print(
        f"Provider={provider} Model={model} temperature={args.temperature} "
        f"N_REPEAT={args.n_repeat} -> up to {total_possible} calls"
    )
    print(f"Output: {out_path}")
    if provider == "anthropic":
        print("Note: Anthropic has no seed param; repeats vary via temperature only.")

    client = None
    if args.live:
        client = make_client(provider)

    done = load_done(out_path)
    if done:
        print(f"Resuming: {len(done)} existing task-condition-seed records will be skipped.")

    n_new = 0
    n_parse_fail = 0
    started = time.time()
    with open(out_path, "a", encoding="utf-8") as out:
        for p in prompts:
            for seed in range(args.n_repeat):
                key = (
                    int(p["store_nbr"]),
                    int(p["item_nbr"]),
                    str(p["condition"]),
                    seed,
                )
                if key in done:
                    continue

                if args.live:
                    raw, error = call_model(
                        client, provider,
                        prompt=p["prompt"],
                        seed=seed,
                        model=model,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        max_retries=args.max_retries,
                        backoff_base=args.backoff_base,
                    )
                    if error and error.startswith("AUTH ERROR"):
                        raise SystemExit(
                            f"\n{error}\nFix the key and re-run; completed calls are saved."
                        )
                else:
                    raw, error = mock_response(p["prompt"], seed), None

                lift = parse_lift(raw)
                if lift is None:
                    n_parse_fail += 1

                rec = {
                    "task_id": p["task_id"],
                    "store_nbr": p["store_nbr"],
                    "item_nbr": p["item_nbr"],
                    "condition": p["condition"],
                    "hint_mode": p["hint_mode"],
                    "history_condition": p["history_condition"],
                    "seed": seed,
                    "provider": provider,
                    "model": model,
                    "temperature": args.temperature,
                    "family": p["family"],
                    "baseline_median": p["baseline_median"],
                    "mu_i_true": p["mu_i_true"],
                    "source_hint": p["source_hint"],
                    "injected_hint": p["injected_hint"],
                    "left_censored": p.get("left_censored"),
                    "ci_excludes_zero": p.get("ci_excludes_zero"),
                    "original_n_history_days": p["original_n_history_days"],
                    "n_history_days": p["n_history_days"],
                    "donor_task_id": p.get("donor_task_id"),
                    "lift_pred": lift,
                    "raw": raw,
                    "parse_ok": lift is not None,
                    "error": error,
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                done.add(key)
                n_new += 1

    elapsed = time.time() - started
    print(f"\nDone. Wrote {n_new} new records in {elapsed:.1f}s.")
    print(
        f"Parse failures among new records: {n_parse_fail}/"
        f"{max(n_new, 1)} ({100 * n_parse_fail / max(n_new, 1):.1f}%)"
    )
    if not args.live:
        print(
            "This was a dry run. Inspect the output JSONL, then use --live for provider calls."
        )


if __name__ == "__main__":
    main()
