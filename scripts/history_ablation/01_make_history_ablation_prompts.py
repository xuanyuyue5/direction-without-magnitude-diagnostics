#!/usr/bin/env python3
"""
Make_history_ablation_prompts.py  (local, no API, no cost)

Builds a 2 x 5 history-sensitivity experiment from Prompts_series.jsonl.
No train.csv is required: the original history, baseline, category, synthetic
series-level hint, and post-hoc reference lift are already stored in that file.

Factor 1: numerical magnitude
  - named_entity : category is shown, but NO numerical magnitude is shown
  - series_hint  : category and the SAME series-level magnitude hint are shown

Factor 2: history supplied to the model
  - original_history
  - shuffled_history
  - other_series_history
  - mean_only_history
  - no_history

The primary question is whether adding a numerical hint reduces sensitivity to
changes in the supplied sales history. Both magnitude arms retain the category
name, so the only difference between them is the numerical hint.

The other-series history is baseline-normalised before being inserted:
    transformed_t = donor_t / donor_baseline * target_baseline
This changes temporal shape while avoiding an artificial scale mismatch.
Donors are chosen deterministically. Prompt history length is kept unchanged.

Input:
  Prompts_series.jsonl

Output:
  history_ablation_prompts.jsonl  (100 tasks x 2 x 5 = 1000 prompts)

Run:
  python Make_history_ablation_prompts.py
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics as st
from pathlib import Path
from typing import Any

DEFAULT_INPUT = "Prompts_series.jsonl"
DEFAULT_OUTPUT = "history_ablation_prompts.jsonl"
DEFAULT_SEED = 20260701

HINT_MODES = ("named_entity", "series_hint")
HISTORY_CONDITIONS = (
    "original_history",
    "shuffled_history",
    "other_series_history",
    "mean_only_history",
    "no_history",
)

HISTORY_RE = re.compile(
    r"Daily unit sales for the\s+(?P<n>\d+)\s+days before a promotion starts:\s*\n"
    r"\[(?P<values>.*?)\]\s*\n"
    r"baseline \(typical non-promo daily sales\):\s*(?P<baseline>[-+]?\d*\.?\d+)",
    flags=re.DOTALL,
)

OUTPUT_INSTRUCTION = (
    "Predict the promotional lift, defined as "
    "(promo_day_sales - baseline) / baseline.\n"
    "Output only a single decimal number (e.g. 0.5 means +50%). "
    "No explanation, no words, just the number."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default=DEFAULT_INPUT, help="Input series-hint JSONL")
    p.add_argument("--output", default=DEFAULT_OUTPUT, help="Output prompt JSONL")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
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


def extract_history(prompt: str) -> tuple[list[float], float]:
    m = HISTORY_RE.search(prompt)
    if not m:
        raise ValueError("Could not locate the history and baseline in a prompt.")
    values_text = m.group("values").strip()
    values = [] if not values_text else [float(x.strip()) for x in values_text.split(",")]
    baseline = float(m.group("baseline"))
    declared_n = int(m.group("n"))
    if len(values) != declared_n:
        raise ValueError(
            f"Prompt declares {declared_n} history values but contains {len(values)}."
        )
    return values, baseline


def fmt_number(value: float) -> str:
    """Compact two-decimal formatting for histories and coarse hints."""
    if not math.isfinite(value):
        raise ValueError(f"Non-finite numeric value: {value}")
    rounded = round(float(value), 2)
    if rounded == 0:
        rounded = 0.0
    if rounded.is_integer():
        return str(int(rounded))
    return f"{rounded:.2f}".rstrip("0").rstrip(".")


def fmt_metadata_number(value: float) -> str:
    """Preserve baseline precision from the task metadata (typically 3 decimals)."""
    if not math.isfinite(value):
        raise ValueError(f"Non-finite numeric value: {value}")
    rounded = round(float(value), 8)
    if rounded == 0:
        rounded = 0.0
    if rounded.is_integer():
        return str(int(rounded))
    return f"{rounded:.8f}".rstrip("0").rstrip(".")


def build_prompt(
    *,
    history: list[float] | None,
    baseline: float,
    family: str,
    hint_mode: str,
    source_hint: float,
) -> str:
    lines = ["You are forecasting retail sales for a single product at a single store."]
    if history is None:
        lines.append("Historical daily unit sales before the promotion are not provided.")
    else:
        hist_str = ", ".join(fmt_number(v) for v in history)
        lines.extend(
            [
                f"Daily unit sales for the {len(history)} days before a promotion starts:",
                f"[{hist_str}]",
            ]
        )
    lines.append(f"baseline (typical non-promo daily sales): {fmt_metadata_number(baseline)}")
    lines.append(f"The product belongs to the category: {family}.")
    if hint_mode == "series_hint":
        lines.append(
            "An estimate for THIS specific product's promotional lift is "
            f"approximately {fmt_number(source_hint)}."
        )
    elif hint_mode != "named_entity":
        raise ValueError(f"Unknown hint_mode: {hint_mode}")
    lines.append("On the next day, this product goes on promotion.")
    lines.append(OUTPUT_INSTRUCTION)
    return "\n".join(lines)


def choose_donor(
    target_index: int,
    tasks: list[dict[str, Any]],
    histories: list[list[float]],
    rng: random.Random,
) -> int:
    """Prefer a different task with the same length; otherwise use >= target length."""
    target_len = len(histories[target_index])
    exact = [
        j for j, h in enumerate(histories)
        if j != target_index and len(h) == target_len
    ]
    if exact:
        return rng.choice(exact)

    long_enough = [
        j for j, h in enumerate(histories)
        if j != target_index and len(h) >= target_len
    ]
    if long_enough:
        # Prefer the smallest amount of truncation, then choose deterministically.
        min_len = min(len(histories[j]) for j in long_enough)
        candidates = [j for j in long_enough if len(histories[j]) == min_len]
        return rng.choice(candidates)

    # Defensive fallback; not expected here because the maximum length has peers.
    candidates = [j for j in range(len(tasks)) if j != target_index]
    if not candidates:
        raise ValueError("At least two tasks are needed for other_series_history.")
    return rng.choice(candidates)


def transform_donor_history(
    donor_history: list[float],
    donor_baseline: float,
    target_baseline: float,
    target_len: int,
) -> list[float]:
    if donor_baseline <= 0:
        raise ValueError(f"Donor baseline must be positive, got {donor_baseline}")

    # Keep the most recent target_len observations. If an unexpected shorter donor
    # appears, repeat cyclically so the target prompt length remains unchanged.
    if len(donor_history) >= target_len:
        seq = donor_history[-target_len:]
    else:
        repeats = (target_len + len(donor_history) - 1) // len(donor_history)
        seq = (donor_history * repeats)[-target_len:]

    return [max(0.0, (v / donor_baseline) * target_baseline) for v in seq]


def main() -> None:
    args = parse_args()
    raw = read_jsonl(args.input)
    if not raw:
        raise SystemExit(f"No records found in {args.input}")

    # Deduplicate defensively by task. Prompts_series.jsonl should contain one row/task.
    by_task: dict[tuple[int, int], dict[str, Any]] = {}
    for r in raw:
        key = (int(r["store_nbr"]), int(r["item_nbr"]))
        if key in by_task:
            raise ValueError(f"Duplicate task in input: {key}")
        by_task[key] = r
    tasks = list(by_task.values())

    histories: list[list[float]] = []
    baselines: list[float] = []
    for r in tasks:
        hist, parsed_baseline = extract_history(r["prompt"])
        metadata_baseline = float(r["baseline_median"])
        if abs(parsed_baseline - metadata_baseline) > 1e-6:
            raise ValueError(
                f"Baseline mismatch for {(r['store_nbr'], r['item_nbr'])}: "
                f"prompt={parsed_baseline}, metadata={metadata_baseline}"
            )
        histories.append(hist)
        baselines.append(metadata_baseline)

    rng = random.Random(args.seed)
    donor_index: dict[int, int] = {
        i: choose_donor(i, tasks, histories, rng) for i in range(len(tasks))
    }

    n_written = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for i, r in enumerate(tasks):
            s = int(r["store_nbr"])
            item = int(r["item_nbr"])
            task_id = f"{s}_{item}"
            original = histories[i]
            baseline = baselines[i]
            source_hint = float(r["family_hint"])

            # Deterministic per-task shuffle, independent of record ordering.
            shuffled = list(original)
            task_rng = random.Random(f"{args.seed}:{s}:{item}")
            task_rng.shuffle(shuffled)

            mean_value = st.fmean(original)
            mean_only = [mean_value] * len(original)

            d_idx = donor_index[i]
            donor = tasks[d_idx]
            other_series = transform_donor_history(
                donor_history=histories[d_idx],
                donor_baseline=baselines[d_idx],
                target_baseline=baseline,
                target_len=len(original),
            )

            variants: dict[str, list[float] | None] = {
                "original_history": original,
                "shuffled_history": shuffled,
                "other_series_history": other_series,
                "mean_only_history": mean_only,
                "no_history": None,
            }

            for hint_mode in HINT_MODES:
                for history_condition in HISTORY_CONDITIONS:
                    history = variants[history_condition]
                    condition = f"{hint_mode}__{history_condition}"
                    rec = {
                        "task_id": task_id,
                        "store_nbr": s,
                        "item_nbr": item,
                        "condition": condition,
                        "hint_mode": hint_mode,
                        "history_condition": history_condition,
                        "family": r["family"],
                        "baseline_median": baseline,
                        "mu_i_true": float(r["mu_i_true"]),
                        "source_hint": source_hint,
                        "injected_hint": source_hint if hint_mode == "series_hint" else None,
                        "left_censored": r.get("left_censored"),
                        "ci_excludes_zero": r.get("ci_excludes_zero"),
                        "original_n_history_days": len(original),
                        "n_history_days": 0 if history is None else len(history),
                        "history_values": history,
                        "history_mean": None if history is None else st.fmean(history),
                        "donor_task_id": (
                            f"{int(donor['store_nbr'])}_{int(donor['item_nbr'])}"
                            if history_condition == "other_series_history" else None
                        ),
                        "donor_family": (
                            donor["family"] if history_condition == "other_series_history" else None
                        ),
                        "donor_original_n_history_days": (
                            len(histories[d_idx])
                            if history_condition == "other_series_history" else None
                        ),
                        "donor_baseline": (
                            baselines[d_idx]
                            if history_condition == "other_series_history" else None
                        ),
                        "other_series_baseline_normalised": (
                            True if history_condition == "other_series_history" else None
                        ),
                        "prompt": build_prompt(
                            history=history,
                            baseline=baseline,
                            family=r["family"],
                            hint_mode=hint_mode,
                            source_hint=source_hint,
                        ),
                    }
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_written += 1

    expected = len(tasks) * len(HINT_MODES) * len(HISTORY_CONDITIONS)
    if n_written != expected:
        raise AssertionError(f"Expected {expected} prompts, wrote {n_written}")

    print(f"Loaded {len(tasks)} unique tasks from {args.input}")
    print(
        f"Wrote {n_written} prompts to {args.output} "
        f"({len(tasks)} tasks x {len(HINT_MODES)} hint modes x "
        f"{len(HISTORY_CONDITIONS)} history conditions)"
    )
    print("No API calls were made.")
    print("Inspect the output JSONL before running the LLM stage.")


if __name__ == "__main__":
    main()
