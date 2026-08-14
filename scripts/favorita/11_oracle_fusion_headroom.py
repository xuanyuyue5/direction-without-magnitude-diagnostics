#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
11_oracle_fusion_headroom.py -- diagnostic fusion-headroom analysis.

This script asks whether the autonomous category-label estimate and the injected
series-level hint contain complementary ranking information. It reports:

  * Spearman correlations for autonomous estimate, hint, and pipe output.
  * Rank-OLS oracle fusion, both in-sample and leave-one-out.
  * Parameter-free rank-average fusion.
  * Bootstrap confidence interval for fusion headroom over the hint alone.

The analysis is diagnostic only: it uses post-hoc series-level hints to quantify
how much information the transmission-pipe condition discards.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

N_BOOT = 10_000
PRED_CANDIDATES = [
    "lift_pred", "prediction", "pred", "parsed", "parsed_lift", "lift",
    "predicted_lift", "value", "answer", "output", "response", "content",
    "text", "raw",
]
HINT_CANDIDATES = [
    "family_hint", "series_hint", "hint", "stated_hint", "hint_value",
    "magnitude_hint", "h_i", "series_magnitude",
]
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
HINT_TEXT_RE = re.compile(r"approximately\s+(-?\d+(?:\.\d+)?)", re.IGNORECASE)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    root = repo_root()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompts", default=root / "artifacts" / "prompts" / "prompts.jsonl")
    p.add_argument("--responses", default=root / "artifacts" / "responses" / "responses.jsonl")
    p.add_argument("--series-prompts", default=root / "artifacts" / "prompts" / "prompts_series.jsonl")
    p.add_argument("--series-responses", default=root / "artifacts" / "responses" / "responses_series.jsonl")
    p.add_argument("--n-boot", type=int, default=N_BOOT)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pred-field", default=None)
    p.add_argument("--hint-field", default=None)
    return p.parse_args()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        sys.exit(f"[error] file not found: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        sys.exit(f"[error] empty file: {path}")
    return rows


def task_key(rec: dict[str, Any]) -> tuple[int, int]:
    try:
        return int(rec["store_nbr"]), int(rec["item_nbr"])
    except KeyError:
        sys.exit(f"[error] record lacks store_nbr/item_nbr. keys={list(rec)}")


def parse_number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        match = NUM_RE.search(value.replace(",", ""))
        if match:
            return float(match.group())
    return None


def detect_field(rows: list[dict[str, Any]], candidates: list[str], override: str | None, what: str, fname: str) -> str:
    if override:
        return override
    sample = rows[: min(50, len(rows))]
    for candidate in candidates:
        hits = sum(1 for row in sample if candidate in row and parse_number(row[candidate]) is not None)
        if hits >= 0.8 * len(sample):
            print(f"[info] {fname}: using field '{candidate}' as {what}")
            return candidate
    print(f"[error] cannot auto-detect {what} in {fname}")
    print(f"        available keys in first record: {list(rows[0])}")
    sys.exit(1)


def extract_hint_from_prompt_text(rows: list[dict[str, Any]]) -> dict[tuple[int, int], float]:
    hints: dict[tuple[int, int], float] = {}
    multi_match = 0
    for row in rows:
        matches = HINT_TEXT_RE.findall(row.get("prompt", ""))
        if not matches:
            continue
        if len(matches) > 1:
            multi_match += 1
        hints[task_key(row)] = float(matches[-1])
    if multi_match:
        print(f"[warn] {multi_match} prompts had multiple hint matches; used the last match")
    return hints


def median_predictions(rows: list[dict[str, Any]], pred_field: str, condition: str | None = None) -> dict[tuple[int, int], float]:
    bucket: dict[tuple[int, int], list[float]] = {}
    for row in rows:
        if condition is not None and row.get("condition") != condition:
            continue
        if row.get("parse_ok") is False:
            continue
        value = parse_number(row.get(pred_field))
        if value is None:
            continue
        bucket.setdefault(task_key(row), []).append(value)
    return {key: float(np.median(values)) for key, values in bucket.items() if values}


def to_ranks(values: np.ndarray) -> np.ndarray:
    return stats.rankdata(values, method="average")


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return float(stats.spearmanr(x, y).statistic)


def rank_ols_fit(features: np.ndarray, target: np.ndarray) -> np.ndarray:
    x = np.column_stack([np.ones(len(target)), features])
    beta, *_ = np.linalg.lstsq(x, target, rcond=None)
    return beta


def rank_ols_predict(features: np.ndarray, beta: np.ndarray) -> np.ndarray:
    x = np.column_stack([np.ones(features.shape[0]), features])
    return x @ beta


def oracle_fusion_insample(auto: np.ndarray, hint: np.ndarray, truth: np.ndarray) -> float:
    ar, hr, tr = to_ranks(auto), to_ranks(hint), to_ranks(truth)
    features = np.column_stack([ar, hr])
    fitted = rank_ols_predict(features, rank_ols_fit(features, tr))
    return float(np.corrcoef(fitted, tr)[0, 1])


def oracle_fusion_loo(auto: np.ndarray, hint: np.ndarray, truth: np.ndarray) -> float:
    ar, hr, tr = to_ranks(auto), to_ranks(hint), to_ranks(truth)
    features = np.column_stack([ar, hr])
    preds = np.empty(len(truth))
    for i in range(len(truth)):
        keep = np.arange(len(truth)) != i
        beta = rank_ols_fit(features[keep], tr[keep])
        preds[i] = rank_ols_predict(features[i:i + 1], beta)[0]
    return float(np.corrcoef(preds, tr)[0, 1])


def rank_average_fusion(auto: np.ndarray, hint: np.ndarray, truth: np.ndarray) -> float:
    fused = (to_ranks(auto) + to_ranks(hint)) / 2.0
    return spearman(fused, truth)


def bootstrap_headroom(auto: np.ndarray, hint: np.ndarray, truth: np.ndarray, n_boot: int, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot)
    n = len(truth)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ab, hb, tb = auto[idx], hint[idx], truth[idx]
        if np.ptp(ab) == 0 or np.ptp(hb) == 0 or np.ptp(tb) == 0:
            deltas[b] = np.nan
            continue
        deltas[b] = oracle_fusion_insample(ab, hb, tb) - spearman(hb, tb)
    deltas = deltas[~np.isnan(deltas)]
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return float(np.mean(deltas)), float(lo), float(hi)


def main() -> None:
    args = parse_args()
    prompts = read_jsonl(args.prompts)
    responses = read_jsonl(args.responses)
    series_prompts = read_jsonl(args.series_prompts)
    series_responses = read_jsonl(args.series_responses)

    truth = {task_key(row): float(row["mu_i_true"]) for row in prompts if "mu_i_true" in row}
    if not truth:
        sys.exit("[error] no mu_i_true values found")

    if args.hint_field:
        hint_field = args.hint_field
        hints = {
            task_key(row): parse_number(row.get(hint_field))
            for row in series_prompts
            if parse_number(row.get(hint_field)) is not None
        }
    else:
        sample = series_prompts[: min(50, len(series_prompts))]
        detected = None
        for candidate in HINT_CANDIDATES:
            hits = sum(1 for row in sample if candidate in row and parse_number(row[candidate]) is not None)
            if hits >= 0.8 * len(sample):
                detected = candidate
                break
        if detected:
            print(f"[info] {Path(args.series_prompts).name}: using field '{detected}' as series hint")
            hints = {
                task_key(row): parse_number(row.get(detected))
                for row in series_prompts
                if parse_number(row.get(detected)) is not None
            }
        else:
            hints = extract_hint_from_prompt_text(series_prompts)

    pred_field = detect_field(responses, PRED_CANDIDATES, args.pred_field, "prediction", Path(args.responses).name)
    auto = median_predictions(responses, pred_field, condition="named_entity")
    series_pred_field = detect_field(
        series_responses, PRED_CANDIDATES, args.pred_field, "prediction", Path(args.series_responses).name
    )
    pipe = median_predictions(series_responses, series_pred_field, condition="stated_magnitude_series")
    if not pipe:
        pipe = median_predictions(series_responses, series_pred_field, condition=None)

    keys = sorted(set(truth) & set(hints) & set(auto))
    if len(keys) < 30:
        sys.exit(f"[error] only {len(keys)} tasks align across truth, hint, and autonomous predictions")

    auto_arr = np.array([auto[key] for key in keys], dtype=float)
    hint_arr = np.array([hints[key] for key in keys], dtype=float)
    truth_arr = np.array([truth[key] for key in keys], dtype=float)

    print(f"[info] aligned N={len(keys)} tasks")
    print("\n=== base correlations (Spearman) ===")
    print(f"  r(autonomous, truth) = {spearman(auto_arr, truth_arr):6.3f}")
    print(f"  r(hint, truth)       = {spearman(hint_arr, truth_arr):6.3f}")
    print(f"  r(autonomous, hint)  = {spearman(auto_arr, hint_arr):6.3f}")

    pipe_keys = [key for key in keys if key in pipe]
    if pipe_keys:
        pipe_arr = np.array([pipe[key] for key in pipe_keys], dtype=float)
        truth_pipe = np.array([truth[key] for key in pipe_keys], dtype=float)
        hint_pipe = np.array([hints[key] for key in pipe_keys], dtype=float)
        print(f"  r(pipe, truth)       = {spearman(pipe_arr, truth_pipe):6.3f}")
        print(f"  r(pipe, hint)        = {spearman(pipe_arr, hint_pipe):6.3f}")

    r_insample = oracle_fusion_insample(auto_arr, hint_arr, truth_arr)
    r_loo = oracle_fusion_loo(auto_arr, hint_arr, truth_arr)
    r_avg = rank_average_fusion(auto_arr, hint_arr, truth_arr)
    print("\n=== diagnostic fusion headroom ===")
    print(f"  rank-OLS in-sample = {r_insample:6.3f}")
    print(f"  rank-OLS LOO       = {r_loo:6.3f}")
    print(f"  rank average       = {r_avg:6.3f}")

    point = r_insample - spearman(hint_arr, truth_arr)
    mean_delta, lo, hi = bootstrap_headroom(auto_arr, hint_arr, truth_arr, args.n_boot, args.seed)
    print("\n=== headroom over hint alone ===")
    print(f"  point estimate = {point:6.3f}")
    print(f"  bootstrap mean = {mean_delta:6.3f}")
    print(f"  95% CI         = [{lo:.3f}, {hi:.3f}]  ({args.n_boot} resamples)")

    print("\n=== interpretation ===")
    if lo > 0 and point >= 0.05:
        print("  Headroom is positive: the autonomous estimate and hint contain complementary ranking information.")
    elif lo > 0:
        print("  Headroom is positive but small: complementarity exists, but the expected gain is limited.")
    else:
        print("  The interval includes zero: this run does not establish reliable headroom.")


if __name__ == "__main__":
    main()
