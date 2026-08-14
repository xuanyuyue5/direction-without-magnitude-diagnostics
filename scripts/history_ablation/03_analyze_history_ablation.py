#!/usr/bin/env python3
"""
Analyze_history_ablation.py  (local, no API)

Collapses repeated calls by the median and tests whether numerical magnitude
injection suppresses sensitivity to history perturbations.

Primary per-task score for each hint mode:
    S_i = mean_c |pred(i,c) - pred(i,original)|
where c ranges over shuffled, other-series, mean-only, and no-history.

Primary paired contrast:
    suppression = mean_i [ S_i(named_entity) - S_i(series_hint) ]
Positive values mean the numerical hint makes predictions less history-sensitive.
A task-level paired bootstrap gives a 95% CI. A paired sign-flip randomisation
test gives a two-sided p-value.

Outputs:
  history_ablation_collapsed.csv
  history_ablation_summary.csv
  history_ablation_pairwise.csv
  history_ablation_task_sensitivity.csv
  history_ablation_primary.txt

Run:
  python Analyze_history_ablation.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics as st
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

DEFAULT_INPUT = "history_ablation_responses.jsonl"
DEFAULT_BOOT = 10000
DEFAULT_SEED = 42
EPS = 0.05

HINT_MODES = ("named_entity", "series_hint")
HISTORY_CONDITIONS = (
    "original_history",
    "shuffled_history",
    "other_series_history",
    "mean_only_history",
    "no_history",
)
ALTERED = tuple(c for c in HISTORY_CONDITIONS if c != "original_history")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default=DEFAULT_INPUT)
    p.add_argument("--n-boot", type=int, default=DEFAULT_BOOT)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--collapsed-out", default="history_ablation_collapsed.csv")
    p.add_argument("--summary-out", default="history_ablation_summary.csv")
    p.add_argument("--pairwise-out", default="history_ablation_pairwise.csv")
    p.add_argument("--task-out", default="history_ablation_task_sensitivity.csv")
    p.add_argument("--report-out", default="history_ablation_primary.txt")
    return p.parse_args()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {lineno}: {exc}") from exc
    return rows


def mean(xs: Iterable[float]) -> float:
    values = list(xs)
    return sum(values) / len(values) if values else float("nan")


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return float("nan")
    mx, my = mean(xs), mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(
        sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)
    )
    return num / den if den else float("nan")


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            out[order[k]] = rank
        i = j + 1
    return out


def spearman(xs: list[float], ys: list[float]) -> float:
    return pearson(ranks(xs), ranks(ys)) if len(xs) >= 2 else float("nan")


def ols_slope(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return float("nan")
    mx, my = mean(xs), mean(ys)
    den = sum((x - mx) ** 2 for x in xs)
    return (
        sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
        if den else float("nan")
    )


def quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    pos = (len(sorted_values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    frac = pos - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def ci_mean(values: list[float], n_boot: int, rng: random.Random) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    boots = []
    n = len(values)
    for _ in range(n_boot):
        boots.append(mean(values[rng.randrange(n)] for _ in range(n)))
    boots.sort()
    return quantile(boots, 0.025), quantile(boots, 0.975)


def sign_flip_pvalue(values: list[float], n_perm: int, rng: random.Random) -> float:
    if not values:
        return float("nan")
    observed = abs(mean(values))
    extreme = 0
    for _ in range(n_perm):
        perm_mean = mean(v if rng.random() < 0.5 else -v for v in values)
        if abs(perm_mean) >= observed - 1e-15:
            extreme += 1
    return (extreme + 1) / (n_perm + 1)


def collapse(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if not r.get("parse_ok") or r.get("lift_pred") is None:
            continue
        key = (str(r["task_id"]), str(r["hint_mode"]), str(r["history_condition"]))
        grouped[key].append(r)

    out: list[dict[str, Any]] = []
    for (task_id, hint_mode, hist_cond), group in grouped.items():
        first = group[0]
        preds = [float(r["lift_pred"]) for r in group]
        out.append(
            {
                "task_id": task_id,
                "store_nbr": int(first["store_nbr"]),
                "item_nbr": int(first["item_nbr"]),
                "hint_mode": hint_mode,
                "history_condition": hist_cond,
                "condition": first["condition"],
                "family": first["family"],
                "mu_i_true": float(first["mu_i_true"]),
                "source_hint": float(first["source_hint"]),
                "injected_hint": first.get("injected_hint"),
                "prediction": st.median(preds),
                "n_parsed_repeats": len(preds),
                "prediction_repeat_sd": st.pstdev(preds) if len(preds) > 1 else 0.0,
                "left_censored": first.get("left_censored"),
                "ci_excludes_zero": first.get("ci_excludes_zero"),
            }
        )
    return out


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.n_boot <= 0:
        raise SystemExit("--n-boot must be positive")
    rows = read_jsonl(args.input)
    collapsed = collapse(rows)
    if not collapsed:
        raise SystemExit("No parsed responses found.")

    write_csv(args.collapsed_out, collapsed)

    lookup = {
        (r["task_id"], r["hint_mode"], r["history_condition"]): r
        for r in collapsed
    }

    # Per-cell descriptive summary.
    summary_rows: list[dict[str, Any]] = []
    for hint_mode in HINT_MODES:
        for hist_cond in HISTORY_CONDITIONS:
            cell = [
                r for r in collapsed
                if r["hint_mode"] == hint_mode and r["history_condition"] == hist_cond
            ]
            preds = [r["prediction"] for r in cell]
            truth = [r["mu_i_true"] for r in cell]
            hints = [r["source_hint"] for r in cell]
            abs_to_truth = [abs(p - y) for p, y in zip(preds, truth)]
            abs_to_hint = [abs(p - h) for p, h in zip(preds, hints)]
            glued = sum(d <= EPS for d in abs_to_hint)
            summary_rows.append(
                {
                    "hint_mode": hint_mode,
                    "history_condition": hist_cond,
                    "n": len(cell),
                    "pearson_pred_truth": pearson(preds, truth),
                    "spearman_pred_truth": spearman(preds, truth),
                    "slope_truth_to_pred": ols_slope(truth, preds),
                    "mae_to_truth": mean(abs_to_truth),
                    "pearson_pred_source_hint": pearson(preds, hints),
                    "mean_abs_pred_minus_source_hint": mean(abs_to_hint),
                    "within_0.05_of_source_hint_n": glued,
                    "within_0.05_of_source_hint_fraction": glued / len(cell) if cell else float("nan"),
                    "mean_repeat_sd": mean(r["prediction_repeat_sd"] for r in cell),
                }
            )
    write_csv(args.summary_out, summary_rows)

    # Pair each altered history with original, within each hint arm.
    pairwise_rows: list[dict[str, Any]] = []
    for hint_mode in HINT_MODES:
        task_ids = sorted({r["task_id"] for r in collapsed if r["hint_mode"] == hint_mode})
        for hist_cond in ALTERED:
            deltas: list[float] = []
            signed: list[float] = []
            n_nearly_same = 0
            for task_id in task_ids:
                orig = lookup.get((task_id, hint_mode, "original_history"))
                alt = lookup.get((task_id, hint_mode, hist_cond))
                if orig is None or alt is None:
                    continue
                d = alt["prediction"] - orig["prediction"]
                signed.append(d)
                deltas.append(abs(d))
                n_nearly_same += abs(d) <= EPS
            rng = random.Random(f"{args.seed}:{hint_mode}:{hist_cond}")
            lo, hi = ci_mean(deltas, args.n_boot, rng)
            pairwise_rows.append(
                {
                    "hint_mode": hint_mode,
                    "history_condition": hist_cond,
                    "n_paired_tasks": len(deltas),
                    "mean_abs_change_from_original": mean(deltas),
                    "median_abs_change_from_original": st.median(deltas) if deltas else float("nan"),
                    "bootstrap_95_ci_lo": lo,
                    "bootstrap_95_ci_hi": hi,
                    "mean_signed_change_from_original": mean(signed),
                    "within_0.05_of_original_n": n_nearly_same,
                    "within_0.05_of_original_fraction": (
                        n_nearly_same / len(deltas) if deltas else float("nan")
                    ),
                }
            )
    write_csv(args.pairwise_out, pairwise_rows)

    # Primary per-task sensitivity and paired suppression contrast.
    all_task_ids = sorted({r["task_id"] for r in collapsed})
    task_rows: list[dict[str, Any]] = []
    suppression_values: list[float] = []
    for task_id in all_task_ids:
        row: dict[str, Any] = {"task_id": task_id}
        complete = True
        for hint_mode in HINT_MODES:
            orig = lookup.get((task_id, hint_mode, "original_history"))
            altered = [lookup.get((task_id, hint_mode, c)) for c in ALTERED]
            if orig is None or any(x is None for x in altered):
                complete = False
                break
            assert all(x is not None for x in altered)
            diffs = [abs(x["prediction"] - orig["prediction"]) for x in altered if x is not None]
            all_preds = [orig["prediction"]] + [x["prediction"] for x in altered if x is not None]
            row[f"sensitivity_{hint_mode}"] = mean(diffs)
            row[f"range_{hint_mode}"] = max(all_preds) - min(all_preds)
            row[f"original_prediction_{hint_mode}"] = orig["prediction"]
            row["store_nbr"] = orig["store_nbr"]
            row["item_nbr"] = orig["item_nbr"]
            row["family"] = orig["family"]
            row["mu_i_true"] = orig["mu_i_true"]
            row["source_hint"] = orig["source_hint"]
        if not complete:
            continue
        suppression = row["sensitivity_named_entity"] - row["sensitivity_series_hint"]
        row["sensitivity_suppression_difference"] = suppression
        no_hint_s = row["sensitivity_named_entity"]
        row["sensitivity_ratio_hint_over_no_hint"] = (
            row["sensitivity_series_hint"] / no_hint_s if no_hint_s > 0 else float("nan")
        )
        task_rows.append(row)
        suppression_values.append(suppression)

    write_csv(args.task_out, task_rows)

    rng_boot = random.Random(args.seed)
    ci_lo, ci_hi = ci_mean(suppression_values, args.n_boot, rng_boot)
    rng_perm = random.Random(args.seed + 1)
    p_value = sign_flip_pvalue(suppression_values, args.n_boot, rng_perm)

    s_named = [r["sensitivity_named_entity"] for r in task_rows]
    s_hint = [r["sensitivity_series_hint"] for r in task_rows]
    ratio_of_means = mean(s_hint) / mean(s_named) if mean(s_named) > 0 else float("nan")
    relative_reduction = 1 - ratio_of_means if math.isfinite(ratio_of_means) else float("nan")

    report_lines = [
        "=" * 74,
        "HISTORY-SENSITIVITY SUPPRESSION TEST",
        "=" * 74,
        f"Input responses: {args.input}",
        f"Parsed task-condition cells: {len(collapsed)}",
        f"Complete paired tasks: {len(task_rows)}",
        "",
        "Per-task sensitivity:",
        "  S_i = mean absolute prediction change from original history",
        "        across shuffled / other-series / mean-only / no-history.",
        "",
        f"  mean S_i without numerical hint (named_entity) = {mean(s_named):.4f}",
        f"  mean S_i with numerical series hint           = {mean(s_hint):.4f}",
        f"  hint/no-hint sensitivity ratio                = {ratio_of_means:.4f}",
        f"  relative sensitivity reduction                = {relative_reduction:.1%}",
        "",
        "Primary paired contrast:",
        "  suppression = mean[S_i(named_entity) - S_i(series_hint)]",
        f"  estimate = {mean(suppression_values):.4f}",
        f"  95% task-bootstrap CI = [{ci_lo:.4f}, {ci_hi:.4f}]",
        f"  paired sign-flip p-value = {p_value:.6f}",
        "",
    ]
    if ci_lo > 0:
        report_lines.extend(
            [
                "READING:",
                "  Numerical magnitude injection significantly reduces sensitivity",
                "  to the supplied history. This supports behavioural suppression of",
                "  the model's autonomous history-dependent contribution.",
            ]
        )
    elif ci_hi < 0:
        report_lines.extend(
            [
                "READING:",
                "  Numerical magnitude injection increases history sensitivity in this run.",
                "  The proposed suppression hypothesis is not supported.",
            ]
        )
    else:
        report_lines.extend(
            [
                "READING:",
                "  The paired CI includes zero. This run does not establish that the",
                "  numerical hint suppresses sensitivity to history.",
            ]
        )
    report_lines.append("=" * 74)
    report = "\n".join(report_lines) + "\n"
    Path(args.report_out).write_text(report, encoding="utf-8")
    print(report)
    print("Wrote:")
    for path in (
        args.collapsed_out,
        args.summary_out,
        args.pairwise_out,
        args.task_out,
        args.report_out,
    ):
        print(f"  {path}")


if __name__ == "__main__":
    main()
