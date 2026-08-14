#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_cross_model.py  --  cross-model replication figure for the transmission-pipe effect.

Reproduces the three core diagnostics across GPT-4o,
DeepSeek, and Claude Opus 4.8, in one three-panel figure:

  (a) near-copy rate: fraction of tasks whose series-hint prediction sits
      within +/-0.05 of the injected hint  (higher = more copying)
  (b) dose-response (Spearman DR) against truth, autonomous vs series-hint
      (the pipe: injection fails to lift ranking above the autonomous estimate)
  (c) history sensitivity S_i, without vs with the numerical hint
      (injection suppresses response to history perturbation)

DeepSeek and Claude values are computed from response JSONL files. GPT-4o values are stored as a fixed reference block so the cross-model figure can be regenerated from the released artifacts.

Inputs (same folder, or pass --dir):
  prompts.jsonl                              (ground-truth mu_i)
  responses_{deepseek,claude}.jsonl          (autonomous: named_entity, no_ctx)
  responses_series_{deepseek,claude}.jsonl   (pipe: stated_magnitude_series)
  history_ablation_responses_{deepseek,claude}.jsonl  (S_i + recovers hint value)

Output:
  fig_cross_model.png / .pdf

Run:
  python plot_cross_model.py
  python plot_cross_model.py --dir . --out fig_cross_model
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ----------------------------------------------------------------------
# GPT-4o reference values used in the cross-model summary.
# ----------------------------------------------------------------------
GPT4O_REFERENCE = {
    "near_copy": 0.83,
    "dr_autonomous": 0.53,   # named_entity DR
    "dr_pipe": 0.50,         # series-hint DR
    "S_no_hint": 0.77,       # mean S_i, named_entity arm
    "S_hint": 0.1691,    # mean S_i, series-hint arm   (from Analyze_history_ablation.py)
}

# Consistent palette for the diagnostic figures
BLUE = "#4C72B0"    # autonomous / no-hint
ORANGE = "#DD8452"  # injected / pipe
EPS = 0.05

MODELS = ["GPT-4o", "DeepSeek", "Claude Opus 4.8"]
SUFFIX = {"DeepSeek": "deepseek", "Claude Opus 4.8": "claude"}


# ----------------------------------------------------------------------
# io / metric helpers
# ----------------------------------------------------------------------
def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def key(r) -> tuple[int, int]:
    return (int(r["store_nbr"]), int(r["item_nbr"]))


def median_by_task(rows, cond) -> dict:
    b: dict = {}
    for r in rows:
        if r.get("condition") != cond:
            continue
        v = r.get("lift_pred")
        if v is None:
            continue
        b.setdefault(key(r), []).append(v)
    return {k: float(np.median(v)) for k, v in b.items()}


def sp(x, y) -> float:
    return stats.spearmanr(x, y).statistic


def compute_model(d: Path, suffix: str, prompts_path: Path) -> dict:
    """Compute the three diagnostics for one model from its raw files."""
    mu = {key(r): float(r["mu_i_true"]) for r in load(prompts_path)}

    resp = load(d / f"responses_{suffix}.jsonl")
    ne = median_by_task(resp, "named_entity")

    ser = load(d / f"responses_series_{suffix}.jsonl")
    pipe = median_by_task(ser, "stated_magnitude_series")

    # recover injected hint from the ablation file (original_history rows)
    abl = load(d / f"history_ablation_responses_{suffix}.jsonl")
    hint = {}
    for r in abl:
        if r.get("condition") == "series_hint__original_history" and r.get("injected_hint") is not None:
            hint[key(r)] = float(r["injected_hint"])

    ks = sorted(set(mu) & set(ne) & set(pipe) & set(hint))
    a = np.array([ne[k] for k in ks])
    p = np.array([pipe[k] for k in ks])
    h = np.array([hint[k] for k in ks])
    m = np.array([mu[k] for k in ks])

    near_copy = float(np.mean(np.abs(p - h) <= EPS))
    dr_auto = sp(a, m)
    dr_pipe = sp(p, m)

    # history sensitivity S_i per arm, from the ablation file
    S_no, S_hint = history_sensitivity(abl)

    return {
        "near_copy": near_copy,
        "dr_autonomous": dr_auto,
        "dr_pipe": dr_pipe,
        "S_no_hint": S_no,
        "S_hint": S_hint,
        "n": len(ks),
    }


def history_sensitivity(abl_rows) -> tuple[float, float]:
    """Mean S_i for the named_entity (no-hint) and series_hint arms.

    S_i = mean over altered histories of |pred(alt) - pred(original)|,
    predictions being the median over repeats. Matches Analyze_history_ablation.py.
    """
    altered = ["shuffled_history", "other_series_history", "mean_only_history", "no_history"]
    # collapse to median per (task, hint_mode, history_condition)
    cell: dict = {}
    for r in abl_rows:
        if not r.get("parse_ok") or r.get("lift_pred") is None:
            continue
        k = (str(r["task_id"]), str(r["hint_mode"]), str(r["history_condition"]))
        cell.setdefault(k, []).append(float(r["lift_pred"]))
    med = {k: float(np.median(v)) for k, v in cell.items()}

    out = {}
    for hint_mode in ("named_entity", "series_hint"):
        tasks = sorted({k[0] for k in med if k[1] == hint_mode})
        s_vals = []
        for t in tasks:
            orig = med.get((t, hint_mode, "original_history"))
            if orig is None:
                continue
            diffs = [abs(med[(t, hint_mode, c)] - orig)
                     for c in altered if (t, hint_mode, c) in med]
            if diffs:
                s_vals.append(np.mean(diffs))
        out[hint_mode] = float(np.mean(s_vals)) if s_vals else float("nan")
    return out["named_entity"], out["series_hint"]


# ----------------------------------------------------------------------
# plotting
# ----------------------------------------------------------------------
def make_figure(results: dict, out_stem: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    x = np.arange(len(MODELS))
    w = 0.36

    # ---- panel (a): near-copy rate -----------------------------------
    ax = axes[0]
    vals = [results[m]["near_copy"] for m in MODELS]
    bars = ax.bar(x, vals, color=ORANGE, width=0.55)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel(r"fraction with $|\hat\mu - h| \leq 0.05$")
    ax.set_title("(a) near-copy rate under series hint", fontsize=11)
    ax.axhline(0, color="black", lw=0.6)

    # ---- panel (b): DR autonomous vs pipe ----------------------------
    ax = axes[1]
    auto = [results[m]["dr_autonomous"] for m in MODELS]
    pipe = [results[m]["dr_pipe"] for m in MODELS]
    ax.bar(x - w / 2, auto, w, label="autonomous (category label)", color=BLUE)
    ax.bar(x + w / 2, pipe, w, label="series hint injected", color=ORANGE)
    for xi, v in zip(x - w / 2, auto):
        ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    for xi, v in zip(x + w / 2, pipe):
        ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, max(auto + pipe) * 1.25)
    ax.set_ylabel("dose-response (Spearman DR)")
    ax.set_title("(b) injection clamps ranking near the hint", fontsize=11)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)

    # ---- panel (c): history sensitivity ------------------------------
    ax = axes[2]
    s_no = [results[m]["S_no_hint"] for m in MODELS]
    s_hi = [results[m]["S_hint"] for m in MODELS]
    ax.bar(x - w / 2, s_no, w, label="no numerical hint", color=BLUE)
    ax.bar(x + w / 2, s_hi, w, label="series hint injected", color=ORANGE)
    for xi, v in zip(x - w / 2, s_no):
        ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    for xi, v in zip(x + w / 2, s_hi):
        ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, max(s_no + s_hi) * 1.25)
    ax.set_ylabel(r"history sensitivity $S_i$")
    ax.set_title("(c) injection suppresses history response", fontsize=11)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(MODELS, fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("The transmission-pipe effect replicates across models",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_stem}.{ext}", dpi=200, bbox_inches="tight")
    print(f"Wrote {out_stem}.png and {out_stem}.pdf")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".", help="folder with model response JSONL files")
    ap.add_argument("--prompts", default=None, help="path to prompts.jsonl; defaults to --dir/prompts.jsonl")
    ap.add_argument("--out", default="fig_cross_model", help="output filename stem")
    args = ap.parse_args()
    d = Path(args.dir)
    prompts_path = Path(args.prompts) if args.prompts else d / "prompts.jsonl"

    results = {"GPT-4o": GPT4O_REFERENCE}
    for m in ("DeepSeek", "Claude Opus 4.8"):
        results[m] = compute_model(d, SUFFIX[m], prompts_path)

    # Echo the numbers for a quick reproducibility check.
    print(f"\n{'model':16s} {'nearcopy':>9s} {'DR_auto':>8s} {'DR_pipe':>8s} "
          f"{'S_nohint':>9s} {'S_hint':>7s}")
    for m in MODELS:
        r = results[m]
        n = r.get("n", "-")
        print(f"{m:16s} {r['near_copy']:9.2f} {r['dr_autonomous']:8.2f} "
              f"{r['dr_pipe']:8.2f} {r['S_no_hint']:9.2f} {r['S_hint']:7.2f}"
              f"   {'(reference)' if m=='GPT-4o' else f'(N={n})'}")
    print()

    make_figure(results, Path(args.out))


if __name__ == "__main__":
    main()
