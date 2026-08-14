#!/usr/bin/env python3
"""
Plot_history_ablation_revised.py  (local, no API)

Publication-oriented version of the history-ablation figure.

Changes from the original plotting script:
  1. Uses the experimentally precise labels "category only" and
     "category + numerical hint".
  2. States explicitly that panel (a) shows means and paired-bootstrap 95% CIs.
  3. Zooms panel (b) into the dense region and adds a full-range inset, so a few
     outliers do not compress the other tasks into the lower-left corner.
  4. Reports the number of off-axis points in panel (b).
  5. Clarifies that panel (c) uses the median prediction across repeated calls.

Run Analyze_history_ablation.py first.

Outputs:
  fig_history_ablation_revised.png
  fig_history_ablation_revised.pdf
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

HISTORY_ORDER = (
    "shuffled_history",
    "other_series_history",
    "mean_only_history",
    "no_history",
)
LABELS = {
    "shuffled_history": "shuffled",
    "other_series_history": "other series",
    "mean_only_history": "mean only",
    "no_history": "no history",
    "original_history": "original",
}

C_NO_HINT = "#2166AC"
C_HINT = "#D55E00"
C_REF = "#888780"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pairwise", default="history_ablation_pairwise.csv")
    p.add_argument("--tasks", default="history_ablation_task_sensitivity.csv")
    p.add_argument("--summary", default="history_ablation_summary.csv")
    p.add_argument("--out-png", default="fig_history_ablation_revised.png")
    p.add_argument("--out-pdf", default="fig_history_ablation_revised.pdf")
    p.add_argument(
        "--zoom-max",
        type=float,
        default=None,
        help=(
            "Upper limit for the main panel-(b) zoom. By default it is chosen "
            "from the 95th percentile and capped at 2.0 when large outliers exist."
        ),
    )
    return p.parse_args()


def read_csv(path: str | Path) -> list[dict[str, Any]]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def f(row: dict[str, Any], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def panel_label(ax: Any, text: str) -> None:
    ax.text(
        -0.12, 1.05, text, transform=ax.transAxes,
        ha="left", va="bottom", fontsize=13, fontweight="bold",
    )


def percentile(values: Iterable[float], q: float) -> float:
    vals = sorted(v for v in values if math.isfinite(v))
    if not vals:
        return float("nan")
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    weight = pos - lo
    return vals[lo] * (1 - weight) + vals[hi] * weight


def metric_box(ax: Any, lines: list[str], loc: tuple[float, float] = (0.04, 0.96)) -> None:
    ax.text(
        loc[0], loc[1], "\n".join(lines), transform=ax.transAxes,
        va="top", ha="left", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#D3D1C7", alpha=0.92),
    )


def paired_bootstrap_mean_reduction(
    xs: Iterable[float],
    ys: Iterable[float],
    n_boot: int = 10_000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Return mean(xs - ys) and a task-level paired-bootstrap 95% CI."""
    diffs = [x - y for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if not diffs:
        return float("nan"), float("nan"), float("nan")

    observed = sum(diffs) / len(diffs)
    rng = random.Random(seed)
    boot_means = []
    for _ in range(n_boot):
        sample = [rng.choice(diffs) for _ in diffs]
        boot_means.append(sum(sample) / len(sample))

    lo = percentile(boot_means, 0.025)
    hi = percentile(boot_means, 0.975)
    return observed, lo, hi


def main() -> None:
    args = parse_args()
    pairwise = read_csv(args.pairwise)
    tasks = read_csv(args.tasks)
    summary = read_csv(args.summary)
    if not pairwise or not tasks or not summary:
        raise SystemExit("Analysis CSV files are empty. Run Analyze_history_ablation.py first.")

    pair_lookup = {(r["hint_mode"], r["history_condition"]): r for r in pairwise}
    summary_lookup = {(r["hint_mode"], r["history_condition"]): r for r in summary}

    fig, axes = plt.subplots(1, 3, figsize=(14.3, 4.6))
    ax_a, ax_b, ax_c = axes

    # ------------------------------------------------------------------
    # (a) Mean absolute change from original history, paired-bootstrap CI
    # ------------------------------------------------------------------
    x = list(range(len(HISTORY_ORDER)))
    offset = 0.09
    for mode, label, color, dx in (
        ("named_entity", "category only", C_NO_HINT, -offset),
        ("series_hint", "category + numerical hint", C_HINT, offset),
    ):
        means = [f(pair_lookup[(mode, c)], "mean_abs_change_from_original") for c in HISTORY_ORDER]
        los = [f(pair_lookup[(mode, c)], "bootstrap_95_ci_lo") for c in HISTORY_ORDER]
        his = [f(pair_lookup[(mode, c)], "bootstrap_95_ci_hi") for c in HISTORY_ORDER]
        yerr = [
            [max(0.0, m - lo) for m, lo in zip(means, los)],
            [max(0.0, hi - m) for m, hi in zip(means, his)],
        ]
        ax_a.errorbar(
            [v + dx for v in x], means, yerr=yerr,
            fmt="o", markersize=5.5, linestyle="none",
            elinewidth=1.6, capsize=3.5,
            label=label, color=color,
        )

    ax_a.axhline(0, lw=0.8, color=C_REF, alpha=0.65)
    ax_a.set_xticks(x, [LABELS[c] for c in HISTORY_ORDER], rotation=22, ha="right")
    ax_a.set_ylabel(r"mean $|\hat{\mu}_{\mathrm{perturbed}}-\hat{\mu}_{\mathrm{original}}|$")
    ax_a.set_title("response to history perturbation")
    ax_a.legend(fontsize=7.8, loc="upper right", framealpha=0.92)
    ax_a.text(
        0.02, -0.27, "Points: task mean; error bars: paired-bootstrap 95% CI",
        transform=ax_a.transAxes, fontsize=7.4, color="#555555",
    )
    panel_label(ax_a, "(a)")

    # ------------------------------------------------------------------
    # (b) Task-level sensitivity: zoomed main panel + full-range inset
    # ------------------------------------------------------------------
    raw_xs = [f(r, "sensitivity_named_entity") for r in tasks]
    raw_ys = [f(r, "sensitivity_series_hint") for r in tasks]
    finite = [(xv, yv) for xv, yv in zip(raw_xs, raw_ys) if math.isfinite(xv) and math.isfinite(yv)]
    if not finite:
        raise SystemExit("No finite task-level sensitivity values.")
    xs, ys = zip(*finite)

    full_hi = max(max(xs), max(ys), 0.05) * 1.05
    if args.zoom_max is not None:
        zoom_hi = args.zoom_max
    else:
        p95 = percentile(list(xs) + list(ys), 0.95)
        candidate = max(1.25, p95 * 1.15)
        zoom_hi = min(candidate, 2.0) if full_hi > 2.5 else full_hi
        zoom_hi = min(zoom_hi, full_hi)

    ax_b.plot([0, zoom_hi], [0, zoom_hi], ls="--", lw=1, color=C_REF)
    ax_b.scatter(xs, ys, s=25, alpha=0.66, color=C_NO_HINT, edgecolors="none")
    ax_b.set_xlim(0, zoom_hi)
    ax_b.set_ylim(0, zoom_hi)
    ax_b.set_aspect("equal", adjustable="box")
    ax_b.set_xlabel("sensitivity: category only")
    ax_b.set_ylabel("sensitivity: category + numerical hint")
    ax_b.set_title("task-level sensitivity")

    below = sum(y < x for x, y in zip(xs, ys))
    off_axis = sum(x > zoom_hi or y > zoom_hi for x, y in zip(xs, ys))
    mean_reduction, reduction_lo, reduction_hi = paired_bootstrap_mean_reduction(xs, ys)

    lines = [
        f"{below}/{len(xs)} below $y=x$",
        f"mean reduction = {mean_reduction:.2f}",
        f"95% CI [{reduction_lo:.2f}, {reduction_hi:.2f}]",
    ]
    if off_axis:
        lines.append(f"{off_axis} task{'s' if off_axis != 1 else ''} outside zoom")
    metric_box(ax_b, lines)

    # Full-range inset prevents the zoom from hiding outliers.
    if full_hi > zoom_hi * 1.15:
        ax_in = inset_axes(ax_b, width="36%", height="36%", loc="upper right", borderpad=1.0)
        ax_in.plot([0, full_hi], [0, full_hi], ls="--", lw=0.8, color=C_REF)
        ax_in.scatter(xs, ys, s=10, alpha=0.55, color=C_NO_HINT, edgecolors="none")
        ax_in.set_xlim(0, full_hi)
        ax_in.set_ylim(0, full_hi)
        ax_in.set_aspect("equal", adjustable="box")
        ax_in.set_title("full range", fontsize=7)
        ax_in.tick_params(labelsize=6, length=2)

    panel_label(ax_b, "(b)")

    # ------------------------------------------------------------------
    # (c) Near-copying under each history condition
    # ------------------------------------------------------------------
    all_hist = ("original_history",) + HISTORY_ORDER
    copy_fracs = [
        f(summary_lookup[("series_hint", c)], "within_0.05_of_source_hint_fraction")
        for c in all_hist
    ]
    bars = ax_c.bar(range(len(all_hist)), copy_fracs, color=C_HINT, alpha=0.82)
    ax_c.set_xticks(range(len(all_hist)), [LABELS[c] for c in all_hist], rotation=22, ha="right")
    ax_c.set_ylim(0, 1.0)
    ax_c.set_ylabel(r"fraction with $|\hat{\mu}-h|\leq0.05$")
    ax_c.set_title("near-copy rate across histories")
    for bar, value in zip(bars, copy_fracs):
        if math.isfinite(value):
            ax_c.text(
                bar.get_x() + bar.get_width() / 2,
                min(0.97, value + 0.025),
                f"{value:.0%}", ha="center", fontsize=8,
            )
    ax_c.text(
        0.02, -0.27, "Based on the median prediction across repeated calls",
        transform=ax_c.transAxes, fontsize=7.4, color="#555555",
    )
    panel_label(ax_c, "(c)")

    for ax in axes:
        ax.tick_params(labelsize=8.5)

    fig.suptitle(
        "Numerical magnitude injection suppresses sensitivity to sales history",
        fontsize=13.8, y=1.03,
    )
    fig.tight_layout(w_pad=2.3)
    fig.savefig(args.out_png, dpi=200, bbox_inches="tight")
    fig.savefig(args.out_pdf, bbox_inches="tight")
    print(f"Wrote {args.out_png} and {args.out_pdf}")
    print(
        f"Panel (b): {below}/{len(xs)} below y=x; "
        f"mean reduction={mean_reduction:.3f}, "
        f"95% CI [{reduction_lo:.3f}, {reduction_hi:.3f}]; "
        f"{off_axis} outside main zoom; zoom max={zoom_hi:.2f}"
    )


if __name__ == "__main__":
    main()
