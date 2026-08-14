#!/usr/bin/env python3
"""
plot_crossdataset_summary.py -- cross-dataset diagnostic summary figure

Cross-dataset replication figure, styled to match Plot_history_ablation_final.py
and the cross-model diagnostic figure. Three panels, one bar group
per dataset (Favorita / Rossmann / Dominick's):

  (a) near-copy rate under the series hint          -> generalises (all high)
  (b) cue DR vs. injected-prediction DR             -> clamping (bars pinned together)
  (c) history sensitivity S_i, no-number vs. hint   -> Rossmann strong,
                                                       Dominick's heterogeneous (CI crosses 0)

All numbers are fixed cross-dataset diagnostic values. Favorita values come from the checked-in artifacts; Rossmann and Dominick's values come from the external replication summaries. Near-copy in panel (a) uses the Stage-1 series-hint definition for all
three datasets (Favorita 0.83, Rossmann 0.804, Dominick's 0.65); the separate
history-ablation near-copy numbers (Rossmann 0.832, Dominick's 0.59) are NOT used
here -- they belong on the ablation figure only. Keep that split consistent when comparing the summary figure with the ablation analyses.

Panel (c) draws the paired reduction Delta S with its 95% bootstrap CI as a
caret between the two bars. Dominick's CI crosses zero and is drawn explicitly
so the heterogeneity is visible rather than hidden.

Outputs:
  fig_crossdataset.png
  fig_crossdataset.pdf
"""

from __future__ import annotations

import argparse
import math
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Palette: identical to Plot_history_ablation_final.py
# ----------------------------------------------------------------------
C_NO_HINT = "#2166AC"   # blue  -> "autonomous / no-number" side
C_HINT = "#D55E00"      # orange-> "cue / injected" side
C_REF = "#888780"       # grey  -> reference lines / zero line

DATASETS = ("Favorita", "Rossmann", "Dominick's")

# ----------------------------------------------------------------------
# Fixed cross-dataset summary values.
# ----------------------------------------------------------------------
# (a) near-copy rate under the series hint (Stage-1 definition)
NEARCOPY = {
    "Favorita":   0.83,
    "Rossmann":   0.804,
    "Dominick's": 0.65,
}

# (b) clamping: cue's own DR (hint alone) vs. injected-prediction DR
CUE_DR = {          # Spearman DR of the injected series hint against mu_i
    "Favorita":   0.48,
    "Rossmann":   0.401,
    "Dominick's": 0.481,
}
INJECTED_DR = {     # Spearman DR of the model's injected-condition prediction
    "Favorita":   0.50,
    "Rossmann":   0.406,
    "Dominick's": 0.429,
}

# (c) history sensitivity S_i: no-number (category only) vs. series-hint
S_NONUM = {
    "Favorita":   0.77,    # Favorita history-ablation response summary
    "Rossmann":   0.310,
    "Dominick's": 0.775,
}
S_HINT = {
    "Favorita":   0.17,
    "Rossmann":   0.077,
    "Dominick's": 0.710,
}
# paired mean reduction Delta S and its 95% bootstrap CI (for the caret annotation)
DELTA_S = {
    "Favorita":   (0.60, 0.37, 0.88),      # Favorita: mean 0.60, CI [0.37,0.88]
    "Rossmann":   (0.232, 0.205, 0.259),   # Rossmann external replication summary
    "Dominick's": (0.066, -0.209, 0.341),  # Dominick's external replication summary; CI crosses 0
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-png", default="fig_crossdataset.png")
    p.add_argument("--out-pdf", default="fig_crossdataset.pdf")
    return p.parse_args()


def panel_label(ax: Any, text: str) -> None:
    ax.text(
        -0.12, 1.05, text, transform=ax.transAxes,
        ha="left", va="bottom", fontsize=13, fontweight="bold",
    )


def main() -> None:
    args = parse_args()
    x = list(range(len(DATASETS)))
    width = 0.38

    fig, axes = plt.subplots(1, 3, figsize=(14.3, 4.6))
    ax_a, ax_b, ax_c = axes

    # ------------------------------------------------------------------
    # (a) near-copy rate under the series hint
    # ------------------------------------------------------------------
    vals_a = [NEARCOPY[d] for d in DATASETS]
    bars = ax_a.bar(x, vals_a, color=C_HINT, alpha=0.82, width=0.62)
    ax_a.set_ylim(0, 1.0)
    ax_a.set_xticks(x, DATASETS)
    ax_a.set_ylabel(r"fraction with $|\hat{\mu}-h|\leq0.05$")
    ax_a.set_title("near-copy rate under series hint")
    for bar, v in zip(bars, vals_a):
        ax_a.text(bar.get_x() + bar.get_width() / 2, min(0.97, v + 0.025),
                  f"{v:.0%}", ha="center", fontsize=8.5)
    ax_a.text(0.02, -0.22,
              "Stage-1 series-hint definition (not the ablation run)",
              transform=ax_a.transAxes, fontsize=7.4, color="#555555")
    panel_label(ax_a, "(a)")

    # ------------------------------------------------------------------
    # (b) clamping: cue DR (hint alone) vs. injected-prediction DR
    # ------------------------------------------------------------------
    cue = [CUE_DR[d] for d in DATASETS]
    inj = [INJECTED_DR[d] for d in DATASETS]
    b1 = ax_b.bar([v - width / 2 for v in x], cue, width, label="cue alone (hint DR)",
                  color=C_REF, alpha=0.85)
    b2 = ax_b.bar([v + width / 2 for v in x], inj, width, label="prediction after injection",
                  color=C_HINT, alpha=0.82)
    ax_b.set_ylim(0, 0.6)
    ax_b.set_xticks(x, DATASETS)
    ax_b.set_ylabel("dose-response (Spearman DR)")
    ax_b.set_title("injection clamps ranking to the cue")
    ax_b.legend(fontsize=7.6, loc="upper right", framealpha=0.92)
    for bars_ in (b1, b2):
        for bar in bars_:
            h = bar.get_height()
            ax_b.text(bar.get_x() + bar.get_width() / 2, h + 0.012,
                      f"{h:.2f}", ha="center", fontsize=7.6)
    panel_label(ax_b, "(b)")

    # ------------------------------------------------------------------
    # (c) history sensitivity: no-number vs. series-hint, with Delta S CI
    # ------------------------------------------------------------------
    s0 = [S_NONUM[d] for d in DATASETS]
    s1 = [S_HINT[d] for d in DATASETS]
    b1 = ax_c.bar([v - width / 2 for v in x], s0, width, label="no numerical hint",
                  color=C_NO_HINT, alpha=0.82)
    b2 = ax_c.bar([v + width / 2 for v in x], s1, width, label="series hint injected",
                  color=C_HINT, alpha=0.82)
    top = max(s0) * 1.30
    ax_c.set_ylim(0, top)
    ax_c.set_xticks(x, DATASETS)
    ax_c.set_ylabel(r"history sensitivity $S_i$")
    ax_c.set_title("injection suppresses history response")
    ax_c.legend(fontsize=7.6, loc="upper left", framealpha=0.92)
    for bars_ in (b1, b2):
        for bar in bars_:
            h = bar.get_height()
            ax_c.text(bar.get_x() + bar.get_width() / 2, h + top * 0.012,
                      f"{h:.2f}", ha="center", fontsize=7.6)

    # Annotate Delta S and whether its 95% CI crosses zero, per dataset.
    for xi, d in zip(x, DATASETS):
        ds, lo, hi = DELTA_S[d]
        crosses = lo <= 0.0 <= hi
        tag = (r"$\Delta S=%.2f$" % ds) + ("\nCI crosses 0" if crosses else "")
        col = "#B00000" if crosses else "#333333"
        ax_c.text(xi, top * 0.93, tag, ha="center", va="top",
                  fontsize=7.2, color=col,
                  bbox=dict(boxstyle="round,pad=0.25", fc="white",
                            ec="#D3D1C7", alpha=0.9))
    ax_c.text(0.02, -0.22,
              r"$\Delta S$: paired mean reduction (95% bootstrap CI)",
              transform=ax_c.transAxes, fontsize=7.4, color="#555555")
    panel_label(ax_c, "(c)")

    for ax in axes:
        ax.tick_params(labelsize=8.5)

    fig.suptitle(
        "The transmission-pipe signature replicates across datasets",
        fontsize=13.8, y=1.03,
    )
    fig.tight_layout(w_pad=2.3)
    fig.savefig(args.out_png, dpi=200, bbox_inches="tight")
    fig.savefig(args.out_pdf, bbox_inches="tight")
    print(f"Wrote {args.out_png} and {args.out_pdf}")
    print("Panel (a) near-copy:", {d: NEARCOPY[d] for d in DATASETS})
    print("Panel (b) cue/inj DR:",
          {d: (CUE_DR[d], INJECTED_DR[d]) for d in DATASETS})
    print("Panel (c) S0/S1:", {d: (S_NONUM[d], S_HINT[d]) for d in DATASETS})


if __name__ == "__main__":
    main()
