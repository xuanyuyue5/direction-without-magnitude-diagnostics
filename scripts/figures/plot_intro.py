#!/usr/bin/env python3
"""
plot_intro.py  (local, no API)

INTRODUCTION / motivation figure: the magnitude gap.

Two panels, shared axes, y=x calibration line:
  (a) no context   | the model predicts lift with no entity/magnitude info.
  (b) named entity  | the model is told the product family only.

In both, predictions get the DIRECTION roughly right but ride well below the
y=x line and barely climb with true mu: the model knows promotions lift sales
but cannot calibrate BY HOW MUCH (shallow slope, modest DR). That gap is what
the paper is about.

Axes match fig_finding.py (window clipped to mu<=3.2 for legibility; one task at
mu=6.25 and a few high predictions sit off-axis). Metrics use ALL 100 tasks.
ASCII-only console output.

Run:  python plot_intro.py --responses responses.jsonl
"""
import argparse
import json
import math
import statistics as st
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESP_MAIN = "responses.jsonl"
OUT_PNG   = "fig_intro.png"
OUT_PDF   = "fig_intro.pdf"

CONDS = [("no_ctx", "no context"), ("named_entity", "named entity")]
LO, HI = -1.2, 3.2
C_PRED = "#2166AC"
C_REF  = "#888780"



def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses", default=RESP_MAIN)
    parser.add_argument("--out-png", default=OUT_PNG)
    parser.add_argument("--out-pdf", default=OUT_PDF)
    return parser.parse_args()


def load(path):
    try:
        return [json.loads(l) for l in open(path, encoding="utf-8")]
    except FileNotFoundError:
        print("WARNING: %s not found." % path)
        return []


def corr(a, b):
    n = len(a); ma = sum(a) / n; mb = sum(b) / n
    num = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return num / den if den else float("nan")


def spearman(xs, ys):
    n = len(xs)
    def rank(a):
        order = sorted(range(n), key=lambda k: a[k]); rk = [0] * n; i = 0
        while i < n:
            j = i
            while j + 1 < n and a[order[j + 1]] == a[order[i]]:
                j += 1
            for k in range(i, j + 1):
                rk[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return rk
    return corr(rank(xs), rank(ys))


def ols_slope(xs, ys):
    n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
    den = sum((xs[k] - mx) ** 2 for k in range(n))
    return sum((xs[k] - mx) * (ys[k] - my) for k in range(n)) / den if den else float("nan")


def main():
    args = parse_args()
    recs = [r for r in load(args.responses) if r.get("parse_ok")]
    if not recs:
        print("No parsed records found.")
        return

    cell = defaultdict(list); mu = {}
    for r in recs:
        cell[(r["store_nbr"], r["item_nbr"], r["condition"])].append(r["lift_pred"])
        mu[(r["store_nbr"], r["item_nbr"])] = r["mu_i_true"]
    med = {k: st.median(v) for k, v in cell.items()}

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.5), sharex=True, sharey=True)

    for ax, (cond, nice), lab in zip(axes, CONDS, ["(a)", "(b)"]):
        X = [mu[(k[0], k[1])] for k in med if k[2] == cond]
        Y = [med[k] for k in med if k[2] == cond]
        dr, sl = spearman(X, Y), ols_slope(X, Y)

        ax.plot([LO, HI], [LO, HI], ls="--", lw=1, color=C_REF, zorder=1)
        ax.scatter(X, Y, s=20, color=C_PRED, alpha=0.65, edgecolors="none", zorder=3)

        ax.set_title(nice, fontsize=12)
        ax.set_xlim(LO, HI); ax.set_ylim(LO, HI)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(r"true post-hoc $\mu_i$", fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel("predicted lift", fontsize=10)
        ax.tick_params(labelsize=9)
        ax.text(0.04, 0.96, "DR = %.2f\nslope = %.2f" % (dr, sl),
                transform=ax.transAxes, va="top", ha="left", fontsize=9.5,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#D3D1C7", alpha=0.92))
        ax.text(-0.02, 1.06, lab, transform=ax.transAxes, va="bottom", ha="left",
                fontsize=13, fontweight="bold")
        k = sum(1 for i in range(len(X)) if not (LO <= X[i] <= HI and LO <= Y[i] <= HI))
        if k:
            ax.text(0.03, 0.03, "%d pt%s off-axis" % (k, "s" if k > 1 else ""),
                    transform=ax.transAxes, va="bottom", ha="left",
                    fontsize=7.5, color="#888780", style="italic")
        print("  %-14s DR=%.2f slope=%.2f" % (nice, dr, sl))

    fig.suptitle("The model gets promotion direction right but magnitude wrong",
                 fontsize=13.5, y=1.04)
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=150, bbox_inches="tight")
    fig.savefig(args.out_pdf, bbox_inches="tight")
    print("Wrote %s and %s" % (args.out_png, args.out_pdf))


if __name__ == "__main__":
    main()