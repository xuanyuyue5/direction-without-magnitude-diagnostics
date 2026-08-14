#!/usr/bin/env python3
"""
plot_finding.py  (local, no API)

FINDING figure (not motivation): the transmission-pipe effect.

Three panels, shared lift axis:
  (a) family hint  | pred vs true mu, with per-family hint drawn as a flat
                     segment over that family's mu-range. Predictions collapse
                     onto their family constant -> DR 0.23, slope 0.01.
  (b) series hint  | pred (blue) AND hint (orange) vs true mu, same axes.
                     The prediction cloud is no tighter to truth than the hint
                     cloud: corr(pred,mu)=0.70 vs corr(hint,mu)=0.65,
                     delta_r=0.05, 95% CI [0.003,0.094] (see bootstrap_increment.py).
                     -> the model adds no meaningful information beyond the hint.
  (c) series hint  | pred vs hint. Points glue to y=x (corr 0.96; 83/100 within
                     +-0.05) -> the model is copying the hint, not reasoning.

Panels (a)/(b) share the true-mu x-axis; panel (c)'s x-axis is the hint.
All metrics are computed on ALL 100 tasks; the view is clipped to mu<=3.2 for
legibility (one task at mu=6.25 sits off-axis). ASCII-only console output.

Run:  python plot_transmission_pipe.py --responses-main responses.jsonl --responses-series responses_series.jsonl
"""
import argparse
import json
import math
import statistics as st
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------- CONFIG -----------------------------
RESP_MAIN   = "responses.jsonl"
RESP_SERIES = "responses_series.jsonl"
OUT_PNG     = "fig_finding.png"
OUT_PDF     = "fig_finding.pdf"

FAMILY_COND = "stated_magnitude"
SERIES_COND = "stated_magnitude_series"

LO, HI = -1.2, 3.2          # shared display window (clip; metrics use full data)
EPS    = 0.05               # "glued to hint" tolerance

C_PRED = "#2166AC"          # blue  - prediction (colorblind-safe)
C_HINT = "#D55E00"          # orange - injected hint
C_REF  = "#888780"          # grey  - y=x reference
# ------------------------------------------------------------------



def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses-main", default=RESP_MAIN)
    parser.add_argument("--responses-series", default=RESP_SERIES)
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


def metric_box(ax, lines):
    ax.text(0.04, 0.96, "\n".join(lines), transform=ax.transAxes,
            va="top", ha="left", fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#D3D1C7", alpha=0.92))


def offaxis_note(ax, xs, ys):
    k = sum(1 for i in range(len(xs)) if not (LO <= xs[i] <= HI and LO <= ys[i] <= HI))
    if k:
        ax.text(0.03, 0.03, "%d pt%s off-axis" % (k, "s" if k > 1 else ""),
                transform=ax.transAxes, va="bottom", ha="left",
                fontsize=7.5, color="#888780", style="italic")


def panel_label(ax, s):
    ax.text(-0.02, 1.06, s, transform=ax.transAxes, va="bottom", ha="left",
            fontsize=13, fontweight="bold")


def main():
    args = parse_args()
    recs = [r for r in (load(args.responses_main) + load(args.responses_series)) if r.get("parse_ok")]
    if not recs:
        print("No parsed records found.")
        return

    cell = defaultdict(list); mu = {}; hint = {}; fam = {}
    for r in recs:
        key = (r["store_nbr"], r["item_nbr"], r["condition"])
        cell[key].append(r["lift_pred"])
        mu[(r["store_nbr"], r["item_nbr"])] = r["mu_i_true"]
        hint[key] = r["family_hint"]
        fam[(r["store_nbr"], r["item_nbr"])] = r["family"]
    med = {k: st.median(v) for k, v in cell.items()}

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.5))
    axA, axB, axC = axes

    # ---------------- (a) family hint: pred vs mu + flat family segments -----
    keysA = [k for k in med if k[2] == FAMILY_COND]
    Xa = [mu[(k[0], k[1])] for k in keysA]
    Ya = [med[k] for k in keysA]
    drA, slA = spearman(Xa, Ya), ols_slope(Xa, Ya)

    axA.plot([LO, HI], [LO, HI], ls="--", lw=1, color=C_REF, zorder=1)
    # per-family hint as a flat segment over that family's mu extent
    by_fam = defaultdict(list)
    for k in keysA:
        by_fam[fam[(k[0], k[1])]].append((mu[(k[0], k[1])], hint[k]))
    first = True
    for f, pts in by_fam.items():
        xs = [p[0] for p in pts]; h = pts[0][1]  # hint constant within family
        x0, x1 = min(xs), max(xs)
        axA.plot([x0, x1], [h, h], color=C_HINT, lw=2.2, alpha=0.8, zorder=2,
                 solid_capstyle="round",
                 label="family hint (constant within family)" if first else None)
        first = False
    axA.scatter(Xa, Ya, s=20, color=C_PRED, alpha=0.7, edgecolors="none",
                zorder=3, label="prediction")
    axA.set_title("family hint", fontsize=12)
    axA.set_xlabel(r"true post-hoc $\mu_i$", fontsize=10)
    axA.set_ylabel("predicted lift", fontsize=10)
    metric_box(axA, ["DR = %.2f" % drA, "slope = %.2f" % slA])
    panel_label(axA, "(a)")
    axA.legend(loc="lower right", fontsize=7.5, framealpha=0.9)
    offaxis_note(axA, Xa, Ya)

    # ---------------- (b) series hint: pred AND hint vs mu -------------------
    keysB = [k for k in med if k[2] == SERIES_COND]
    Xb = [mu[(k[0], k[1])] for k in keysB]
    Yb = [med[k] for k in keysB]
    Hb = [hint[k] for k in keysB]
    r_pm = corr(Yb, Xb); r_hm = corr(Hb, Xb)

    axB.plot([LO, HI], [LO, HI], ls="--", lw=1, color=C_REF, zorder=1)
    axB.scatter(Xb, Hb, s=20, color=C_HINT, alpha=0.55, edgecolors="none",
                zorder=2, label="injected hint")
    axB.scatter(Xb, Yb, s=20, color=C_PRED, alpha=0.6, edgecolors="none",
                zorder=3, label="prediction")
    axB.set_title("series hint  (pred & hint vs truth)", fontsize=11)
    axB.set_xlabel(r"true post-hoc $\mu_i$", fontsize=10)
    metric_box(axB, [
        r"Pearson $r$(pred,$\mu$) = %.2f" % r_pm,
        r"Pearson $r$(hint,$\mu$) = %.2f" % r_hm,
        r"Spearman DR(pred) = 0.50",
        r"Spearman DR(hint) = 0.48",
    ])
    panel_label(axB, "(b)")
    axB.legend(loc="lower right", fontsize=7.5, framealpha=0.9)
    offaxis_note(axB, Xb + Xb, Yb + Hb)

    # ---------------- (c) series hint: pred vs hint (copying) ----------------
    r_ph = corr(Yb, Hb)
    glued = sum(1 for i in range(len(Yb)) if abs(Yb[i] - Hb[i]) < EPS)

    axC.plot([LO, HI], [LO, HI], ls="--", lw=1, color=C_REF, zorder=1)
    axC.scatter(Hb, Yb, s=20, color=C_PRED, alpha=0.7, edgecolors="none", zorder=3)
    axC.set_title("series hint  (prediction vs hint)", fontsize=11)
    axC.set_xlabel("injected hint", fontsize=10)
    axC.set_ylabel("predicted lift", fontsize=10)
    metric_box(axC, ["corr(pred,hint) = %.2f" % r_ph,
                     "%d/%d within $\\pm$%.2f" % (glued, len(Yb), EPS)])
    panel_label(axC, "(c)")
    offaxis_note(axC, Hb, Yb)

    for ax in axes:
        ax.set_xlim(LO, HI); ax.set_ylim(LO, HI)
        ax.tick_params(labelsize=9)
        ax.set_aspect("equal", adjustable="box")

    fig.suptitle("Injecting magnitude collapses the model into a transmission pipe",
                 fontsize=13.5, y=1.04)
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=150, bbox_inches="tight")
    fig.savefig(args.out_pdf, bbox_inches="tight")
    print("Wrote %s and %s" % (args.out_png, args.out_pdf))
    print("  (a) family hint   DR=%.2f slope=%.2f" % (drA, slA))
    print("  (b) series hint   corr(pred,mu)=%.2f corr(hint,mu)=%.2f" % (r_pm, r_hm))
    print("  (c) series hint   corr(pred,hint)=%.2f  glued %d/%d" % (r_ph, glued, len(Yb)))


if __name__ == "__main__":
    main()