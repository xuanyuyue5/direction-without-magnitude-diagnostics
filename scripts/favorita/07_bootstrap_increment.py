#!/usr/bin/env python3
"""
bootstrap_increment.py  (local, no API)

Quantifies whether the model adds ANY information beyond the injected series-level
hint, in the stated_magnitude_series condition.

Core quantity:
    delta_r = corr(pred, mu) - corr(hint, mu)

If the model were a pure transmission pipe (copies the hint, contributes nothing),
delta_r should be ~0. We bootstrap a 95% CI over tasks (resample series with
replacement) to see whether the small observed delta_r is distinguishable from 0.

Also reports:
    corr(pred, hint)        -> how tightly predictions track the hint ("copying")
    fraction within +-EPS   -> how many predictions are glued to the hint
    per-condition DR / slope -> sanity check vs the figures

Run:  python3 bootstrap_increment.py
"""
import json
import math
import random
import statistics as st
from collections import defaultdict

RESP_MAIN   = "responses.jsonl"
RESP_SERIES = "responses_series.jsonl"
SERIES_COND = "stated_magnitude_series"
EPS         = 0.05      # "glued to hint" tolerance
N_BOOT      = 10000
SEED        = 42        # fixed for reproducibility


def load(path):
    try:
        return [json.loads(l) for l in open(path, encoding="utf-8")]
    except FileNotFoundError:
        print("WARNING: %s not found, skipping." % path)
        return []


def corr(a, b):
    n = len(a)
    ma = sum(a) / n
    mb = sum(b) / n
    num = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return num / den if den else float("nan")


def spearman(xs, ys):
    n = len(xs)

    def rank(a):
        order = sorted(range(n), key=lambda k: a[k])
        rk = [0] * n
        i = 0
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
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    den = sum((xs[k] - mx) ** 2 for k in range(n))
    return sum((xs[k] - mx) * (ys[k] - my) for k in range(n)) / den if den else float("nan")


def collapse(recs):
    """3 seeds -> per-(task,condition) median prediction; carry mu and hint."""
    cell = defaultdict(list)
    mu = {}
    hint = {}
    for r in recs:
        key = (r["store_nbr"], r["item_nbr"], r["condition"])
        cell[key].append(r["lift_pred"])
        mu[(r["store_nbr"], r["item_nbr"])] = r["mu_i_true"]
        hint[key] = r["family_hint"]   # in the series file this field holds the series-level hint
    med = {k: st.median(v) for k, v in cell.items()}
    return med, mu, hint


def main():
    recs = [r for r in (load(RESP_MAIN) + load(RESP_SERIES)) if r.get("parse_ok")]
    if not recs:
        print("No parsed records found.")
        return
    med, mu, hint = collapse(recs)

    # ---- series condition, aligned arrays over the 100 tasks ----
    keys = [k for k in med if k[2] == SERIES_COND]
    P = [med[k] for k in keys]                 # prediction (seed-median)
    M = [mu[(k[0], k[1])] for k in keys]       # true post-hoc mu_i
    H = [hint[k] for k in keys]                # injected series hint
    n = len(P)

    r_pm = corr(P, M)
    r_hm = corr(H, M)
    r_ph = corr(P, H)
    delta = r_pm - r_hm
    glued = sum(1 for i in range(n) if abs(P[i] - H[i]) < EPS)

    # ---- bootstrap delta_r over tasks ----
    random.seed(SEED)
    idx = list(range(n))
    deltas = []
    for _ in range(N_BOOT):
        bs = [random.choice(idx) for _ in idx]
        pp = [P[i] for i in bs]
        mm = [M[i] for i in bs]
        hh = [H[i] for i in bs]
        deltas.append(corr(pp, mm) - corr(hh, mm))
    deltas.sort()
    lo = deltas[int(0.025 * N_BOOT)]
    hi = deltas[int(0.975 * N_BOOT)]

    print("=" * 64)
    print("SERIES-HINT INCREMENT TEST  (n=%d tasks, %d bootstrap resamples)" % (n, N_BOOT))
    print("=" * 64)
    print("  corr(pred, mu)   = %.3f   <- how well predictions track truth" % r_pm)
    print("  corr(hint, mu)   = %.3f   <- how well the injected hint tracks truth" % r_hm)
    print("  corr(pred, hint) = %.3f   <- how tightly predictions copy the hint" % r_ph)
    print("  predictions within +-%.2f of hint: %d/%d" % (EPS, glued, n))
    print("-" * 64)
    print("  delta_r = corr(pred,mu) - corr(hint,mu) = %.3f" % delta)
    print("  95%% bootstrap CI for delta_r: [%.3f, %.3f]" % (lo, hi))
    print("  CI crosses zero: %s" % ("YES" if lo < 0 < hi else "NO"))
    print("-" * 64)
    if lo > 0:
        print("  READING: delta_r is small but statistically > 0.")
        print("  The model is NOT a perfect pipe: a thin residual of its own")
        print("  series reasoning survives the injection and is weakly aligned")
        print("  with truth. Report as 'negligible but nonzero gain', NOT 'zero'.")
    else:
        print("  READING: delta_r is indistinguishable from 0 at 95%.")
        print("  Consistent with the model adding no information beyond the hint.")
    print("=" * 64)

    # sanity: per-condition DR / slope (should match the figures)
    print("\nper-condition sanity (all data, not the clipped view):")
    print("  condition                  DR     slope")
    for c in ["no_ctx", "named_entity", "stated_magnitude", "stated_magnitude_series"]:
        X = [mu[(k[0], k[1])] for k in med if k[2] == c]
        Y = [med[k] for k in med if k[2] == c]
        if X:
            print("    %-24s %5.2f  %5.2f" % (c, spearman(X, Y), ols_slope(X, Y)))


if __name__ == "__main__":
    main()