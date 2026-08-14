#!/usr/bin/env python3
"""
Cold-start promo series filter + post-hoc mu_i computation for Corporacion Favorita.

What it does (no LLM, no API cost):
  1. Streams train.csv with polars (lazy, low memory).
  2. For each (store_nbr, item_nbr) series, finds t0 = first onpromotion=True date.
  3. Keeps the series only if it is a PROMO COLD-START:
        - no onpromotion=True anywhere before t0 (full visible history)
        - >= MIN_BASELINE_DAYS non-promo days with a record before t0
        - matched (same day-of-week) pre-t0 baseline median > 0
  4. Groups post-t0 promo days into EVENTS (consecutive promo days, gap <= EVENT_GAP
     days merged into one event). Requires >= MIN_PROMO_EVENTS events.
  5. Computes a stable post-hoc mu_i as the median per-event lift, plus a bootstrap CI.
  6. Flags left-censored series (first record within LEFT_CENSOR_DAYS of dataset start).
  7. Writes:
        - coldstart_series_table.csv  (one row per surviving series, with mu_i, CI, n_events, flags)
        - coldstart_health_report.txt (funnel: how many dropped at each stage, family mu_i dispersion)

Lift definition (matches research plan §1.2):
    For a promo event, event_sales = median(unit_sales over the event's promo days).
    baseline = median(pre-t0 unit_sales on the SAME day-of-week as the event's first day),
               falling back to overall pre-t0 median if same-dow sample is too small.
    event_lift = (event_sales - baseline) / baseline
    mu_i = median over the series' events ; CI via bootstrap over events.

Hygiene (matches §1 discussion):
    - negative unit_sales clipped to 0 (returns are not negative sales)
    - missing calendar days are NOT zero-filled (avoids deflating baseline -> fake huge lift)
    - baseline median must be > 0 or the series is dropped

Edit the CONFIG block, then:  python3 filter_coldstart_series.py
"""

import polars as pl
import numpy as np

# ----------------------------- CONFIG -----------------------------
TRAIN_CSV   = "train.csv"      # path to Favorita train.csv
ITEMS_CSV   = "items.csv"      # path to Favorita items.csv (for family dispersion check)
OUT_TABLE   = "coldstart_series_table.csv"
OUT_REPORT  = "coldstart_health_report.txt"

MIN_BASELINE_DAYS = 30   # >= this many non-promo recorded days before t0
MIN_PROMO_EVENTS  = 5    # >= this many post-t0 promo EVENTS to estimate stable mu_i
EVENT_GAP         = 1    # promo days <= this many days apart merge into one event
SAME_DOW_MIN      = 4    # need >= this many same-dow baseline points, else fall back to overall median
LEFT_CENSOR_DAYS  = 7    # first record within this many days of dataset start -> flag left-censored
N_BOOT            = 1000 # bootstrap resamples for mu_i CI
RNG_SEED          = 0
# ------------------------------------------------------------------


def merge_events(promo_dates_sorted, gap):
    """List of date objects (sorted) -> list of events, each a list of dates."""
    if not promo_dates_sorted:
        return []
    events, cur = [], [promo_dates_sorted[0]]
    for d in promo_dates_sorted[1:]:
        if (d - cur[-1]).days <= gap:
            cur.append(d)
        else:
            events.append(cur)
            cur = [d]
    events.append(cur)
    return events


def bootstrap_ci(values, n_boot, seed):
    """Bootstrap CI of the median of per-event lifts."""
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    meds = np.empty(n_boot)
    n = len(v)
    for b in range(n_boot):
        meds[b] = np.median(v[rng.integers(0, n, n)])
    return (float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5)))


def main():
    print("Scanning train.csv (lazy)...")
    lf = pl.scan_csv(TRAIN_CSV, try_parse_dates=False,
                      schema_overrides={"onpromotion": pl.Utf8, "date": pl.Utf8})

    # Normalize column presence / types. Favorita onpromotion can be true/false/empty.
    lf = lf.with_columns([
        pl.col("date").str.to_date("%Y-%m-%d", strict=False),
        pl.col("store_nbr").cast(pl.Int32),
        pl.col("item_nbr").cast(pl.Int32),
        pl.col("unit_sales").cast(pl.Float64).clip(lower_bound=0.0),  # clip negatives (returns) to 0
        # onpromotion in raw Favorita train.csv is a STRING column with many blanks/NaNs.
        # schema_overrides above forces it to stay Utf8 at scan time (polars would otherwise
        # auto-infer it as Boolean when it sees only True/False/blank, which then silently
        # breaks the .eq("True") comparison below). Map to boolean manually: anything not
        # exactly "True" (including blank/NaN) -> False.
        pl.col("onpromotion").eq("True").fill_null(False),
    ])

    dataset_start = lf.select(pl.col("date").min()).collect().item()
    print("Dataset start date:", dataset_start)

    # Per-series first-promo date (t0) and first-record date.
    print("Computing t0 and first-record per series...")
    per_series = (
        lf.group_by(["store_nbr", "item_nbr"])
          .agg([
              pl.col("date").filter(pl.col("onpromotion")).min().alias("t0"),
              pl.col("date").min().alias("first_record"),
              pl.col("onpromotion").any().alias("ever_promo"),
          ])
          .collect()
    )

    n_total = per_series.height
    n_ever_promo = per_series.filter(pl.col("ever_promo")).height
    survivors = per_series.filter(pl.col("ever_promo"))  # must have at least one promo

    # We need full per-(series) day-level data only for survivors; collect once, filtered.
    survivor_keys = survivors.select(["store_nbr", "item_nbr"]).unique()
    print(f"{n_total} series total; {n_ever_promo} ever promoted. Pulling day-level data for those...")

    day = (
        lf.join(survivor_keys.lazy(), on=["store_nbr", "item_nbr"], how="inner")
          .select(["store_nbr", "item_nbr", "date", "unit_sales", "onpromotion"])
          .collect()
    )

    t0_map = {(r["store_nbr"], r["item_nbr"]): r["t0"] for r in survivors.iter_rows(named=True)}
    firstrec_map = {(r["store_nbr"], r["item_nbr"]): r["first_record"] for r in survivors.iter_rows(named=True)}

    # Funnel counters
    drop = {"no_clean_preperiod": 0, "too_few_baseline_days": 0,
            "baseline_median_zero": 0, "too_few_events": 0}
    kept_rows = []

    rng_seed = RNG_SEED
    for (store, item), sub in day.group_by(["store_nbr", "item_nbr"]):
        store = int(store); item = int(item)
        t0 = t0_map[(store, item)]
        sub = sub.sort("date")

        pre = sub.filter(pl.col("date") < t0)
        post = sub.filter(pl.col("date") >= t0)

        # (a) clean pre-period: no promo before t0
        if pre.select(pl.col("onpromotion").any()).item():
            drop["no_clean_preperiod"] += 1
            continue

        # (b) enough baseline (non-promo, recorded) days before t0
        base_days = pre.height
        if base_days < MIN_BASELINE_DAYS:
            drop["too_few_baseline_days"] += 1
            continue

        # pre-period sales with day-of-week for matched baseline
        pre = pre.with_columns(pl.col("date").dt.weekday().alias("dow"))
        overall_base_median = float(pre.select(pl.col("unit_sales").median()).item())

        # (c) baseline median must be > 0
        if not (overall_base_median > 0):
            drop["baseline_median_zero"] += 1
            continue

        # same-dow baseline lookup table
        dow_med = {r["dow"]: r["med"] for r in
                   pre.group_by("dow").agg(pl.col("unit_sales").median().alias("med")).iter_rows(named=True)}
        dow_cnt = {r["dow"]: r["c"] for r in
                   pre.group_by("dow").agg(pl.len().alias("c")).iter_rows(named=True)}

        # promo events post t0
        promo_dates = (post.filter(pl.col("onpromotion"))
                          .select("date").to_series().to_list())
        promo_dates = sorted(promo_dates)
        events = merge_events(promo_dates, EVENT_GAP)
        if len(events) < MIN_PROMO_EVENTS:
            drop["too_few_events"] += 1
            continue

        # per-event lift
        sales_by_date = {r["date"]: r["unit_sales"] for r in sub.iter_rows(named=True)}
        event_lifts = []
        for ev in events:
            ev_sales = np.median([sales_by_date[d] for d in ev if d in sales_by_date])
            dow0 = ev[0].isoweekday()  # 1..7 ; polars dt.weekday() is also 1=Mon..7=Sun
            if dow_cnt.get(dow0, 0) >= SAME_DOW_MIN and dow_med.get(dow0, 0) > 0:
                base = dow_med[dow0]
            else:
                base = overall_base_median
            event_lifts.append((ev_sales - base) / base)

        mu = float(np.median(event_lifts))
        lo, hi = bootstrap_ci(event_lifts, N_BOOT, rng_seed)
        rng_seed += 1

        first_record = firstrec_map[(store, item)]
        left_censored = (first_record - dataset_start).days <= LEFT_CENSOR_DAYS

        kept_rows.append({
            "store_nbr": store, "item_nbr": item,
            "t0": t0, "first_record": first_record,
            "left_censored": left_censored,
            "n_baseline_days": base_days,
            "n_promo_events": len(events),
            "baseline_median": round(overall_base_median, 4),
            "mu_i": round(mu, 4),
            "mu_ci_lo": round(lo, 4), "mu_ci_hi": round(hi, 4),
            "ci_width": round(hi - lo, 4),
        })

    table = pl.DataFrame(kept_rows) if kept_rows else pl.DataFrame(
        schema={"store_nbr": pl.Int32, "item_nbr": pl.Int32})

    # attach family for dispersion check
    fam_disp_lines = []
    if table.height > 0:
        try:
            items = pl.read_csv(ITEMS_CSV).select(["item_nbr", "family"])
            table = table.join(items, on="item_nbr", how="left")
        except Exception as e:
            fam_disp_lines.append(f"(items.csv not joined: {e})")

    table.write_csv(OUT_TABLE)

    # ---------------- health report ----------------
    lines = []
    lines.append("COLD-START SERIES -- HEALTH REPORT")
    lines.append("=" * 50)
    lines.append(f"Config: MIN_BASELINE_DAYS={MIN_BASELINE_DAYS}, MIN_PROMO_EVENTS={MIN_PROMO_EVENTS}, "
                 f"EVENT_GAP={EVENT_GAP}, SAME_DOW_MIN={SAME_DOW_MIN}")
    lines.append("")
    lines.append("FUNNEL")
    lines.append(f"  total series                : {n_total}")
    lines.append(f"  ever promoted               : {n_ever_promo}")
    lines.append(f"  dropped: promo before t0    : {drop['no_clean_preperiod']}")
    lines.append(f"  dropped: < {MIN_BASELINE_DAYS} baseline days   : {drop['too_few_baseline_days']}")
    lines.append(f"  dropped: baseline median 0  : {drop['baseline_median_zero']}")
    lines.append(f"  dropped: < {MIN_PROMO_EVENTS} promo events    : {drop['too_few_events']}")
    lines.append(f"  >>> SURVIVING candidates    : {table.height}")
    lines.append("")
    if table.height > 0:
        lc = int(table.select(pl.col("left_censored").sum()).item())
        lines.append(f"  left-censored (flagged)     : {lc}  "
                     f"({100*lc/table.height:.1f}% of survivors -- first record near dataset start, "
                     f"upstream history unknown)")
        lines.append("")
        lines.append("mu_i DISTRIBUTION (survivors)")
        q = table.select([
            pl.col("mu_i").min().alias("min"), pl.col("mu_i").quantile(0.25).alias("q25"),
            pl.col("mu_i").median().alias("med"), pl.col("mu_i").quantile(0.75).alias("q75"),
            pl.col("mu_i").max().alias("max"), pl.col("ci_width").median().alias("med_ci_width"),
        ]).to_dicts()[0]
        for k, v in q.items():
            lines.append(f"    {k:14s}: {v:.4f}")
        lines.append("")
        lines.append("FAMILY-INTERNAL mu_i DISPERSION  (the key Step-3 check:")
        lines.append("  do series WITHIN the same family spread out, or collapse to one blob?)")
        if "family" in table.columns:
            fam = (table.group_by("family")
                        .agg([pl.len().alias("n"),
                              pl.col("mu_i").median().alias("mu_med"),
                              pl.col("mu_i").std().alias("mu_std"),
                              (pl.col("mu_i").quantile(0.75) - pl.col("mu_i").quantile(0.25)).alias("mu_iqr")])
                        .filter(pl.col("n") >= 3)
                        .sort("n", descending=True))
            lines.append(f"    {'family':22s} {'n':>4s} {'mu_med':>8s} {'mu_std':>8s} {'mu_iqr':>8s}")
            for r in fam.iter_rows(named=True):
                lines.append(f"    {str(r['family'])[:22]:22s} {r['n']:>4d} "
                             f"{r['mu_med']:>8.3f} {(r['mu_std'] or 0):>8.3f} {(r['mu_iqr'] or 0):>8.3f}")
            lines.append("")
            lines.append("  READ: large mu_std / mu_iqr within a family = real cross-series magnitude")
            lines.append("        variation to test. If they collapse near 0, the magnitude task is")
            lines.append("        not present at this granularity -- stop before spending API budget.")
        else:
            lines.extend("    " + x for x in fam_disp_lines)
    else:
        lines.append("  No surviving candidates -- loosen thresholds or inspect data.")

    report = "\n".join(lines)
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print("\n" + report)
    print(f"\nWrote {OUT_TABLE} ({table.height} rows) and {OUT_REPORT}")


if __name__ == "__main__":
    main()