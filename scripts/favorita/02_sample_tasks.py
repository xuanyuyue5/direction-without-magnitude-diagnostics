#!/usr/bin/env python3
"""
Stratified task sampler v2 for the cold-start magnitude experiment.

Adds a FAMILY CAP on top of mu_i stratification so the 100 tasks cover
7-8 families instead of being dominated by GROCERY I / BEVERAGES / PRODUCE.

Pipeline (no API, no LLM):
  1. Load coldstart_series_table.csv.
  2. Hard filter: baseline_median >= MIN_BASELINE.
  3. ci_excludes_zero flag.
  4. Stratify by mu_i into LOW / MID / HIGH / XHIGH (LOW band is CI-gated).
  5. Within each band, sample toward the band target BUT never let any single
     family exceed FAMILY_CAP across the whole 100-task set. When a family is
     capped, its slot flows to the next-eligible family in that band.
  6. Only families with >= FAMILY_MIN_POOL survivors are eligible to receive
     "overflow" coverage (don't force tiny families like HARDWARE with 7 rows).
  7. Write coldstart_tasks_100.csv + print mu_i spread and family spread.

Edit CONFIG, then:  python3 sample_tasks_v2.py
"""
import polars as pl
import random

# ----------------------------- CONFIG -----------------------------
TABLE         = "coldstart_series_table.csv"
OUT_TASKS     = "coldstart_tasks_100.csv"
MIN_BASELINE  = 2.0
RNG_SEED      = 42

BANDS = [
    ("LOW",   -0.10, 0.15, 30),
    ("MID",    0.15, 0.65, 35),
    ("HIGH",   0.65, 2.00, 25),
    ("XHIGH",  2.00, 1e9,  10),
]
CI_GATE_LOW_BAND = True

FAMILY_CAP      = 18    # no single family may exceed this many of the 100 tasks
FAMILY_MIN_POOL = 200   # a family must have >= this many survivors (post-baseline-filter)
                        # to be eligible; prevents forcing in tiny families
# ------------------------------------------------------------------


def main():
    df = pl.read_csv(TABLE)
    n0 = df.height
    print(f"Loaded {n0} survivors.")

    df = df.with_columns(
        (~((pl.col("mu_ci_lo") <= 0) & (pl.col("mu_ci_hi") >= 0))).alias("ci_excludes_zero")
    )
    df = df.filter(pl.col("baseline_median") >= MIN_BASELINE)
    print(f"After baseline >= {MIN_BASELINE}: {df.height} ({100*df.height/n0:.1f}% of pool)")

    if "family" not in df.columns:
        print("ERROR: no 'family' column — was items.csv joined in the filter step?")
        return

    # eligible families (enough pool to be worth covering)
    fam_pool = (df.group_by("family").agg(pl.len().alias("n")))
    eligible = set(fam_pool.filter(pl.col("n") >= FAMILY_MIN_POOL)["family"].to_list())
    print(f"Eligible families (pool >= {FAMILY_MIN_POOL}): {len(eligible)}")

    rng = random.Random(RNG_SEED)
    fam_count = {}          # family -> how many picked so far (global, across bands)
    picked_keys = set()     # (store,item) already picked
    picked_rows = []

    print("\nBAND SAMPLING (with family cap)")
    print(f"  {'band':6s} {'range':>16s} {'avail':>8s} {'target':>7s} {'taken':>6s}  note")

    for i, (name, lo, hi, target) in enumerate(BANDS):
        band = df.filter((pl.col("mu_i") >= lo) & (pl.col("mu_i") < hi))
        note = ""
        if name == "LOW" and CI_GATE_LOW_BAND:
            before = band.height
            band = band.filter(pl.col("ci_excludes_zero"))
            note = f"CI-gated ({before}->{band.height})"

        rows = band.to_dicts()
        rng.shuffle(rows)

        # Greedy pick: prefer rows whose family is not yet capped.
        # Two passes: first respect the cap strictly; if still short and only
        # capped families remain, we stop (cap is a hard ceiling).
        taken = 0
        # sort so that under-represented eligible families get picked first,
        # which spreads coverage; ties broken by the shuffle order already applied
        def fam_priority(r):
            f = r["family"]
            # lower current count = higher priority; ineligible families go last
            return (fam_count.get(f, 0), 0 if f in eligible else 1)
        rows.sort(key=fam_priority)

        for r in rows:
            if taken >= target:
                break
            key = (r["store_nbr"], r["item_nbr"])
            if key in picked_keys:
                continue
            f = r["family"]
            if fam_count.get(f, 0) >= FAMILY_CAP:
                continue
            picked_keys.add(key)
            picked_rows.append({**r, "band": name})
            fam_count[f] = fam_count.get(f, 0) + 1
            taken += 1

        if taken < target:
            note = (note + "  " if note else "") + f"SHORT by {target-taken} (family caps bind)"
        print(f"  {name:6s} {f'[{lo}, {hi})':>16s} {band.height:>8d} {target:>7d} {taken:>6d}  {note}")

    if not picked_rows:
        print("\nNo tasks sampled.")
        return

    tasks = pl.DataFrame(picked_rows)
    band_order = {b[0]: i for i, b in enumerate(BANDS)}
    tasks = (tasks
             .with_columns(pl.col("band").replace_strict(band_order, return_dtype=pl.Int32).alias("_bo"))
             .sort(["_bo", "mu_i"]).drop("_bo"))
    tasks.write_csv(OUT_TASKS)

    print(f"\nTOTAL TASKS: {tasks.height}")
    mu = tasks.select([
        pl.col("mu_i").min().alias("min"), pl.col("mu_i").quantile(0.25).alias("q25"),
        pl.col("mu_i").median().alias("med"), pl.col("mu_i").quantile(0.75).alias("q75"),
        pl.col("mu_i").max().alias("max"),
    ]).to_dicts()[0]
    print("  mu_i spread: " + "  ".join(f"{k}={v:.3f}" for k, v in mu.items()))
    lc = int(tasks.select(pl.col("left_censored").sum()).item())
    print(f"  left-censored: {lc} ({100*lc/tasks.height:.1f}%)")
    cz = int(tasks.select(pl.col("ci_excludes_zero").sum()).item())
    print(f"  CI-excludes-zero: {cz} ({100*cz/tasks.height:.1f}%)")

    fam = tasks.group_by("family").agg(pl.len().alias("n")).sort("n", descending=True)
    print(f"  family spread ({fam.height} families):")
    for r in fam.iter_rows(named=True):
        print(f"    {str(r['family'])[:24]:24s} {r['n']}")

    print(f"\nWrote {OUT_TASKS} ({tasks.height} rows).")


if __name__ == "__main__":
    main()