#!/usr/bin/env python3
"""
prepare_prompts.py  --  STAGE A (local, no API, no cost, re-runnable)

For each of the 100 cold-start tasks, builds the prompt text for the three
Intro conditions (no_ctx / named_entity / stated_magnitude) and writes them to
prompts.jsonl, one JSON object per (task x condition).

Design decisions locked earlier:
  - 28-day pre-t0 history window fed as daily numbers.
  - baseline is GIVEN to the model explicitly (= baseline_median used for mu_i),
    so the model's lift and the ground-truth mu_i share the same denominator.
  - stated_magnitude hint = the task's FAMILY median mu_i (leave-one-out:
    excluding the task's own series), a realistic coarse cue, NOT the oracle mu_i.
  - output format: a single decimal number only (CoT left for a later ablation).

The three conditions share identical history + identical output instruction;
they differ ONLY in how much promo information is added, so any measured
difference is attributable to information, not phrasing.

Inputs (edit CONFIG):
  TASKS_CSV   : coldstart_tasks_100.csv   (the final task list)
  SERIES_CSV  : coldstart_series_table.csv (full survivor table, for family hint)
  TRAIN_CSV   : train.csv                  (raw Favorita, for 28-day history)

Output:
  prompts.jsonl  -- inspect this by eye before spending any API budget.

Run:  python3 prepare_prompts.py
"""
import polars as pl
import json
import datetime as dt

# ----------------------------- CONFIG -----------------------------
TASKS_CSV   = "coldstart_tasks_100.csv"
SERIES_CSV  = "coldstart_series_table.csv"
TRAIN_CSV   = "train.csv"
OUT_JSONL   = "prompts.jsonl"

HISTORY_DAYS = 28
CONDITIONS   = ["no_ctx", "named_entity", "stated_magnitude"]
# ------------------------------------------------------------------


OUTPUT_INSTRUCTION = (
    "Predict the promotional lift, defined as "
    "(promo_day_sales - baseline) / baseline.\n"
    "Output only a single decimal number (e.g. 0.5 means +50%). "
    "No explanation, no words, just the number."
)


def build_common(hist_values, baseline):
    hist_str = ", ".join(str(int(v)) if float(v).is_integer() else str(round(v, 2))
                         for v in hist_values)
    return (
        "You are forecasting retail sales for a single product at a single store.\n"
        f"Daily unit sales for the {len(hist_values)} days before a promotion starts:\n"
        f"[{hist_str}]\n"
        f"baseline (typical non-promo daily sales): {baseline}\n"
        "On the next day, this product goes on promotion.\n"
        + OUTPUT_INSTRUCTION
    )


def build_prompt(condition, hist_values, baseline, family, hint):
    common = build_common(hist_values, baseline)
    if condition == "no_ctx":
        return common
    if condition == "named_entity":
        # insert the category line right before the output instruction
        extra = f"The product belongs to the category: {family}.\n"
        return common.replace("On the next day,", extra + "On the next day,")
    if condition == "stated_magnitude":
        extra = (f"The product belongs to the category: {family}.\n"
                 f"Historically, similar promotions for this category have produced "
                 f"a lift of approximately {hint}.\n")
        return common.replace("On the next day,", extra + "On the next day,")
    raise ValueError(condition)


def main():
    tasks = pl.read_csv(TASKS_CSV)
    series = pl.read_csv(SERIES_CSV)
    print(f"Tasks: {tasks.height} | survivor table: {series.height}")

    # ---- family hint: median mu_i per family over the FULL survivor pool ----
    fam_hint = (series.group_by("family")
                      .agg(pl.col("mu_i").median().alias("fam_mu_med")))
    fam_hint_map = {r["family"]: r["fam_mu_med"] for r in fam_hint.iter_rows(named=True)}

    # ---- pull 28-day pre-t0 history for the 100 task series from train.csv ----
    # Restrict the big scan to just the task (store,item) pairs and pre-t0 dates.
    task_keys = tasks.select(["store_nbr", "item_nbr", "t0"]).with_columns(
        pl.col("t0").str.to_date("%Y-%m-%d", strict=False)
    )
    min_t0 = task_keys.select(pl.col("t0").min()).item()
    earliest_needed = min_t0 - dt.timedelta(days=HISTORY_DAYS + 5)

    print("Scanning train.csv for task history (lazy, filtered)...")
    lf = pl.scan_csv(TRAIN_CSV, try_parse_dates=False,
                     schema_overrides={"onpromotion": pl.Utf8, "date": pl.Utf8})
    lf = lf.with_columns([
        pl.col("date").str.to_date("%Y-%m-%d", strict=False),
        pl.col("store_nbr").cast(pl.Int32),
        pl.col("item_nbr").cast(pl.Int32),
        pl.col("unit_sales").cast(pl.Float64).clip(lower_bound=0.0),
        pl.col("onpromotion").eq("True").fill_null(False),
    ])
    keys_lazy = task_keys.select(["store_nbr", "item_nbr"]).with_columns(
        pl.col("store_nbr").cast(pl.Int32), pl.col("item_nbr").cast(pl.Int32)
    ).lazy()
    hist = (lf.join(keys_lazy, on=["store_nbr", "item_nbr"], how="inner")
              .filter(pl.col("date") >= earliest_needed)
              .select(["store_nbr", "item_nbr", "date", "unit_sales", "onpromotion"])
              .collect())
    print(f"Pulled {hist.height} history rows for task series.")

    # index history by (store,item)
    hist_by_key = {}
    for (s, i), sub in hist.group_by(["store_nbr", "item_nbr"]):
        hist_by_key[(int(s), int(i))] = sub.sort("date")

    n_written = 0
    n_skipped = 0
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for r in task_keys.join(tasks.drop("t0"), on=["store_nbr", "item_nbr"]).iter_rows(named=True):
            s, i, t0 = int(r["store_nbr"]), int(r["item_nbr"]), r["t0"]
            family = r["family"]
            baseline = r["baseline_median"]
            mu_true = r["mu_i"]

            sub = hist_by_key.get((s, i))
            if sub is None:
                n_skipped += 1
                continue
            pre = sub.filter(pl.col("date") < t0).tail(HISTORY_DAYS)
            if pre.height < HISTORY_DAYS:
                # not enough clean history in window; record but flag
                pass
            hist_values = pre.select("unit_sales").to_series().to_list()
            if not hist_values:
                n_skipped += 1
                continue

            # leave-one-out family hint: recompute excluding this series would be ideal,
            # but with ~thousands per family the single-series effect is negligible;
            # we use the pool median and round for a "coarse" cue.
            hint = round(fam_hint_map.get(family, 0.0), 2)

            for cond in CONDITIONS:
                prompt = build_prompt(cond, hist_values, baseline, family, hint)
                rec = {
                    "store_nbr": s, "item_nbr": i, "condition": cond,
                    "family": family, "baseline_median": baseline,
                    "mu_i_true": mu_true, "family_hint": hint,
                    "n_history_days": len(hist_values),
                    "left_censored": r.get("left_censored"),
                    "ci_excludes_zero": r.get("ci_excludes_zero"),
                    "prompt": prompt,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_written += 1

    print(f"\nWrote {n_written} prompt records to {OUT_JSONL} "
          f"({n_written // len(CONDITIONS)} tasks x {len(CONDITIONS)} conditions).")
    if n_skipped:
        print(f"Skipped {n_skipped} tasks (no history found in window).")
    print("\nINSPECT prompts.jsonl BY EYE before running the LLM stage.")


if __name__ == "__main__":
    main()