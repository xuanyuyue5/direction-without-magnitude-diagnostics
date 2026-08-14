#!/usr/bin/env python3
"""
run_llm.py -- model-call stage for lift-prediction prompts.

Reads a prompt JSONL file, queries the selected provider `N_REPEAT` times per
prompt, parses the first numeric lift in each response, and appends every result
immediately to the output JSONL file.

Operational safeguards:
  - Dry-run mode is the default. Use `--live` for real provider calls.
  - API keys are read only from environment variables and are never hardcoded.
  - Runs are resumable: existing (store, item, condition, seed) records are
    skipped, and each new record is flushed immediately.
  - Parse failures are recorded with `lift_pred=null` and excluded by scorers.
  - Transient provider errors use exponential-backoff retries.

Examples:
  python 05_run_llm.py
  python 05_run_llm.py --live --provider openai
  python 05_run_llm.py --prompts prompts_anchor.jsonl --output responses_anchor.jsonl --live
"""
import os
import re
import sys
import json
import time
import random
import argparse

from llm_providers import (
    call_model, make_client, default_model, out_path_for, PROVIDER_SUFFIX,
)

# ----------------------------- CONFIG -----------------------------
# Defaults chosen so the ORIGINAL behaviour is reproduced when you pass no new
# flags except --provider. prompts/output are overridable on the CLI because
# this script is run twice (named_entity file and series file).
PROMPTS_JSONL = "prompts_series.jsonl"      # override with --prompts
OUT_LIVE      = "responses_series.jsonl"    # override with --output
OUT_DRYRUN    = "responses_DRYRUN.jsonl"
N_REPEAT      = 3            # queries per prompt (median taken later)
TEMPERATURE   = 0.7          # >0 so repeats actually vary; median stabilizes
MAX_TOKENS    = 16           # single number; keep tiny
MAX_RETRIES   = 5
BACKOFF_BASE  = 2.0          # seconds
# ------------------------------------------------------------------


NUM_RE = re.compile(r'[-+]?\d*\.?\d+')

def parse_lift(text):
    """Extract the first numeric value from the model's text. None if none found."""
    if text is None:
        return None
    m = NUM_RE.search(text.replace(",", ""))
    if not m:
        return None
    try:
        return float(m.group())
    except ValueError:
        return None


def load_done(path):
    """Set of (store,item,condition,seed) already present, for resume."""
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add((r["store_nbr"], r["item_nbr"], r["condition"], r["seed"]))
                except Exception:
                    continue
    return done


def mock_response(prompt, seed):
    """Deterministic dry-run stand-in for parser and resume tests."""
    rng = random.Random(hash((prompt, seed)) & 0xffffffff)
    val = round(rng.uniform(-0.2, 1.2), 2)
    if rng.random() < 0.1:
        return f"approximately {val}"
    if rng.random() < 0.03:
        return "no numeric estimate"
    return str(val)


def main():
    ap = argparse.ArgumentParser(description="STAGE B: query a model, parse a lift per response.")
    ap.add_argument("--live", action="store_true", help="make live provider API calls")
    ap.add_argument("--provider", choices=["openai", "deepseek", "anthropic"],
                    default="openai", help="which model provider to call")
    ap.add_argument("--prompts", default=PROMPTS_JSONL,
                    help="input prompts jsonl (default: %(default)s)")
    ap.add_argument("--output", default=None,
                    help="live output path; default derived from OUT_LIVE + provider suffix")
    ap.add_argument("--model", default=None,
                    help="override model string (default: provider's default)")
    args = ap.parse_args()

    live = args.live
    provider = args.provider
    model = args.model or default_model(provider)
    prompts_path = args.prompts

    # output: openai keeps the bare name; deepseek/anthropic get a suffix so
    # GPT-4o files are never touched. --output overrides everything.
    if live:
        base_out = args.output or out_path_for(OUT_LIVE, provider)
    else:
        base_out = out_path_for(OUT_DRYRUN, provider)
    out_path = base_out

    if not os.path.exists(prompts_path):
        sys.exit(f"Missing {prompts_path}. Run prepare_prompts.py first.")

    prompts = [json.loads(l) for l in open(prompts_path, encoding="utf-8")]
    print(f"Loaded {len(prompts)} prompts from {prompts_path}. "
          f"Mode: {'LIVE API calls' if live else 'DRY RUN'}")
    print(f"Provider={provider}  Model={model}  N_REPEAT={N_REPEAT}  "
          f"-> up to {len(prompts)*N_REPEAT} calls")
    print(f"Output: {out_path}")
    if provider == "anthropic":
        print("Note: Anthropic has no seed param; repeats vary via temperature only.")

    client = None
    if live:
        client = make_client(provider)

    done = load_done(out_path)
    if done:
        print(f"Resuming: {len(done)} (task,condition,seed) results already on disk, skipping those.")

    n_new = n_parsefail = 0
    t_start = time.time()
    with open(out_path, "a", encoding="utf-8") as f:
        for p in prompts:
            for seed in range(N_REPEAT):
                key = (p["store_nbr"], p["item_nbr"], p["condition"], seed)
                if key in done:
                    continue

                if live:
                    raw, err = call_model(
                        client, provider, prompt=p["prompt"], seed=seed,
                        model=model, temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
                        max_retries=MAX_RETRIES, backoff_base=BACKOFF_BASE,
                    )
                    if err and "AUTH" in err:
                        sys.exit(f"\n{err}\nStopping. Fix the key and re-run --live to resume.")
                else:
                    raw, err = mock_response(p["prompt"], seed), None

                lift = parse_lift(raw)
                if lift is None:
                    n_parsefail += 1

                rec = {
                    "store_nbr": p["store_nbr"], "item_nbr": p["item_nbr"],
                    "condition": p["condition"], "seed": seed,
                    "provider": provider, "model": model,
                    "family": p["family"], "baseline_median": p["baseline_median"],
                    "mu_i_true": p["mu_i_true"], "family_hint": p["family_hint"],
                    "left_censored": p.get("left_censored"),
                    "ci_excludes_zero": p.get("ci_excludes_zero"),
                    "lift_pred": lift, "raw": raw,
                    "parse_ok": lift is not None, "error": err,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()                      # <- flush every record: crash-safe
                n_new += 1

    dt = time.time() - t_start
    print(f"\nDone. Wrote {n_new} new results to {out_path} in {dt:.1f}s.")
    print(f"Parse failures: {n_parsefail} ({100*n_parsefail/max(n_new,1):.1f}%) "
          f"(recorded with lift_pred=null, excluded from main results).")
    if not live:
        print(f"\nThis was a dry run. Inspect {out_path}, then add --live for provider calls.")


if __name__ == "__main__":
    main()