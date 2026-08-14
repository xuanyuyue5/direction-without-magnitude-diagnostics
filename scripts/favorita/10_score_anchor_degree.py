#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
score_anchor_degree.py -- score the anchoring-control and degree-label conditions.
  * per-task median lift over repeated calls
  * parse_ok filter
  * mu_i_true as truth
  * DR = Spearman(pred, mu), K_dir = mean(sign(pred)==sign(mu))

Usage:
  python score_anchor_degree.py --responses responses_anchor.jsonl \
          --condition anchoring_control --anchor 3.0
  python score_anchor_degree.py --responses responses_degree.jsonl \
          --condition degree_label --prompts prompts_degree.jsonl
"""
import argparse, json, sys
import numpy as np
from scipy.stats import spearmanr, pearsonr

COND_OF = {
    "anchoring_control": "anchoring_control",
    "degree_label": "degree_label",
}


def load(path, cond):
    per = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("condition") != cond:
                continue
            if not r.get("parse_ok"):
                continue
            k = (r["store_nbr"], r["item_nbr"])
            d = per.setdefault(k, {"mu": r["mu_i_true"],
                                   "inj": r.get("family_hint"),
                                   "label": r.get("degree_label"),
                                   "preds": []})
            d["preds"].append(float(r["lift_pred"]))
    return per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", required=True)
    ap.add_argument("--condition", required=True, choices=list(COND_OF))
    ap.add_argument("--anchor", type=float, default=3.0)
    ap.add_argument("--prompts", default=None,
                    help="degree_label only: prompts_degree.jsonl, to recover "
                         "labels (Run_llm.py does not carry degree_label through)")
    args = ap.parse_args()

    cond = COND_OF[args.condition]
    per = load(args.responses, cond)
    if not per:
        sys.exit(f"no '{cond}' rows with parse_ok in {args.responses}")
    keys = sorted(per.keys())
    mu = np.array([per[k]["mu"] for k in keys], float)
    pred = np.array([np.median(per[k]["preds"]) for k in keys], float)
    nseed = np.mean([len(per[k]["preds"]) for k in keys])
    print(f"[score] {len(keys)} tasks, median over {nseed:.1f} seeds\n")

    dr = spearmanr(pred, mu)[0]
    slope = np.polyfit(mu, pred, 1)[0]
    kdir = float(np.mean(np.sign(pred) == np.sign(mu)))

    print("=" * 54)
    print(f"{args.condition}  --  GPT-4o")
    print("=" * 54)
    print(f"  DR (Spearman vs mu)   = {dr:+.3f}")
    print(f"  slope (OLS)           = {slope:+.3f}")
    print(f"  K_dir                 = {kdir:.3f}   [base rate 0.90]")

    if args.condition == "anchoring_control":
        anchor = np.full_like(pred, args.anchor)
        copy = float(np.mean(np.abs(pred - anchor) <= 0.05))
        gt1 = float(np.mean(pred > 1.0))
        gt2 = float(np.mean(pred > 2.0))
        print(f"  Copy(0.05) w.r.t. {args.anchor} = {copy:.3f}")
        print(f"  share pred > 1.0      = {gt1:.3f}")
        print(f"  share pred > 2.0      = {gt2:.3f}")
        print(f"  pred median / mean    = {np.median(pred):+.3f} / {np.mean(pred):+.3f}")
        print("=" * 54)
        print("\n[reading]")
        print(f"  Autonomous predictions sit ~0.1-0.5. Any mass near {args.anchor} "
              f"is anchoring toward an absurd value (Valencia-Clavijo). High copy "
              f"or large share>2.0 = anti-rational-copying: the model relays the "
              f"slot content, not a value it judges plausible.")

    elif args.condition == "degree_label":
        # Run_llm.py does not carry degree_label into responses; recover it from
        # the prompts file (falls back to any label present in responses).
        label_map = {}
        if args.prompts:
            for r in (json.loads(l) for l in open(args.prompts, encoding="utf-8")
                      if l.strip()):
                label_map[(r["store_nbr"], r["item_nbr"])] = r.get("degree_label")
        labels = np.array([label_map.get(k) or per[k]["label"] for k in keys])
        if any(x is None for x in labels):
            sys.exit("missing degree labels; pass --prompts prompts_degree.jsonl")
        print(f"  Copy(0.05)            = N/A (no number in prompt -- the point)")
        print("=" * 54)
        print("\n[per-label prediction distribution -- degree-collapse check]")
        order = ["a slight decrease", "a slight increase", "a moderate increase",
                 "a large increase", "a very large, several-fold increase"]
        seen = list(dict.fromkeys(list(labels)))
        for lab in order + [x for x in seen if x not in order]:
            mask = labels == lab
            if mask.sum() == 0:
                continue
            vs = np.sort(pred[mask])
            q1 = float(np.percentile(vs, 25))
            q3 = float(np.percentile(vs, 75))
            print(f"  {lab:<40} n={int(mask.sum()):<3} "
                  f"median={np.median(vs):+.3f}  IQR=[{q1:+.3f},{q3:+.3f}]")
        print("\n[reading]")
        print("  Tight, well-separated per-label medians with small IQR = "
              "quantitative degree-collapse: the model maps a coarse verbal "
              "label to a near-constant number per bucket. Between named_entity "
              "(0.53) and family hint (0.23) on DR is expected.")


if __name__ == "__main__":
    main()
