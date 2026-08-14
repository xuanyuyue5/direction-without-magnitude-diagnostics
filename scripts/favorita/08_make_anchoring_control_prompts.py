# -*- coding: utf-8 -*-
"""
make_anchoring_control_prompts.py

Condition: anchoring_control  (Sec 3 "planned" rung, after Valencia-Clavijo).
Base: Prompts_series.jsonl  (probe series-hint template; this rung is Sec 3/4).

Transformation: identical prompt, hint number replaced by a single absurd
constant (default 3.0 = +300%), rendered in the SAME numeric format.

Reading of results:
  - high Copy(0.05) w.r.t. 3.0 / predictions piling near 3.0 = the model
    relays whatever occupies the cue slot even when absurd (anti-rational-copy);
  - report K_dir, mean/median pred, share > 1.0 / > 2.0; DR ~ 0 (constant cue).

Output condition label: "anchoring_control".
"""
import argparse
from ladder_common import read_jsonl, write_jsonl, replace_hint_value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="Prompts_series.jsonl")
    ap.add_argument("--out", dest="out", default="prompts_anchor.jsonl")
    ap.add_argument("--anchor", type=float, default=3.0,
                    help="absurd anchor on lift scale (3.0 = +300%%)")
    args = ap.parse_args()

    rows = read_jsonl(args.inp)
    out = []
    for r in rows:
        r = dict(r)
        tid = (r["store_nbr"], r["item_nbr"])
        r["prompt"] = replace_hint_value(r["prompt"], args.anchor, tid)
        r["hint_original"] = r.get("family_hint")
        r["family_hint"] = args.anchor      # so Run_llm records the injected val
        r["injected_cue"] = args.anchor
        r["condition"] = "anchoring_control"
        out.append(r)

    write_jsonl(args.out, out)
    print(f"\nanchor = {args.anchor}")
    print("--- first transformed prompt ---")
    print(out[0]["prompt"])


if __name__ == "__main__":
    main()
