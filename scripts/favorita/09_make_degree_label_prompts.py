# -*- coding: utf-8 -*-
"""
make_degree_label_prompts.py

Condition: degree_label  (Sec 3, Jang-et-al.-style qualitative rung).
Base: Prompts_series.jsonl.

Transformation: the numeric hint SENTENCE is replaced by a verbal degree
phrase derived from the SAME hint value, coarsening the series hint by one
rung. The output-format line ("Output only a single decimal number...") is
unchanged, so the model must still return a number -- letting us measure
degree-collapse (predictions forming discrete bands per label).

Bucketing reuses the mu_i sampling-band edges from Appendix A (0.15 / 0.65 /
2.0), so the buckets are an existing published design choice, not post-hoc
tuning. The thresholds are kept fixed across runs.

Note on what to compare against: named_entity DR ~0.53 (upper neighbour),
family hint DR ~0.23 (lower). degree_label should land between if the label
carries bucket-level rank info. near-copy is UNDEFINED (no number to copy) --
that absence is the point: the anchoring surface is the copyable number.

Output condition label: "degree_label".
"""
import argparse
from ladder_common import (read_jsonl, write_jsonl, find_hint_sentence,
                           replace_hint_sentence_whole)

BUCKETS = [   # (lower bound inclusive, phrase); checked top-down, first match wins
    (2.0,   "a very large, several-fold increase"),
    (0.65,  "a large increase"),
    (0.15,  "a moderate increase"),
    (0.0,   "a slight increase"),
    (-99.0, "a slight decrease"),
]

# The replacement sentence frame. Keep it declarative and parallel to the
# numeric version's role so only the number->phrase change differs.
SENTENCE = ("An estimate for THIS specific product's promotional lift is "
            "{phrase} relative to its baseline.")


def label_for(h):
    h = float(h)
    for lo, phrase in BUCKETS:
        if h >= lo:
            return phrase
    return BUCKETS[-1][1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="Prompts_series.jsonl")
    ap.add_argument("--out", dest="out", default="prompts_degree.jsonl")
    args = ap.parse_args()

    rows = read_jsonl(args.inp)
    out, counts = [], {}
    for r in rows:
        r = dict(r)
        tid = (r["store_nbr"], r["item_nbr"])
        phrase = label_for(r["family_hint"])
        counts[phrase] = counts.get(phrase, 0) + 1
        new_sentence = SENTENCE.format(phrase=phrase)
        r["prompt"] = replace_hint_sentence_whole(r["prompt"], new_sentence, tid)
        r["hint_original"] = r.get("family_hint")
        r["degree_label"] = phrase
        r["condition"] = "degree_label"
        out.append(r)

    write_jsonl(args.out, out)
    print(f"\nlabel distribution: {counts}")
    print("--- first transformed prompt ---")
    print(out[0]["prompt"])
    print("\n^^ read the hint sentence: it should be grammatical and end in "
          "a period before 'On the next day...'.")


if __name__ == "__main__":
    main()
