#!/usr/bin/env python3
"""
make_series_hint_prompts.py  (local, no API)
Generates ONE new condition `stated_magnitude_series` for the 100 tasks:
identical to stated_magnitude but the hint is a SERIES-LEVEL, distinct-per-series
estimate (mu_true + noise, corr~0.62, clipped to >= -0.9), instead of the flat
family-median hint. Writes prompts_series.jsonl for run_llm.py.
"""
import json, random, math, statistics as st

PROMPTS_IN  = "prompts.jsonl"
OUT         = "prompts_series.jsonl"
TARGET_CORR = 0.65
CLIP_MIN    = -0.9
SEED        = 20240624

recs=[json.loads(l) for l in open(PROMPTS_IN,encoding="utf-8")]
# base body from no_ctx (history + baseline + output instruction), per task
base={}
for r in recs:
    if r["condition"]=="no_ctx":
        base[(r["store_nbr"],r["item_nbr"])]=r
ks=list(base.keys())
mu={k:base[k]["mu_i_true"] for k in ks}
mu_vals=[mu[k] for k in ks]

rng=random.Random(SEED)
sd_mu=st.pstdev(mu_vals)
sd_noise=sd_mu*math.sqrt(1/TARGET_CORR**2 - 1)
hint={k: max(CLIP_MIN, round(mu[k]+rng.gauss(0,sd_noise),2)) for k in ks}

def build(b, h):
    # reuse the no_ctx prompt, inject category + series hint before the promo line
    fam=b["family"]
    extra=(f"The product belongs to the category: {fam}.\n"
           f"An estimate for THIS specific product's promotional lift is "
           f"approximately {h}.\n")
    return b["prompt"].replace("On the next day,", extra+"On the next day,")

n=0
with open(OUT,"w",encoding="utf-8") as f:
    for k in ks:
        b=base[k]
        rec={**{x:b[x] for x in ["store_nbr","item_nbr","family","baseline_median",
                                  "mu_i_true","left_censored","ci_excludes_zero"]},
             "condition":"stated_magnitude_series",
             "family_hint":hint[k],          # now the series-level hint
             "n_history_days":b["n_history_days"],
             "prompt":build(b,hint[k])}
        f.write(json.dumps(rec,ensure_ascii=False)+"\n"); n+=1
print(f"wrote {n} prompts to {OUT}")
# report realized corr after clipping
def corr(a,b):
    nn=len(a);ma=sum(a)/nn;mb=sum(b)/nn
    num=sum((a[i]-ma)*(b[i]-mb) for i in range(nn))
    den=math.sqrt(sum((x-ma)**2 for x in a)*sum((y-mb)**2 for y in b))
    return num/den if den else float('nan')
H=[hint[k] for k in ks]
print(f"realized corr(hint,mu) after clip = {corr(H,mu_vals):.3f}")