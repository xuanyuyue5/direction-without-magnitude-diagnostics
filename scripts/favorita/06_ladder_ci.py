"""Task-level paired bootstrap CIs for all ladder DR point estimates
and paired tests for adjacent rungs. 10^4 resamples, seed fixed.

Inputs : responses.jsonl (no_ctx / named_entity / stated_magnitude),
         responses_series.jsonl (stated_magnitude_series)
Output : per-condition DR with 95% CI; paired Delta-DR with 95% CI and
         a bootstrap two-sided p-ish criterion (CI excludes 0).
"""
import json, numpy as np
from collections import defaultdict
from scipy.stats import spearmanr

rng = np.random.default_rng(0)
B = 10_000

def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]

rows  = load('responses.jsonl') + load('responses_series.jsonl')
byct  = defaultdict(lambda: defaultdict(list))   # cond -> task -> [preds]
mu, hint = {}, {}
for x in rows:
    k = (x['store_nbr'], x['item_nbr'])
    byct[x['condition']][k].append(x['lift_pred'])
    mu[k] = x['mu_i_true']
    if x['condition'] == 'stated_magnitude_series':
        hint[k] = x['family_hint']          # series-level hint stored here
    elif x['condition'] == 'stated_magnitude':
        pass

# family hint value: take from stated_magnitude rows' family_hint field
fam_hint = {}
for x in load('responses.jsonl'):
    if x['condition'] == 'stated_magnitude':
        fam_hint[(x['store_nbr'], x['item_nbr'])] = x['family_hint']

tasks = sorted(mu)
N = len(tasks)
truth = np.array([mu[k] for k in tasks])

cols = {}
for c in ['no_ctx', 'named_entity', 'stated_magnitude', 'stated_magnitude_series']:
    cols[c] = np.array([np.median(byct[c][k]) for k in tasks])
cols['series_hint_itself'] = np.array([hint[k]     for k in tasks])
cols['family_hint_itself'] = np.array([fam_hint[k] for k in tasks])
# parameter-free rank-average fusion of autonomous (named_entity) and series hint
def ranks(v):
    from scipy.stats import rankdata
    return rankdata(v)
cols['rank_avg_fusion'] = (ranks(cols['named_entity']) + ranks(cols['series_hint_itself'])) / 2

def dr(pred, y):
    return spearmanr(pred, y).statistic

# point estimates
point = {c: dr(v, truth) for c, v in cols.items()}

# paired bootstrap: one index draw per replicate, applied to every condition
idx = rng.integers(0, N, size=(B, N))
boot = {c: np.empty(B) for c in cols}
for b in range(B):
    ii = idx[b]
    yb = truth[ii]
    for c, v in cols.items():
        boot[c][b] = dr(v[ii], yb)

def ci(a): 
    return np.percentile(a, [2.5, 97.5])

print(f'{"condition":28s}  DR      95% CI')
for c in ['no_ctx','named_entity','family_hint_itself','stated_magnitude',
          'series_hint_itself','stated_magnitude_series','rank_avg_fusion']:
    lo, hi = ci(boot[c])
    print(f'{c:28s}  {point[c]:+.3f}  [{lo:+.3f}, {hi:+.3f}]')

pairs = [
    ('named_entity','no_ctx',                 'category label vs no context'),
    ('stated_magnitude','named_entity',       'family hint vs category label (non-monotonicity)'),
    ('stated_magnitude','family_hint_itself', 'family-hint output vs its cue (clamping)'),
    ('stated_magnitude_series','series_hint_itself','series-hint output vs its cue (Spearman increment)'),
    ('stated_magnitude_series','named_entity','series injection vs autonomous'),
    ('rank_avg_fusion','stated_magnitude_series','rank-avg fusion vs naive injection'),
    ('rank_avg_fusion','named_entity',        'rank-avg fusion vs autonomous'),
]
print()
print(f'{"paired difference":52s}  Delta   95% CI          excl. 0')
for a, b_, label in pairs:
    d = boot[a] - boot[b_]
    lo, hi = ci(d)
    excl = 'YES' if (lo > 0 or hi < 0) else 'no'
    print(f'{label:52s}  {point[a]-point[b_]:+.3f}  [{lo:+.3f}, {hi:+.3f}]  {excl}')
