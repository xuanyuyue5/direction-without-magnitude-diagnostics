# -*- coding: utf-8 -*-
"""
ladder_common.py  --  helpers for prompt-ladder control conditions
  anchoring_control / degree_label / optional instructed_injection

Shared row schema used by the Favorita prompt/response pipeline:
  row keys: store_nbr, item_nbr, family, baseline_median, mu_i_true,
            left_censored, ci_excludes_zero, condition, family_hint,
            [injected_cue], n_history_days, prompt

The hint always appears in the prompt as exactly one sentence:
    "An estimate for THIS specific product's promotional lift is approximately {V}."
so we locate/replace via that sentence, not by raw number search.

Number rendering in the source prompts is mostly %.2f with a %g minority; we
reproduce the source rendered string's format when substituting a value, so
prompts stay stylistically identical.
"""

import json
import re
from pathlib import Path

HINT_SENTENCE_RE = re.compile(
    r"An estimate for THIS specific product's promotional lift is approximately "
    r"(-?\d+\.?\d*)\."
)


def task_key(r):
    return (r["store_nbr"], r["item_nbr"])


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"[{path}] line {ln}: bad JSON ({e})")
    if not rows:
        raise SystemExit(f"[{path}] empty file")
    return rows


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows -> {path}")


def find_hint_sentence(prompt, task_id="?"):
    m = HINT_SENTENCE_RE.search(prompt)
    if not m:
        raise SystemExit(
            f"[task {task_id}] hint sentence not found. This script assumes the "
            f"series/naive template. First 300 chars:\n{prompt[:300]}")
    return m, m.group(1)


def rendered_format(rendered_string):
    if "." in rendered_string:
        decimals = len(rendered_string.split(".")[1])
        return lambda v: f"{v:.{decimals}f}"
    return lambda v: f"{v:g}"


def replace_hint_value(prompt, new_value, task_id="?"):
    """Replace the numeric hint with new_value in the SAME format."""
    m, rendered = find_hint_sentence(prompt, task_id)
    fmt = rendered_format(rendered)
    new_rendered = fmt(float(new_value))
    start, end = m.span(1)
    return prompt[:start] + new_rendered + prompt[end:]


def replace_hint_sentence_whole(prompt, new_sentence, task_id="?"):
    """Replace the ENTIRE hint sentence (through its period) with new_sentence."""
    m, _ = find_hint_sentence(prompt, task_id)
    return prompt[:m.start()] + new_sentence + prompt[m.end():]
