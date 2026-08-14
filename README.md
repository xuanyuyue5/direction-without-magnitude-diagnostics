# Direction Without Magnitude Diagnostics

This repository contains the diagnostic experiments for the first part of
**Direction Without Magnitude: Anti-Anchoring Injection for LLM-Based Cold-Start
Promotional Forecasting**. The code here covers the empirical findings up to
Section 4.6: the magnitude gap, the transmission-pipe effect, history-sensitivity
suppression, anchoring-surface controls, fusion headroom, and cross-model /
cross-dataset replication summaries.

The repository intentionally does **not** include the later method implementation
or the full raw retail datasets. Raw data should be obtained from the original
sources and placed outside version control.

## Repository Layout

```text
artifacts/
  prompts/       Prompt JSONL files and small task metadata used by the runs.
  responses/     Model response JSONL files used to reproduce the diagnostics.
  results/       Collapsed history-ablation tables and summary reports.

data/
  derived/       Small derived Favorita task lists and health reports.

figures/         Rendered figures used for quick inspection.

scripts/
  common/        Shared JSONL, prompt-editing, and provider helpers.
  favorita/      Favorita filtering, prompt construction, scoring, and headroom.
  history_ablation/
                 History-perturbation prompt generation, model runs, and analysis.
  figures/       Plotting scripts for the paper-style diagnostic figures.
```

## Setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The live model-call scripts read API keys only from environment variables:

```bash
export OPENAI_API_KEY=...
export DEEPSEEK_API_KEY=...
export ANTHROPIC_API_KEY=...
```

On Windows PowerShell, use `$env:OPENAI_API_KEY="..."` instead.

## Import Path

The scripts are kept as simple research scripts rather than a packaged Python
module. When running a script that imports `llm_providers` or `ladder_common`, add
`scripts/common` to `PYTHONPATH`:

```bash
export PYTHONPATH=scripts/common
```

On Windows PowerShell:

```powershell
$env:PYTHONPATH = "scripts/common"
```

## Reproducing The Main Diagnostics

The checked-in artifacts are enough to recompute the reported diagnostic metrics
and redraw the included figures.

Magnitude gap and transmission-pipe figure:

```bash
python scripts/figures/plot_intro.py \
  --responses artifacts/responses/responses.jsonl \
  --out-png figures/fig_intro.png \
  --out-pdf figures/fig_intro.pdf

python scripts/figures/plot_transmission_pipe.py \
  --responses-main artifacts/responses/responses.jsonl \
  --responses-series artifacts/responses/responses_series.jsonl \
  --out-png figures/fig_finding.png \
  --out-pdf figures/fig_finding.pdf
```

History-sensitivity suppression:

```bash
python scripts/history_ablation/03_analyze_history_ablation.py \
  --input artifacts/responses/history_ablation_responses.jsonl \
  --collapsed-out artifacts/results/history_ablation_collapsed.csv \
  --summary-out artifacts/results/history_ablation_summary.csv \
  --pairwise-out artifacts/results/history_ablation_pairwise.csv \
  --task-out artifacts/results/history_ablation_task_sensitivity.csv \
  --report-out artifacts/results/history_ablation_primary.txt

python scripts/figures/plot_history_ablation.py \
  --pairwise artifacts/results/history_ablation_pairwise.csv \
  --tasks artifacts/results/history_ablation_task_sensitivity.csv \
  --summary artifacts/results/history_ablation_summary.csv \
  --out-png figures/fig_history_ablation_revised.png \
  --out-pdf figures/fig_history_ablation_revised.pdf
```

Anchoring-surface controls:

```bash
python scripts/favorita/10_score_anchor_degree.py \
  --responses artifacts/responses/responses_anchor.jsonl \
  --condition anchoring_control

python scripts/favorita/10_score_anchor_degree.py \
  --responses artifacts/responses/responses_degree.jsonl \
  --condition degree_label \
  --prompts artifacts/prompts/prompts_degree.jsonl
```

Fusion headroom diagnostic:

```bash
python scripts/favorita/11_oracle_fusion_headroom.py
```

Cross-model and cross-dataset summaries:

```bash
python scripts/figures/plot_cross_model.py \
  --dir artifacts/responses \
  --prompts artifacts/prompts/prompts.jsonl \
  --out figures/fig_cross_model

python scripts/figures/plot_crossdataset_summary.py \
  --out-png figures/fig_crossdataset.png \
  --out-pdf figures/fig_crossdataset.pdf
```

## Regenerating Prompts And Responses

Prompt-generation scripts are included for transparency. They can be rerun after
placing raw Favorita data outside version control and passing paths explicitly.
The response-generation scripts default to a dry run; pass `--live` only when you
intend to call a model provider.

Example:

```bash
python scripts/favorita/05_run_llm.py \
  --provider openai \
  --prompts artifacts/prompts/prompts_series.jsonl \
  --output artifacts/responses/responses_series.jsonl \
  --live
```

## Data Notes

Full raw datasets are intentionally excluded. The small checked-in JSONL/CSV
artifacts are derived diagnostic inputs and outputs for reproducing the analyses
shown here. See `data/README.md` for more detail.

## Scope

This repository is a diagnostics-only release. It is meant to show the empirical
phenomena and analysis code behind Sections 4.1-4.6, not the later method or
system design experiments.
