# IRE Assignment 1 — Lexical & Semantic Retrieval on EB-NeRD and MIND

CS4.406 Information Retrieval & Extraction, Assignment 1. Build and evaluate a lexical (BM25)
+ semantic (embedding) candidate-generation pipeline for news click prediction on the EB-NeRD
and MIND datasets. Full spec: `A1.pdf`.

## Repo layout

```
data/                    (git-ignored) raw + processed datasets
notebooks/exploration/   reference EDA notebooks (schema exploration, memory strategy, a
                          popularity-baseline submission worked example) — not the pipeline
scripts/                 CLI entrypoints: download, build pipeline
src/ire_a1/              pipeline library code (clean, split, feature store, BM25, ANN, eval)
tests/                   incl. the anti-leakage test required by the assignment (Q9)
configs/                 paths / split parameters
```

## Setup

```bash
# uv (recommended -- this machine has no python3-venv system package installed, uv doesn't need it)
uv venv .venv
uv pip install -p .venv -r requirements.txt
source .venv/bin/activate

# or plain venv/pip, if python3-venv is available on your machine:
# python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

# Only on a GPU box (Kaggle/Colab), for BERT/XLM-R embeddings at large-bundle scale:
# uv pip install -p .venv -r requirements-gpu.txt
```

## Data

Small/demo tiers are for local development; the **large** tiers are required for Codabench
submission. Both fit comfortably on a normal disk (~7GB total, checked against real
`Content-Length` headers, not the "several GB each" guess in the PDF) and download fine locally
-- no Kaggle needed just to fetch/store them. Kaggle is only used for the one GPU-bound step in
this assignment (Q3: computing our own BERT/XLM-RoBERTa article embeddings); everything else,
including large-test-set prediction generation, runs locally via Polars/PyArrow batching.

```bash
# EB-NeRD (S3, no auth needed)
python scripts/download_ebnerd.py --tier demo
python scripts/download_ebnerd.py --tier small
python scripts/download_ebnerd.py --tier large   # ~5GB zipped, ~7GB extracted

# MIND (HuggingFace, dataset is *gated* -- one-time: accept terms on the dataset page at
# https://huggingface.co/datasets/yjw1029/MIND, create a token at
# https://huggingface.co/settings/tokens, then `huggingface-cli login` or export HF_TOKEN)
python scripts/download_mind.py --tier small
python scripts/download_mind.py --tier large-test   # required for Codabench, ~1.5GB
```

## Competition registration (manual, one-time)

- MIND: https://www.codabench.org/competitions/13967/
- RecSys 2024 Challenge (EB-NeRD): https://www.codabench.org/competitions/2469/

## Reproduce (Q1: data pipeline)

```bash
python scripts/build_pipeline.py   # defaults: EB-NeRD small + MIND small, configs/pipeline.yaml
```

Cleans both datasets into a unified schema (`src/ire_a1/schema.py`: articles / impressions /
history with identical columns, ids cast to string), does a **temporal** train/val/test split
per dataset (quantile-based cutoffs on impression timestamps, never random -- see
`src/ire_a1/split.py`), and writes the feature store to `data/processed/<dataset>/`:
`articles.parquet`, `user_history.parquet`, `impressions_{train,val,test}.parquet`.

Options: `--ebnerd-raw-dir`, `--mind-train-dir` / `--mind-dev-dir`, `--out-dir`, `--test-frac`,
`--val-frac`, `--skip-ebnerd`, `--skip-mind` (see `--help`). Defaults come from
`configs/pipeline.yaml`.

Run the tests (unit tests on synthetic data + integration checks against the downloaded
demo/small data, including a leakage assertion baked into `temporal_split` itself):

```bash
python -m pytest tests/ -v
```

*(Q2-Q6 land in later commits; this README is updated as each part is implemented)*
