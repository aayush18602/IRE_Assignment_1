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
submission (multi-GB) and are meant to be downloaded/run on Kaggle, not on this machine — see
`ai_usage_log.md` / assignment PDF for the compute rationale.

```bash
# EB-NeRD (S3, no auth needed)
python scripts/download_ebnerd.py --tier demo
python scripts/download_ebnerd.py --tier small
python scripts/download_ebnerd.py --tier large   # Kaggle: multi-GB, needs disk

# MIND (HuggingFace, dataset is *gated* -- one-time: accept terms on the dataset page at
# https://huggingface.co/datasets/yjw1029/MIND, create a token at
# https://huggingface.co/settings/tokens, then `huggingface-cli login` or export HF_TOKEN)
python scripts/download_mind.py --tier small
python scripts/download_mind.py --tier large-test   # Kaggle: required for Codabench
```

## Competition registration (manual, one-time)

- MIND: https://www.codabench.org/competitions/13967/
- RecSys 2024 Challenge (EB-NeRD): https://www.codabench.org/competitions/2469/

## Reproduce

```bash
python scripts/build_pipeline.py   # one command: raw data -> cleaned -> split -> feature store
```

*(pipeline script lands with Q1; this README is updated as each part is implemented)*
