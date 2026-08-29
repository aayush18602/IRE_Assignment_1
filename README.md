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

## Reproduce (Q2: BM25 lexical retrieval)

```bash
python scripts/run_bm25.py --dataset ebnerd
python scripts/run_bm25.py --dataset mind
```

Builds an inverted index (`src/ire_a1/bm25.py`: hand-rolled Okapi BM25 over article title +
abstract, not a library wrapper) and, for every impression in the chosen split (default
`test`), constructs a query from the user's `--n-recent` (default 10) most recently clicked
article titles **as of that impression's timestamp** (`feature_store.recent_history_asof`, so
no future click ever leaks into the query), retrieves the top-K candidates, and reports
recall@K for K in `--k-values` (default `50,100,200`). Writes `bm25/recall_<split>.json` and
`bm25/candidates_<split>.parquet` (the actual top-K retrieved ids + scores, for reuse in Q3/Q4)
under `data/processed/<dataset>/`. Use `--limit N` for a quick smoke test before a full run.

## Reproduce (Q3: semantic/embedding retrieval)

Split across two environments, since embedding computation is the one GPU-bound step in this
assignment (see the compute-strategy discussion above / `ai_usage_log.md`):

```bash
# 1. On Kaggle (GPU): pip install -r requirements.txt -r requirements-gpu.txt, then
python scripts/compute_embeddings.py --dataset ebnerd   # xlm-roberta-base by default, both datasets
python scripts/compute_embeddings.py --dataset mind
# -> copy the resulting data/processed/<dataset>/embeddings.parquet back to this machine

# 2. Locally (CPU): ANN index + recall@K, same metric/format as Q2 so they're comparable
python scripts/run_embeddings.py --dataset ebnerd
python scripts/run_embeddings.py --dataset mind

# 3. Locally: side-by-side comparison table (Q3.5)
python scripts/compare_retrieval.py --dataset ebnerd
```

`compute_embeddings.py` mean-pools XLM-RoBERTa token embeddings over title+abstract (one
multilingual model for both datasets, not two dataset-specific ones -- it natively handles
EB-NeRD's Danish and MIND's English, keeping the downstream ANN/eval code dataset-agnostic like
the rest of the pipeline). `run_embeddings.py` builds a FAISS flat (exact) index
(`src/ire_a1/ann.py`) and represents each user as the mean-pooled embedding of their recent
click history -- same leak-safe `recent_history_asof` cutoff as Q2, and the same
`candidate_eval.recall_at_k` metric, so BM25 and embedding results are directly comparable.
Verified locally end-to-end (including `compute_embeddings.py` itself) on a small CPU smoke run
before handing the real run off to Kaggle.

## Results so far (Q2 vs Q3, full test splits)

See `results/comparison.md` for the full table + discussion. Two embedding variants were run:
raw `xlm-roberta-base` (no similarity fine-tuning) and `sentence-transformers/paraphrase-
multilingual-mpnet-base-v2` (same architecture, fine-tuned for semantic similarity). Headline:
fine-tuning roughly doubles recall@200 either way, and the lexical-vs-semantic winner flips by
*language* -- BM25 wins on EB-NeRD (Danish), the fine-tuned model wins on MIND (English),
likely because that model's fine-tuning data was English-centric before multilingual
distillation.

| Dataset | recall@200 BM25 | recall@200 XLM-R (raw) | recall@200 mpnet (fine-tuned) |
|---|---|---|---|
| EB-NeRD | **2.01%** | 0.95% | 1.69% |
| MIND | 1.43% | 0.37% | **2.27%** |

## Reproduce (Q4: offline evaluation harness)

```bash
python scripts/run_eval.py --dataset ebnerd --method bm25
python scripts/run_eval.py --dataset ebnerd --method embeddings
# a second embedding variant (e.g. the mpnet ablation) needs its own --embeddings-file/--candidates-dir:
python scripts/run_eval.py --dataset ebnerd --method embeddings \
    --embeddings-file embeddings_mpnet.parquet --candidates-dir data/processed/ebnerd/embeddings_eval_mpnet
```

Two kinds of metric, computed differently on purpose:

- **AUC, MRR, nDCG@5, nDCG@10** -- computed the way the official MIND/EB-NeRD leaderboards do
  it: re-rank each impression's own *shown* candidates (`article_ids_inview` / MIND's
  `impressions` field, already in `impressions_<split>.parquet` from Q1) by BM25/embedding
  score, compare against the binary click label (`src/ire_a1/eval.py:ranking_metrics`, using
  sklearn's `roc_auc_score`/`ndcg_score` -- reference implementations, not reimplemented, since
  the assignment's "build it yourself" instruction was specifically about Q2's inverted index).
- **Diversity, novelty, coverage** -- computed over each method's actual top-K *recommended*
  list (Q2/Q3's saved `candidates_<split>.parquet`), since these are about the system's output
  as a whole, not a re-ranking of a handful of given candidates. Diversity = fraction of unique
  categories in a list (comparable across BM25/embeddings without needing embeddings for BM25
  candidates). Novelty = mean self-information vs. TRAIN-only popularity (never test, to avoid
  leakage). Coverage = fraction of the catalog touched by any recommendation.
- **Slicing**: cold vs. warm users, split at a **percentile** (default 25th) of each dataset's
  own `history_length` distribution, not a fixed count -- EB-NeRD's small/demo tiers are
  pre-filtered by the dataset's creators to a minimum history_length of 5 for every user, so a
  fixed threshold produced an empty cold slice there until this was caught and fixed.
- **Bootstrap 95% CIs** on every metric, resampling impressions with replacement. Coverage's CI
  has a documented quirk: the point estimate can legitimately fall outside its own bootstrap CI
  (known downward bias of the naive bootstrap for set-cardinality statistics) -- both the direct
  point estimate (`mean`) and the bootstrap distribution's own mean (`boot_mean`) are reported
  so this is visible rather than looking like a bug.

### Results (Q4, full test splits)

See `results/eval_comparison.md` for the full tables + slice breakdown + discussion. Headline:
**the winner reverses from Q2/Q3.** There, BM25 won candidate-generation recall@K on EB-NeRD
and only lost to fine-tuned embeddings on MIND. Here, on the *official ranking task*
(re-ranking each impression's own small, already-curated candidate set, not searching the full
catalog), embeddings win the accuracy metrics on EB-NeRD outright and are roughly tied with
BM25 on MIND. Candidate generation ("find the needle in a 20K-65K article haystack") rewards
exact term matching; re-ranking a handful of already-similar candidates rewards the subtler
distinctions continuous semantic similarity can pick up on that binary term overlap can't -- a
natural argument for a two-stage system using both. Coverage tells the opposite story: BM25
covers 58-96% more of the catalog than embeddings on both datasets, a real beyond-accuracy
trade-off the accuracy metrics alone don't surface.

| Dataset | AUC BM25 | AUC Embeddings | MRR BM25 | MRR Embeddings |
|---|---|---|---|---|
| EB-NeRD | 0.497 | **0.526** | 0.319 | **0.334** |
| MIND | 0.545 | **0.557** | **0.282** | 0.276 |

## Reproduce (Q5: Codabench submission, large test sets)

```bash
python scripts/generate_submission.py --dataset ebnerd   # ~18 min, 13.5M impressions
python scripts/generate_submission.py --dataset mind     # ~15 min, 2.37M impressions
```

The large test sets have no labels and are ~190x (EB-NeRD) / ~69x (MIND) bigger than the small
test splits Q2-Q4 ran on -- naively reusing the per-impression BM25 scoring from those would
take **~47 hours (EB-NeRD) / ~50 hours (MIND)**, worked out by extrapolating the measured
per-impression cost. Made tractable by two changes, both in `src/ire_a1/`:

1. **`BM25Index.score_candidates` was rewritten to be candidate-first, term-second** (loop over
   the ~15-30 given candidates, and only their query-term overlaps) instead of scanning every
   document that shares a term with the query and then discarding all but the requested
   candidates. A query built from 10 article titles routinely contains common words with
   postings lists in the thousands; the old approach touched all of them, the new one touches
   zero documents outside the actual candidate set. Benchmarked ~1000x faster on real data.
2. **Each user's query is computed once and cached**, not rebuilt per impression. Verified this
   is safe, not just fast: EB-NeRD's test history is confirmed entirely pre-test-period (max
   history timestamp `2023-06-01 06:59:59`, min test impression timestamp
   `2023-06-01 07:00:00`) -- so the leak-safe as-of cutoff never actually excludes anything
   within the test period, meaning a user's query is identical across all their impressions
   (16.76 impressions/user on average for EB-NeRD, 3.38 for MIND).

Combined, the full scoring pass benchmarks at ~15 min/dataset. EB-NeRD's 13.5M-row behaviors
file is streamed in row-group batches (PyArrow) to keep memory bounded; MIND's 2.37M rows fit
comfortably in memory as one frame.

Output: `data/submissions/predictions.zip` containing exactly `predictions.txt` (EB-NeRD) and
`data/submissions/mind_prediction.zip` containing exactly `prediction.txt` (MIND) -- **the
required filename *inside* the zip differs between the two competitions and does not match
either dataset's own naming**; confirmed directly against each competition's actual Codabench
"Submission Guidelines" page, not assumed from the reference notebooks (which used
`mind_prediction.txt` for MIND -- would have been rejected). Upload these to the two Codabench
competitions from the PDF; both competitions require you to register/create an account there
first (manual, one-time).

**Full runs completed and validated**: 13,536,710 EB-NeRD predictions in 19.0 min, 2,370,727
MIND predictions in 14.3 min. Every line format-checked against the source data with a random
2,000-row sample spanning the *entire* output file for each dataset (0 malformed lines either
dataset) -- valid permutation of the correct length, row order preserved (both competitions
require this), full impression_id coverage confirmed via exact line-count match.

### Second variant: embeddings

For Q6's "tried two approaches" comparison, `generate_submission.py` also supports
`--method embeddings`, needing large-tier article embeddings (a *different* catalog than Q3's
small-tier ones -- 125,541/120,961 articles, computed via `compute_embeddings.py` on Kaggle with
the fine-tuned `paraphrase-multilingual-mpnet-base-v2`, our strongest performer from Q3/Q4):

```bash
python scripts/generate_submission.py --dataset ebnerd --method embeddings
python scripts/generate_submission.py --dataset mind --method embeddings
```

Output: `predictions_embeddings.zip` / `mind_prediction_embeddings.zip` (same required internal
filenames as the BM25 variant -- only the local working filename differs, so nothing clobbers
the original submission). Completed and validated the same way: 13,536,710 EB-NeRD predictions
in **10.3 min**, 2,370,727 MIND predictions in **4.4 min** -- both faster than BM25, since
cosine similarity against a handful of candidates has no term/postings work at all.

## Design Note (Q6)

`design_note.md` (source) / `design_note.pdf` (rendered, 3 pages -- regenerate with any
markdown-to-PDF tool, e.g. the pure-Python `markdown`+`xhtml2pdf` toolchain used here) covers
what was built, design choices and alternatives, experimental observations (candidate
generation vs. official ranking task, dataset differences, real Codabench results), and where
the pipeline breaks at 10x scale. Leaderboard screenshot at
`results/screenshots/mind_leaderboard.png`.

## Reproduce (Q9: Anti-Gaming)

```bash
python scripts/run_anti_gaming.py --dataset ebnerd
python scripts/run_anti_gaming.py --dataset mind
```

Our actual retrieval methods (Q2/Q3) never touch a serving-time-unavailable feature -- the
query is built only from a user's *past* clicks, candidates are scored on static text. So
there's no naturally-occurring leaky version of the pipeline to report against. This script
builds one deliberately, to measure and report the risk Q9 asks about: blends BM25's score with
an article-popularity feature computed two ways -- "safe" (click counts from the TRAIN split
only, which entirely precedes val/test in `temporal_split`) vs. "LEAKY" (click counts including
val+test's own clicks -- literally the period being evaluated). Reports AUC/MRR/nDCG@5/nDCG@10
for BM25-alone / BM25+safe / BM25+LEAKY. `tests/test_no_leakage.py` is the dedicated,
consolidated test for the "no future-click leakage" guarantee Q9 also asks for (the underlying
logic is exercised across the codebase, but this is the one file to look for it) -- includes
both synthetic-data unit tests and an integration-level check against the real generated
EB-NeRD/MIND splits.

## Tests

Unit tests on synthetic data (BM25 ranking sanity, recall@K edge cases, split
ordering/fractions) + integration checks against the downloaded demo/small data (schema shape,
clicked-subset-of-candidates, no leakage), including the leakage assertion baked into
`temporal_split` itself:

```bash
python -m pytest tests/ -v
```

*(Q3-Q6 land in later commits; this README is updated as each part is implemented)*
