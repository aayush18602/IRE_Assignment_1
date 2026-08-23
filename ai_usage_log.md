# AI Usage Log

Per Q7.4: prompts used, and which code is AI-generated vs. human-written, for CS4.406
Assignment 1. Updated continuously as work progresses; entries are in chronological order.

## Tooling

Claude Code (Anthropic), used interactively in the terminal against this repo.

## Log

### 2026-08-23 — Part 0: repo scaffolding & setup plan

- Prompt (paraphrased): asked Claude to read `A1.pdf` in full, cross-reference the two
  pre-existing analysis notebooks (`ebnerd_analysis.ipynb`, `mind_analysis.ipynb`), and produce
  a plan for Part 0 (datasets & setup), noting the code submission target is a plain git repo
  (not GitHub Classroom) and asking for a compute recommendation (local GPU vs. Kaggle).
- AI-generated: this file, `.gitignore`, `requirements.txt`, `requirements-gpu.txt`,
  `scripts/download_ebnerd.py`, `scripts/download_mind.py`, README scaffold, directory layout.
- Human-written: none yet at this stage (setup only).
- Human review: reviewed and approved the plan before implementation (see
  `/home/aayush/.claude/plans/fluttering-snacking-pumpkin.md` for the plan as approved).

### 2026-08-23 — Part 0: dataset downloads, Kaggle/GPU decision

- Prompt (paraphrased): user provided a HuggingFace access token to unblock the gated MIND
  dataset download, confirmed Codabench registration was done, and asked Claude to re-verify
  (by re-reading the PDF in depth) whether Kaggle/GPU is actually needed, since local downloads
  seemed to conflict with an earlier "use Kaggle for large-scale work" recommendation.
- AI-generated: downloaded and extracted EB-NeRD demo/small/large/testset and MIND small
  train+dev + large-test locally; found and fixed a bug in `download_ebnerd.py` (a dead
  `articles_large_only.zip` URL was silently aborting the run before the testset download);
  added skip-if-exists + non-fatal-per-file-error handling to the script.
- Human decision: given a genuine 3-way choice (load EB-NeRD's provided embeddings / compute
  our own BERT-XLM-RoBERTa embeddings from article text / do both) for Q3, the user chose to
  **compute our own embeddings**. This is the one part of the assignment that needs a GPU, so
  Kaggle will be used specifically for that step; BM25 and large-test-set prediction generation
  stay local/CPU (Polars/PyArrow batching, per the PDF's own guidance).
- Security note: the HF token was pasted directly in chat. It was used only via an `HF_TOKEN`
  env var for the download commands (never written into any repo file); the user was told to
  treat it as exposed and optionally rotate it.

### 2026-08-23 — Q1: reproducible data pipeline

- Prompt: "do the Q1 code only, nothing else" -- user stepped away, asked for Q1 (Reproducible
  Data Pipeline) to be implemented unattended.
- AI-generated, entirely: `src/ire_a1/schema.py` (unified article/impression/history schema
  shared by both datasets), `src/ire_a1/clean.py` (`clean_ebnerd`, `clean_mind` -- parse raw
  files into the unified schema; ids cast to string so EB-NeRD's ints and MIND's "N12345"
  strings compare equal; MIND's per-impression `history` field verified empirically to be a
  fixed pre-collection-period snapshot, constant per user, not per-impression), `src/ire_a1/
  split.py` (`temporal_split` -- quantile-based time cutoffs, never random, with a leakage
  assertion built into the function itself), `src/ire_a1/feature_store.py` (article + user
  feature store, `history_asof()` helper for future leak-safe history truncation),
  `scripts/build_pipeline.py` (one-command CLI entrypoint), `configs/pipeline.yaml`,
  `tests/test_split.py` + `tests/test_clean.py` (7 tests, all passing -- synthetic-data unit
  tests for the split's ordering/fraction/validation behavior, integration tests against the
  real downloaded EB-NeRD demo + MIND small data checking schema shape and clicked-subset-of-
  candidates correctness).
- Verification performed: ran `python scripts/build_pipeline.py` end-to-end on EB-NeRD small +
  MIND small (no errors, no leakage-assertion failures); manually inspected sample rows of every
  output table (articles, impressions, user_history) for both datasets to confirm entity
  parsing, click/candidate consistency, and correct null handling (e.g. MIND's `last_history_
  time` is correctly null since MIND's history has no per-item timestamps); ran the full pytest
  suite (7/7 passed).
- Human review: not yet reviewed by the user (in progress, unattended run per their request).
