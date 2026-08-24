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
- Human review: user reviewed the summary and asked to continue to Q2 (implicit approval).

### 2026-08-23 — Q2: lexical candidate generation (BM25)

- Prompt: "yeah lets do Q2".
- AI-generated, entirely: `src/ire_a1/bm25.py` (hand-rolled inverted index + Okapi BM25 over
  title+abstract, not a library wrapper -- chosen deliberately since the assignment explicitly
  says "build an inverted index"), `src/ire_a1/candidate_eval.py` (`recall_at_k`, shared with
  Q3), `feature_store.recent_history_asof()` (query construction: user's N most recent clicks
  strictly before the impression's own timestamp), `scripts/run_bm25.py`, `tests/test_bm25.py`
  (8 tests).
- Verification performed: smoke-tested on 500 EB-NeRD test impressions before committing to a
  full run (recall@50/100/200 = 0.6%/1.4%/2.4% -- low but expected for pure lexical
  title-matching against a ~20K-article catalog); ran the full pytest suite (16/16 passed); full
  test-split runs for both datasets launched in the background.
- Human review: user asked for status later; final numbers reported (EB-NeRD recall@50/100/200
  = 0.49%/1.03%/2.01%, MIND = 0.45%/0.82%/1.43%), no objections raised.

### 2026-08-23 — Q3: semantic candidate generation (embeddings)

- Prompt: "now what to do next?" -> user chose "write everything now, you [Claude] run Kaggle
  later" when asked how to sequence Q3 given the GPU step needs Kaggle.
- AI-generated, entirely: `src/ire_a1/ann.py` (FAISS flat/exact ANN index -- the "brute-force
  for small scale" option the PDF allows; `user_embedding()` mean-pools a leak-safe click
  history the same way Q2's BM25 query is built), `scripts/compute_embeddings.py` (Kaggle-bound:
  mean-pooled XLM-RoBERTa embeddings over title+abstract, xlm-roberta-base by default for BOTH
  datasets -- one multilingual model instead of two dataset-specific ones, since it natively
  handles EB-NeRD's Danish and MIND's English alike), `scripts/run_embeddings.py` (local: builds
  the ANN index, scores every impression, reports recall@K using the *same*
  `candidate_eval.recall_at_k` as Q2 so results are directly comparable),
  `scripts/compare_retrieval.py` (Q3.5's "which works better" side-by-side table),
  `tests/test_ann.py` (6 tests).
- Verification performed: rather than trust the Kaggle-bound script untested, temporarily
  installed CPU-only torch/transformers/sentencepiece locally (`requirements-gpu.txt` deps,
  normally Kaggle-only) purely to smoke-test `compute_embeddings.py` end-to-end -- first
  attempted a tiny test model (`prajjwal1/bert-tiny`) which turned out to have a broken
  tokenizer config on the HF Hub (unrelated bug in that model repo, not in our code), then
  smoke-tested with the real `xlm-roberta-base` on 8 articles (confirmed 768-dim, non-zero
  embeddings, ~56s including one-time model download) and `run_embeddings.py` against those 8
  embeddings (confirmed the cold-start code path and output format, no crashes). Ran the full
  pytest suite (22/22 passed). Deleted the smoke-test scratch files afterward.
- Human review: not yet reviewed by the user at time of writing; Kaggle run itself still pending
  (user needs to actually execute `compute_embeddings.py` there and hand back the embeddings).

### 2026-08-24 — Q3: Kaggle run completed, results in

- Prompt: user ran `compute_embeddings.py` on Kaggle. First attempt hit `FileNotFoundError` --
  the `--processed-dir` guess in the instructions Claude gave didn't match Kaggle's actual mount
  path (`/kaggle/input/datasets/<username>/<dataset-slug>/...`, deeper than assumed). Fixed by
  asking the user to run `!find /kaggle/input -name "articles.parquet"` and using the real path.
  Also found and fixed a real dead-code bug while writing those instructions: `compute_
  embeddings.py` had an unused `sys.path.insert(...)` for `src/` that it never actually needed
  (the script is fully self-contained) -- removed, confirmed via `grep` there was no `ire_a1`
  import anywhere in the file, re-verified `--help` still works.
- User then ran both datasets on Kaggle and copied `embeddings.parquet` back for each.
- AI-generated: validated the returned embeddings before trusting them (row counts match
  articles, no NaNs, no all-zero rows, full article_id overlap, correct 768-dim) before running
  anything downstream; ran `scripts/run_embeddings.py` (smoke-tested on 500 rows first, then
  full test splits: EB-NeRD 71,631 impressions in ~200s, MIND 34,519 impressions -- noticeably
  slower per-impression than EB-NeRD despite fewer rows, due to MIND's larger 65K-article
  corpus) and `scripts/compare_retrieval.py` for both datasets; wrote `results/comparison.md` +
  copied the four `recall_test.json` result files into `results/<dataset>/` since
  `data/processed/` (where the scripts write by default) is entirely gitignored and the actual
  recall numbers are a real deliverable worth having in git, not just reproducible-in-theory.
- Finding: BM25 beats the raw (not fine-tuned) xlm-roberta-base embedding baseline on every K,
  both datasets (EB-NeRD ~2x, MIND ~4-5x). Not a bug -- documented in `results/comparison.md`
  as the expected behavior of un-fine-tuned mean-pooled transformer embeddings vs. lexical
  retrieval (the reason Sentence-BERT-style fine-tuning exists).
- Human review: results reported to the user; no objections raised yet.

### 2026-08-24 — Q3 ablation: fine-tuned sentence-embedding model vs. raw XLM-R

- Prompt: user asked "shouldn't semantic models be better than lexical?" after seeing BM25 win.
  Claude explained that raw pretrained transformer mean-pooling is a documented weak retrieval
  baseline and offered a fine-tuned sentence-embedding model as a fairer comparison. User then
  asked specifically about `all-MiniLM-v8` (likely meaning `all-MiniLM-L6-v2`, English-only --
  flagged as unsuitable for Danish EB-NeRD), then about `paraphrase-multilingual-mpnet-base-v2`
  and `intfloat/multilingual-e5-base`. User asked Claude to re-check the PDF for whether
  non-BERT/XLM-RoBERTa-named models are "allowed" -- Claude quoted the exact PDF wording (only
  "BERT, XLM-RoBERTa" named, no explicit allow/deny list) and gave an honest ambiguous-but-likely-fine
  read, recommending framing any such model as an *additional* ablation alongside the compliant
  XLM-RoBERTa result rather than a replacement. User then said "yes lets implement", after
  Claude had recommended mpnet over e5 as the lower-risk pick (drop-in, no architecture change,
  vs. e5's query/passage-prefix convention which our mean-pooled-history-embedding design
  doesn't follow).
- AI-generated: no changes needed to `compute_embeddings.py` (model already a CLI parameter);
  added `--embeddings-file` to `run_embeddings.py` (evaluate a second embeddings.parquet without
  overwriting the first) and `--variant LABEL=DIR` (repeatable) to `compare_retrieval.py`
  (N-way comparison table, not just BM25-vs-one-embedding), verified backward-compatible against
  the existing xlm-roberta-base results before committing. Gave the user updated Kaggle
  instructions for `--model sentence-transformers/paraphrase-multilingual-mpnet-base-v2`.
- User ran it on Kaggle and returned `embeddings_mpnet.parquet` for both datasets. AI-generated:
  validated the returned embeddings (same checks as the first Kaggle round: row counts, no
  NaN/zero vectors, full id overlap, 768-dim) before use; ran `run_embeddings.py` and the new
  multi-variant `compare_retrieval.py` for both datasets; rewrote `results/comparison.md` with
  the three-way table + analysis, copied the two new `recall_test.json` files into `results/`.
- Finding (genuinely interesting, not anticipated): fine-tuning roughly doubles recall@200 on
  both datasets vs. raw XLM-R, confirming the earlier hypothesis. But the BM25-vs-embeddings
  *winner flips by language* -- BM25 still wins on EB-NeRD (Danish) by ~1.2x over the
  fine-tuned model, while the fine-tuned model *beats* BM25 on MIND (English) by ~1.6x. Likely
  cause documented in results/comparison.md: `paraphrase-multilingual-mpnet-base-v2` was
  distilled from an English-only teacher, so its English quality likely exceeds its Danish
  quality (multilingual extension via distillation, not native multilingual training).
- Human review: results reported to the user.
