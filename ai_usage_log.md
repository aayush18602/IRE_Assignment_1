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

### 2026-08-24 — Q4: offline evaluation harness

- Prompt: "yes" (to Claude's suggestion, after the Q3 ablation, to move to Q4 given the 2026-08-27
  deadline and Q4/Q5/Q6/Q9 all still being unbuilt).
- AI-generated, entirely: `src/ire_a1/eval.py` (`ranking_metrics` -- AUC/MRR/nDCG@5/nDCG@10 via
  sklearn over each impression's own shown candidates, not a full-catalog list; `intra_list_
  category_diversity`, `novelty` vs. train-only popularity, `coverage`; `bootstrap_ci`/
  `bootstrap_coverage_ci`; `percentile_threshold` for the cold/warm slice); refactored `bm25.py`
  (`BM25Index._score_all` extracted so `query()` and a new `score_candidates()` share scoring
  logic) and `ann.py` (new `score_candidates()`) so both methods can score a *given* candidate
  set (Q4's official-metric re-ranking) as well as their existing full-catalog top-K search (Q2/
  Q3's candidate generation); `scripts/run_eval.py` (orchestrator); `tests/test_eval.py` (14
  tests) + new tests in `test_bm25.py`/`test_ann.py` for the candidate-scoring methods.
- Bug caught and fixed before the full run: cold/warm slicing with a fixed absolute
  `history_length` threshold (5) produced an empty cold slice for EB-NeRD on a 500-impression
  smoke test. Investigated with real data rather than assuming a fixed threshold would just
  work -- confirmed EB-NeRD's small/demo tiers are pre-filtered by the dataset's own creators to
  a minimum history_length of exactly 5 for every single user (checked via `.describe()` on the
  full user_history table), while MIND genuinely ranges 0-558. Fixed by switching to a
  percentile-based threshold (`percentile_threshold`, default 25th percentile of each dataset's
  own distribution) instead of a fixed count.
- Second issue caught (not a bug, but confusing if unexplained): the smoke test showed
  coverage's point estimate (0.5087) falling *outside* its own bootstrap CI [0.4460, 0.4763].
  Investigated and identified this as a known property of the naive bootstrap applied to
  set-cardinality statistics (resampling with replacement can only shrink the union of
  recommended items, never grow it past the true value, biasing the bootstrap distribution
  downward). Documented explicitly in `bootstrap_coverage_ci`'s docstring and added a
  `boot_mean` field alongside the point-estimate `mean` so this is visible rather than looking
  like a bug to a reader.
- Verification performed: full pytest suite (38/38 passed) before launching real runs;
  smoke-tested `run_eval.py` on 500 EB-NeRD impressions for both fixes above before committing
  to full-split runs; full runs for BM25 x {ebnerd, mind} and embeddings x {ebnerd, mind}
  launched in the background (BM25 expected to take as long as Q2's runs did, since
  `score_candidates` shares the same `_score_all` cost as `query()`).
- Human review: not yet reviewed by the user at time of writing (full-run numbers pending).
- Update once all 4 background runs (BM25/embeddings x EB-NeRD/MIND) completed: results copied
  to `results/{ebnerd,mind}/eval/` and written up in `results/eval_comparison.md`. Headline
  finding: the winner reverses from Q2/Q3's candidate-generation recall@K. There, BM25 won on
  EB-NeRD and only lost to fine-tuned embeddings on MIND. On Q4's official ranking task
  (re-ranking each impression's own small, already-curated candidate set), embeddings win the
  accuracy metrics on EB-NeRD outright and are roughly tied with BM25 on MIND -- documented as
  the difference between "find a needle in a large haystack" (favors exact term matching) vs.
  "distinguish among already-similar candidates" (favors continuous semantic similarity).
  Coverage tells the opposite story (BM25 covers 58-96% more of the catalog on both datasets),
  a genuine beyond-accuracy trade-off the accuracy metrics alone don't surface. One drafting
  error caught before finalizing: an initial claim that "AUC drops for cold users in every
  method" was checked against the actual numbers and found false for EB-NeRD BM25 (cold AUC
  0.499 vs. warm 0.497, statistically indistinguishable) -- corrected in results/
  eval_comparison.md rather than left as an overclaim.

### 2026-08-24 — Q5: Codabench submission at scale

- Prompt: "what to do for Q5?" -> Claude flagged that naively reusing Q2-Q4's per-impression
  BM25 scoring on the large test sets (13.5M EB-NeRD / 2.37M MIND impressions, ~190x/~69x
  bigger than the small splits) would take an estimated ~47-50 hours, infeasible with the
  deadline. Presented two options via AskUserQuestion (fast scalable baseline vs. engineering
  BM25 to run at scale); user chose to engineer BM25 for scale.
- Investigation before writing any code: inspected the real large-tier schemas
  (`ebnerd_testset`, `MINDlarge_test`) directly rather than assuming they matched the small-tier
  schemas clean.py already handles (they don't -- different directory structure, no labels,
  different/larger article catalogs: 125,541 EB-NeRD articles vs. 20,738 in the small tier,
  120,961 MIND articles vs. 65,238). Checked impressions-per-user ratio empirically (EB-NeRD
  16.76x, MIND 3.38x) and, critically, verified EB-NeRD's test history is entirely
  pre-test-period (max history timestamp 2023-06-01 06:59:59, min test impression timestamp
  2023-06-01 07:00:00, an exact 1-second boundary) before relying on that fact to cache each
  user's query -- caching without that verification would have risked silently reusing a stale,
  leaky, or simply wrong query.
- AI-generated: refactored `BM25Index` (`src/ire_a1/bm25.py`) so `score_candidates()` is
  candidate-first/term-second (touches only the given ~15-30 candidates and their query-term
  overlaps) instead of reusing `_score_all`'s full-corpus scan; `postings` restructured from
  `term -> [(doc_idx, tf), ...]` to `term -> {doc_idx: tf}` for O(1) point lookups, with
  `query()`/`_score_all` (Q2/Q3, unaffected) updated to iterate `.items()` instead of a list --
  same data, no duplicated storage. New `src/ire_a1/submission.py` (`ranks_from_scores`,
  `format_submission_line`) extracted out of the script so the ranking/formatting logic is unit
  tested (4 new tests) rather than only validated by eyeballing script output.
  `scripts/generate_submission.py`: dataset-specific loaders for the large/unlabeled bundles
  (self-contained, don't reuse `clean_ebnerd`/`clean_mind` which assume the labeled train/
  validation structure), per-user query caching, EB-NeRD behaviors streamed in PyArrow
  row-group batches to bound memory (13.5M rows), MIND loaded as one frame (2.37M rows, modest).
- Verification performed, in order: (1) full pytest suite (42/42) after the bm25.py refactor,
  confirming the rewrite didn't change scoring semantics; (2) isolated micro-benchmark on the
  small-tier index showed a ~1157x speedup, but was flagged as unrealistic and re-benchmarked
  against the real large-tier index (125,541 articles) with real user-history-derived queries
  before trusting any number -- realistic estimate ~0.065ms/impression EB-NeRD,
  ~0.37ms/impression MIND, extrapolating to ~15 min/dataset for the full scoring pass, not the
  isolated benchmark's misleadingly large multiplier; (3) exact output format re-verified
  against the reference notebooks' markdown cells (not assumed from memory) before running
  anything at scale -- confirmed `impression_id [rank_order]`, comma-separated 1-indexed
  permutation in original candidate order, and matched the reference notebooks' exact output
  filenames (`predictions.txt`/`.zip`, `mind_prediction.txt`/`.zip`); (4) 5,000-row smoke test
  for both datasets, then programmatically validated every output line against the real source
  data -- confirmed every line is a valid permutation of the correct length and every
  impression_id in the sample is covered (0 malformed lines either dataset) -- before launching
  the full 13.5M/2.37M-row runs.
- Human review: not yet reviewed by the user at time of writing (full runs in progress).

### 2026-08-24 — Q5: filename bug caught by the user, full runs validated and complete

- Prompt: user pasted MIND's actual official Codabench submission guidelines and asked "check
  if zip we made is fine with this?" -- a real, load-bearing question: the guidelines required
  the file *inside the zip* to be named exactly `prediction.txt`, but the code (following the
  reference notebook's convention without independently checking it against the real
  competition rules) used `mind_prediction.txt`, which would have been rejected on upload.
- AI-generated: fixed `scripts/generate_submission.py` so the zip's internal arcname is
  dataset-specific and confirmed rather than derived from the local working filename. Attempted
  to independently verify EB-NeRD's equivalent requirement via WebFetch (the Codabench
  competition page and the ebnerd-benchmark GitHub repo) before assuming the reference
  notebook's `predictions.txt` convention was correct there too -- both attempts hit JS-rendered
  content WebFetch couldn't read, so explicitly told the user "I can't reliably access this,
  please paste it" rather than guessing. User then pasted EB-NeRD's actual guidelines too,
  confirming `predictions.txt` was in fact already correct there (happened to match the
  reference notebook's convention, unlike MIND).
- Since the two full-scale background runs (started before the fix) were using stale in-process
  code, manually recreated the MIND zip afterward with the corrected arcname rather than
  re-running the 14-minute generation; verified via `unzip -l` that the fixed zip contains
  exactly one file named `prediction.txt`, no `__MACOSX`, no folder nesting.
- Final verification once both full runs completed: exact line-count match against known
  impression counts (13,536,710 EB-NeRD, 2,370,727 MIND); a 2,000-row *random* sample spanning
  the entire output file for each dataset (not just head/tail) checked against the real source
  data -- confirmed every sampled line is a valid permutation of the correct length, 0 malformed
  lines either dataset; row order preserved (both competitions' guidelines require this
  explicitly); MIND's last line (impression_id 2,370,727) matches the total row count, ruling
  out silent truncation.
- Human review: user caught the filename bug via the pasted MIND guidelines -- a real error
  that would not have been caught by any of the automated validation already in place (which
  checked format correctness, not competition-specific naming rules). Both submission files
  are now generated, fixed, and validated. Remaining manual steps for the user: register/upload
  both zips to the two Codabench competitions, screenshot the leaderboard results for Q6.

### 2026-08-26 — Q5: second submission variant (embeddings) for Q6's comparison

- Prompt: MIND leaderboard score came back (AUC 0.5675, MRR 0.2731, nDCG@5 0.2886, nDCG@10
  0.3436 -- recorded in results/mind/codabench_leaderboard.md, three of four metrics higher
  than our own offline Q4 estimate on the small split, good evidence the pipeline generalizes).
  User then asked which method the submissions used (BM25 only, confirmed via grep -- no
  ann.py/embeddings import in generate_submission.py) and asked for a second, embeddings-based
  submission "to show that I tried two approaches" for Q6.
- Identified before writing code: our existing embeddings.parquet files only cover the
  small-tier catalogs (20,738/65,238 articles) -- the large test sets reference a different,
  bigger catalog (125,541/120,961 articles), so a fresh Kaggle embedding run was needed against
  newly-extracted large-tier article files, not reusable from Q3.
- Asked the user which model to use for this (AskUserQuestion: fine-tuned mpnet vs. raw
  xlm-roberta-base) rather than defaulting silently, recommending mpnet since it was the
  stronger performer in Q3/Q4 (beat BM25 outright on MIND) -- user agreed.
- AI-generated: extracted lean large-tier article files (article_id/title/abstract only, 7.9MB/
  14MB) to `data/processed_large/{ebnerd,mind}/articles.parquet` for Kaggle upload; refactored
  `generate_submission.py` to add `--method {bm25,embeddings}` via a shared build_user_repr/
  score closure pattern (same structure as Q4's run_eval.py) so both methods reuse the same
  batching/caching/output logic instead of a duplicated script; local output filenames now
  include the method to avoid clobbering the already-submitted BM25 files, while the zip's
  required internal filename stays fixed per competition regardless of method.
- Verification performed, same rigor as the first Q5 round: full pytest suite (42/42) after the
  refactor; regression-tested the BM25 path still works (500-row smoke test) and the embeddings
  path fails cleanly with an actionable error when the large-tier embeddings file doesn't exist
  yet (correct behavior, not a bug) -- both checked *before* asking the user to spend Kaggle
  time; validated the returned large-tier embeddings the same way as the first round (row
  counts, no NaN/zero vectors, full id overlap, 768-dim) before use; 5,000-row smoke test for
  both datasets, format-validated against real source data (0 malformed lines either dataset)
  before launching the full runs. Embeddings scoring benchmarked faster than BM25 in the smoke
  tests (23,412 imp/s EB-NeRD vs. BM25's 13,526 imp/s; 8,900 imp/s MIND vs. BM25's ~2,800 imp/s)
  -- expected, since cosine similarity against a handful of candidates has no term/postings
  work at all, unlike BM25's targeted-but-still-per-term scoring.
- Human review: not yet reviewed by the user at time of writing (full runs in progress).

### 2026-08-26 — Q5: embeddings submission runs interrupted then completed

- Mid-run, VSCode closed unexpectedly ("actually i was out for a while and vscode close
  automatically"), killing both background generation jobs before either wrote meaningful
  output (MIND's file was 0 bytes, EB-NeRD's didn't exist yet). Checked git status, running
  processes, and output file state directly rather than assuming anything -- found one
  uncommitted ai_usage_log.md edit (committed, a4691f9), confirmed the large-tier embeddings
  inputs were untouched on disk (process kills don't affect already-saved files), deleted the
  two partial/empty outputs, and relaunched both.
- User then said "dont relaunch both, launch one by one" -- had already relaunched both in
  parallel; stopped the MIND job (TaskStop), cleaned up its partial output again, and let
  EB-NeRD finish alone before starting MIND.
- Both completed and validated with the same rigor as every other Q5 run: exact line-count
  match (13,536,710 EB-NeRD, 2,370,727 MIND), 2,000-row random-sample permutation check against
  real source data (0 malformed lines either dataset), correct zip contents. EB-NeRD embeddings
  took 10.3 min (vs BM25's 19.0 min), MIND embeddings took 4.4 min (vs BM25's 14.3 min) --
  faster than BM25 as predicted, since cosine similarity has no term/postings work at all.
- Human review: user asked "check once" when unsure whether a scheduled status check had fired
  -- it had, just concurrently with their message; reported the actual state rather than
  guessing. Both embedding-based submission files now ready alongside the original BM25 ones:
  `predictions_embeddings.zip` (EB-NeRD) and `mind_prediction_embeddings.zip` (MIND). Remaining
  manual steps: user uploads both to Codabench (EB-NeRD allows 5/day, MIND only 1/day) and
  screenshots both leaderboard results for Q6's two-approach comparison.
