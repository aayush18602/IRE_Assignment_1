# CONTEXT — shared working state

Running handoff log for **Aayush** and **Anurag**. Read the top three sections before you start
work; add an entry to *your own* session log before you stop.

**This file is committed** (unlike `A2_PLAN.md`, which is git-ignored) — it only works if it
travels through git. Pull before you read it, push after you write it.

**Why the log is split by person:** you only ever edit your own section, so two people writing on
the same day never produce a merge conflict. Do not edit the other person's log.

---

## Task ledger

**This is the file's main job.** Every row is a code item defined in `A2_PLAN.md`, keyed by the
same ID. Work is divided **by assignment question**: Q1 and Q2 Option A are Aayush's, Q2 Option B
and Q3 are Anurag's, Q2.4 metrics are shared, and Q4/Q5 are deliberately unassigned for now.

When you finish an item, set it `done`, put your name and the date in the last column, and add a
session-log entry. Whoever sits down next reads this table and knows what is finished, what is
open, and what is still unassigned.

Status is one of: `todo`, `wip`, `blocked`, `done`. Owner is `Aayush`, `Anurag`, `Both`, or `TBD`.
`scripts/check_sync.py` rejects anything else, and also fails if an ID here isn't in the plan, if a
plan item is missing here, if an owner disagrees with the plan, if an item is `done` with nobody's
name against it, if a `TBD` item is marked `done` without first being assigned, or if an item is
`done` while something it depends on is still open.

| ID | Item | Owner | Status | Done by / when |
|---|---|---|---|---|
| Q1-A | Widen schema and `clean.py` for behavioural columns | Aayush | done | Aayush, 2026-09-11 |
| Q1-B | Click-history features, exponential recency decay, category match | Aayush | done | Aayush, 2026-09-11 |
| Q1-C | Session context, dwell, position bias | Aayush | done | Aayush, 2026-09-11 |
| Q1-D | Article features: trailing-window popularity, CTR, freshness | Aayush | done | Aayush, 2026-09-11 |
| Q1-E | Behaviour-window boundary enforcement + leakage tests | Aayush | done | Aayush, 2026-09-11 |
| Q2-A1 | Feature matrix builder and impression grouping — **shared substrate**. Also carries Q1.1's title (BM25) and embedding similarity features, deferred here because A1's indexes are loaded at this point | Aayush | done | Aayush, 2026-09-11 |
| Q2-A2 | Option A: LightGBM LambdaRank training, scoring, CLI | Aayush | todo | — |
| Q2-B1 | Option B: neural ranker (MLP over the same feature matrix) | Anurag | todo | — |
| Q2-B2 | Option B: training loop and CLI | Anurag | todo | — |
| Q2-M | Q2.4 before/after metrics for both re-rankers | Both | todo | — |
| Q3-A | NRMS: news and user encoders | Anurag | todo | — |
| Q3-B | NRMS: training loop and CLI, both datasets | Anurag | todo | — |
| Q3-C | Improvement: recency-weighted history attention | Anurag | todo | — |
| Q3-D | Paired bootstrap CI and unit tests | Anurag | todo | — |
| Q3-E | Ablation harness: half-life on vs. off, with paired CIs | Anurag | todo | — |
| Q4-A | Index memory and staged latency instrumentation | TBD | todo | — |
| Q4-B | Serving benchmark CLI and cost/QPS arithmetic | TBD | todo | — |
| Q5-A | Eval harness: `--method reranker` arm | TBD | todo | — |
| Q5-B | Head/tail article slicing | Aayush | todo | — |
| Q5-C | Large-tier prediction generation via the re-ranker | TBD | todo | — |

Non-code deliverables (Codabench uploads, screenshots, `design_note.md`, `README.md`) are not
tracked here — both of you do them together at the end.

### Repo state (not ledger items)

| Thing | State |
|---|---|
| A1 pipeline (Q1–Q9) | Complete, on `main`, tests pass |
| `a2` branch | Not created yet |
| EB-NeRD Codabench | A1 submission never scored — resubmit early |
| MIND Codabench | A1 scored: BM25 0.5675, embeddings 0.6218 |
| Q4 / Q5 ownership | **Undecided** — assign once Q2-A2 lands |

---

## Blocked / waiting on the other person

The most important section. If it is empty, nobody is stuck.

- ~~Anurag waiting on `Q1-A`~~ — **done and pushed. `Q3-A` (NRMS encoders) is unblocked.** Pull
  `a2` and re-run `python scripts/build_pipeline.py` to regenerate your local
  `data/processed/` with the widened columns.
- **Anurag is waiting on `Q2-A1`** (feature matrix, Aayush) before `Q2-B1` can start. **Agree the
  `build_matrix` output contract before Q2-A1 is written** — it is the only cross-person interface
  in the plan, and the shape is specified in `A2_PLAN.md`.
- **`Q4-A`, `Q4-B`, `Q5-A`, `Q5-C` have no owner yet.** All depend on `Q2-A2`; decide when it
  lands. `Q5-B` is Aayush's.

---

## Decisions taken

Append-only. One entry per decision that someone else would otherwise re-litigate.

### 2026-09-11 — The `RankingMatrix` contract, and the group array that can fail silently

`MatrixBuilder.transform()` returns a frozen `RankingMatrix`:

```
X              float32 [n_rows, n_features]     rows are (impression, candidate)
y              int8    [n_rows] or None         None on the submission path
groups         int32   [n_impressions]          candidates per impression, IN ROW ORDER
feature_names  list[str]                        index-aligned with X's columns
impression_ids [n_impressions]                  aligned with groups
article_ids    [n_rows]                         aligned with X
```

**Why this is validated rather than documented.** A learning-to-rank model is told "the first g0
rows are one impression, the next g1 the next". If that is out of step with the row order, nothing
raises — it trains on impressions stitched together from unrelated candidates and scores nonsense.
So `__post_init__` checks `groups.sum() == n_rows` and every alignment, and the run-length encoder
checks the rows really *are* contiguous per impression rather than assuming three separate feature
joins preserved order. Mutation-tested: sorting the exploded rows by `article_id` breaks 10 tests.

`select(names)` subsets columns while keeping groups and labels intact — that is how the Q9
with/without run and the submission-safe run are built from one matrix rather than two pipelines
that could drift apart.

**For Anurag (`Q2-B1`):** two columns contain NaN by design — `freshness_hours` (MIND: 100%, since
it has no `published_time`) and `hours_since_first_seen` (0.05% on EB-NeRD, for articles never seen
before the impression). LightGBM treats NaN as a first-class split direction; **an MLP will emit
NaN loss and never recover.** Impute or mask them in your input layer, and say which you chose —
the choice is a legitimate difference between the two models rather than a bug in either.

### 2026-09-11 — Q1.1's title and embedding features are deferred to Q2-A1, on purpose

Audited Q1 against the PDF after finishing `Q1-E`. Q1.1 asks for the user's recent clicked
articles as **titles, categories and embeddings**. Categories, click count and exponential decay
all ship in `behaviour.py`. Titles and embeddings do not — they are BM25 and ANN similarity, which
are A1 components (`bm25.score_candidates`, `ann.score_candidates`), and `reranker.build_matrix`
calls them at the point where those indexes are already built. Recomputing them inside
`behaviour.py` would mean loading a 226MB embedding table twice.

Defensible engineering, but it was only a sentence in a docstring, which is how a graded
requirement goes missing. `Q2-A1`'s ledger row now names them explicitly, so the item is not done
until they exist. **Q1 is complete except for these two, and they are tracked rather than
forgotten.**

### 2026-09-11 — The two-index "future-blind" test found a real leak

`hours_since_first_seen` was computed as `as_of − first_seen`, where `first_seen` is a min over the
whole indexed period with no as-of filter. For an article whose first appearance falls *after* the
impression being scored, that returns a **negative age** — which encodes "this article turns up in
the future". Measured −5.0 on a synthetic case; the two indexes disagreed, which is the signature.

It survived review because in practice a candidate in the current slate contributes its own
in-view event, so `first_seen ≤ as_of` and the value is non-negative. **A correctness guarantee
must not rest on that coincidence.** Fixed with a `ref < as_of` guard.

Worth the method, not just the fix: spot-checking individual boundaries would never have found
this, because nobody writes a test for the case they did not think of. Building the index twice
and demanding identical output tests the *property* rather than an enumeration of cases.

Both guards are mutation-tested — flipping `side="left"` to `side="right"`, and removing the
`ref < as_of` guard, each make specific named tests fail. The first pass of that exercise also
showed the synthetic fixture was too weak to catch the second mutation (only the real-data test
did), so the fixture was strengthened until the fast test catches what the slow one does.

### 2026-09-11 — Measure features by PER-IMPRESSION AUC, never pooled

I had been reporting pooled AUC across all candidate rows. That is the wrong metric and it
flatters features that cannot possibly help. The leaderboard re-ranks each impression's *own*
slate, so **a feature that is constant within an impression cannot reorder anything**, whatever its
pooled number says.

| feature | pooled AUC | per-impression AUC |
|---|---|---|
| `slate_size` | 0.3088 | **0.5000** (constant within impression) |
| `hist_len` | 0.5157 | **0.5000** (constant) |
| `is_session_start` | 0.5267 | **0.5000** (constant) |
| `ctr_1h` | 0.7139 | **0.7387** (varies — genuinely strong) |

**This does not mean the constant features are worthless** — a GBDT can use them as interaction
context ("when `hist_len` is high, trust `cat_affinity` more"), and LambdaRank groups by impression
so it will not be fooled the way pooled AUC was. It means they have no *standalone* ranking power
and must never be reported as though they do. Every feature number in the design note must be
per-impression.

Fast way to compute it, since a million `roc_auc_score` calls does not finish: the rank identity
`AUC = (Σ positive ranks − npos(npos+1)/2) / (npos·nneg)`, one vectorised pass grouped by
impression.

### 2026-09-11 — Neither dataset exposes usable position bias

Q1.2 asks for position bias. Implemented it, measured it, and the data does not support it:
`pos_in_slate` scores per-impression AUC **0.5006** (EB-NeRD) and **0.4990** (MIND) — random,
despite varying in 100% of impressions.

Its pooled AUC of 0.3779 is real but comes entirely from `slate_size`: small slates have both low
positions and a high per-candidate click rate, and that correlation vanishes once you compare
candidates *within* one slate. The candidate lists are not sorted (0.1% / 2.5% are in ascending id
order), so this is not an artifact of our cleaning — the ordering in `article_ids_inview` simply is
not display order. Report it as a measured negative result; the feature stays in because its cost
is zero and its absence would need explaining anyway.

### 2026-09-11 — Recency decay on the click history is close to inert

Swept the half-life against test-split AUC for `cat_affinity`, and the principled-sounding choice
is not the one that wins:

| EB-NeRD half-life | 6h | 24h | 72h | **168h** | 336h | 720h | undecayed |
|---|---|---|---|---|---|---|---|
| AUC | 0.5578 | 0.5667 | 0.5699 | **0.5702** | 0.5699 | 0.5696 | 0.5693 |

The peak beats no decay at all by **+0.0009**. On MIND it is worse — decay *monotonically hurts*
(3 clicks 0.6000 → 100 clicks 0.6079, undecayed 0.6080), converging on the undecayed value from
below. Defaults are set to the measured best (168h / 50 positions) and both the decayed and
undecayed affinities ship as features so the GBDT can pick.

Likely why: EB-NeRD's history is a 4-week snapshot ending 5+ days before the test window, so
everything in it is "old" and relative recency inside a stale snapshot says little about what the
user wants now. MIND has no timestamps at all, and position is a weak proxy.

**For Anurag, on `Q3-C`:** this is a caution, not a veto. Your recency-weighted attention operates
on history *embeddings* inside a learned model, which is a different mechanism from reweighting a
category histogram — it may well behave differently. But if the ablation comes out flat, this is
the reason, and it is worth predicting in the design note rather than explaining afterwards.

### 2026-09-11 — Popularity is indexed over ALL impressions, not train-only

The instinctive "safe" choice — index only the train split — is **structurally broken for trailing
windows** and fails silently. The window is relative to each impression's own timestamp, and our
temporal split leaves a 2-day gap between the end of train and the start of test, so a 24h window
at test time reaches back into empty space. Measured on the real EB-NeRD test split, every
popularity feature scored exactly **AUC 0.5000** that way.

Indexing all impressions and relying on the as-of cutoff instead:

| feature | train-only | all splits, as-of filtered |
|---|---|---|
| `ctr_1h` (EB-NeRD) | 0.5000 | **0.7139** |
| `ctr_24h` (MIND) | 0.5033 | **0.7141** |
| `pop_clicks_1h` (EB-NeRD) | 0.5000 | **0.6861** |

This is safe *because* the cutoff is strict, not despite it — a click that has already happened is
knowable at serving time regardless of which evaluation split it belongs to. Verified it is not
boundary leakage by shifting `as_of` backwards: AUC decays smoothly (0.7139 → 0.7138 at 1s → 0.638
at 1h → 0.502 at 6h) with no cliff, which a bug counting the impression's own click would produce.
For context, A1's entire BM25 pipeline scored AUC 0.497 / 0.545 — one feature here beats it.

### 2026-09-11 — Click-derived features are unavailable at submission time

Both Codabench test sets ship candidates but **no click labels** — EB-NeRD's test
`behaviors.parquet` has no `article_ids_clicked` column at all, MIND's impressions field is bare
ids. So `pop_clicks_*` and `ctr_*` cannot be computed over the test period at submission time;
`pop_inviews_*` and freshness can.

`ArticleFeatures.submission_safe_feature_names` names the surviving subset so this is mechanical
rather than a comment someone has to remember. **Q2 must not train a model that silently depends
on inputs the submission cannot supply** — either train a submission-safe variant, or report the
gap. This is Q9's "with and without features unavailable at serving time" arising from a real
constraint rather than a synthetic one, and worth saying precisely: the limitation is a property of
the released files, not of serving — a deployed system sees clicks as they happen.

### 2026-09-10 — The re-ranker ranks the impression's shown slate, not a full-catalog top-K

Q2.1 reads as "retrieve top-200 with A1's generator, then re-rank those". Measured against our own
A1 numbers, BM25 recall@200 is 2.01% (EB-NeRD) / 1.43% (MIND) — the clicked article is absent from
the retrieved pool ~98% of the time, and 99.4% of EB-NeRD impressions have exactly one click. A
re-ranker over that pool cannot win, and a 200-article list is not a valid Codabench submission
(both competitions want the shown list returned reordered).

**So:** rank the impression's own `candidates` list (mean 12.0 items EB-NeRD, 38.4 MIND). A1's
BM25 and embedding scorers become *features* — one score per candidate — instead of choosing the
candidates. A1's generator is still in the pipeline; it contributes its opinion, not its shortlist.

**Also:** run the literal full-catalog path anyway and report recall@200 as the measured ceiling.
That measurement is the justification for the design and belongs in the design note.

### 2026-09-10 — Re-ranker is LightGBM (Option A); NRMS is the Q3 baseline (Option B)

Q2 asks for GBDT *or* a neural ranker. Q3 separately asks for a reproduced official baseline,
which is NRMS. Taking Option A for Q2 and NRMS for Q3 means the assignment gets both model
families without either being redundant, and keeps the GPU-bound work confined to one track.

### 2026-09-10 — Keep the `src/ire_a1/` package name

A2 code lands inside the A1 package. Renaming rewrites every import in every script and test,
creates cross-track merge conflicts, and gains nothing. Note the continuity in the README instead.

---

## Gotchas

Append-only. Things that cost someone time — write them down so they cost only one person.

- **`clean.py` drops 10 of EB-NeRD's 17 behaviour columns.** `session_id`, `read_time`,
  `scroll_percentage`, demographics and the `*_fixed` history arrays are all discarded at cleaning
  time. They exist in the raw parquet; the unified schema throws them away.
- **MIND has no `published_time`** — 100% null across all 65,238 articles. No article freshness
  feature on MIND without a proxy (first appearance in the train logs).
- **MIND has no per-item history timestamps** — `history_timestamps` is null for all 94,057 rows.
  Exponential time decay is EB-NeRD-only; MIND needs positional decay.
- **`eval.bootstrap_ci` is unpaired** and will not satisfy Q3.4. Two overlapping CIs are not a
  paired CI. Needs `stats.paired_bootstrap_ci()`.
- **EB-NeRD Codabench is slow to score.** A1's submission sat in "submitted" for over six days and
  never scored before the deadline. Submit something early.
- **MIND on HuggingFace is gated.** Each person needs their own accepted terms and token; tokens do
  not transfer, and the failure message is unhelpful.
- **`data/` is git-ignored and ~9 GB.** The shared working set is only 91 MB — see the tier-1
  tarball in `A2_PLAN.md`. Do not try to sync `data/` through git.
- **Local machine has no GPU.** Anything torch-heavy goes to Kaggle.
- **EB-NeRD demographics are nearly all null** — measured on the test split after `Q1-A`:
  `age` 2.5%, `gender` 6.8%, `postcode` 1.9% non-null. Only `is_subscriber` is 100%. So the
  "demographics" feature group is effectively one boolean; do not budget real work for the rest.
- **The safe dwell column is sparse, the leaky one is dense.** `scroll_percentage` is 28.9%
  non-null; `next_scroll_percentage` (which describes the *next* article read, pure future
  information) is 89.1%. A tidy illustration for the Q9 write-up of why leakage is tempting.
- **`read_time` on the impression row is 100% non-null and must not be used as a feature.** It is
  the dwell on the article clicked *in that impression* — measured after the click, unknowable at
  ranking time. The serving-safe dwell signal is `history_read_times` (dwell on past clicks).
  `schema.LEAKY_IMPRESSION_COLS` names the whole group.
- **`total_inviews` / `total_pageviews` are only 48.1% non-null** on EB-NeRD articles, on top of
  being leaky. Q9 material only.
- **EB-NeRD's fixed history snapshot causes a real train/test skew.** The snapshot ends
  2023-05-25 while impressions run to 06-01, so history gets staler as the impression date
  advances. Measured means: `hours_since_last_click` 12.6 (train) → 113.9 (val) → 159.8 (test),
  and an as-of-relative `hist_clicks_24h` read 17.0 → 0.0 → 0.0, i.e. dead everywhere it mattered.
  A feature alive in train and dead in test is *worse* than a useless one — the model learns to
  lean on it. Fixed by anchoring the trailing counts to the user's own last click
  (`hist_burst_*`): now 19.7 / 19.6 / 20.3. `hours_since_last_click` is kept but named in
  `HistoryFeatures.split_unstable_feature_names` so Q2 can exclude it by name.
- **Dwell beats recency as a history weight.** On EB-NeRD `dwell_affinity` (0.5720) is the
  strongest user-side feature, above both decayed (0.5702) and raw (0.5693) category affinity.
  MIND has no dwell instrumentation, so it has no equivalent.
- **Item-side features dominate user-side ones.** Per-impression AUC on EB-NeRD: `ctr_1h` 0.7387,
  `pop_clicks_24h` 0.7370, `ctr_24h` 0.7298 — versus `dwell_affinity` 0.5633 and `cat_affinity`
  0.5598 as the best user-side features. What is popular right now predicts clicks far better than
  who the user is. Worth stating plainly in the design note rather than letting the tables imply it.
- **MIND's derived sessions are 93.3% singletons** (EB-NeRD: 48.5% session starts). Every MIND
  session feature scores per-impression AUC 0.5000. Implemented, measured, reported — not
  special-cased.
- **BM25 is genuinely at chance on EB-NeRD, and that is not a bug.** Checked before reporting it:
  3.7% zeros, mean 5.28, 57,161 distinct values, Danish tokenization working
  (`"Natascha var ikke den første"` → `['natascha','var','ikke','den','første']`). Its pooled AUC
  of 0.500007 is coincidence. Note this *reverses* A1's candidate-generation result, where BM25
  beat embeddings on EB-NeRD (recall@200 2.01% vs 1.69%) — the same "the winner reverses between
  retrieval and re-ranking" effect A1's design note already documents, now visible at feature level.
- **A1's language split survives into the feature set.** Per-impression AUC of the two similarity
  features: `embed_cos` 0.5273 (EB-NeRD) vs **0.6168** (MIND, 3rd strongest feature overall);
  `bm25_score` **0.4970** (EB-NeRD — chance) vs 0.5449 (MIND). Same direction A1 measured for
  candidate generation: the multilingual model's English-centric fine-tuning shows up on MIND, and
  BM25 over Danish titles adds essentially nothing on EB-NeRD. Worth one line in the design note,
  since it is the same finding arriving by a second route.
- **Transform cost splits roughly 30/45/25** across Q1 features / BM25 / embeddings — measured
  0.23s, +0.36s, +0.20s per 5,000 EB-NeRD impressions. Useful for Q4: the similarity features are
  ~70% of per-request feature cost and are also the weakest, so they are the first thing to drop
  under a latency budget.
- **News popularity has a ~1h half-life.** Shifting the observation point back from an impression:
  `ctr_1h` AUC 0.714 at lag 0, 0.638 at 1h, 0.502 at 6h, 0.467 at 24h. Trailing windows must be
  hours, not days — and EB-NeRD's train split spans only 10 days, MIND's 6, so a 7d window is
  nearly the whole split anyway.
- **`freshness_hours` AUC is ~0.395, i.e. inverted, and that is correct.** Higher value = older
  article = less likely clicked. A GBDT does not care about direction; do not "fix" it by flipping
  the sign.
- **Category match moved from `Q1-D` to `Q1-B`.** It needs the as-of-filtered, decay-weighted
  history, and duplicating that boundary logic in the item-side module is exactly where a leak
  would be hardest to spot.
- **Rebuilding the feature store is cheap** — `python scripts/build_pipeline.py` on the small
  tiers takes ~8s, and the temporal splits are deterministic (verified byte-identical
  impression-id hashes across a rebuild), so re-running it never invalidates A1's results.

---

## Session log — Aayush

Newest entry at the top. Only Aayush edits this section.

### 2026-09-11 — Q1 evidence made reproducible

**Item(s):** none — closing out Q1 rather than a ledger item.
**Did:** Every Q1 measurement lived in scratch scripts that would be deleted, so none of the
design note's numbers were reproducible from the repo. Added `eval.batch_ranking_auc()` (vectorised
per-impression AUC via the rank identity, 7 tests) and `scripts/run_feature_analysis.py`, which
writes `results/<ds>/feature_analysis.json` and `results/feature_analysis.md` — 52 feature rows
across both datasets, each with per-impression AUC, pooled AUC, constant-within-impression share,
NaN share and submission-safety. Runs in 46s for both datasets.
**State:** Works. 121 tests pass, was 114. **Q1 is now genuinely done** — implemented, verified,
and its findings reproducible by anyone with the repo.
**Next:** `Q2-A2` — LightGBM LambdaRank.
**For Anurag:** use `eval.batch_ranking_auc` for any feature or model comparison rather than
`sklearn.roc_auc_score` over pooled rows — see the decision entry on why. It lives in `eval.py`,
which is `Q5-B`; I have taken that item.

### 2026-09-11 — Q2-A1: the shared feature matrix

**Item(s):** `Q2-A1` — done. **Q1 is now literally complete**: the deferred title and embedding
features ship here.
**Did:** New `src/ire_a1/reranker.py`. `MatrixBuilder` fits the three Q1 sources once and
`transform()` returns a `RankingMatrix` (X, y, groups, feature_names, impression_ids,
article_ids). 28 features on EB-NeRD, 24 on MIND. BM25 over the user's recent titles and cosine
against their mean-pooled recent embeddings are computed per impression, with the derived query and
vector cached on the as-of-filtered recent-history tuple. 16 tests (114 total, was 98).
**State:** Works. Warm throughput 77K rows/s with everything on; ~36 min extrapolated to the
13.5M-impression large tier.
**Next:** `Q2-A2` — LightGBM LambdaRank over this matrix.
**For Anurag: `Q2-B1` is unblocked.** Read the contract note below before you build the input
layer — there is one thing that will break an MLP and not a GBDT.

### 2026-09-11 — Q1-E: boundary enforcement, and a leak it found

**Item(s):** `Q1-E` — done. **Q1 is complete** (`Q1-A` … `Q1-E`).
**Did:** Extended `tests/test_no_leakage.py` from A1's split-level checks to the A2 feature
builders. The load-bearing one is *future-blind*: build the same index twice — once over all
impressions, once over only those strictly before a cutoff — and assert features at or before the
cutoff come out identical. Any path by which future data reaches a feature makes the two disagree,
whether or not anyone wrote a case for it. Synthetic and real-data variants of it, plus a proof by
construction that nothing reads `schema.LEAKY_*_COLS` (drop the columns entirely; the builders
still work). 13 tests in that file; 98 total, was 91.
**State:** Works, and mutation-tested — see below.
**Next:** Q1 is done. `Q2-A1` (feature matrix builder) is the next item and unblocks Anurag's
`Q2-B1`.
**For Anurag:** `Q2-A1` is mine and you are waiting on it. Worth agreeing the `build_matrix`
output contract now rather than when I push it.

### 2026-09-11 — Q1-C: session context, dwell, slate position

**Item(s):** `Q1-C` — done.
**Did:** Added `explode_candidates()` and `session_features()` to `behaviour.py` — stateless, so
functions rather than another builder. Sessions come from `session_id` on EB-NeRD and from
30-minute timestamp gaps on MIND. Every in-session counter is strictly-previous
(`cum_sum() - value`, no shift-null case). 9 tests (91 total, was 82).
**State:** Works, and it surfaced the measurement mistake below — worth reading before Q2.
**Next:** `Q1-E` — boundary and leakage tests across all three feature modules.
**For Anurag:** the per-impression AUC point below applies to your `Q2-B` and `Q3` evaluation too.
Pooled AUC will flatter your model; the leaderboard metric is per-impression.

### 2026-09-11 — Q1-B: user-side history features

**Item(s):** `Q1-B` — done.
**Did:** New `src/ire_a1/behaviour.py`. Two decay clocks: EB-NeRD decays in elapsed time
(per-(user, as_of) state, chunked explode), MIND in position from the end (per-user state, since
weights are as-of independent there). Category affinity in decayed / raw / dwell-weighted variants,
history counts, `hours_since_last_click`. 14 tests (82 total, was 68).
**State:** Works. EB-NeRD 0.47M rows/s, MIND 16.4M rows/s — the gap is the per-impression explode
EB-NeRD needs and MIND doesn't. ~5.7 min for the 162M-row large tier; fine, but it is the slowest
feature stage so far and worth remembering for Q4.
**Next:** `Q1-C` — session context, dwell, position bias.
**For Anurag:** the recency-decay result below is worth reading before you build `Q3-C`.

### 2026-09-11 — Q1-D: item-side features

**Item(s):** `Q1-D` — done. Taken before `Q1-B`, reversing the plan's order.
**Did:** New `src/ire_a1/article_features.py`: trailing-window click/in-view counts, smoothed CTR,
freshness. Boundary lives inside the lookup (`np.searchsorted(..., side="left")`), so an event at
exactly `as_of` is excluded and there is no argument-less variant to call by accident. Counting is
vectorised per *article* rather than per row, so numpy call count tracks catalogue size (20-65K)
rather than row count (5-9M). 13 tests in `tests/test_article_features.py` (68 total, was 54).
**State:** Works and measured. Throughput 2.2-2.6M rows/s, so the 162M-row large tier is ~65s.
**Next:** `Q1-B` — history features, now also carrying category match (moved out of Q1-D, see below).
**For Anurag:** two findings below change what your Q2-B model can rely on — worth reading before
you design the input layer.

### 2026-09-11 — Q1-A: widened the schema

**Item(s):** `Q1-A` — done.
**Did:** Added the behavioural columns A1's cleaning was discarding. `IMPRESSION_COLS` 7 → 16
(`session_id`, demographics, and the four post-click dwell columns), `HISTORY_COLS` 6 → 8
(`history_read_times`, `history_scroll_percentages`), `ARTICLE_COLS` 11 → 14 (the lifetime
counters). Added `schema.LEAKY_IMPRESSION_COLS` / `LEAKY_ARTICLE_COLS` so Q9's serving-unavailable
columns are named in one place and the Q1 feature builders can assert against them. Three new tests
in `test_clean.py` (54 passing, was 51). Rebuilt `data/processed/` for both datasets.
**State:** Works. Splits verified byte-identical to before the change — same impression counts,
same id hashes, same time boundaries — so every A1 result still reproduces.
**Next:** `Q1-D` (trailing-window popularity) before `Q1-B`, reversing the plan's order — the
popularity index is the piece with real design risk and `Q1-C`/`Q1-E` both lean on it, whereas the
history features are more mechanical.
**For Anurag:** `Q3-A` is unblocked. Pull `a2` and re-run `python scripts/build_pipeline.py`
(~8s) before you start, or your local parquets will be missing the new columns. Note
`data/processed_large/` is still on the old schema — it only matters for submission generation, so
it can wait.

### 2026-09-10 — Planning

**Item(s):** none — planning only.
**Did:** Read A2.pdf against the A1 codebase. Produced `A2_PLAN.md` (git-ignored) with the full
build plan, task split, and handoff instructions. Settled the candidate-set question (see
Decisions). Audited the repo for A2 gaps.

**State:** No code written. `.gitignore` modified but not committed. `A2.pdf` untracked.

**Next:** Create the `a2` branch, land the schema widening, do the four chores (lightgbm dep,
gitignore model artefacts, EB-NeRD resubmit, tier-1 bundle for Anurag).

**For Anurag:** Nothing yet — you are blocked on the schema widening. Do Kaggle setup and read
`src/ire_a1/` in the meantime.

---

## Session log — Anurag

Newest entry at the top. Only Anurag edits this section.

*(no entries yet)*

---

## Entry template

Copy this into your own log section. Keep it short — a template nobody fills in is worthless.

```
### YYYY-MM-DD — <what you worked on>

**Item(s):** the plan IDs, e.g. `Q2-A1` — and set their status in the ledger above.
**Did:** what changed, in one or two lines.
**State:** works / half-done / broken — and if broken, how.
**Next:** what you would do first if you sat back down right now.
**For <other person>:** anything that affects their work, or "nothing".
```
