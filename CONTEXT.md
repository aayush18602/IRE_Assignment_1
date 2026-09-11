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
| Q2-A2 | Option A: LightGBM LambdaRank training, scoring, CLI | Aayush | done | Aayush, 2026-09-11 |
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

### 2026-09-11 — The re-ranker works, and the size of the win needs its caveats

Test split, AUC, against A1's own scorers used alone:

| dataset | variant | `bm25_score` | `embed_cos` | **re-ranker** |
|---|---|---|---|---|
| EB-NeRD | full | 0.4970 | 0.5273 | **0.8078** |
| EB-NeRD | submission-safe | 0.4970 | 0.5273 | **0.7594** |
| MIND | full | 0.5449 | 0.6168 | **0.6898** |
| MIND | submission-safe | 0.5449 | 0.6168 | **0.5991** ✗ |

The plan said to interrogate a jump this size rather than celebrate it. Four checks on EB-NeRD's
+0.31: val nDCG@10 at the best iteration (0.6736) matches test (0.6855), so it is not fitted to the
val split; the model beats its own strongest feature `ctr_1h` (0.7387) by +0.069, so it is not a
passthrough; the `as_of`-shift probe on that feature decays smoothly with no cliff; and the
future-blind test covers the index. The win looks real.

**Full-set numbers are not submittable.** 48.0% of EB-NeRD's model gain comes from the five
click-derived features (`ctr_1h`, `pop_clicks_*`, `ctr_24h`, `clicks_so_far_in_session`), and
neither Codabench test set ships click labels. The submission-safe row is the honest headline:
still a large win on EB-NeRD (0.5273 → 0.7594), at a cost of −0.048.

### 2026-09-11 — The literal Q2.1 pipeline, measured: ~270x worse than what we ship

Q2.1 says "use A1's candidate generator to retrieve top-K, then re-rank". We rank the shown slate
instead and feed A1's scorers in as features. That decision was justified by A1's recall@200
(2.01% EB-NeRD / 1.43% MIND) — a ceiling no re-ranker can lift, since it can reorder a pool but not
add to it. Now it is also *demonstrated*, on the full EB-NeRD test split:

| scope | method | MRR | nDCG@5 | nDCG@10 |
|---|---|---|---|---|
| **end-to-end** (every impression; a miss scores 0) | BM25's own top-200 order | 0.0004 | 0.0001 | 0.0002 |
| **end-to-end** | re-ranker over the top-200 | **0.0022** | 0.0021 | 0.0023 |
| conditional (the 2.0% where stage 1 retrieved the click) | BM25 order | 0.0210 | 0.0026 | 0.0091 |
| conditional | re-ranker | 0.1079 | 0.1026 | 0.1146 |

Against the shipped pipeline's MRR of **0.5919**. The clicked article is in the retrieved pool for
2.03% of impressions; stage 2 cannot recover the other 98%.

Two things worth reading off the conditional rows rather than the headline. First, the re-ranker
is *not* the problem: on the 2% where stage 1 did retrieve the click, it lifts MRR 5x over BM25's
own ordering (0.0210 → 0.1079). Stage 1 is the problem, and that is a cleaner argument than "the
numbers are bad". Second, the conditional rows are explicitly **not a system metric** — the
script labels them so — because quoting them alone would be exactly the favourable slicing Q9
exists to discourage.

End-to-end omits AUC on purpose: MRR and nDCG are genuinely 0 on a miss, but AUC with no positive
present is *undefined*, not 0, and averaging a 0 in would manufacture a number. An earlier draft of
the script reported end-to-end AUC 0.0107, which was exactly that mistake.

**MIND was not run, and does not need to be.** Its ceiling is already measured by A1 on the full
split (recall@200 = 1.43%, *lower* than EB-NeRD's), so the argument transfers. Running it would
also require regenerating a cache — see Gotchas.

### 2026-09-11 — A GBDT cannot reproduce its own continuous input feature

**This corrects an earlier entry.** MIND's submission-safe re-ranker scores 0.5991 against
`embed_cos`'s 0.6168 — losing to one of its own 19 inputs. I first wrote that off as "the data",
which was incomplete. The dominant cause is mechanical and applies everywhere.

The decisive test: train the model on **`embed_cos` alone**, nothing else.

| | MIND test AUC | distinct scores per impression |
|---|---|---|
| raw `embed_cos` | **0.6168** | 97.4% |
| GBDT given only `embed_cos` | 0.6067 | 87.4% |
| … 4x finer bins (`max_bin=1024`) | 0.6080 | 94.4% |
| … 32x finer bins | 0.5979 | 94.8% |

With a single input and nothing to be confused by, it still loses. **LightGBM bins every feature**
(`max_bin`, default 255) and each tree emits one value per leaf, so a continuous score's ordering
survives only to that resolution. Candidates the raw feature cleanly separates come out tied, and
ties break arbitrarily. Finer bins raise the distinct-score rate but start overfitting before they
close the gap.

So the ~0.018 shortfall decomposes into a **quantisation tax of ~0.010** — paid by any re-ranker
over a strong continuous feature, on any dataset — plus ~0.008 from 18 near-random companions.

**Why it only shows up on MIND submission-safe:** the tax is fixed, the gains are not. EB-NeRD's
model gains +0.23 from combining features, so 0.01 is invisible. Strip MIND's click features and
there is nothing left to gain, so the tax is the entire result.

Still true from the earlier diagnosis, now as the *second* cause rather than the only one: it is
not hyperparameters (truncation 20/50/100, lr 0.05/0.01, subsampling, `min_data_in_leaf` all give
the same shape; `lr=0.01` reproduces `lr=0.05`'s round-1 score exactly), not an unrepresentative
val split (val tracks test at every fixed round count), and not the three zero-variance columns
(dropping them changes nothing).

**Mitigation shipped:** `LambdaRanker.predict(tie_breaker=...)` orders candidates the model scored
*identically* by a continuous input. It cannot reverse any ordering the model expressed, so it is
non-harmful by construction, and it is reported as its own row rather than folded into the model.
Measured effect is exactly where the theory predicts — it moves only the weak model:

| run | re-ranker | + tie-break |
|---|---|---|
| EB-NeRD full | 0.8078 | 0.8078 |
| EB-NeRD submission-safe | 0.7594 | 0.7594 |
| MIND full | 0.6898 | 0.6898 |
| MIND submission-safe | 0.5991 | **0.6024** |

A no-op in three runs out of four is the confirmation that it is not quietly reshuffling anything:
EB-NeRD slates average 12 candidates with a confident model, so there is almost nothing tied to
break; MIND's 38-candidate slates with a one-round model are full of ties.

**The Q9 conclusion is unchanged**, only better explained. Even with the tie-break, MIND's
submission-safe re-ranker (0.6024) still loses to `embed_cos` (0.6168): on MIND the entire value of
the re-ranker comes from click feedback the released test set withholds. On EB-NeRD it does not.

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
- **Namespacing MIND's ids made A1's cached candidate files unjoinable.** The fix for the
  4,116 colliding ids renamed every MIND `impression_id` to `train:N` / `dev:N`; the three caches
  under `data/processed/mind/{bm25,embeddings_eval,embeddings_eval_mpnet}/candidates_test.parquet`
  were written in August under the old ids and now join to zero rows. EB-NeRD is unaffected
  (71,631/71,631). **A1's reported numbers are safe** — they are aggregates in `results/`, not
  keyed on ids. Regenerating is `python scripts/run_bm25.py --dataset mind` and it is *slow*: it
  uses A1's full-catalogue `query()` path, and was killed after 40 min of CPU with no visible
  progress. Run it with `python -u` so stdout is not buffered, and only if something actually needs
  the file. `run_two_stage.py` exits with this exact instruction if it hits the stale cache.
  **Note:** `data/processed/mind/bm25/candidates_test.parquet` currently holds only 400 rows from
  an aborted `--limit` run — it is git-ignored and nothing reads it, but do not mistake it for real.
- **Materialising 200 candidates per impression OOM-killed the machine.** 71,631 × 200 = 14.3M
  rows; the feature transform hit 8.3GB resident and the kernel killed it, taking VS Code with it.
  Anything that expands impressions by a large factor must batch — `run_two_stage.py` now does
  4,000 impressions at a time and holds steady at ~6GB free. Worth remembering for the large-tier
  submission (13.5M impressions × ~12 candidates ≈ 162M rows).
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

### 2026-09-11 — Q2.1 run literally, so the substitution is measured rather than argued

**Item(s):** closes the one open caveat on `Q2-A2`. **Q2 Option A is now complete on all four
sub-items.**
**Did:** `scripts/run_two_stage.py` — A1's full-catalogue top-200, re-ranked by the trained model,
against A1's cached candidate lists. Full EB-NeRD test split, 14.3M (impression, candidate) rows,
batched at 4,000 impressions so it never materialises the whole matrix. Result in
`results/ebnerd/two_stage_literal.json`.
**State:** Works. **It took the machine down first** — see the OOM note in Gotchas.
**Next:** `Q5-A`/`Q5-B`, or `Q4`. Also see the MIND cache note below before touching anything that
joins on MIND's `impression_id`.
**For Anurag:** if `Q2-B` ever needs the literal pipeline for comparison, the script takes any
model exposing `predict(matrix)`; the retrieved lists are already on disk for EB-NeRD.

### 2026-09-11 — Q2-A2: LightGBM LambdaRank

**Item(s):** `Q2-A2` — done. Q2 Option A complete.
**Did:** `LambdaRanker` in `reranker.py` (train / predict / save / load / feature_importance /
degenerate_features) plus `scripts/run_reranker.py`. Four runs: both datasets x full and
submission-safe feature sets. `results/rerank_comparison.md` holds the table,
`results/<ds>/rerank_before_after*.json` the detail. Added `lightgbm>=4.0,<5` and the `*.lgb` /
`models/` gitignore entries. 129 tests, was 121.
**State:** Works. EB-NeRD trains in ~50s, MIND in ~18s; feature building dominates at ~2 min a
dataset. **Read the two decisions below before quoting any number from this** — one result is a
failure, and it is a real one rather than a bug.
**Next:** `Q4`/`Q5` are still unassigned, and `Q5-A` (`--method reranker` in `run_eval.py`) is the
natural follow-on.
**For Anurag:** your `Q2-B` neural ranker consumes the same matrix. The MIND submission-safe
finding below is the one to read — it may well hit your model too, and if it does that is a finding
about the data rather than about your architecture.

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
