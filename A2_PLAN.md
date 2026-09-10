# Learning from Click-Logs — A2 Build Plan

**CS4.406 · Assignment 2 · Build Plan**

Adding behavioural signals, a two-stage re-ranker, and a reproduced-then-beaten neural baseline
on top of the Assignment 1 retrieval pipeline — for EB-NeRD and MIND.

| | |
|---|---|
| **Due** | Sep 20, 2026 |
| **Working days** | 10 |
| **Team** | 2 people |
| **Branch** | `a2` |
| **Reused from A1** | ~80% |

> This file is git-ignored. It is the internal working plan; the graded write-up is
> `design_note.md`.

---

## Base — What A1 already gives you

A2 is explicitly a continuation, and the existing repo covers more of it than it looks. The
honest split is: the data layer, the metrics, and the large-scale scoring loop are done; the
modelling layer is entirely new.

### Reuse as-is

- `bm25.py` — the fast candidate-first `score_candidates`; becomes a re-ranker *feature*
- `ann.py` — FAISS index + user embedding; another feature, and the Q4 memory measurement
- `split.py` — temporal split with the leakage assertion baked in
- `eval.py` — AUC/MRR/nDCG, diversity/novelty/coverage, cold-warm slicing
- `generate_submission.py` — the batched 13.5M-row scoring loop; add a `--method reranker`
- `feature_store.history_asof` — the as-of cutoff primitive Q1.4 is asking for

### Must be extended or written

- `schema.py` / `clean.py` — currently *drop* every behavioural column (see below)
- `behaviour.py` **new** — Q1 feature engineering
- `reranker.py` **new** — Q2 LightGBM LambdaRank
- `nrms.py` **new** — Q3 baseline + improvement
- `stats.py` **new** — *paired* bootstrap; A1 only has the unpaired one
- `serving.py` **new** — Q4 latency/memory instrumentation

### ▸ Load-bearing decision — settle this first

**Which articles does the re-ranker actually put in order?** Start from one row of the data:

```
candidates : [A, B, C, D, E, F, G, H, I, J, K, L]  ← the 12 the site showed
clicked    : [F]                                   ← the one they clicked
```

The task is to put those 12 in order with **F** first. That is also exactly what both Codabench
competitions score: they hand you the shown list and want the same list back, reordered.

A1 did something different. It ignored the 12 and searched the whole 20,738-article catalogue for
200 the user might like. Measured against ground truth, **F was among those 200 only 2 times in
100** — that is what recall@200 = 2.01% means.

So Q2.1 read literally — retrieve 200, then re-rank them — hands the re-ranker a pile that *does
not contain the right answer* 98% of the time. Sorting cannot add articles, so the system is wrong
before the model runs. It is also unsubmittable: 200 catalogue articles is not a reordering of the
12 Codabench asked about.

**Rank the shown list instead, and demote A1 to a feature.** The clicked article is guaranteed to
be in there. A1's BM25 and embedding scorers stop *choosing* articles and start *rating* them: one
number per candidate — "how well does this match what the user read recently?" — sitting in the
feature table next to popularity, freshness, and category match. A1's generator is still in the
pipeline; it contributes its opinion rather than its shortlist.

Then run the literal version anyway and report it, because the measurement is the justification:
*"recall@200 of 2.01% caps any full-catalog two-stage system at 2% regardless of re-ranker
quality, so we rank the impression's own slate and use the A1 retriever as a scoring feature."*
That paragraph earns more than silently picking one.

---

## Q1 — Click-history & session features

**Blocking prerequisite:** A1's `clean.py` keeps only seven columns per impression. Every
behavioural signal A2 asks for is thrown away during cleaning. Widening the schema is task zero,
it touches shared code, and it should be merged within the first day so nobody is blocked behind
it.

### Columns A1 discards that A2 needs

All present in `data/ebnerd_*/train/behaviors.parquet` and `history.parquet`.

| Raw column | Carries | Feeds |
|---|---|---|
| `session_id` | Session grouping | Q1.2 session context |
| `read_time`, `scroll_percentage` | Dwell on the impression row | Q1.2 dwell — **and Q9** |
| `read_time_fixed`, `scroll_percentage_fixed` | Per-item dwell in history | Q1.1 weighted history |
| `age`, `gender`, `is_subscriber`, `postcode` | User demographics | Q1 static user features |
| `next_read_time`, `next_scroll_percentage` | The *next* article read | **Q9 leakage trap only** |
| `total_inviews`, `total_pageviews` | Corpus-lifetime article counts | **Q9 leakage trap only** |

### Feature families to build

Everything goes in `src/ire_a1/behaviour.py`, and **every builder takes an explicit `as_of`
datetime** — not a default, not an optional. That single API choice is what makes Q1.4 and Q9
provable rather than asserted.

| Feature | EB-NeRD | MIND | MIND fallback |
|---|---|---|---|
| History length, click count | yes | yes | — |
| Exp-decay recency weighting | real timestamps | no per-item time | Positional decay over the ordered history |
| Time since last click | yes | no | Drop; flag as unavailable |
| Dwell / scroll depth | yes | no | Drop |
| Session context | `session_id` | derive | Segment on a timestamp-gap threshold (e.g. 30 min) |
| Position bias in the in-view list | yes | yes | — |
| Article freshness | `published_time` | 100% null | Proxy: first appearance in the *train* logs |
| Trailing-window popularity | yes | yes | — |
| Category match with history | yes | yes | — |
| A1 BM25 / embedding score | yes | yes | — |

The availability gap is the real design work here — half these features only exist on one dataset.

### ▸ Design it in, don't discover it late

The two datasets do not support the same feature set, and pretending otherwise produces silent
all-null columns that LightGBM will happily ignore while you wonder why MIND underperforms. Build
a per-dataset feature manifest, assert it at train time, and put the availability table straight
into the design note — the asymmetry is a finding, not an embarrassment.

### Boundary enforcement (Q1.4)

The one genuinely new primitive is **trailing-window popularity**: click counts over
`[as_of − window, as_of)`, strictly exclusive on the right. Implement it once as a
sorted-timestamp prefix-sum over the train impressions so it stays O(1) per lookup at 13.5M rows,
and add the boundary case to `tests/test_no_leakage.py`: an article clicked at *exactly* `as_of`
must not appear in the count.

---

## Q2 — The re-ranker

Take **Option A: LightGBM with the LambdaRank objective**, grouped by impression. It needs no GPU,
trains in minutes on 24 cores, handles the large-tier scoring pass, and hand-crafted behavioural
features are exactly what Q1 produces. Option B is not skipped — it becomes Q3's neural track, so
the assignment gets both without doubling the work.

| What | Where | Note |
|---|---|---|
| Feature matrix builder | `src/ire_a1/reranker.py → build_matrix()` | One row per (impression, candidate). Group array = candidate count per impression — LightGBM ranking breaks silently if this is misaligned. |
| LambdaRank training on the temporal train split | `scripts/run_reranker.py --dataset {ebnerd,mind}` | Early-stop on the val split's nDCG@10. Fixed seed, pinned lightgbm version, model saved outside git. |
| Before/after table (Q2.4) | `results/<ds>/rerank_before_after.md` | Stage-1-only scores vs. re-ranked, same impressions, AUC/MRR/nDCG@5/nDCG@10. This is your headline result. |
| Feature importance dump | `results/<ds>/feature_importance.json` | Gain-based. Costs one line, earns a design-note figure, and is how you catch a leaky feature dominating the split. |

### ▸ Expect a large jump — and interrogate it

A1's stage-1 AUC on the official ranking task was 0.497–0.557, barely above chance on EB-NeRD. A
GBDT over popularity and recency features will beat that comfortably. If the jump looks *too*
good, check the feature importances before celebrating: a trailing popularity window whose
boundary is off by one impression will look like a brilliant model.

---

## Q3 — Baseline reproduced, then beaten

Reproduce **NRMS** — multi-head self-attention over title tokens, additive attention over click
history — on both datasets. It is the canonical baseline for both the `ebnerd-benchmark` repo and
MIND, so one implementation satisfies Q3.1 twice. This is the GPU track and the critical path; it
runs on Kaggle.

### The one principled change

**Recency-weighted history attention.** NRMS attends over a user's clicked articles with no notion
of when they were clicked. News decays in hours — the assignment's own framing. Multiply the
attention logits by an exponential decay in `as_of − click_time`, with a single learnable
half-life.

It is one change, it is motivated by the domain rather than by hyperparameter luck, the ablation
is exactly "half-life on vs. off" with everything else frozen, and it degrades cleanly to
positional decay on MIND where per-item timestamps do not exist.

### ▸ Q3.4 is where claims get thrown out

"Paired bootstrap 95% CI that excludes zero" is not the same as two overlapping CIs. Resample
*impressions*, and for each resample compute the **difference** `metric(improved) −
metric(baseline)` on the *same* impressions, then take percentiles of that difference
distribution.

A1's `eval.bootstrap_ci` is unpaired and will not do this. Write `stats.paired_bootstrap_ci()`,
unit-test it against a known-zero-difference case, and use it for every A/B claim in the report.
Same seed for both arms.

---

## Q4 — Serving & scale

Fully local, no GPU, one afternoon. The design note already has a real 10× story from A1 — the
~1000× `score_candidates` rewrite that took a 47-hour run down to 15 minutes — so this section
extends a measured narrative rather than starting one.

| What | Where | Note |
|---|---|---|
| Index memory | `scripts/run_serving_bench.py --measure memory` | FAISS index bytes (ntotal × d × 4), BM25 postings via a recursive sizeof walk, feature-store parquet resident size. Report per dataset. |
| p99 latency, single user request | `--measure latency --n 2000` | Time the whole path: history lookup → candidate generation → feature build → LightGBM predict. Report the p50/p95/p99 breakdown by stage, not one aggregate number — the stage split is the interesting part. |
| Cost per 1000 queries at p99 < 100 ms | `design_note.md §Serving` | Measured QPS per core → cores needed for the SLA → a public cloud instance-hour rate. Show the arithmetic; a back-of-envelope with visible assumptions beats a confident single number. |
| 10× scaling argument | `design_note.md §10×` | What breaks first, with a measurement behind it. Strong candidates: the FAISS flat index is linear in catalog size, and per-impression feature building is the new bottleneck once BM25 stops being one. |

---

## Q5 — Extended evaluation & leaderboards

Mostly plumbing: teach `run_eval.py` a `--method reranker` and run the full metric suite over the
two-stage pipeline. Two things are genuinely new.

**Head/tail slicing.** A1 has cold/warm on `history_length`; Q5 also wants head vs. tail
*articles*. Split on a percentile of train-split click count, reusing the existing
`percentile_threshold` helper so both slices are derived from the data rather than hard-coded.

**Both leaderboards, with screenshots.** Non-negotiable deliverable, and the scheduling risk below
is the single most likely way this assignment loses marks.

### ▸ Submit EB-NeRD in week one, not on the last day

In A1, `predictions.zip` sat in "submitted" status on the RecSys Codabench for over six days and
never scored before the deadline — the design note had to report EB-NeRD leaderboard results as
unavailable. **Re-submit the existing A1 embeddings predictions to EB-NeRD on day one** to start
that clock immediately, then submit the improved re-ranker output when it exists. Worst case you
have a score to screenshot; best case you have two and can show the delta.

---

## Q9 — Anti-gaming, for real this time

A1 had to *construct* a leaky feature because the retrieval pipeline never touched one. A2 does
not have that problem: EB-NeRD ships three features that are in the logs and genuinely unavailable
at serving time.

- `next_read_time`, `next_scroll_percentage` — describe the article read *after* this impression.
  Pure future information.
- `total_inviews`, `total_pageviews`, `total_read_time` — lifetime article counters aggregated over
  the whole corpus period, test window included.
- `read_time`, `scroll_percentage` on the impression row — measured after the click. Available in
  the log, unknowable when you rank.

So Q9 becomes an honest experiment rather than a synthetic demo: **train the same re-ranker with
and without this group, report both metric sets side by side, and ship the paired CI on the gap.**
Then state plainly that the reported system excludes them. That is a stronger answer to "report
metrics with and without features unavailable at serving time" than anything you could stage.

---

## Task division — by assignment question

Divided the way the assignment is: **whole questions to one person**, except Q2, where the two
options split naturally.

| Question | Owner | What it covers |
|---|---|---|
| **Q1** — click-history & session features | Aayush | All four sub-parts, plus the schema widening that unblocks them |
| **Q2 Option A** — GBDT re-ranker | Aayush | Feature matrix + LightGBM LambdaRank |
| **Q2 Option B** — neural ranker | Anurag | MLP over the *same* feature matrix |
| **Q2.4** — before/after metrics | Both | One table covering both re-rankers |
| **Q3** — baseline reproduced, then beaten | Anurag | NRMS, the improvement, the ablation, the paired CI |
| **Q4** — serving & scale | *undecided* | Assign once Q2 lands |
| **Q5** — extended evaluation | *undecided* | Assign once Q2 lands |

Two things worth knowing before you start.

**Doing both Q2 options is more than the assignment requires.** Q2.2 says "Option A *or* Option B".
Building both is a legitimate and better answer — it gives a real GBDT-vs-neural comparison on
identical features — but it is optional scope. If the deadline tightens, Option B is the first
thing to cut, not Q3.

**Q2 Option B and Q3 are different models, not the same one twice.** Option B is an MLP over the
*engineered features* from Q1. NRMS encodes *raw titles* with self-attention and never sees those
features. Anurag owns both, so it is worth being deliberate: they share a training harness but not
an architecture.

### The seam between Q2-A and Q2-B

`Q2-A1` builds the feature matrix — one row per (impression, candidate), plus the group array. Both
re-rankers consume it. **Aayush owns it, Anurag depends on it**, so agree the output shape before
Q2-A1 is written:

```python
# reranker.build_matrix(...) -> (X, y, groups, feature_names)
#   X            : float32 [n_rows, n_features]
#   y            : int8    [n_rows]        1 if clicked
#   groups       : int32   [n_impressions] candidates per impression, in row order
#   feature_names: list[str]               index-aligned with X's columns
```

That contract is the whole interface. Once it is fixed, Q2-A2 and Q2-B1 proceed independently.

### Aayush — Q1 and Q2 Option A

| ID | Item | Files | Size | Owner | Needs | Kind |
|---|---|---|---|---|---|---|
| Q1-A | Widen schema and `clean.py` for behavioural columns | `src/ire_a1/schema.py`, `src/ire_a1/clean.py` | M | **Aayush** | — | modify |
| Q1-B | Click-history features + exponential recency decay | `src/ire_a1/behaviour.py` | L | **Aayush** | Q1-A | new |
| Q1-C | Session context, dwell, position bias | `src/ire_a1/behaviour.py` | M | **Aayush** | Q1-A | new |
| Q1-D | Article features: trailing-window popularity, freshness, category match | `src/ire_a1/article_features.py` | L | **Aayush** | Q1-A | new |
| Q1-E | Behaviour-window boundary enforcement + leakage tests | `tests/test_no_leakage.py`, `tests/test_behaviour.py` | M | **Aayush** | Q1-B, Q1-C, Q1-D | modify |
| Q2-A1 | Feature matrix builder and impression grouping — **shared substrate** | `src/ire_a1/reranker.py` | L | **Aayush** | Q1-E | new |
| Q2-A2 | Option A: LightGBM LambdaRank training, scoring, CLI | `src/ire_a1/reranker.py`, `scripts/run_reranker.py` | L | **Aayush** | Q2-A1 | new |

### Anurag — Q2 Option B and Q3

| ID | Item | Files | Size | Owner | Needs | Kind |
|---|---|---|---|---|---|---|
| Q2-B1 | Option B: neural ranker (MLP over the same feature matrix) | `src/ire_a1/neural_ranker.py` | L | **Anurag** | Q2-A1 | new |
| Q2-B2 | Option B: training loop and CLI | `scripts/run_neural_ranker.py` | M | **Anurag** | Q2-B1 | new |
| Q3-A | NRMS: news and user encoders | `src/ire_a1/nrms.py` | L | **Anurag** | Q1-A | new |
| Q3-B | NRMS: training loop and CLI, both datasets | `src/ire_a1/nrms.py`, `scripts/train_nrms.py` | L | **Anurag** | Q3-A | new |
| Q3-C | Improvement: recency-weighted history attention | `src/ire_a1/nrms.py` | M | **Anurag** | Q3-B | new |
| Q3-D | Paired bootstrap CI and unit tests | `src/ire_a1/stats.py`, `tests/test_stats.py` | M | **Anurag** | — | modify |
| Q3-E | Ablation harness: half-life on vs. off, with paired CIs | `scripts/run_ablation.py` | M | **Anurag** | Q3-C, Q3-D | new |

### Shared

| ID | Item | Files | Size | Owner | Needs | Kind |
|---|---|---|---|---|---|---|
| Q2-M | Q2.4 before/after metrics for both re-rankers | `results/*/rerank_before_after.md` | M | **Both** | Q2-A2, Q2-B2 | new |

### Undecided — Q4 and Q5

Left as `TBD` deliberately. Both depend on `Q2-A2`, so there is no cost to deciding later, and by
then you will know who has capacity.

| ID | Item | Files | Size | Owner | Needs | Kind |
|---|---|---|---|---|---|---|
| Q4-A | Index memory and staged latency instrumentation | `src/ire_a1/serving.py` | M | **TBD** | Q2-A2 | new |
| Q4-B | Serving benchmark CLI and cost/QPS arithmetic | `scripts/run_serving_bench.py` | M | **TBD** | Q4-A | new |
| Q5-A | Eval harness: `--method reranker` arm | `scripts/run_eval.py` | M | **TBD** | Q2-A2 | modify |
| Q5-B | Head/tail article slicing | `src/ire_a1/eval.py`, `scripts/run_eval.py` | M | **TBD** | — | modify |
| Q5-C | Large-tier prediction generation via the re-ranker | `scripts/generate_submission.py` | M | **TBD** | Q2-A2 | modify |

**Worth knowing when you assign these:** Q5 is the cheapest question in the assignment. Three of
its four code parts are `modify`, and the metric suite, bootstrap CIs and cold/warm slicing are
already built and working in `run_eval.py`. Q4 is entirely new code but small and self-contained.

### Balance so far

Aayush 7 items, Anurag 7, shared 1, undecided 5. The decided half is even; the five TBD items are
what you have left to trade.

### Rules

1. **Fix the `build_matrix` contract before Q2-A1 is written.** It is the only cross-person
   interface in the whole plan.
2. **Pull before you start, push when a piece works.** Work is parallel again, so use short
   branches off `a2` (`a2/q1-features`, `a2/q3-nrms`) and merge with `--no-ff`.
3. **Never hand over broken code.** `python -m pytest tests/ -q` passes before you push.
4. **Update `CONTEXT.md` before you stop** — set your item's status in the ledger and add a session
   log entry. Run `python scripts/check_sync.py`.
5. **Commit messages start with the item ID:** `Q2-A1: feature matrix builder and impression
   grouping`.
6. **`ai_usage_log.md` is appended by both of you, every session.** Q7.4 grades it and asks for
   AI-generated vs. human-written marking.

### Outside the Q1–Q5 division, still required

Two easy-to-miss deliverables that are not part of the Q1–Q5 split above:

- **Q7.1 — one-command reproduce.** The README documents a command *per question*, not one command.
  Q7.1 requires a single entry point: add `scripts/reproduce.sh` (or a Makefile) that runs the whole
  chain end to end on the small tiers, and run it clean once. Assign it with Q4/Q5.
- **Q8 — gitignore gap.** `*.pt` and `*.ckpt` are covered; LightGBM artefacts are not. Add `*.lgb`,
  `*.txt.model` and `models/` **before the first model is saved**, not after it is committed.
  Whoever writes `Q2-A2` does this in the same commit.

### Not divided — do these together at the end

Not code, so not in the ledger: Codabench uploads to both competitions, leaderboard screenshots,
`design_note.md` (6 pages), the `README.md` rewrite, and the final `ai_usage_log.md` pass.

## Sync contract

Every code item above has a **stable ID** — `Q1-A`, `Q2-A1`, `Q3-D` and so on, keyed to the
assignment question it belongs to. Those IDs are the link between the two documents:

- **`A2_PLAN.md` defines the items.** ID, what it is, which files, how big, who owns it, what it
  needs first, and whether it is new code or a modification. This changes rarely.
- **`CONTEXT.md` tracks their completion.** The same IDs, plus status, who finished it, and when.
  This changes every session.

When either of you finishes an item, mark it `done` in `CONTEXT.md`'s task ledger and add a
session-log entry naming it. Whoever sits down next reads the ledger and knows what is finished,
what is open, and what is still unassigned.

**Status vocabulary:** `todo`, `wip`, `blocked`, `done`.
**Owner vocabulary:** `Aayush`, `Anurag`, `Both`, `TBD`.

**Enforcement.** `scripts/check_sync.py` fails if:

- a plan item has no ledger row, or a ledger row invents an ID the plan doesn't define
- an owner or status is outside the vocabularies above
- the ledger's owner disagrees with the plan's
- an item is `done` with nobody's name and date against it
- an item is `done` while still owned by `TBD` — assign Q4/Q5 in both files before marking them
- an item is `done` while something in its **Needs** column is still open, which would mean
  someone built on code that was never pushed

It prints per-person progress and names the next item. Run it after editing either file:

```bash
python scripts/check_sync.py
```

To have git enforce it, install the hook once:

```bash
printf '#!/bin/sh\nexec python scripts/check_sync.py\n' > .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

**Assigning Q4 or Q5 later?** Change `**TBD**` to the owner in this file's table *and* in
`CONTEXT.md`'s ledger, in the same commit. The checker will catch you if you do only one.

**Adding an item later?** Add the row here with the next free ID and its prerequisites, add the
matching ledger row in `CONTEXT.md` with status `todo`, same commit.

## Git — Branch strategy

One `a2` branch on the existing repo, not a new repo. A2 builds on A1 by design, and keeping the
lineage visible in one history is worth more to a grader than a clean slate.

Work on short branches off `a2` and merge with `--no-ff`, so each question's work stays visible as
a unit in the history:

```bash
git checkout a2 && git pull
git checkout -b a2/q1-features        # or a2/q3-nrms
# ... commit as you go ...
git push -u origin a2/q1-features
git checkout a2 && git pull && git merge --no-ff a2/q1-features && git push
```

File ownership makes conflicts rare: the only file both of you touch is `reranker.py` at the
`Q2-A1` / `Q2-B1` seam, and even there Anurag's Option B lives in its own `neural_ranker.py`.

```bash
# once -- Aayush
git checkout main && git pull
git checkout -b a2
git push -u origin a2

# once -- Anurag
git clone https://github.com/aayush18602/IRE_Assignment_1.git
cd IRE_Assignment_1 && git checkout a2
```

Add Anurag under *Settings → Collaborators* on `aayush18602/IRE_Assignment_1` so he can push. Do
not rename the `src/ire_a1/` package — a rename rewrites every import in every script and test and
buys nothing but a tidier name.

---

## Handoff — Getting it running on your teammate's machine

The code is the easy half. The awkward half is that `data/` is git-ignored and holds about 9 GB —
but your teammate does not need most of it. The neural track reads processed parquets only, and
that working set is **91 MB**.

| Tier | Contents | Size | Unlocks |
|---|---|---|---|
| **1 — send now** | `data/processed/*/{articles,impressions_*,user_history}.parquet` | 91 MB | Q1, Q2, Q3, Q5 — the whole neural track |
| 2 — if needed | `data/processed/*/embeddings*.parquet` | 477 MB | Embedding-similarity features, A1 stage-1 scores |
| 3 — you keep | `data/ebnerd_*`, `data/MIND*`, `data/processed_large/` | 7.8 GB | Codabench submissions only — run these locally |

Send tier 1 today. Tiers 2 and 3 only when the work actually reaches them.

### Build the tier-1 bundle

```bash
tar czf a2_working_set.tar.gz \
  data/processed/*/articles.parquet \
  data/processed/*/impressions_*.parquet \
  data/processed/*/user_history.parquet
```

Drop it on Google Drive and send the link. The paths inside the archive are repo-relative, so it
untars straight into the right place with no path fixing.

### Your teammate's setup, start to finish

```bash
git clone https://github.com/aayush18602/IRE_Assignment_1.git
cd IRE_Assignment_1
git checkout a2

# uv, not venv — python3-venv is missing on at least one of your machines
uv venv .venv
uv pip install -p .venv -r requirements.txt
source .venv/bin/activate

# restore the shared working set at the repo root
tar xzf ~/Downloads/a2_working_set.tar.gz

# confirm the environment is real, not just installed
python -m pytest tests/ -v
python scripts/run_eval.py --dataset mind --method bm25 --limit 200
```

### ▸ The one thing they cannot inherit from you

**MIND is gated on HuggingFace.** If they ever need to regenerate MIND from raw, they must accept
the terms at `huggingface.co/datasets/yjw1029/MIND` under their own account and create their own
token — your token will not transfer and the download fails with an unhelpful error. Tier 1
sidesteps this entirely, which is another reason to send it rather than telling them to
re-download.

Two loose ends worth closing in the same pass. `requirements.txt` needs `lightgbm` added, and
`requirements-gpu.txt` is what the Kaggle notebook installs for the NRMS track — pin both so the
two of you are not debugging a version skew at 11pm on the 19th. And the `scripts/reproduce.sh`
from Q7.1 doubles as the real onboarding test: if it runs clean on their machine, the handoff
worked.

---

*CS4.406 Information Retrieval & Extraction · Assignment 2 · built on
aayush18602/IRE_Assignment_1 @ e6d8099*
