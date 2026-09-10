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
| Q1-A | Widen schema and `clean.py` for behavioural columns | Aayush | todo | — |
| Q1-B | Click-history features + exponential recency decay | Aayush | todo | — |
| Q1-C | Session context, dwell, position bias | Aayush | todo | — |
| Q1-D | Article features: trailing-window popularity, freshness, category match | Aayush | todo | — |
| Q1-E | Behaviour-window boundary enforcement + leakage tests | Aayush | todo | — |
| Q2-A1 | Feature matrix builder and impression grouping — **shared substrate** | Aayush | todo | — |
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
| Q5-B | Head/tail article slicing | TBD | todo | — |
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

- **Anurag is waiting on `Q1-A`** (schema widening, Aayush) before `Q3-A` can start.
- **Anurag is waiting on `Q2-A1`** (feature matrix, Aayush) before `Q2-B1` can start. **Agree the
  `build_matrix` output contract before Q2-A1 is written** — it is the only cross-person interface
  in the plan, and the shape is specified in `A2_PLAN.md`.
- **Q4 and Q5 have no owner yet.** Both depend on `Q2-A2`; decide when it lands.

---

## Decisions taken

Append-only. One entry per decision that someone else would otherwise re-litigate.

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

---

## Session log — Aayush

Newest entry at the top. Only Aayush edits this section.

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
