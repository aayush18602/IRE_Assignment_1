# Q4: Offline Evaluation Harness -- BM25 vs. Embeddings (raw XLM-R)

Full test splits, official-style metrics (re-ranking each impression's own shown candidates,
not a full-catalog list -- see `scripts/run_eval.py`). Full per-metric bootstrap 95% CIs in the
raw JSON files in this directory (`{ebnerd,mind}/eval/{bm25,embeddings}_eval_test.json`); the
tables below show point estimates only.

## EB-NeRD (71,631 test impressions)

| Metric | BM25 | Embeddings (XLM-R) | Winner |
|---|---|---|---|
| AUC | 0.497 | **0.526** | Embeddings |
| MRR | 0.319 | **0.334** | Embeddings |
| nDCG@5 | 0.348 | **0.367** | Embeddings |
| nDCG@10 | 0.432 | **0.449** | Embeddings |
| Diversity | **0.059** | 0.049 | BM25 |
| Novelty | 17.21 | **17.70** | Embeddings |
| Coverage | **0.936** | 0.478 | BM25 |

## MIND (34,519 test impressions)

| Metric | BM25 | Embeddings (XLM-R) | Winner |
|---|---|---|---|
| AUC | 0.545 | **0.557** | Embeddings |
| MRR | **0.282** | 0.276 | BM25 |
| nDCG@5 | **0.257** | 0.251 | BM25 |
| nDCG@10 | **0.318** | 0.315 | BM25 |
| Diversity | 0.059 | **0.061** | Embeddings |
| Novelty | 17.42 | **17.50** | Embeddings |
| Coverage | **0.982** | 0.620 | BM25 |

## Cold (bottom 25th percentile of history_length) vs. warm

AUC drops for cold users in 3 of the 4 (dataset, method) runs -- EB-NeRD embeddings (0.526 ->
0.519 cold), MIND BM25 (0.545 -> 0.522 cold), MIND embeddings (0.557 -> 0.537 cold) -- roughly
matching the expected pattern of less personalization signal to work with. The exception is
EB-NeRD BM25, where cold and warm AUC are statistically indistinguishable (0.499 vs 0.497,
both near-random) -- not surprising given BM25's *overall* AUC there is already ~random, so
there's little personalization signal for the cold/warm split to visibly degrade further.
Interestingly nDCG@5/@10 sometimes goes the *other* way (MIND, both methods: cold nDCG@10 >
warm nDCG@10) -- plausibly because cold users are shown candidate sets the original system
already biased toward safer/more-popular picks, which are easier to rank correctly with less
personalization. Full slice numbers in the JSON files.

## Reading

**The headline reversal from Q2/Q3's candidate-generation recall@K:** there, BM25 won on
EB-NeRD and only lost (to fine-tuned embeddings) on MIND. Here, on the *official ranking task* --
re-ranking a small, already-curated candidate set rather than searching the full catalog --
**embeddings win the accuracy metrics (AUC/MRR/nDCG) on EB-NeRD outright**, and are roughly tied
with BM25 on MIND (embeddings ahead on AUC, BM25 ahead on MRR/nDCG@5/@10, all by small margins).

This makes sense once the two tasks are pulled apart. Candidate generation (Q2/Q3) is "find the
few needles in a 20K-65K article haystack" -- exact term matching is a strong signal there.
Re-ranking (Q4) is "these ~10-30 candidates were already selected as broadly relevant by
whatever produced the historical impression -- which one specifically got clicked?" -- a subtler
distinction among already-similar items, where continuous semantic similarity can pick up on
nuance that binary term overlap can't. **A two-stage system (BM25 for candidate generation, an
embedding-based re-ranker on top) plays to both methods' actual strengths** -- a natural
architecture takeaway for the design note (Q6).

**Coverage is the other clear, consistent story: BM25 covers 58-96% more of the catalog than
embeddings (MIND / EB-NeRD respectively).** Exact term matching depends on which specific words appear in
each user's own history, which varies a lot across users and naturally spreads recommendations
across the catalog. Mean-pooled continuous embeddings cluster candidates from a narrower
"semantic neighborhood" repeatedly, favoring the same central/popular articles across many
different users -- a real filter-bubble risk for the embedding-only system that the accuracy
metrics alone don't surface. This is exactly why the assignment asks for beyond-accuracy metrics
alongside AUC/MRR/nDCG, not instead of them.
