"""Q4: offline evaluation harness.

- Ranking metrics (AUC, MRR, nDCG@5, nDCG@10): computed the way the official MIND/EB-NeRD
  leaderboards do it -- re-rank each impression's own *shown* candidates (not a full-catalog
  top-K list) by score, compare against the binary click label. Uses sklearn's roc_auc_score/
  ndcg_score (well-tested reference implementations; the assignment's "build it yourself"
  instruction was specifically about the inverted index in Q2, not about reimplementing AUC).
- Beyond-accuracy (diversity, novelty, coverage): computed over each method's actual top-K
  *recommended* list (the Q2/Q3 candidate-generation output), which is what these metrics are
  about -- they don't make sense evaluated against a handful of officially-shown candidates.
- Bootstrap 95% CIs: resample impressions with replacement, recompute the mean each time.
"""
from collections import defaultdict

import numpy as np
from sklearn.metrics import ndcg_score, roc_auc_score


def article_popularity(clicked_lists: list[list[str]]) -> dict[str, int]:
    """Click counts per article_id, from whatever set of impressions' `clicked` columns the
    caller passes in -- e.g. only the train split (Q9's serving-time-safe popularity, since
    train entirely precedes val/test in the temporal split) vs. train+val+test combined (Q9's
    deliberately-leaky popularity, since it uses clicks from the very period being evaluated).
    Caller decides which impressions belong in the count; this function just counts."""
    counts: dict[str, int] = defaultdict(int)
    for clicked in clicked_lists:
        for article_id in clicked:
            counts[article_id] += 1
    return dict(counts)


def ranking_metrics(scores: list[float], labels: list[int]) -> dict[str, float | None]:
    """Per-impression AUC/MRR/nDCG@5/nDCG@10 given per-candidate scores and binary click labels
    in matching order. None for metrics undefined for this impression (e.g. AUC needs both a
    clicked and a non-clicked candidate present) -- caller should drop Nones before aggregating,
    not treat them as 0."""
    labels_arr = np.asarray(labels)
    scores_arr = np.asarray(scores, dtype=float)
    n_pos = int(labels_arr.sum())
    result: dict[str, float | None] = {}

    if n_pos == 0 or n_pos == len(labels_arr):
        result["auc"] = None
    else:
        result["auc"] = float(roc_auc_score(labels_arr, scores_arr))

    if n_pos == 0:
        result["mrr"] = None
        result["ndcg@5"] = None
        result["ndcg@10"] = None
    else:
        order = np.argsort(-scores_arr)
        ranked_labels = labels_arr[order]
        first_hit_rank = int(np.argmax(ranked_labels)) + 1
        result["mrr"] = 1.0 / first_hit_rank
        result["ndcg@5"] = float(ndcg_score([labels_arr], [scores_arr], k=5))
        result["ndcg@10"] = float(ndcg_score([labels_arr], [scores_arr], k=10))

    return result


def intra_list_category_diversity(article_ids: list[str], category_lookup: dict[str, str]) -> float | None:
    """Fraction of unique categories in a recommended list -- 1.0 = every item a different
    category, low = concentrated in a few categories. Category-based (not embedding-based) so
    it's directly comparable between BM25 and embedding recommendation lists alike."""
    cats = [category_lookup.get(a) for a in article_ids]
    cats = [c for c in cats if c is not None]
    if not cats:
        return None
    return len(set(cats)) / len(cats)


def novelty(article_ids: list[str], train_popularity: dict[str, int], n_train_clicks: int) -> float | None:
    """Mean self-information (-log2 popularity) of a recommended list, using click counts from
    the TRAIN split only (never test) as the popularity reference -- higher = the system is
    recommending less-obvious, less-popular items."""
    if not article_ids:
        return None
    scores = []
    for a in article_ids:
        c = train_popularity.get(a, 0)
        p = (c + 1) / (n_train_clicks + 1)  # +1 smoothing avoids log(0) for never-clicked articles
        scores.append(-np.log2(p))
    return float(np.mean(scores))


def coverage(recommended_lists: list[list[str]], catalog_size: int) -> float:
    """Catalog coverage: fraction of the article catalog that appears in *any* recommended list
    across the given impressions. A system-level (not per-impression) statistic."""
    if catalog_size == 0:
        return 0.0
    unique_recommended: set[str] = set()
    for lst in recommended_lists:
        unique_recommended.update(lst)
    return len(unique_recommended) / catalog_size


def bootstrap_ci(
    values: list[float], n_boot: int = 1000, alpha: float = 0.05, seed: int = 42
) -> dict[str, float]:
    """Bootstrap 95% CI (default) for the mean of a list of per-impression metric values.
    Caller must have already dropped None/undefined values."""
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}

    rng = np.random.default_rng(seed)
    n = len(arr)
    boot_means = np.array([rng.choice(arr, size=n, replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": float(arr.mean()), "ci_low": float(lo), "ci_high": float(hi), "n": n}


def bootstrap_coverage_ci(
    recommended_lists: list[list[str]], catalog_size: int,
    n_boot: int = 1000, alpha: float = 0.05, seed: int = 42,
) -> dict[str, float]:
    """Bootstrap 95% CI for coverage -- resamples *impressions* (not individual items) with
    replacement and recomputes coverage over each resampled set of recommended lists.

    `mean` is the actual point estimate (coverage over the real, non-resampled impressions) --
    it can legitimately fall outside [ci_low, ci_high]. This is a known property of the naive
    bootstrap applied to a set-cardinality statistic like coverage: resampling *with
    replacement* duplicates some impressions and drops others, which can only shrink or hold
    the union of recommended items, never grow it beyond the original -- so the bootstrap
    distribution is systematically biased downward relative to the true point estimate.
    `boot_mean` (the mean of the resampled distribution) is reported alongside `mean` so this
    is visible rather than looking like a bug."""
    n = len(recommended_lists)
    if n == 0:
        return {"mean": float("nan"), "boot_mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}

    rng = np.random.default_rng(seed)
    point = coverage(recommended_lists, catalog_size)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = coverage([recommended_lists[j] for j in idx], catalog_size)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": point, "boot_mean": float(boots.mean()), "ci_low": float(lo), "ci_high": float(hi), "n": n}


def cold_warm_slice(history_lengths: list[int], threshold: float) -> list[str]:
    """"cold" (< threshold clicks in raw history) vs "warm" label per impression, aligned by
    position with `history_lengths`."""
    return ["cold" if h < threshold else "warm" for h in history_lengths]


def percentile_threshold(values: list[int], percentile: float = 25.0) -> float:
    """A cold/warm threshold derived from the data itself (default: 25th percentile of
    history_length), rather than a fixed absolute count -- some datasets are pre-filtered by
    their creators to only include "active" users (e.g. EB-NeRD's minimum history_length is
    exactly 5 for every user in the small/demo tiers, so a fixed threshold of 5 produces an
    empty cold slice there), so what counts as "cold" has to be relative to each dataset's own
    distribution to be meaningful."""
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values), percentile))
