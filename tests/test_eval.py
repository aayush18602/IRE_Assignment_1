import numpy as np

from ire_a1.eval import (
    article_popularity,
    bootstrap_ci,
    bootstrap_coverage_ci,
    cold_warm_slice,
    coverage,
    intra_list_category_diversity,
    novelty,
    percentile_threshold,
    ranking_metrics,
)


def test_ranking_metrics_perfect_ranking():
    scores = [0.9, 0.5, 0.1]
    labels = [1, 0, 0]
    m = ranking_metrics(scores, labels)
    assert m["auc"] == 1.0
    assert m["mrr"] == 1.0
    assert m["ndcg@5"] == 1.0


def test_ranking_metrics_worst_ranking():
    scores = [0.9, 0.5, 0.1]
    labels = [0, 0, 1]  # relevant item ranked last
    m = ranking_metrics(scores, labels)
    assert m["auc"] == 0.0
    assert m["mrr"] == 1.0 / 3


def test_ranking_metrics_undefined_cases_return_none():
    all_negative = ranking_metrics([0.5, 0.3], [0, 0])
    assert all_negative["auc"] is None
    assert all_negative["mrr"] is None
    assert all_negative["ndcg@5"] is None

    all_positive = ranking_metrics([0.5, 0.3], [1, 1])
    assert all_positive["auc"] is None
    assert all_positive["mrr"] == 1.0  # MRR well-defined even if AUC isn't


def test_intra_list_category_diversity():
    lookup = {"a": "sport", "b": "sport", "c": "news"}
    assert intra_list_category_diversity(["a", "b", "c"], lookup) == 2 / 3
    assert intra_list_category_diversity(["a", "a"], lookup) == 0.5
    assert intra_list_category_diversity([], lookup) is None
    assert intra_list_category_diversity(["unknown"], lookup) is None


def test_novelty_prefers_less_popular_items():
    popularity = {"popular": 1000, "rare": 1}
    n_train = 1000
    novelty_popular = novelty(["popular"], popularity, n_train)
    novelty_rare = novelty(["rare"], popularity, n_train)
    novelty_unseen = novelty(["never_clicked"], popularity, n_train)
    assert novelty_rare > novelty_popular
    assert novelty_unseen > novelty_rare  # unseen items are the most novel
    assert novelty([], popularity, n_train) is None


def test_coverage_basic():
    lists = [["a", "b"], ["b", "c"]]
    assert coverage(lists, catalog_size=10) == 3 / 10
    assert coverage([], catalog_size=10) == 0.0
    assert coverage(lists, catalog_size=0) == 0.0


def test_bootstrap_ci_matches_mean_and_has_sane_bounds():
    values = [0.1, 0.2, 0.3, 0.4, 0.5]
    result = bootstrap_ci(values, n_boot=500, seed=1)
    assert abs(result["mean"] - np.mean(values)) < 1e-9
    assert result["ci_low"] <= result["mean"] <= result["ci_high"]
    assert result["n"] == 5


def test_bootstrap_ci_empty_returns_nan():
    result = bootstrap_ci([], n_boot=100)
    assert np.isnan(result["mean"])
    assert result["n"] == 0


def test_bootstrap_coverage_ci_mean_matches_direct_point_estimate():
    # `mean` must equal the direct (non-resampled) point estimate. `boot_mean`/CI come from
    # resampling and can legitimately differ (known downward bias for set-cardinality stats
    # under naive bootstrap -- see the docstring) so they're checked only for sane bounds/type,
    # not for straddling `mean`.
    lists = [["a", "b"], ["c"], ["a"]]
    result = bootstrap_coverage_ci(lists, catalog_size=10, n_boot=500, seed=1)
    assert abs(result["mean"] - coverage(lists, 10)) < 1e-9
    assert 0.0 <= result["boot_mean"] <= 1.0
    assert result["ci_low"] <= result["ci_high"]


def test_bootstrap_coverage_ci_empty_returns_nan():
    result = bootstrap_coverage_ci([], catalog_size=10)
    assert np.isnan(result["mean"])
    assert result["n"] == 0


def test_cold_warm_slice():
    assert cold_warm_slice([0, 4, 5, 10], threshold=5) == ["cold", "cold", "warm", "warm"]


def test_percentile_threshold():
    values = list(range(1, 101))  # 1..100
    assert abs(percentile_threshold(values, 25.0) - 25.75) < 0.1
    assert percentile_threshold([], 25.0) == 0.0


def test_article_popularity_counts_clicks_per_article():
    clicked_lists = [["a"], ["a", "b"], ["b"], []]
    assert article_popularity(clicked_lists) == {"a": 2, "b": 2}


def test_article_popularity_empty_input():
    assert article_popularity([]) == {}


def test_article_popularity_safe_vs_leaky_scoping():
    # Q9: "safe" popularity (train only) must not be affected by clicks that only occur in
    # val/test -- an article clicked only in test must be invisible to the safe count, and
    # only appear once val/test clicks are deliberately included (the leaky variant).
    train_clicked = [["a"], ["a"]]
    test_clicked = [["b"], ["b"], ["b"]]
    safe = article_popularity(train_clicked)
    leaky = article_popularity(train_clicked + test_clicked)
    assert "b" not in safe
    assert safe["a"] == 2
    assert leaky["b"] == 3
    assert leaky["a"] == 2
