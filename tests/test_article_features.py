"""A2 Q1.3/Q1.4: item-side features, and the behaviour-window boundary they must respect.

The boundary case that matters is an event landing at *exactly* `as_of`. Off-by-one there is the
classic leak: it looks like a working feature, it inflates offline metrics, and nothing crashes.
"""
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from ire_a1.article_features import ArticleFeatures

T0 = datetime(2024, 1, 10, 12, 0, 0)


def _impressions(rows: list[tuple[datetime, list[str], list[str]]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "timestamp": [t for t, _, _ in rows],
            "candidates": [c for _, c, _ in rows],
            "clicked": [k for _, _, k in rows],
        },
        schema={"timestamp": pl.Datetime("us"), "candidates": pl.List(pl.Utf8),
                "clicked": pl.List(pl.Utf8)},
    )


def _articles(ids: list[str], published: list[datetime | None]) -> pl.DataFrame:
    return pl.DataFrame({"article_id": ids, "published_time": published},
                        schema={"article_id": pl.Utf8, "published_time": pl.Datetime("us")})


def _rows(article_id: str, as_of: list[datetime]) -> pl.DataFrame:
    return pl.DataFrame({"article_id": [article_id] * len(as_of), "as_of": as_of},
                        schema={"article_id": pl.Utf8, "as_of": pl.Datetime("us")})


@pytest.fixture
def three_clicks() -> ArticleFeatures:
    """Article "a" clicked at T0, T0+1h, T0+2h; article "b" never clicked."""
    imps = _impressions([
        (T0, ["a", "b"], ["a"]),
        (T0 + timedelta(hours=1), ["a", "b"], ["a"]),
        (T0 + timedelta(hours=2), ["a", "b"], ["a"]),
    ])
    arts = _articles(["a", "b"], [T0 - timedelta(hours=5), None])
    return ArticleFeatures.build(imps, arts, windows_h=(1, 24))


def test_click_at_exactly_as_of_is_excluded(three_clicks):
    """THE boundary case. At as_of == T0 the click at T0 has not happened yet from the ranker's
    point of view; one microsecond later it has."""
    at_t0 = three_clicks.transform(_rows("a", [T0]))
    assert at_t0["pop_clicks_24h"][0] == 0

    just_after = three_clicks.transform(_rows("a", [T0 + timedelta(microseconds=1)]))
    assert just_after["pop_clicks_24h"][0] == 1


def test_window_lower_bound_is_inclusive(three_clicks):
    """[as_of - window, as_of): a click exactly `window` old is still inside it."""
    as_of = T0 + timedelta(hours=1)          # click at T0 is exactly 1h old
    out = three_clicks.transform(_rows("a", [as_of]))
    assert out["pop_clicks_1h"][0] == 1      # the T0 click, not the one at as_of itself


def test_counts_accumulate_over_time(three_clicks):
    out = three_clicks.transform(_rows("a", [
        T0 - timedelta(hours=1),
        T0 + timedelta(minutes=30),
        T0 + timedelta(hours=1, minutes=30),
        T0 + timedelta(hours=3),
    ]))
    assert out["pop_clicks_24h"].to_list() == [0, 1, 2, 3]


def test_counts_are_monotonic_in_as_of(three_clicks):
    """A cumulative count that ever decreases as `as_of` advances means the boundary logic is
    wrong -- a cheap invariant that catches sign and side= errors."""
    times = [T0 + timedelta(minutes=7 * i) for i in range(40)]
    counts = three_clicks.transform(_rows("a", times))["pop_clicks_24h"].to_numpy()
    assert np.all(np.diff(counts) >= 0)


def test_never_clicked_article_scores_zero_not_null(three_clicks):
    out = three_clicks.transform(_rows("b", [T0 + timedelta(hours=3)]))
    assert out["pop_clicks_24h"][0] == 0
    assert out["pop_inviews_24h"][0] == 3      # shown three times, clicked never


def test_unknown_article_is_zero_rather_than_an_error(three_clicks):
    out = three_clicks.transform(_rows("never-seen", [T0 + timedelta(hours=3)]))
    assert out["pop_clicks_24h"][0] == 0
    assert np.isnan(out["freshness_hours"][0])


def test_ctr_is_smoothed(three_clicks):
    """clicks/(inviews+1), so a 1-of-1 article cannot outrank a 100-of-200 one."""
    out = three_clicks.transform(_rows("a", [T0 + timedelta(hours=3)]))
    assert out["ctr_24h"][0] == pytest.approx(3 / 4)


def test_freshness_uses_published_time_when_present(three_clicks):
    out = three_clicks.transform(_rows("a", [T0]))
    assert out["freshness_hours"][0] == pytest.approx(5.0)
    assert out["has_published_time"][0] == 1.0


def test_mind_style_null_published_time_falls_back_to_first_seen(three_clicks):
    """MIND's published_time is 100% null, so freshness_hours is NaN there and the first-seen
    proxy carries the signal instead. Both are reported so the model can tell them apart."""
    out = three_clicks.transform(_rows("b", [T0 + timedelta(hours=2)]))
    assert np.isnan(out["freshness_hours"][0])
    assert out["has_published_time"][0] == 0.0
    assert out["hours_since_first_seen"][0] == pytest.approx(2.0)


def test_feature_names_match_transform_output(three_clicks):
    out = three_clicks.transform(_rows("a", [T0]))
    assert out.columns == three_clicks.feature_names


def test_submission_safe_subset_excludes_click_derived_features(three_clicks):
    """Both Codabench test sets ship candidates but no click labels, so anything derived from
    `clicked` cannot be computed over the test period at submission time. In-view counts and
    freshness survive."""
    safe = three_clicks.submission_safe_feature_names
    assert not [n for n in safe if "clicks" in n or n.startswith("ctr")]
    assert "pop_inviews_1h" in safe and "freshness_hours" in safe
    assert set(safe).issubset(three_clicks.feature_names)


def test_index_built_without_clicks_still_yields_inview_features():
    """The submission-time shape: impressions with candidates and no clicks at all. In-view
    popularity must still work -- if this regresses, the submitted model loses its only
    popularity signal."""
    unlabelled = _impressions([
        (T0, ["a", "b"], []),
        (T0 + timedelta(hours=1), ["a"], []),
    ])
    af = ArticleFeatures.build(unlabelled, _articles(["a", "b"], [None, None]), windows_h=(24,))
    out = af.transform(_rows("a", [T0 + timedelta(hours=2)]))
    assert out["pop_inviews_24h"][0] == 2
    assert out["pop_clicks_24h"][0] == 0


def test_train_only_index_misses_a_split_gap():
    """Regression guard for the finding that shaped this module: indexing only the earlier split
    yields all-zero trailing counts once the gap to the scored period exceeds the window. This is
    the failure that made every popularity feature score AUC 0.5000 on the real EB-NeRD split."""
    train = _impressions([(T0, ["a"], ["a"]), (T0 + timedelta(hours=1), ["a"], ["a"])])
    af = ArticleFeatures.build(train, _articles(["a"], [None]), windows_h=(24,))

    within = af.transform(_rows("a", [T0 + timedelta(hours=6)]))
    assert within["pop_clicks_24h"][0] == 2          # gap smaller than the window: fine

    beyond = af.transform(_rows("a", [T0 + timedelta(days=3)]))
    assert beyond["pop_clicks_24h"][0] == 0          # gap exceeds the window: silently empty


def test_transform_rejects_rows_without_as_of(three_clicks):
    with pytest.raises(ValueError, match="as_of"):
        three_clicks.transform(pl.DataFrame({"article_id": ["a"]}))
