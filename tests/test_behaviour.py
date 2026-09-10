"""A2 Q1.1/Q1.4: user-side history features, their decay clocks, and the as-of boundary.

Two datasets, two clocks: EB-NeRD decays in elapsed time, MIND (which has no per-item history
timestamps at all) decays in position from the end. Both paths are exercised here.
"""
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from ire_a1.behaviour import HistoryFeatures

T0 = datetime(2024, 3, 1, 12, 0, 0)

_HIST_SCHEMA = {
    "user_id": pl.Utf8,
    "history_article_ids": pl.List(pl.Utf8),
    "history_timestamps": pl.List(pl.Datetime("us")),
    "history_length": pl.Int64,
    "history_read_times": pl.List(pl.Float32),
}


def _history(user, ids, times, read_times=None) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "user_id": [user],
            "history_article_ids": [ids],
            "history_timestamps": [times],
            "history_length": [len(ids)],
            "history_read_times": [read_times],
        },
        schema=_HIST_SCHEMA,
    )


def _articles(pairs) -> pl.DataFrame:
    return pl.DataFrame({"article_id": [a for a, _ in pairs], "category": [c for _, c in pairs]},
                        schema={"article_id": pl.Utf8, "category": pl.Utf8})


def _rows(user, article_ids, as_of) -> pl.DataFrame:
    return pl.DataFrame(
        {"user_id": [user] * len(article_ids), "article_id": article_ids,
         "as_of": [as_of] * len(article_ids)},
        schema={"user_id": pl.Utf8, "article_id": pl.Utf8, "as_of": pl.Datetime("us")},
    )


ARTS = _articles([("a", "sport"), ("b", "sport"), ("c", "news"), ("d", "news"), ("e", "tech")])


def _timed_hf(**kw) -> HistoryFeatures:
    """User clicked sport at T0-3h, sport at T0-2h, news at T0-1h."""
    hist = _history("u", ["a", "b", "c"],
                    [T0 - timedelta(hours=3), T0 - timedelta(hours=2), T0 - timedelta(hours=1)],
                    [10.0, 20.0, 30.0])
    return HistoryFeatures.build(hist, ARTS, **kw)


# --- the boundary -------------------------------------------------------------------------

def test_history_item_at_exactly_as_of_is_excluded():
    """The Q1.4 boundary. At as_of == the click time, that click has not happened yet."""
    hist = _history("u", ["a"], [T0], [5.0])
    hf = HistoryFeatures.build(hist, ARTS)
    assert hf.transform(_rows("u", ["e"], T0))["hist_len_asof"][0] == 0
    assert hf.transform(_rows("u", ["e"], T0 + timedelta(microseconds=1)))["hist_len_asof"][0] == 1


def test_as_of_filter_is_load_bearing_when_history_overlaps_impressions():
    """EB-NeRD's history snapshot overlaps the train period, so this is not hypothetical:
    scoring an impression mid-history must not see the later half of that history."""
    hf = _timed_hf()
    early = hf.transform(_rows("u", ["e"], T0 - timedelta(hours=2, minutes=30)))
    assert early["hist_len_asof"][0] == 1        # only the T0-3h click
    late = hf.transform(_rows("u", ["e"], T0))
    assert late["hist_len_asof"][0] == 3


# --- decay --------------------------------------------------------------------------------

def test_time_decay_weights_recent_clicks_more():
    hf = _timed_hf(half_life_hours=1.0)
    out = hf.transform(_rows("u", ["e"], T0))
    # ages 3h, 2h, 1h at a 1h half-life -> 0.125 + 0.25 + 0.5
    assert out["decay_weight_sum"][0] == pytest.approx(0.875, rel=1e-4)


def test_longer_half_life_approaches_undecayed_counts():
    """Sanity on the parameterisation: as the half-life grows the decayed affinity converges on
    the raw one. This is what makes the measured sweep interpretable rather than arbitrary."""
    out = _timed_hf(half_life_hours=100_000.0).transform(_rows("u", ["a"], T0))
    assert out["cat_affinity"][0] == pytest.approx(out["cat_affinity_raw"][0], abs=1e-4)


def test_mind_style_history_uses_positional_decay():
    """history_timestamps null throughout -- the MIND shape. Decay falls back to position from
    the end, and the as-of filter has nothing to filter on."""
    hist = pl.DataFrame(
        {"user_id": ["u"], "history_article_ids": [["a", "b", "c"]], "history_timestamps": [None],
         "history_length": [3], "history_read_times": [None]}, schema=_HIST_SCHEMA)
    hf = HistoryFeatures.build(hist, ARTS, half_life_positions=1.0)
    out = hf.transform(_rows("u", ["e"], T0))
    # positions from the end are 2, 1, 0 -> 0.25 + 0.5 + 1.0
    assert out["decay_weight_sum"][0] == pytest.approx(1.75, rel=1e-4)
    assert "hours_since_last_click" not in hf.feature_names
    assert hf.split_unstable_feature_names == []


# --- category affinity --------------------------------------------------------------------

def test_category_affinity_is_the_decayed_share_of_that_category():
    hf = _timed_hf(half_life_hours=1.0)          # weights: a=0.125, b=0.25, c=0.5
    out = hf.transform(_rows("u", ["a", "c", "e"], T0))
    assert out["cat_affinity"][0] == pytest.approx(0.375 / 0.875, rel=1e-3)   # sport
    assert out["cat_affinity"][1] == pytest.approx(0.500 / 0.875, rel=1e-3)   # news
    assert out["cat_affinity"][2] == pytest.approx(0.0, abs=1e-6)             # tech, unseen


def test_raw_affinity_ignores_decay():
    out = _timed_hf(half_life_hours=1.0).transform(_rows("u", ["a", "c"], T0))
    assert out["cat_affinity_raw"][0] == pytest.approx(2 / 3, rel=1e-3)
    assert out["cat_affinity_raw"][1] == pytest.approx(1 / 3, rel=1e-3)


def test_dwell_affinity_weights_by_time_spent_not_recency():
    """Dwell comes from history_read_times -- engagement on *past* clicks, which is serving-safe.
    The impression row's own read_time is post-click and lives in schema.LEAKY_IMPRESSION_COLS."""
    out = _timed_hf().transform(_rows("u", ["a", "c"], T0))
    assert out["dwell_affinity"][0] == pytest.approx(30 / 60, rel=1e-3)   # sport: 10+20 of 60
    assert out["dwell_affinity"][1] == pytest.approx(30 / 60, rel=1e-3)   # news: 30 of 60


# --- train/test stability -----------------------------------------------------------------

def test_burst_counts_are_measured_from_the_last_click_not_as_of():
    """The fix for the measured train/test skew: an as-of-relative count read 17.0 on train and
    exactly 0.0 on val and test, because EB-NeRD's history snapshot ends days before the test
    window. Anchoring to the user's own last click makes the count independent of when we score."""
    hf = _timed_hf()
    near = hf.transform(_rows("u", ["e"], T0))
    far = hf.transform(_rows("u", ["e"], T0 + timedelta(days=30)))
    assert near["hist_burst_24h"][0] == 3
    assert far["hist_burst_24h"][0] == 3                     # unchanged despite scoring 30d later
    assert far["hours_since_last_click"][0] > near["hours_since_last_click"][0]


def test_unstable_features_are_named_for_exclusion():
    assert _timed_hf().split_unstable_feature_names == ["hours_since_last_click"]


# --- edges --------------------------------------------------------------------------------

def test_unknown_user_scores_zero_rather_than_failing():
    out = _timed_hf().transform(_rows("stranger", ["a"], T0))
    assert out["hist_len_asof"][0] == 0
    assert out["cat_affinity"][0] == pytest.approx(0.0)


def test_empty_history_does_not_divide_by_zero():
    hist = _history("u", [], [], [])
    out = HistoryFeatures.build(hist, ARTS).transform(_rows("u", ["a"], T0))
    assert out["cat_affinity"][0] == pytest.approx(0.0)
    assert np.isfinite(out["cat_affinity"][0])


def test_output_is_row_aligned_with_input():
    hf = _timed_hf()
    rows = _rows("u", ["a", "c", "e", "b"], T0)
    out = hf.transform(rows)
    assert out.height == rows.height
    assert out.columns == hf.feature_names


def test_transform_rejects_rows_without_user_id():
    with pytest.raises(ValueError, match="user_id"):
        _timed_hf().transform(pl.DataFrame({"article_id": ["a"], "as_of": [T0]}))
