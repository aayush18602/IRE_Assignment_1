from datetime import datetime
from pathlib import Path

import pytest

from ire_a1.clean import clean_ebnerd, clean_mind
from ire_a1.feature_store import history_asof, recent_history_asof
from ire_a1.schema import (
    ARTICLE_COLS,
    HISTORY_COLS,
    IMPRESSION_COLS,
    LEAKY_ARTICLE_COLS,
    LEAKY_IMPRESSION_COLS,
)

EBNERD_DEMO = Path("data/ebnerd_demo")
MIND_TRAIN = Path("data/MINDsmall_train/MINDsmall_train")
MIND_DEV = Path("data/MINDsmall_dev/MINDsmall_dev")

skip_if_no_ebnerd = pytest.mark.skipif(not EBNERD_DEMO.exists(), reason="EB-NeRD demo not downloaded")
skip_if_no_mind = pytest.mark.skipif(
    not (MIND_TRAIN.exists() and MIND_DEV.exists()), reason="MIND small not downloaded"
)


def _assert_clicked_subset_of_candidates(impressions, n=500):
    for cand, clicked in zip(
        impressions["candidates"].to_list()[:n], impressions["clicked"].to_list()[:n]
    ):
        assert set(clicked).issubset(set(cand))


@skip_if_no_ebnerd
def test_clean_ebnerd_schema_and_labels():
    tables = clean_ebnerd(EBNERD_DEMO)
    assert tables["articles"].columns == ARTICLE_COLS
    assert tables["impressions"].columns == IMPRESSION_COLS
    assert tables["history"].columns == HISTORY_COLS

    assert tables["articles"].height > 0
    assert tables["impressions"]["clicked"].list.len().sum() > 0
    _assert_clicked_subset_of_candidates(tables["impressions"])

    # ids must be strings so EB-NeRD and MIND ids can be compared/joined uniformly
    assert str(tables["articles"]["article_id"].dtype) == "String"


@skip_if_no_mind
def test_clean_mind_schema_and_labels():
    tables = clean_mind(MIND_TRAIN, MIND_DEV)
    assert tables["articles"].columns == ARTICLE_COLS
    assert tables["impressions"].columns == IMPRESSION_COLS
    assert tables["history"].columns == HISTORY_COLS

    assert tables["articles"].height > 0
    assert tables["impressions"]["clicked"].list.len().sum() > 0
    _assert_clicked_subset_of_candidates(tables["impressions"])

    # MIND history has no per-item timestamps -- history_length must still be correct
    row = tables["history"].row(0, named=True)
    assert row["history_length"] == len(row["history_article_ids"])
    assert row["last_history_time"] is None


@skip_if_no_ebnerd
def test_ebnerd_behavioural_columns_are_populated():
    """A2 Q1: the behavioural columns exist in EB-NeRD's raw logs but were dropped by A1's
    cleaning. Assert they now survive it -- a silently all-null column would look like a working
    feature to LightGBM while carrying no signal at all."""
    tables = clean_ebnerd(EBNERD_DEMO)
    impressions, history = tables["impressions"], tables["history"]

    assert impressions["session_id"].null_count() == 0
    assert impressions["is_subscriber"].null_count() == 0
    assert impressions["read_time"].null_count() < impressions.height

    # position-aligned with history_article_ids, or the per-item dwell weighting is meaningless
    row = history.row(0, named=True)
    assert len(row["history_read_times"]) == len(row["history_article_ids"])
    assert len(row["history_scroll_percentages"]) == len(row["history_article_ids"])


@skip_if_no_mind
def test_mind_behavioural_columns_are_null_not_absent():
    """MIND ships none of the behavioural instrumentation. The columns must still exist with the
    right dtype and be null throughout -- behaviour.py branches on null to select MIND's
    fallbacks (pseudo-sessions from timestamp gaps, positional decay instead of time decay), so
    an absent column and a null one are not interchangeable."""
    tables = clean_mind(MIND_TRAIN, MIND_DEV)
    impressions, history = tables["impressions"], tables["history"]

    for col in ("session_id", "age", "gender", "postcode", "is_subscriber"):
        assert impressions[col].null_count() == impressions.height, f"{col} unexpectedly populated"
    for col in LEAKY_IMPRESSION_COLS:
        assert impressions[col].null_count() == impressions.height
    for col in ("history_read_times", "history_scroll_percentages"):
        assert history[col].null_count() == history.height
    for col in LEAKY_ARTICLE_COLS:
        assert tables["articles"][col].null_count() == tables["articles"].height


def test_leaky_columns_are_declared_and_disjoint_from_features():
    """Q9's leaky columns are carried on purpose, so the guard against using them has to be
    explicit: they must be real columns of the schema, and the two lists must not overlap."""
    assert set(LEAKY_IMPRESSION_COLS).issubset(IMPRESSION_COLS)
    assert set(LEAKY_ARTICLE_COLS).issubset(ARTICLE_COLS)
    assert not set(LEAKY_IMPRESSION_COLS) & set(LEAKY_ARTICLE_COLS)


def test_history_asof_filters_future_entries():
    ids = ["a", "b", "c", "d"]
    times = [datetime(2024, 1, i) for i in (1, 2, 3, 4)]
    cutoff = datetime(2024, 1, 3)
    assert history_asof(ids, times, cutoff) == ["a", "b"]


def test_history_asof_passthrough_when_no_timestamps():
    ids = ["a", "b", "c"]
    assert history_asof(ids, None, datetime(2024, 1, 1)) == ids


def test_recent_history_asof_takes_last_n_after_cutoff_filtering():
    ids = ["a", "b", "c", "d", "e"]
    times = [datetime(2024, 1, i) for i in (1, 2, 3, 4, 5)]
    cutoff = datetime(2024, 1, 5)  # excludes "e"
    assert recent_history_asof(ids, times, cutoff, n_recent=2) == ["c", "d"]
    assert recent_history_asof(ids, times, cutoff, n_recent=10) == ["a", "b", "c", "d"]
