from datetime import datetime
from pathlib import Path

import pytest

from ire_a1.clean import clean_ebnerd, clean_mind
from ire_a1.feature_store import history_asof
from ire_a1.schema import ARTICLE_COLS, HISTORY_COLS, IMPRESSION_COLS

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


def test_history_asof_filters_future_entries():
    ids = ["a", "b", "c", "d"]
    times = [datetime(2024, 1, i) for i in (1, 2, 3, 4)]
    cutoff = datetime(2024, 1, 3)
    assert history_asof(ids, times, cutoff) == ["a", "b"]


def test_history_asof_passthrough_when_no_timestamps():
    ids = ["a", "b", "c"]
    assert history_asof(ids, None, datetime(2024, 1, 1)) == ids
