"""A2 Q2-A1: the feature matrix both re-rankers consume.

The features themselves are tested in test_behaviour.py / test_article_features.py. What matters
here is the *contract*: row order, the group array, label alignment, and the feature-name subsets
the Q9 and submission comparisons depend on. A group array out of step with the rows does not
raise on its own -- a ranking model just trains on impressions stitched together from unrelated
candidates -- so those invariants get explicit tests rather than trust.
"""
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from ire_a1.reranker import MatrixBuilder, RankingMatrix, _contiguous_groups

T0 = datetime(2024, 5, 1, 9, 0, 0)


def _articles() -> pl.DataFrame:
    ids = ["a", "b", "c", "d"]
    return pl.DataFrame(
        {"article_id": ids,
         "title": ["alpha story", "beta story", "gamma report", "delta report"],
         "abstract": ["", "", "", ""],
         "category": ["news", "news", "sport", "sport"],
         "published_time": [T0 - timedelta(days=1)] * 4},
        schema={"article_id": pl.Utf8, "title": pl.Utf8, "abstract": pl.Utf8,
                "category": pl.Utf8, "published_time": pl.Datetime("us")})


def _history() -> pl.DataFrame:
    return pl.DataFrame(
        {"user_id": ["u1", "u2"],
         "history_article_ids": [["a", "c"], ["d"]],
         "history_timestamps": [[T0 - timedelta(hours=5), T0 - timedelta(hours=4)],
                                [T0 - timedelta(hours=3)]],
         "history_length": [2, 1],
         "history_read_times": [[10.0, 20.0], [30.0]]},
        schema={"user_id": pl.Utf8, "history_article_ids": pl.List(pl.Utf8),
                "history_timestamps": pl.List(pl.Datetime("us")), "history_length": pl.Int64,
                "history_read_times": pl.List(pl.Float32)})


def _impressions() -> pl.DataFrame:
    return pl.DataFrame(
        {"impression_id": ["i1", "i2", "i3"],
         "user_id": ["u1", "u2", "u1"],
         "timestamp": [T0, T0 + timedelta(minutes=10), T0 + timedelta(minutes=20)],
         "candidates": [["a", "b", "c"], ["b", "d"], ["a", "d", "c", "b"]],
         "clicked": [["b"], ["d"], ["c"]],
         "read_time": [12.0, 5.0, 8.0],
         "session_id": ["s1", "s2", "s1"]},
        schema={"impression_id": pl.Utf8, "user_id": pl.Utf8, "timestamp": pl.Datetime("us"),
                "candidates": pl.List(pl.Utf8), "clicked": pl.List(pl.Utf8),
                "read_time": pl.Float32, "session_id": pl.Utf8})


@pytest.fixture
def builder() -> MatrixBuilder:
    imps = _impressions()
    return MatrixBuilder.build(articles=_articles(), history=_history(),
                               index_impressions=imps, with_bm25=True, windows_h=(24,))


# --- the group-array contract ---------------------------------------------------------------

def test_groups_match_candidate_counts_in_row_order(builder):
    m = builder.transform(_impressions())
    assert m.groups.tolist() == [3, 2, 4]
    assert m.impression_ids.tolist() == ["i1", "i2", "i3"]
    assert int(m.groups.sum()) == m.n_rows == 9


def test_ranking_matrix_rejects_groups_that_do_not_sum_to_the_row_count():
    with pytest.raises(ValueError, match="out of step"):
        RankingMatrix(X=np.zeros((5, 2), dtype=np.float32), y=None,
                      groups=np.array([2, 2], dtype=np.int32), feature_names=["f1", "f2"],
                      impression_ids=np.array(["i1", "i2"]), article_ids=np.array(list("abcde")))


def test_ranking_matrix_rejects_a_name_count_mismatch():
    with pytest.raises(ValueError, match="feature columns"):
        RankingMatrix(X=np.zeros((2, 3), dtype=np.float32), y=None,
                      groups=np.array([2], dtype=np.int32), feature_names=["only_one"],
                      impression_ids=np.array(["i1"]), article_ids=np.array(["a", "b"]))


def test_contiguity_guard_catches_rows_split_across_runs():
    """If a join ever reorders rows so one impression's candidates appear in two runs, the group
    array would straddle impressions. That must raise, not train."""
    with pytest.raises(ValueError, match="not contiguous"):
        _contiguous_groups(pl.Series(["i1", "i2", "i1"]))


def test_contiguous_groups_handles_an_empty_frame():
    sizes, ids = _contiguous_groups(pl.Series([], dtype=pl.Utf8))
    assert len(sizes) == 0 and len(ids) == 0


# --- rows, labels, alignment ------------------------------------------------------------------

def test_rows_follow_candidate_order_within_each_impression(builder):
    m = builder.transform(_impressions())
    assert m.article_ids.tolist() == ["a", "b", "c", "b", "d", "a", "d", "c", "b"]


def test_labels_mark_exactly_the_clicked_candidate(builder):
    m = builder.transform(_impressions())
    assert m.y.tolist() == [0, 1, 0, 0, 1, 0, 0, 1, 0]


def test_matrix_shape_matches_names_and_rows(builder):
    m = builder.transform(_impressions())
    assert m.X.shape == (9, len(builder.feature_names))
    assert m.feature_names == builder.feature_names
    assert m.X.dtype == np.float32


def test_feature_columns_land_on_the_right_rows(builder):
    """Alignment across three independently-built feature blocks is the thing most likely to go
    quietly wrong, so pin it against a column whose value is known per row."""
    m = builder.transform(_impressions())
    assert m.column("pos_in_slate").tolist() == [0, 1, 2, 0, 1, 0, 1, 2, 3]
    assert m.column("slate_size").tolist() == [3, 3, 3, 2, 2, 4, 4, 4, 4]


# --- the submission path ----------------------------------------------------------------------

def test_transform_works_without_click_labels(builder):
    """Both Codabench test sets ship candidates and no clicks. The matrix must still build."""
    unlabelled = _impressions().drop("clicked")
    m = builder.transform(unlabelled, with_labels=False)
    assert m.y is None
    assert m.n_rows == 9
    assert int(m.groups.sum()) == 9


def test_submission_safe_subset_drops_click_derived_features(builder):
    safe = builder.submission_safe_feature_names
    assert "clicks_so_far_in_session" not in safe
    assert not [n for n in safe if n.startswith("ctr_") or "pop_clicks" in n]
    assert "pop_inviews_24h" in safe and "bm25_score" in safe
    assert set(safe).issubset(builder.feature_names)


def test_select_subsets_columns_and_keeps_the_grouping(builder):
    """Q9's with/without comparison and the submission-safe run are both `select()` on one
    matrix, so the groups and labels have to survive the subset unchanged."""
    m = builder.transform(_impressions())
    sub = m.select(["pos_in_slate", "cat_affinity"])
    assert sub.feature_names == ["pos_in_slate", "cat_affinity"]
    assert sub.X.shape == (9, 2)
    assert np.array_equal(sub.groups, m.groups)
    assert np.array_equal(sub.y, m.y)
    assert np.allclose(sub.column("pos_in_slate"), m.column("pos_in_slate"))


def test_select_rejects_an_unknown_feature(builder):
    with pytest.raises(ValueError, match="unknown features"):
        builder.transform(_impressions()).select(["not_a_feature"])


# --- Q1.1's deferred title/embedding features -------------------------------------------------

def test_bm25_title_similarity_is_present_and_rewards_the_users_reading(builder):
    """Q1.1 asks for the user's recent clicked articles as titles. That is BM25 over their recent
    titles, deferred here because this is where A1's index is loaded."""
    assert "bm25_score" in builder.feature_names
    m = builder.transform(_impressions())
    scores = m.column("bm25_score")
    # u1's history is "alpha story" + "gamma report"; in i1 the candidates are a, b, c
    assert scores[0] > 0 and scores[2] > 0        # a and c share terms with the history
    assert np.all(np.isfinite(scores))


def test_embedding_similarity_is_omitted_when_no_embeddings_are_supplied(builder):
    assert "embed_cos" not in builder.feature_names
    assert builder.similarity_features == ["bm25_score"]


def test_embedding_similarity_appears_when_embeddings_are_supplied(tmp_path):
    emb = tmp_path / "emb.parquet"
    pl.DataFrame({"article_id": ["a", "b", "c", "d"],
                  "embedding": [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]]}).write_parquet(emb)
    mb = MatrixBuilder.build(articles=_articles(), history=_history(),
                             index_impressions=_impressions(), embeddings_path=emb,
                             with_bm25=False, windows_h=(24,))
    assert mb.similarity_features == ["embed_cos"]
    m = mb.transform(_impressions())
    assert np.all(np.isfinite(m.column("embed_cos")))
