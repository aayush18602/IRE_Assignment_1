import numpy as np
import polars as pl

from ire_a1.ann import ANNIndex, load_embedding_lookup, score_candidates, user_embedding


def test_ann_index_finds_nearest_neighbor():
    doc_ids = ["a", "b", "c"]
    embeddings = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    index = ANNIndex(doc_ids, embeddings)

    results = index.query(np.array([0.9, 0.1, 0.0]), top_k=2)
    assert results[0][0] == "a"
    assert len(results) == 2


def test_ann_index_respects_top_k_and_caps_at_corpus_size():
    doc_ids = ["a", "b", "c"]
    embeddings = np.eye(3, dtype=np.float32)
    index = ANNIndex(doc_ids, embeddings)

    assert len(index.query(np.array([1.0, 0.0, 0.0]), top_k=2)) == 2
    assert len(index.query(np.array([1.0, 0.0, 0.0]), top_k=100)) == 3  # capped, not padded


def test_user_embedding_mean_pools_known_articles():
    lookup = {
        "a": np.array([1.0, 0.0], dtype=np.float32),
        "b": np.array([0.0, 1.0], dtype=np.float32),
    }
    result = user_embedding(["a", "b"], lookup, dim=2)
    assert np.allclose(result, [0.5, 0.5])


def test_user_embedding_skips_unknown_ids():
    lookup = {"a": np.array([2.0, 0.0], dtype=np.float32)}
    result = user_embedding(["a", "unknown_id"], lookup, dim=2)
    assert np.allclose(result, [2.0, 0.0])


def test_user_embedding_none_when_no_ids_match():
    lookup = {"a": np.array([1.0, 0.0], dtype=np.float32)}
    assert user_embedding([], lookup, dim=2) is None
    assert user_embedding(["unknown"], lookup, dim=2) is None


def test_load_embedding_lookup_round_trips(tmp_path):
    df = pl.DataFrame({
        "article_id": ["a", "b"],
        "embedding": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    })
    path = tmp_path / "embeddings.parquet"
    df.write_parquet(path)

    lookup = load_embedding_lookup(str(path))
    assert set(lookup.keys()) == {"a", "b"}
    assert np.allclose(lookup["a"], [1.0, 2.0, 3.0])


def test_score_candidates_cosine_similarity_and_ordering():
    lookup = {
        "a": np.array([1.0, 0.0], dtype=np.float32),
        "b": np.array([0.0, 1.0], dtype=np.float32),
        "c": np.array([-1.0, 0.0], dtype=np.float32),
    }
    user_vec = np.array([1.0, 0.0], dtype=np.float32)
    scores = score_candidates(user_vec, ["a", "b", "c"], lookup)
    assert scores[0] > scores[1] > scores[2]
    assert scores[0] == 1.0
    assert scores[2] == -1.0


def test_score_candidates_unknown_id_scores_zero():
    lookup = {"a": np.array([1.0, 0.0], dtype=np.float32)}
    scores = score_candidates(np.array([1.0, 0.0]), ["a", "missing"], lookup)
    assert scores[1] == 0.0
