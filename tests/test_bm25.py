import polars as pl

from ire_a1.bm25 import BM25Index, build_article_corpus, tokenize
from ire_a1.candidate_eval import recall_at_k


def test_tokenize_lowercases_and_splits_on_punctuation():
    assert tokenize("Prince Harry's DNA-test!") == ["prince", "harry", "s", "dna", "test"]


def test_tokenize_handles_none_and_empty():
    assert tokenize(None) == []
    assert tokenize("") == []


def test_bm25_ranks_matching_doc_above_unrelated_doc():
    doc_ids = ["a", "b", "c"]
    texts = [
        "ice hockey player scores winning goal",
        "royal family dna test controversy",
        "weather forecast rain tomorrow",
    ]
    index = BM25Index(doc_ids, texts)

    results = index.query("ice hockey goal", top_k=3)
    ranked_ids = [doc_id for doc_id, _ in results]
    assert ranked_ids[0] == "a"


def test_bm25_empty_or_unmatched_query_returns_empty():
    index = BM25Index(["a", "b"], ["hello world", "foo bar"])
    assert index.query("", top_k=10) == []
    assert index.query("   ", top_k=10) == []
    assert index.query("zzz nonexistent qqq", top_k=10) == []


def test_bm25_respects_top_k():
    doc_ids = [str(i) for i in range(20)]
    texts = ["common word article " + str(i) for i in range(20)]
    index = BM25Index(doc_ids, texts)
    results = index.query("common word", top_k=5)
    assert len(results) == 5


def test_build_article_corpus_concatenates_title_and_abstract_and_handles_nulls():
    articles = pl.DataFrame({
        "article_id": ["1", "2"],
        "title": ["Hello", None],
        "abstract": [None, "World"],
    })
    doc_ids, texts = build_article_corpus(articles)
    assert doc_ids == ["1", "2"]
    assert texts == ["Hello ", " World"]


def test_recall_at_k_full_and_partial_hit():
    retrieved = ["a", "b", "c", "d", "e"]
    assert recall_at_k(retrieved, ["a"], k=3) == 1.0
    assert recall_at_k(retrieved, ["z"], k=3) == 0.0
    assert recall_at_k(retrieved, ["a", "z"], k=3) == 0.5
    assert recall_at_k(retrieved, ["d"], k=3) == 0.0  # rank 4, outside top-3
    assert recall_at_k(retrieved, ["d"], k=5) == 1.0


def test_recall_at_k_no_ground_truth_returns_none():
    assert recall_at_k(["a", "b"], [], k=10) is None
