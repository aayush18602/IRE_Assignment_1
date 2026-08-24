"""Q2: lexical candidate generation -- inverted index + Okapi BM25 scoring, built from scratch
(not a library wrapper) over article title + abstract text, per the assignment's explicit
"build an inverted index" instruction.
"""
import heapq
import math
import re
from collections import defaultdict

import polars as pl

_TOKEN_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def tokenize(text: str | None) -> list[str]:
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())


def build_article_corpus(articles: pl.DataFrame) -> tuple[list[str], list[str]]:
    """(doc_ids, texts) where text = title + " " + abstract, per Q2's "titles and abstracts"."""
    texts = (
        (pl.col("title").fill_null("") + " " + pl.col("abstract").fill_null("")).alias("text")
    )
    df = articles.select(pl.col("article_id"), texts)
    return df["article_id"].to_list(), df["text"].to_list()


class BM25Index:
    """Classic Okapi BM25 over an inverted index (term -> postings list of (doc_idx, tf)).
    Query time only touches documents that share at least one term with the query, which is
    the actual point of an inverted index over brute-force scoring every document."""

    def __init__(self, doc_ids: list[str], texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.doc_ids = doc_ids
        self.k1 = k1
        self.b = b
        self.n_docs = len(doc_ids)

        self.doc_lengths: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for doc_idx, text in enumerate(texts):
            tokens = tokenize(text)
            self.doc_lengths.append(len(tokens))
            tf: dict[str, int] = defaultdict(int)
            for tok in tokens:
                tf[tok] += 1
            for tok, freq in tf.items():
                self.postings[tok].append((doc_idx, freq))

        self.avgdl = (sum(self.doc_lengths) / self.n_docs) if self.n_docs else 0.0
        self.idf: dict[str, float] = {
            term: math.log((self.n_docs - len(plist) + 0.5) / (len(plist) + 0.5) + 1.0)
            for term, plist in self.postings.items()
        }
        self.doc_id_to_idx: dict[str, int] = {doc_id: i for i, doc_id in enumerate(doc_ids)}

    def _score_all(self, text: str) -> dict[int, float]:
        """doc_idx -> BM25 score, for every document sharing >=1 term with the query. Shared by
        query() (full-catalog top-K, Q2) and score_candidates() (rank a specific given
        candidate set, Q4) so both use the exact same scoring, just different output shaping."""
        q_tokens = tokenize(text)
        if not q_tokens or self.n_docs == 0:
            return {}

        scores: dict[int, float] = defaultdict(float)
        for term in set(q_tokens):
            plist = self.postings.get(term)
            if not plist:
                continue
            idf = self.idf[term]
            for doc_idx, tf in plist:
                dl = self.doc_lengths[doc_idx]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[doc_idx] += idf * (tf * (self.k1 + 1)) / denom
        return scores

    def query(self, text: str, top_k: int = 200) -> list[tuple[str, float]]:
        scores = self._score_all(text)
        if not scores:
            return []
        top = heapq.nlargest(top_k, scores.items(), key=lambda kv: kv[1])
        return [(self.doc_ids[doc_idx], score) for doc_idx, score in top]

    def score_candidates(self, text: str, candidate_ids: list[str]) -> list[float]:
        """BM25 score of `text` against each of a *given* set of candidate doc ids, in the
        same order -- for Q4's official-metric re-ranking of an impression's own shown
        candidates (as opposed to query()'s full-catalog top-K candidate generation). 0.0 for a
        candidate sharing no term with the query, or not present in the index at all."""
        scores = self._score_all(text)
        return [scores.get(self.doc_id_to_idx.get(cid, -1), 0.0) for cid in candidate_ids]
