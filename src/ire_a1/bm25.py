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
        # term -> {doc_idx: tf}, not term -> [(doc_idx, tf), ...]: the dict gives O(1)
        # "does this specific candidate contain this term" lookups, which is what
        # score_candidates() below needs (Q4/Q5's re-ranking of a *given* small candidate set).
        # query()'s full-catalog scan (Q2/Q3's candidate generation) just iterates .items()
        # instead of a list -- equally cheap, no separate storage needed for that path.
        self.postings: dict[str, dict[int, int]] = defaultdict(dict)
        for doc_idx, text in enumerate(texts):
            tokens = tokenize(text)
            self.doc_lengths.append(len(tokens))
            tf: dict[str, int] = defaultdict(int)
            for tok in tokens:
                tf[tok] += 1
            for tok, freq in tf.items():
                self.postings[tok][doc_idx] = freq

        self.avgdl = (sum(self.doc_lengths) / self.n_docs) if self.n_docs else 0.0
        self.idf: dict[str, float] = {
            term: math.log((self.n_docs - len(plist) + 0.5) / (len(plist) + 0.5) + 1.0)
            for term, plist in self.postings.items()
        }
        self.doc_id_to_idx: dict[str, int] = {doc_id: i for i, doc_id in enumerate(doc_ids)}

    def _score_all(self, text: str) -> dict[int, float]:
        """doc_idx -> BM25 score, for every document sharing >=1 term with the query. Used by
        query() (Q2/Q3's full-catalog top-K candidate generation) -- deliberately NOT used by
        score_candidates() (Q4/Q5's re-ranking of a small given candidate set), which would
        waste time scanning every document containing a common query term just to throw away
        all but a handful of scores; see score_candidates()'s docstring."""
        q_tokens = tokenize(text)
        if not q_tokens or self.n_docs == 0:
            return {}

        scores: dict[int, float] = defaultdict(float)
        for term in set(q_tokens):
            plist = self.postings.get(term)
            if not plist:
                continue
            idf = self.idf[term]
            for doc_idx, tf in plist.items():
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
        same order -- for re-ranking an impression's own shown candidates (Q4's official
        metrics, Q5's Codabench submission) as opposed to query()'s full-catalog top-K search.

        Deliberately candidate-first, term-second (the opposite loop order from _score_all):
        cost is O(|candidates| * |query terms present in the index|), touching zero documents
        outside the given candidate set. _score_all's cost is O(sum of postings-list sizes for
        every query term) -- fine when top_k is meant to search the whole catalog, but wasteful
        here: a query built from 10 article titles routinely contains a handful of very common
        words with postings lists in the thousands, and only ~15-30 of those documents are ever
        actually needed. This distinction is what makes Q5 (13.5M/2.37M-impression large test
        sets) tractable on CPU -- see scripts/generate_submission.py."""
        q_tokens = set(tokenize(text))
        scores = [0.0] * len(candidate_ids)
        if not q_tokens:
            return scores

        term_idf_postings = [
            (self.idf[t], self.postings[t]) for t in q_tokens if t in self.postings
        ]
        for i, cid in enumerate(candidate_ids):
            doc_idx = self.doc_id_to_idx.get(cid)
            if doc_idx is None:
                continue
            dl = self.doc_lengths[doc_idx]
            denom_base = self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            s = 0.0
            for idf, postings in term_idf_postings:
                tf = postings.get(doc_idx)
                if tf is None:
                    continue
                s += idf * (tf * (self.k1 + 1)) / (tf + denom_base)
            scores[i] = s
        return scores
