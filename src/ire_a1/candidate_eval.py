"""Recall@K for candidate-generation retrieval. Shared by Q2 (BM25) and Q3 (embeddings) so
lexical vs. semantic results are computed identically and are directly comparable, per Q3.5
("Compare lexical vs. semantic retrieval: which works better?").
"""


def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float | None:
    """Fraction of `relevant` (ground-truth clicked) ids present in the top-k of `retrieved`.
    None if there's no ground truth for this impression (excluded from aggregation, not counted
    as 0)."""
    if not relevant:
        return None
    top_k = set(retrieved[:k])
    hits = sum(1 for r in relevant if r in top_k)
    return hits / len(relevant)
