"""Q5: Codabench submission formatting -- shared by scripts/generate_submission.py, tested
independently of the (slow, large-scale) full pipeline.
"""
import numpy as np


def ranks_from_scores(scores: list[float]) -> list[int]:
    """1-indexed rank per original position (1 = highest score), via the standard
    argsort-of-argsort trick. Ties (e.g. multiple candidates with 0 score) broken by original
    candidate order (stable sort) -- deterministic, not a bug."""
    order = np.argsort(-np.asarray(scores), kind="stable")
    ranks = np.empty(len(scores), dtype=int)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks.tolist()


def format_submission_line(impression_id, ranks: list[int]) -> str:
    """The official MIND/EB-NeRD Codabench format: 'impression_id [rank1,rank2,...]'."""
    return f"{impression_id} [{','.join(map(str, ranks))}]\n"
