"""Q3 step 2-3: ANN index over article embeddings + leak-safe user representation.

FAISS's flat index is used as the "brute-force for small scale" option the PDF explicitly
allows -- exact nearest-neighbour search, no approximation, appropriate at the 20K-65K article
scale these datasets' small/demo tiers sit at. Vectors are L2-normalized so inner product =
cosine similarity, the standard choice for semantic (sentence/article) embeddings.
"""
import numpy as np
import polars as pl

try:
    import faiss
except ImportError:  # pragma: no cover
    faiss = None


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


class ANNIndex:
    def __init__(self, doc_ids: list[str], embeddings: np.ndarray):
        if faiss is None:
            raise ImportError("faiss-cpu is required for ANNIndex (pip install faiss-cpu)")
        self.doc_ids = doc_ids
        embeddings = _normalize(np.asarray(embeddings, dtype=np.float32))
        self.dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embeddings)

    def query(self, vector: np.ndarray, top_k: int = 200) -> list[tuple[str, float]]:
        vector = _normalize(np.asarray(vector, dtype=np.float32).reshape(1, -1))
        top_k = min(top_k, len(self.doc_ids))
        scores, idxs = self.index.search(vector, top_k)
        return [
            (self.doc_ids[idx], float(score))
            for idx, score in zip(idxs[0], scores[0])
            if idx != -1
        ]


def load_embedding_lookup(embeddings_path: str) -> dict[str, np.ndarray]:
    df = pl.read_parquet(embeddings_path)
    return dict(zip(df["article_id"].to_list(), (np.array(e, dtype=np.float32) for e in df["embedding"].to_list())))


def user_embedding(
    article_ids: list[str], embedding_lookup: dict[str, np.ndarray], dim: int
) -> np.ndarray | None:
    """Mean-pooled embedding of the given (already leak-safe, e.g. via recent_history_asof)
    clicked article ids. None for cold-start users / users whose history has no embedded
    articles -- caller should skip retrieval for those rather than querying with a zero vector."""
    vectors = [embedding_lookup[aid] for aid in article_ids if aid in embedding_lookup]
    if not vectors:
        return None
    return np.mean(vectors, axis=0).astype(np.float32)
