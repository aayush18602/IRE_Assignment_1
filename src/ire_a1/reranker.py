"""A2 Q2: the feature matrix both re-rankers consume.

This is the seam between Q1's feature builders and the two Q2 models -- Option A (LightGBM
LambdaRank, `scripts/run_reranker.py`) and Option B (a neural ranker over the same inputs). They
must see identical features or the comparison between them measures plumbing rather than models,
so the matrix is built once here and `RankingMatrix` is the only thing either model is handed.

**The group array is the dangerous part.** A learning-to-rank model is told "the first g0 rows are
one impression, the next g1 rows are the next" -- and if that is out of step with the row order it
does not raise, it silently trains on impressions stitched together from unrelated candidates.
`RankingMatrix` validates the invariant on construction, and `transform()` checks that the rows
really are contiguous per impression rather than assuming the joins preserved order.

Q1.1's title and embedding features live here rather than in `behaviour.py`: they are A1's BM25 and
ANN similarity, and this is where those indexes are already loaded.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from ire_a1.ann import load_embedding_lookup, score_candidates as ann_score_candidates
from ire_a1.ann import user_embedding
from ire_a1.article_features import ArticleFeatures
from ire_a1.behaviour import (
    SESSION_FEATURE_NAMES,
    SLATE_FEATURE_NAMES,
    SUBMISSION_SAFE_SESSION_FEATURES,
    HistoryFeatures,
    explode_candidates,
    session_features,
)
from ire_a1.bm25 import BM25Index, build_article_corpus
from ire_a1.feature_store import recent_history_asof

DEFAULT_N_RECENT = 10
SIMILARITY_FEATURES = ["bm25_score", "embed_cos"]


@dataclass(frozen=True)
class RankingMatrix:
    """One row per (impression, candidate), grouped by impression.

    `groups` holds the candidate count per impression **in row order**, which is the contract
    LightGBM's ranking objective and any listwise neural loss both depend on.
    """

    X: np.ndarray                 # float32 [n_rows, n_features]
    y: np.ndarray | None          # int8 [n_rows]; None when click labels are unavailable
    groups: np.ndarray            # int32 [n_impressions]
    feature_names: list[str]
    impression_ids: np.ndarray    # [n_impressions], aligned with groups
    article_ids: np.ndarray       # [n_rows]

    def __post_init__(self) -> None:
        n_rows, n_feat = self.X.shape
        if len(self.feature_names) != n_feat:
            raise ValueError(f"{n_feat} feature columns but {len(self.feature_names)} names")
        if int(self.groups.sum()) != n_rows:
            raise ValueError(
                f"groups sum to {int(self.groups.sum())} but X has {n_rows} rows -- the group "
                "array is out of step with the matrix, which trains silently and scores nonsense"
            )
        if len(self.impression_ids) != len(self.groups):
            raise ValueError("impression_ids must be aligned with groups, one per impression")
        if len(self.article_ids) != n_rows:
            raise ValueError("article_ids must be aligned with X, one per row")
        if self.y is not None and len(self.y) != n_rows:
            raise ValueError("y must be aligned with X, one per row")

    @property
    def n_rows(self) -> int:
        return self.X.shape[0]

    @property
    def n_impressions(self) -> int:
        return len(self.groups)

    def column(self, name: str) -> np.ndarray:
        return self.X[:, self.feature_names.index(name)]

    def select(self, names: list[str]) -> "RankingMatrix":
        """A copy keeping only `names`, for the Q9 with/without and submission-safe comparisons.
        Order follows `names`, so two matrices built from different subsets stay self-describing."""
        missing = [n for n in names if n not in self.feature_names]
        if missing:
            raise ValueError(f"unknown features: {missing}")
        idx = [self.feature_names.index(n) for n in names]
        return RankingMatrix(
            X=np.ascontiguousarray(self.X[:, idx]), y=self.y, groups=self.groups,
            feature_names=list(names), impression_ids=self.impression_ids,
            article_ids=self.article_ids,
        )


class MatrixBuilder:
    """Fit the Q1 feature sources once, then `transform()` any set of impressions."""

    def __init__(
        self,
        articles: pl.DataFrame,
        history_features: HistoryFeatures,
        article_features: ArticleFeatures,
        bm25: BM25Index | None,
        embedding_lookup: dict[str, np.ndarray] | None,
        n_recent: int,
    ):
        self._titles = dict(zip(articles["article_id"].to_list(), articles["title"].to_list()))
        self._history = history_features
        self._articles = article_features
        self._bm25 = bm25
        self._embeddings = embedding_lookup
        self._embed_dim = (next(iter(embedding_lookup.values())).shape[0]
                           if embedding_lookup else 0)
        self._n_recent = n_recent
        self._history_by_user = None   # populated lazily on first transform

    @classmethod
    def build(
        cls,
        *,
        articles: pl.DataFrame,
        history: pl.DataFrame,
        index_impressions: pl.DataFrame,
        embeddings_path: str | Path | None = None,
        with_bm25: bool = True,
        n_recent: int = DEFAULT_N_RECENT,
        **feature_kwargs,
    ) -> "MatrixBuilder":
        """`index_impressions` is what the popularity/CTR index counts over.

        Pass every impression available, not just the training split: the counts are as-of
        filtered, and indexing train alone leaves a gap wider than the trailing window, which
        makes every popularity feature exactly AUC 0.5000 (measured -- see `article_features`).
        At submission time that is the unlabelled test file, whose clicks are absent but whose
        in-view events are not.
        """
        hist_kwargs = {k: v for k, v in feature_kwargs.items() if "half_life" in k or k == "key_chunk"}
        art_kwargs = {k: v for k, v in feature_kwargs.items() if k == "windows_h"}
        return cls(
            articles=articles,
            history_features=HistoryFeatures.build(history, articles, **hist_kwargs),
            article_features=ArticleFeatures.build(index_impressions, articles, **art_kwargs),
            bm25=BM25Index(*build_article_corpus(articles)) if with_bm25 else None,
            embedding_lookup=(load_embedding_lookup(str(embeddings_path))
                              if embeddings_path else None),
            n_recent=n_recent,
        )

    # -- feature name groups -----------------------------------------------------------------

    @property
    def similarity_features(self) -> list[str]:
        return ([n for n in SIMILARITY_FEATURES
                 if (n == "bm25_score" and self._bm25 is not None)
                 or (n == "embed_cos" and self._embeddings is not None)])

    @property
    def feature_names(self) -> list[str]:
        return (SLATE_FEATURE_NAMES + self._history.feature_names
                + self._articles.feature_names + SESSION_FEATURE_NAMES
                + self.similarity_features)

    @property
    def submission_safe_feature_names(self) -> list[str]:
        """The subset computable when the impressions being scored carry no click labels, which
        is the case for both Codabench test sets. Anything derived from `clicked` drops out."""
        safe = set(SLATE_FEATURE_NAMES) | set(self._history.feature_names) \
            | set(self._articles.submission_safe_feature_names) \
            | set(SUBMISSION_SAFE_SESSION_FEATURES) | set(self.similarity_features)
        return [n for n in self.feature_names if n in safe]

    @property
    def split_unstable_feature_names(self) -> list[str]:
        """Features whose distribution shifts between train and test because of how the datasets
        were assembled rather than anything the model did. See `HistoryFeatures`."""
        return list(self._history.split_unstable_feature_names)

    # -- similarity scores (Q1.1's titles and embeddings) --------------------------------------

    def _similarity_columns(self, impressions: pl.DataFrame) -> dict[str, np.ndarray]:
        """BM25 over the user's recent titles, and cosine against their mean-pooled recent
        embeddings. Both are scored per impression against that impression's own candidates, so
        they are computed in one pass over impressions rather than over the exploded rows."""
        wanted = self.similarity_features
        if not wanted:
            return {}
        if self._history_by_user is None:
            raise RuntimeError("_prepare_history() must run before scoring similarities")

        out = {name: [] for name in wanted}
        # A user's as-of filtered recent history repeats across their impressions whenever the
        # cutoff excludes nothing new, so the derived query and vector are cached on it.
        query_cache: dict[tuple[str, ...], str] = {}
        vec_cache: dict[tuple[str, ...], np.ndarray | None] = {}

        for uid, as_of, candidates in zip(impressions["user_id"].to_list(),
                                          impressions["timestamp"].to_list(),
                                          impressions["candidates"].to_list()):
            hist = self._history_by_user.get(uid)
            recent = tuple(recent_history_asof(hist[0], hist[1], as_of, self._n_recent)
                           if hist else ())

            if "bm25_score" in out:
                if recent not in query_cache:
                    query_cache[recent] = " ".join(self._titles.get(a) or "" for a in recent)
                out["bm25_score"].extend(
                    self._bm25.score_candidates(query_cache[recent], candidates))

            if "embed_cos" in out:
                if recent not in vec_cache:
                    vec_cache[recent] = user_embedding(list(recent), self._embeddings,
                                                       self._embed_dim)
                vec = vec_cache[recent]
                out["embed_cos"].extend(
                    ann_score_candidates(vec, candidates, self._embeddings) if vec is not None
                    else [0.0] * len(candidates))

        return {k: np.asarray(v, dtype=np.float32) for k, v in out.items()}

    def _prepare_history(self) -> None:
        if self._history_by_user is None:
            h = self._history._history
            self._history_by_user = {
                r["user_id"]: (r["history_article_ids"], r["history_timestamps"])
                for r in h.select("user_id", "history_article_ids",
                                  "history_timestamps").iter_rows(named=True)
            }

    # -- public ------------------------------------------------------------------------------

    def transform(
        self,
        impressions: pl.DataFrame,
        *,
        with_labels: bool = True,
        session_feats: pl.DataFrame | None = None,
    ) -> RankingMatrix:
        """Impressions -> a `RankingMatrix`.

        `with_labels=False` is the submission path, where `clicked` is absent.

        Session features are derived from `impressions` itself, so **pass the whole set being
        scored rather than a batch** -- sessions straddle any row-count boundary and would
        silently restart mid-session. When batching a large tier, compute them once over the
        full file and hand them in as `session_feats`.
        """
        self._prepare_history()
        keep = ("clicked",) if with_labels and "clicked" in impressions.columns else ()
        rows = explode_candidates(impressions, keep=keep)

        if session_feats is None:
            session_feats = session_features(impressions)
        sess = (rows.select("impression_id").with_row_index("_row")
                    .join(session_feats, on="impression_id", how="left")
                    .sort("_row").select(SESSION_FEATURE_NAMES))

        blocks = [rows.select(SLATE_FEATURE_NAMES),
                  self._history.transform(rows),
                  self._articles.transform(rows),
                  sess]
        columns: list[np.ndarray] = []
        for block in blocks:
            for name in block.columns:
                columns.append(block[name].to_numpy().astype(np.float32, copy=False))
        for name, values in self._similarity_columns(impressions).items():
            columns.append(values)

        X = np.column_stack(columns) if columns else np.empty((rows.height, 0), dtype=np.float32)
        X = np.ascontiguousarray(X, dtype=np.float32)

        groups, impression_ids = _contiguous_groups(rows["impression_id"])

        y = None
        if keep:
            clicked = rows["clicked"].to_list()
            article_ids = rows["article_id"].to_list()
            y = np.fromiter((1 if a in set(c) else 0 for a, c in zip(article_ids, clicked)),
                            dtype=np.int8, count=rows.height)

        return RankingMatrix(
            X=X, y=y, groups=groups, feature_names=self.feature_names,
            impression_ids=impression_ids,
            article_ids=rows["article_id"].to_numpy(),
        )


def _contiguous_groups(impression_ids: pl.Series) -> tuple[np.ndarray, np.ndarray]:
    """Run-length encode the impression column into (group sizes, impression ids).

    Checks contiguity rather than assuming it: if a join ever reorders rows so an impression's
    candidates are split into two runs, the run count exceeds the distinct-id count and this
    raises, instead of handing a learning-to-rank model groups that straddle impressions.
    """
    ids = impression_ids.to_numpy()
    if len(ids) == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=ids.dtype)
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    sizes = np.diff(np.r_[starts, len(ids)]).astype(np.int32)
    run_ids = ids[starts]
    if len(np.unique(run_ids)) != len(run_ids):
        raise ValueError(
            "rows are not contiguous per impression -- a join reordered them, and the group "
            "array would straddle impressions"
        )
    return sizes, run_ids
