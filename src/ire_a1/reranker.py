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

import json
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
        # A left join fans out when the right side has duplicate keys, which is exactly what a
        # non-unique impression_id produces (MIND numbers from 1 in each of its behaviors files).
        # Catching it here names the cause; without this the symptom is a numpy shape error two
        # hundred lines later that says nothing about impression ids.
        if session_feats["impression_id"].n_unique() != session_feats.height:
            raise ValueError(
                "session features contain duplicate impression_id values -- joining on them "
                "would fan out the feature matrix. Impression ids must be unique within the "
                "set being scored."
            )
        sess = (rows.select("impression_id").with_row_index("_row")
                    .join(session_feats, on="impression_id", how="left")
                    .sort("_row").select(SESSION_FEATURE_NAMES))
        if sess.height != rows.height:
            raise ValueError(
                f"session-feature join changed the row count ({rows.height} -> {sess.height})"
            )

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


# ---------------------------------------------------------------------------
# A2 Q2 Option A: LightGBM LambdaRank
# ---------------------------------------------------------------------------

DEFAULT_PARAMS: dict = {
    "objective": "lambdarank",
    # Optimise the metric the leaderboards actually report rather than a proxy. lambdarank
    # compares candidates only within their own group, which is also why the features that are
    # constant within an impression (see run_feature_analysis.py) cost nothing here: they can
    # still gate a split, but they can never be mistaken for ranking signal.
    "metric": "ndcg",
    "eval_at": [5, 10],
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "lambdarank_truncation_level": 20,
    "verbosity": -1,
    "seed": 42,
    "deterministic": True,
    "force_row_wise": True,   # silences the threading-strategy guess and keeps runs comparable
}


class LambdaRanker:
    """LightGBM LambdaRank over a `RankingMatrix`.

    Option A of Q2.2. Chosen over the neural alternative for the GBDT track because it needs no
    GPU, trains in minutes on CPU, and takes NaN as a first-class split direction -- which matters
    here, since `freshness_hours` is 100% NaN on MIND (no `published_time`) and imputing it would
    invent a publication date rather than admit the field is missing.
    """

    def __init__(self, booster, feature_names: list[str], params: dict):
        self.booster = booster
        self.feature_names = feature_names
        self.params = params

    @classmethod
    def train(
        cls,
        train: RankingMatrix,
        valid: RankingMatrix | None = None,
        *,
        num_boost_round: int = 1000,
        early_stopping_rounds: int = 50,
        params: dict | None = None,
        log_every: int = 50,
    ) -> "LambdaRanker":
        """Train on `train`, early-stopping on `valid`'s nDCG@10.

        `valid` must be a *later* temporal split than `train` -- stopping on a random split would
        pick the round that best fits the period being predicted, which is the same leak the
        temporal split exists to prevent, just moved into hyperparameter selection.
        """
        import lightgbm as lgb

        if train.y is None:
            raise ValueError("training needs labels; build the matrix with with_labels=True")

        merged = {**DEFAULT_PARAMS, **(params or {})}
        train_set = lgb.Dataset(train.X, label=train.y, group=train.groups,
                                feature_name=train.feature_names, free_raw_data=False)
        callbacks = [lgb.log_evaluation(period=log_every)]
        valid_sets, valid_names = [], []
        if valid is not None and valid.y is not None:
            valid_sets.append(lgb.Dataset(valid.X, label=valid.y, group=valid.groups,
                                          feature_name=valid.feature_names, reference=train_set))
            valid_names.append("valid")
            callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=False))

        booster = lgb.train(merged, train_set, num_boost_round=num_boost_round,
                            valid_sets=valid_sets, valid_names=valid_names, callbacks=callbacks)
        return cls(booster, list(train.feature_names), merged)

    def predict(self, matrix: RankingMatrix, tie_breaker: str | None = None) -> np.ndarray:
        """One score per row, in row order. Refuses a matrix whose columns do not match the ones
        the model was trained on -- LightGBM would otherwise happily score positionally and
        return plausible nonsense.

        `tie_breaker` names a continuous input feature used to order candidates the model scored
        *identically*. It cannot change any ordering the model actually expressed, so it is
        non-harmful by construction, and it addresses a measured defect: a GBDT bins its inputs,
        so it cannot reproduce a continuous feature's ordering even when that feature is its only
        input. Measured on MIND, a model trained on `embed_cos` alone scores 0.6067 against the
        raw feature's 0.6168, and produces only 87% distinct scores per impression where the raw
        feature produces 97%. Finer binning does not close it (94% distinct still scores 0.6080).
        That gap is the reason a re-ranker is not guaranteed to beat its own inputs.
        """
        if matrix.feature_names != self.feature_names:
            missing = [n for n in self.feature_names if n not in matrix.feature_names]
            extra = [n for n in matrix.feature_names if n not in self.feature_names]
            raise ValueError(
                "feature mismatch between the model and the matrix being scored"
                + (f"; missing {missing}" if missing else "")
                + (f"; unexpected {extra}" if extra else "")
                + ("; same names, different order" if not missing and not extra else "")
            )
        scores = self.booster.predict(matrix.X, num_iteration=self.booster.best_iteration)
        if tie_breaker is None:
            return scores
        if tie_breaker not in matrix.feature_names:
            raise ValueError(f"tie_breaker {tie_breaker!r} is not one of the matrix's features")

        # Rank the tie-breaker within each impression and add it at a magnitude far below the
        # model's own spread, so it only ever separates candidates the model left equal.
        raw = matrix.column(tie_breaker).astype(np.float64)
        within = np.empty_like(raw)
        offset = 0
        for size in matrix.groups:
            sl = slice(offset, offset + size)
            values = np.nan_to_num(raw[sl], nan=0.0)
            within[sl] = values.argsort().argsort() / max(size - 1, 1)
            offset += size
        epsilon = 1e-6 * (float(np.ptp(scores)) or 1.0)
        return scores + epsilon * within

    def degenerate_features(self, matrix: RankingMatrix) -> list[str]:
        """Columns with no variance at all on this data -- constant or entirely NaN.

        They are not always a bug: `freshness_hours` is 100% NaN on MIND because MIND ships no
        `published_time`, and `session_dwell_so_far` is constant because it ships no dwell
        instrumentation. But a feature that cannot vary cannot inform a split, so a long list
        here means the feature set is smaller than it looks and the model has less to work with
        than the column count suggests.
        """
        dead = []
        for i, name in enumerate(matrix.feature_names):
            column = matrix.X[:, i]
            finite = column[np.isfinite(column)]
            if len(finite) == 0 or finite.std() == 0:
                dead.append(name)
        return dead

    def feature_importance(self) -> list[tuple[str, float]]:
        """Gain-based importance, descending. Worth reading before believing a good result: a
        trailing-popularity window whose boundary is off by one impression looks like a brilliant
        model, and it shows up here as one feature dominating everything else."""
        gains = self.booster.feature_importance(importance_type="gain")
        total = float(gains.sum()) or 1.0
        return sorted(((n, float(g) / total) for n, g in zip(self.feature_names, gains)),
                      key=lambda p: -p[1])

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(path), num_iteration=self.booster.best_iteration)
        path.with_suffix(".features.json").write_text(
            json.dumps({"feature_names": self.feature_names, "params": self.params}, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "LambdaRanker":
        import lightgbm as lgb

        path = Path(path)
        meta = json.loads(path.with_suffix(".features.json").read_text())
        return cls(lgb.Booster(model_file=str(path)), meta["feature_names"], meta["params"])
