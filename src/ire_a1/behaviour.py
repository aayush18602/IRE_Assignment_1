"""A2 Q1.1: user-side features from the click history -- recency decay, counts, category affinity.

Every feature is computed as of an explicit `as_of` and sees only clicks strictly before it
(Q1.4). That matters here more than it looks: EB-NeRD's history snapshot runs to 2023-05-25 while
train impressions start 2023-05-18, so history and impressions **overlap**, and a train impression
must not see history entries recorded after it. The filter is a no-op for val/test (whose windows
start after the snapshot ends) but relying on that coincidence would be a leak waiting for the
next split configuration.

The two datasets need different decay clocks, which is the main structural thing here:

- **EB-NeRD** has per-item history timestamps, so decay is exponential in *elapsed time*, with a
  half-life in hours. Weights depend on `as_of`, so state is per (user, as_of) -- effectively per
  impression.
- **MIND** has none (`history_timestamps` is null for all 94,057 rows) and its history is a fixed
  pre-collection snapshot, so decay is exponential in *position from the end* instead, with a
  half-life in clicks. Weights do not depend on `as_of` at all, so state is per user and gets
  computed once -- 94K users instead of 230K impressions.

Not here, deliberately: BM25 and raw embedding similarity between the candidate and the user's
recent titles. Those are A1 components (`bm25.score_candidates`, `ann.score_candidates`) and
`reranker.build_matrix` calls them directly at the point where their indexes are already loaded --
pulling a 226MB embedding table into this module to recompute them would be the wrong seam.
"""
from __future__ import annotations

import math

import numpy as np
import polars as pl

US_PER_HOUR = 3_600_000_000
LN2 = math.log(2.0)

# Measured, not guessed. Sweeping the EB-NeRD half-life against test-split AUC for cat_affinity:
# 6h 0.5578, 24h 0.5667, 72h 0.5699, 168h 0.5702, 336h 0.5699, 720h 0.5696, undecayed 0.5693.
# 168h is the peak and the margin over no decay at all is +0.0009 -- see the module note in
# CONTEXT.md; recency decay is close to inert on this data, and the honest default is the one that
# measured best rather than the one that sounds most principled.
DEFAULT_HALF_LIFE_HOURS = 168.0
# MIND is worse: decay monotonically *hurts* (3 clicks 0.6000 ... 100 clicks 0.6079, undecayed
# 0.6080), converging to the undecayed value from below. A long half-life is the least-harm choice.
DEFAULT_HALF_LIFE_POSITIONS = 50.0
DEFAULT_KEY_CHUNK = 50_000

_BURST_WINDOWS_H = (24, 168)   # 1 day, 7 days


class HistoryFeatures:
    """Build once from the history table, then `transform()` batches of candidate rows."""

    def __init__(
        self,
        history: pl.DataFrame,
        categories: pl.DataFrame,
        *,
        half_life_hours: float,
        half_life_positions: float,
        key_chunk: int,
    ):
        self._history = history
        self._categories = categories
        self._half_life_hours = half_life_hours
        self._half_life_positions = half_life_positions
        self._key_chunk = key_chunk
        self._timed = history["history_timestamps"].null_count() < history.height
        self._has_dwell = history["history_read_times"].null_count() < history.height
        self._user_state: pl.DataFrame | None = None if self._timed else self._build_user_state()

    @classmethod
    def build(
        cls,
        history: pl.DataFrame,
        articles: pl.DataFrame,
        *,
        half_life_hours: float = DEFAULT_HALF_LIFE_HOURS,
        half_life_positions: float = DEFAULT_HALF_LIFE_POSITIONS,
        key_chunk: int = DEFAULT_KEY_CHUNK,
    ) -> "HistoryFeatures":
        return cls(
            history,
            articles.select("article_id", "category"),
            half_life_hours=half_life_hours,
            half_life_positions=half_life_positions,
            key_chunk=key_chunk,
        )

    @property
    def feature_names(self) -> list[str]:
        names = ["hist_len", "hist_len_asof", "decay_weight_sum", "cat_affinity", "cat_affinity_raw"]
        if self._timed:
            names += ["hours_since_last_click"] + [f"hist_burst_{h}h" for h in _BURST_WINDOWS_H]
        if self._has_dwell:
            names += ["dwell_affinity"]
        return names

    @property
    def split_unstable_feature_names(self) -> list[str]:
        """Features whose distribution shifts between train and test through no fault of the model.

        EB-NeRD's history is a **fixed snapshot** ending 2023-05-25, while impressions run to
        06-01. So the history gets staler as the impression date advances: measured mean
        `hours_since_last_click` is 12.6 on train, 113.9 on val, 159.8 on test. A model trained on
        train learns that recent activity predicts clicks, then meets a test set where every user
        looks dormant -- which is worse than a useless feature, because the model leans on it.

        This is why the trailing counts here are measured backwards from the user's *own last
        click* (`hist_burst_*`) rather than from `as_of`: an as-of-relative count read 17.0 on
        train and exactly 0.0 on val and test, i.e. dead everywhere it mattered.
        `hours_since_last_click` is kept because it is a genuine serving signal and Q9 wants the
        with/without comparison, but Q2 should be able to exclude it by name rather than by memory.
        """
        return ["hours_since_last_click"] if self._timed else []

    # -- history -> one row per (key, history item), with a decay weight ---------------------

    def _explode(self, keys: pl.DataFrame | None) -> pl.DataFrame:
        """`keys` is a frame of (user_id, as_of) for the timed path, or None for the positional
        path (where weights are as-of independent and one row per user suffices)."""
        cols = ["user_id", "history_article_ids", "history_length"]
        if self._timed:
            cols.append("history_timestamps")
        if self._has_dwell:
            cols.append("history_read_times")

        h = self._history.select(cols)
        if keys is not None:
            h = keys.join(h, on="user_id", how="left")

        list_cols = [c for c in ("history_article_ids", "history_timestamps", "history_read_times")
                     if c in h.columns]
        h = h.filter(pl.col("history_article_ids").list.len() > 0)
        h = h.explode(list_cols, empty_as_null=False)

        if self._timed:
            # strictly before as_of -- the Q1.4 boundary, applied before any weighting so a
            # future click cannot contribute even a vanishing amount
            h = h.filter(pl.col("history_timestamps") < pl.col("as_of"))
            age_h = (pl.col("as_of") - pl.col("history_timestamps")).dt.total_microseconds() / US_PER_HOUR
            h = h.with_columns(
                age_hours=age_h,
                weight=(-LN2 * age_h / self._half_life_hours).exp(),
            )
        else:
            # Position from the end: 0 is the most recent click. MIND's history is stored
            # oldest-first (verified in clean.py), so this is a reverse rank within each user.
            h = h.with_columns(
                rank_from_end=(pl.col("history_article_ids").cum_count().reverse().over("user_id") - 1)
                .cast(pl.Float64)
            ).with_columns(
                weight=(-LN2 * pl.col("rank_from_end") / self._half_life_positions).exp()
            )
        return h.join(self._categories, left_on="history_article_ids", right_on="article_id", how="left")

    # -- aggregates ---------------------------------------------------------------------------

    def _aggregate(self, exploded: pl.DataFrame, key: list[str]) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Returns (per-key totals, per-(key, category) weighted mass)."""
        totals = [
            pl.len().alias("hist_len_asof"),
            pl.col("weight").sum().alias("decay_weight_sum"),
            pl.col("history_length").first().alias("hist_len"),
        ]
        if self._timed:
            totals.append(pl.col("age_hours").min().alias("hours_since_last_click"))
            for h in _BURST_WINDOWS_H:
                # relative to the user's own most recent click, not to as_of -- see
                # split_unstable_feature_names for the measurement that forced this
                totals.append(
                    (pl.col("age_hours") <= pl.col("age_hours").min() + h).sum()
                    .alias(f"hist_burst_{h}h")
                )

        cat_aggs = [pl.col("weight").sum().alias("_cat_w"), pl.len().alias("_cat_n")]
        if self._has_dwell:
            # dwell on *past* clicks -- serving-safe, unlike the impression row's own read_time
            totals.append(pl.col("history_read_times").sum().alias("_dwell_total"))
            cat_aggs.append(pl.col("history_read_times").sum().alias("_cat_dwell"))

        return (exploded.group_by(key).agg(totals),
                exploded.group_by(key + ["category"]).agg(cat_aggs))

    def _build_user_state(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        return self._aggregate(self._explode(None), ["user_id"])

    # -- public ------------------------------------------------------------------------------

    def transform(self, rows: pl.DataFrame) -> pl.DataFrame:
        """`rows` needs `user_id`, `article_id` and `as_of`, one row per candidate. Returns one
        column per `feature_names`, index-aligned with `rows`."""
        for col in ("user_id", "article_id", "as_of"):
            if col not in rows.columns:
                raise ValueError(f"transform() needs a `{col}` column; got {rows.columns}")

        indexed = rows.with_row_index("_row").join(
            self._categories.rename({"category": "_cand_category"}),
            on="article_id", how="left",
        )

        if self._timed:
            key = ["user_id", "as_of"]
            keys = rows.select(key).unique()
            parts = [self._aggregate(self._explode(chunk), key)
                     for chunk in keys.iter_slices(self._key_chunk)]
            totals = pl.concat([p[0] for p in parts])
            cats = pl.concat([p[1] for p in parts])
        else:
            key = ["user_id"]
            totals, cats = self._user_state

        out = (indexed.join(totals, on=key, how="left")
                      .join(cats, left_on=key + ["_cand_category"],
                            right_on=key + ["category"], how="left")
                      .sort("_row"))

        exprs = [
            pl.col("hist_len").fill_null(0).cast(pl.Float32),
            pl.col("hist_len_asof").fill_null(0).cast(pl.Float32),
            pl.col("decay_weight_sum").fill_null(0.0).cast(pl.Float32),
            # share of the user's decayed attention that went to this candidate's category;
            # 0 when the user has no usable history, which is different from "wrong category"
            # but is the honest encoding of "we know nothing"
            (pl.col("_cat_w").fill_null(0.0) / (pl.col("decay_weight_sum").fill_null(0.0) + 1e-9))
            .cast(pl.Float32).alias("cat_affinity"),
            (pl.col("_cat_n").fill_null(0) / (pl.col("hist_len_asof").fill_null(0) + 1e-9))
            .cast(pl.Float32).alias("cat_affinity_raw"),
        ]
        if self._timed:
            exprs.append(pl.col("hours_since_last_click").cast(pl.Float32))
            exprs += [pl.col(f"hist_burst_{h}h").fill_null(0).cast(pl.Float32)
                      for h in _BURST_WINDOWS_H]
        if self._has_dwell:
            exprs.append(
                (pl.col("_cat_dwell").fill_null(0.0) / (pl.col("_dwell_total").fill_null(0.0) + 1e-9))
                .cast(pl.Float32).alias("dwell_affinity")
            )
        return out.select(exprs).select(self.feature_names)
