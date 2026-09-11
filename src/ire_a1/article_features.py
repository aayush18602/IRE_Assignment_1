"""A2 Q1.3: time-aware item-side features -- trailing popularity, CTR, and freshness.

Every lookup here takes an explicit `as_of` and counts only events strictly *before* it. That is
the Q1.4 behaviour-window boundary, and putting it inside the lookup rather than leaving it to the
caller is what makes it testable: there is no code path that can accidentally see the future,
because there is no argument-less variant to call.

Two things are deliberately *not* here:

- **Category match with user history**, which A2 lists under Q1.3, lives in `behaviour.py`
  instead. It needs the as-of-filtered, decay-weighted click history, and that machinery belongs
  in one place -- computing a second, cruder version here would duplicate the boundary logic in
  exactly the spot where a leak would be hardest to spot.
- **`total_inviews` / `total_pageviews`** from the articles table. They are lifetime counters
  aggregated over the whole collection period, so no as-of cutoff can be reconstructed for them
  (`schema.LEAKY_ARTICLE_COLS`). Q9 uses them; features never do.

Two measured findings shaped this module; both are counter-intuitive enough to be worth stating
where the code lives rather than only in the design note.

**1. Indexing the train split alone does not work for trailing windows.** It is the instinctive
"safe" choice, and it silently produces all-zero features. The window is relative to each
impression's own timestamp, and our temporal split leaves a 2-day gap (EB-NeRD) between the end of
train and the start of test -- so a 24h window at test time reaches back into empty space. Measured
on the EB-NeRD test split, every popularity feature scored exactly AUC 0.5000 that way. Indexing
*all* impressions and relying on the as-of filter takes ctr_1h to 0.7139. The as-of cutoff is what
makes this safe, and it is enforced here rather than trusted to the caller.

**2. Click-derived and in-view-derived features have different serving-time availability.** Both
Codabench test sets ship candidates but no click labels (EB-NeRD's test `behaviors.parquet` has no
`article_ids_clicked` column at all), so `pop_clicks_*` and `ctr_*` cannot be computed over the
test period at submission time, while `pop_inviews_*` can. That is a property of the released
files rather than a fundamental serving limit -- a deployed system sees clicks as they happen --
but it means the offline-evaluated model and the submitted model do not have the same inputs.
`submission_safe_feature_names` names the subset that survives, so the distinction is mechanical
instead of a comment someone has to remember.
"""
from __future__ import annotations

import numpy as np
import polars as pl

US_PER_HOUR = 3_600_000_000
DEFAULT_WINDOWS_H = (1, 24)


def _to_us(series: pl.Series) -> np.ndarray:
    """Datetime -> int64 microseconds. Microsecond resolution matters: at second resolution a
    click landing in the same second as `as_of` could not be distinguished from one landing just
    before it, and the strict boundary would quietly become approximate."""
    return series.dt.timestamp("us").to_numpy().astype(np.int64)


class _EventIndex:
    """Per-article sorted event times, queried by binary search.

    Counting is vectorised per *article* rather than per row: query rows are grouped by article
    and each group costs one `np.searchsorted` call over that article's own times. So the number
    of numpy calls tracks the catalogue size (20K-65K) rather than the row count (5-9M), which is
    what keeps this usable on the large tiers.
    """

    __slots__ = ("_times",)

    def __init__(self, times: dict[int, np.ndarray]):
        self._times = times

    @classmethod
    def from_pairs(cls, article_idx: np.ndarray, times_us: np.ndarray) -> "_EventIndex":
        # An empty event set is the normal submission-time case, not an error: both Codabench
        # test sets ship candidates with no click labels, so the click index is legitimately
        # empty there and every count it returns should be 0.
        if len(article_idx) == 0:
            return cls({})
        order = np.argsort(article_idx, kind="stable")
        ids, ts = article_idx[order], times_us[order]
        starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
        ends = np.r_[starts[1:], len(ids)]
        return cls({int(ids[s]): np.sort(ts[s:e]) for s, e in zip(starts, ends)})

    def count_in_window(
        self, article_idx: np.ndarray, as_of_us: np.ndarray, window_us: int | None
    ) -> np.ndarray:
        """Events in `[as_of - window, as_of)` per row -- inclusive left, **exclusive right**.
        `window_us=None` counts everything before `as_of` with no lower bound."""
        out = np.zeros(len(article_idx), dtype=np.int32)
        if len(article_idx) == 0:
            return out

        order = np.argsort(article_idx, kind="stable")
        sorted_ids = article_idx[order]
        starts = np.flatnonzero(np.r_[True, sorted_ids[1:] != sorted_ids[:-1]])
        ends = np.r_[starts[1:], len(sorted_ids)]

        for s, e in zip(starts, ends):
            times = self._times.get(int(sorted_ids[s]))
            if times is None:          # article never seen in the index period
                continue
            rows = order[s:e]
            t = as_of_us[rows]
            # side="left" on the upper bound is the boundary: an event at exactly `as_of` sorts
            # to the left of the insertion point, so it is excluded.
            hi = np.searchsorted(times, t, side="left")
            lo = (np.searchsorted(times, t - window_us, side="left") if window_us is not None
                  else np.zeros_like(hi))
            out[rows] = hi - lo
        return out


class ArticleFeatures:
    """Item-side feature builder. `build()` indexes a set of impressions once; `transform()`
    scores a batch of (article, as_of) rows against it.

    **Which impressions to index is the caller's decision, and it is a real one.** Passing only
    the train split is the conservative choice and matches how A1's anti-gaming script defined
    "safe" popularity. Passing every split is defensible too -- and arguably more faithful to
    serving, where clicks that already happened are known regardless of which evaluation split
    they belong to -- precisely *because* every count here is as-of filtered. What is never
    acceptable is an unfiltered count, which is why no such method exists.
    """

    def __init__(
        self,
        article_ids: list[str],
        clicks: _EventIndex,
        inviews: _EventIndex,
        published_us: np.ndarray,
        first_seen_us: np.ndarray,
        windows_h: tuple[int, ...],
    ):
        self._idx = {a: i for i, a in enumerate(article_ids)}
        self._clicks = clicks
        self._inviews = inviews
        self._published_us = published_us
        self._first_seen_us = first_seen_us
        self._windows_h = windows_h

    @classmethod
    def build(
        cls,
        impressions: pl.DataFrame,
        articles: pl.DataFrame,
        windows_h: tuple[int, ...] = DEFAULT_WINDOWS_H,
    ) -> "ArticleFeatures":
        article_ids = articles["article_id"].to_list()
        idx = {a: i for i, a in enumerate(article_ids)}
        n = len(article_ids)

        def explode_events(col: str) -> tuple[np.ndarray, np.ndarray]:
            # Empty lists are filtered out before exploding, which makes `empty_as_null`
            # behaviourally irrelevant here -- so take Polars 2.0's future default rather than
            # pinning the deprecated one.
            ev = (impressions.select(pl.col(col), pl.col("timestamp"))
                             .filter(pl.col(col).list.len() > 0)
                             .explode(col, empty_as_null=False)
                             .drop_nulls(col))
            mapped = np.fromiter((idx.get(a, -1) for a in ev[col].to_list()),
                                 dtype=np.int64, count=ev.height)
            times = _to_us(ev["timestamp"])
            keep = mapped >= 0          # ids absent from the article table (rare, but real)
            return mapped[keep], times[keep]

        click_ids, click_ts = explode_events("clicked")
        view_ids, view_ts = explode_events("candidates")

        # published_time is the real signal on EB-NeRD and 100% null on MIND. first_seen -- the
        # article's earliest appearance anywhere in the indexed impressions -- is MIND's stand-in.
        # It is only a lower bound on the true publish time (an article may have been published
        # well before it was first shown), so it is exposed as its own column rather than
        # silently merged into freshness_hours.
        published_us = np.full(n, np.iinfo(np.int64).min, dtype=np.int64)
        pub = articles["published_time"]
        if pub.null_count() < n:
            have = ~pub.is_null().to_numpy()
            published_us[have] = _to_us(pub.fill_null(strategy="zero"))[have]

        first_seen_us = np.full(n, np.iinfo(np.int64).min, dtype=np.int64)
        if len(view_ids):
            order = np.argsort(view_ids, kind="stable")
            vi, vt = view_ids[order], view_ts[order]
            starts = np.flatnonzero(np.r_[True, vi[1:] != vi[:-1]])
            first_seen_us[vi[starts]] = np.minimum.reduceat(vt, starts)

        return cls(
            article_ids,
            _EventIndex.from_pairs(click_ids, click_ts),
            _EventIndex.from_pairs(view_ids, view_ts),
            published_us,
            first_seen_us,
            windows_h,
        )

    @property
    def feature_names(self) -> list[str]:
        names: list[str] = []
        for h in self._windows_h:
            names += [f"pop_clicks_{h}h", f"pop_inviews_{h}h", f"ctr_{h}h"]
        return names + ["freshness_hours", "hours_since_first_seen", "has_published_time"]

    @property
    def submission_safe_feature_names(self) -> list[str]:
        """The subset computable when the impressions being scored carry no click labels.

        Both Codabench test sets are like that, so anything derived from `clicked` is unavailable
        over the test period at submission time. Training a model on the full set and submitting
        with these is a silent input mismatch; Q9's "with and without serving-time features"
        comparison is exactly the honest way to report it.
        """
        clicky = {n for h in self._windows_h for n in (f"pop_clicks_{h}h", f"ctr_{h}h")}
        return [n for n in self.feature_names if n not in clicky]

    def transform(self, rows: pl.DataFrame) -> pl.DataFrame:
        """`rows` needs `article_id` (str) and `as_of` (Datetime), one row per candidate.
        Returns one column per `feature_names`, index-aligned with `rows`."""
        for col in ("article_id", "as_of"):
            if col not in rows.columns:
                raise ValueError(f"transform() needs an `{col}` column; got {rows.columns}")

        a_idx = np.fromiter((self._idx.get(a, -1) for a in rows["article_id"].to_list()),
                            dtype=np.int64, count=rows.height)
        as_of = _to_us(rows["as_of"])
        known = a_idx >= 0

        out: dict[str, np.ndarray] = {}
        for h in self._windows_h:
            w = h * US_PER_HOUR
            clicks = self._clicks.count_in_window(a_idx, as_of, w).astype(np.float32)
            views = self._inviews.count_in_window(a_idx, as_of, w).astype(np.float32)
            out[f"pop_clicks_{h}h"] = clicks
            out[f"pop_inviews_{h}h"] = views
            # +1 smoothing: an article shown twice and clicked once is not 3x more clickable than
            # one shown 200 times and clicked 100 times, and unsmoothed ratios say it is.
            out[f"ctr_{h}h"] = clicks / (views + 1.0)

        def age_hours(reference: np.ndarray) -> np.ndarray:
            """Age in hours, or NaN when the reference point is not strictly in the past.

            The `ref < as_of` guard is the as-of boundary, not defensive padding. `first_seen` is
            a min over the whole indexed period, so without it an article whose first appearance
            falls *after* the impression being scored returns a negative age -- which encodes
            "this article shows up in the future", exactly the future information the boundary
            exists to exclude. It survives in practice only because a candidate in the current
            slate contributes its own in-view event, and a correctness guarantee must not rest on
            that coincidence. `test_no_leakage.py` pins it by building the index twice.
            """
            ref = np.where(known, reference[np.clip(a_idx, 0, None)], np.iinfo(np.int64).min)
            valid = known & (ref != np.iinfo(np.int64).min) & (ref < as_of)
            age = np.full(len(a_idx), np.nan, dtype=np.float32)
            age[valid] = (as_of[valid] - ref[valid]) / US_PER_HOUR
            return age

        out["freshness_hours"] = age_hours(self._published_us)
        out["hours_since_first_seen"] = age_hours(self._first_seen_us)
        out["has_published_time"] = (~np.isnan(out["freshness_hours"])).astype(np.float32)
        return pl.DataFrame(out)
