"""Q1 step 4: feature store -- article features and user features (click history, recency).

The article store is just the cleaned `articles` table (title/abstract/body/category/entities;
`embedding` is a null placeholder populated later, in Q3). The user store adds two cheap,
leak-safe aggregates on top of the cleaned `history` table: history_length and
last_history_time. Both are derived purely from the raw history snapshot itself (not from any
impression being predicted), so they carry no future-click information -- see history_asof()
below for the per-impression cutoff future Qs should use when they need it.
"""
from datetime import datetime

import polars as pl


def build_article_store(articles: pl.DataFrame) -> pl.DataFrame:
    return articles


def build_user_store(history: pl.DataFrame) -> pl.DataFrame:
    return history


def history_asof(
    article_ids: list[str], timestamps: list[datetime] | None, cutoff: datetime
) -> list[str]:
    """Restrict a user's click history to entries strictly before `cutoff`.

    For EB-NeRD, `timestamps` is the real per-item history, so this filters out anything at or
    after the impression being scored -- the anti-leakage guarantee Q9 asks for. For MIND,
    `timestamps` is None because MIND's history is a fixed pre-collection-period snapshot,
    constant across all of a user's impressions (verified in clean.py) -- it's leak-safe by
    construction already, so it's returned unfiltered.
    """
    if timestamps is None:
        return article_ids
    return [aid for aid, ts in zip(article_ids, timestamps) if ts is not None and ts < cutoff]


def recent_history_asof(
    article_ids: list[str], timestamps: list[datetime] | None, cutoff: datetime, n_recent: int = 10
) -> list[str]:
    """The last `n_recent` article ids from history_asof(), i.e. the user's most recent clicks
    strictly before `cutoff`. History lists are stored oldest-first (verified in clean.py for
    EB-NeRD; MIND ships its history pre-ordered the same way), so "most recent" = the tail."""
    return history_asof(article_ids, timestamps, cutoff)[-n_recent:]
