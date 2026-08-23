"""Unified schema for EB-NeRD and MIND, shared by the whole pipeline (Q1).

Both datasets are cleaned into three tables with identical column sets, so everything
downstream (BM25, embeddings, eval) can be dataset-agnostic. Article/user/impression ids are
always cast to Utf8 so EB-NeRD's integer ids and MIND's "N12345"-style ids compare equal.
"""

ARTICLE_COLS = [
    "dataset",          # "ebnerd" | "mind"
    "article_id",       # str
    "title",            # str
    "abstract",         # str -- EB-NeRD: subtitle, MIND: abstract
    "body",             # str | null -- MIND has no body text
    "category",         # str
    "entities",         # list[str] -- EB-NeRD: NER clusters, MIND: title/abstract entity labels
    "published_time",   # datetime | null -- MIND doesn't provide this
    "sentiment_score",  # f32 | null -- EB-NeRD only
    "sentiment_label",  # str | null -- EB-NeRD only
    "embedding",        # list[f32] | null -- placeholder, populated in Q3
]

IMPRESSION_COLS = [
    "dataset",
    "impression_id",    # str
    "user_id",           # str
    "timestamp",          # datetime
    "candidates",       # list[str] -- articles shown (article_ids_inview / parsed impressions)
    "clicked",           # list[str] -- ground-truth clicked articles (subset of candidates)
    "device_type",       # str | null
]

HISTORY_COLS = [
    "dataset",
    "user_id",
    "history_article_ids",  # list[str], time-ordered
    "history_timestamps",   # list[datetime] | null -- MIND has no per-item timestamps
    "history_length",       # i64
    "last_history_time",    # datetime | null -- "recency" feature, derived from the raw history file
]
