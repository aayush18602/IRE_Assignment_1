"""Unified schema for EB-NeRD and MIND, shared by the whole pipeline (A1 Q1, A2 Q1).

Both datasets are cleaned into three tables with identical column sets, so everything
downstream (BM25, embeddings, eval) can be dataset-agnostic. Article/user/impression ids are
always cast to Utf8 so EB-NeRD's integer ids and MIND's "N12345"-style ids compare equal.

Columns are grouped by what they are for. The behavioural columns were added for A2 Q1 (click
history, sessions, dwell); they exist only in EB-NeRD's logs and are null throughout MIND, which
is a real asymmetry the feature builders have to handle rather than hide -- see
LEAKY_*_COLS below for the separate matter of columns that exist but must not be used as
features.
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
    "embedding",        # list[f32] | null -- placeholder, populated in A1 Q3
    # --- A2 Q9 only: corpus-lifetime counters, NOT serving-safe (see LEAKY_ARTICLE_COLS) ---
    "total_inviews",    # i32 | null -- EB-NeRD only
    "total_pageviews",  # i32 | null -- EB-NeRD only
    "total_read_time",  # f32 | null -- EB-NeRD only
]

IMPRESSION_COLS = [
    "dataset",
    "impression_id",    # str
    "user_id",           # str
    "timestamp",          # datetime
    "candidates",       # list[str] -- articles shown (article_ids_inview / parsed impressions)
    "clicked",           # list[str] -- ground-truth clicked articles (subset of candidates)
    "device_type",       # str | null
    # --- A2 Q1.2: session context. EB-NeRD only; MIND has no session concept, so it is null and
    #     behaviour.py derives pseudo-sessions from timestamp gaps instead. ---
    "session_id",        # str | null -- globally unique in EB-NeRD, not per-user
    # --- A2 Q1: static user attributes, EB-NeRD only ---
    "age",               # i8 | null -- bucketed by the dataset's creators, not a raw age
    "gender",            # i8 | null -- categorical code
    "postcode",          # i8 | null -- categorical code
    "is_subscriber",     # bool | null
    # --- A2 Q9 only: measured AFTER the click, NOT serving-safe (see LEAKY_IMPRESSION_COLS) ---
    "read_time",              # f32 | null
    "scroll_percentage",      # f32 | null
    "next_read_time",         # f32 | null
    "next_scroll_percentage", # f32 | null
]

HISTORY_COLS = [
    "dataset",
    "user_id",
    "history_article_ids",  # list[str], time-ordered
    "history_timestamps",   # list[datetime] | null -- MIND has no per-item timestamps
    "history_length",       # i64
    "last_history_time",    # datetime | null -- "recency" feature, derived from the raw history file
    # --- A2 Q1.1/Q1.2: per-item engagement on PAST clicks. Position-aligned with
    #     history_article_ids. Serving-safe: these describe clicks that already happened, unlike
    #     the impression-row read_time above. EB-NeRD only. ---
    "history_read_times",           # list[f32] | null
    "history_scroll_percentages",   # list[f32] | null
]

# ---------------------------------------------------------------------------
# A2 Q9: columns that exist in the logs but are NOT knowable at ranking time
# ---------------------------------------------------------------------------
# They are carried through cleaning on purpose -- Q9 requires reporting metrics *with* and
# *without* features unavailable at serving time, and building that comparison honestly needs the
# real columns rather than a synthetic stand-in. Nothing in the Q1 feature builders may read them;
# behaviour.py asserts this against these lists.

LEAKY_IMPRESSION_COLS = [
    "read_time",               # dwell on the article clicked IN this impression -- post-click
    "scroll_percentage",       # scroll depth on that same article -- post-click
    "next_read_time",          # dwell on the NEXT article the user read -- pure future
    "next_scroll_percentage",  # scroll depth on that next article -- pure future
]

LEAKY_ARTICLE_COLS = [
    "total_inviews",     # lifetime counters aggregated over the whole collection period, test
    "total_pageviews",   # window included -- no as-of cutoff exists for them, so they cannot be
    "total_read_time",   # reconstructed as of any particular serving time
]

LEAKY_COLS = LEAKY_IMPRESSION_COLS + LEAKY_ARTICLE_COLS
