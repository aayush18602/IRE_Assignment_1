"""A1 Q1 step 2: parse EB-NeRD and MIND raw files into the unified schema (schema.py).

Both `clean_ebnerd` and `clean_mind` return {"articles": DataFrame, "impressions": DataFrame,
"history": DataFrame}, with identical columns (schema.ARTICLE_COLS / IMPRESSION_COLS /
HISTORY_COLS) so downstream code never needs to branch on dataset.
"""
import json
from pathlib import Path

import polars as pl

from ire_a1.schema import ARTICLE_COLS, HISTORY_COLS, IMPRESSION_COLS

MIND_NEWS_COLS = ["news_id", "category", "subcategory", "title", "abstract", "url",
                   "title_entities", "abstract_entities"]
MIND_BEHAVIOR_COLS = ["impression_id", "user_id", "time", "history", "impressions"]


# ---------------------------------------------------------------------------
# EB-NeRD
# ---------------------------------------------------------------------------

def clean_ebnerd(raw_dir: Path) -> dict[str, pl.DataFrame]:
    """raw_dir e.g. data/ebnerd_demo or data/ebnerd_small (must contain articles.parquet,
    train/{behaviors,history}.parquet, validation/{behaviors,history}.parquet). Train and
    validation are concatenated -- Q1 does its own temporal split downstream, so EB-NeRD's
    built-in train/validation partition is not treated as authoritative here."""
    raw_dir = Path(raw_dir)

    articles_raw = pl.read_parquet(raw_dir / "articles.parquet")
    articles = articles_raw.select(
        dataset=pl.lit("ebnerd"),
        article_id=pl.col("article_id").cast(pl.Utf8),
        title=pl.col("title"),
        abstract=pl.col("subtitle"),
        body=pl.col("body"),
        category=pl.col("category_str"),
        entities=pl.col("ner_clusters"),
        published_time=pl.col("published_time"),
        sentiment_score=pl.col("sentiment_score"),
        sentiment_label=pl.col("sentiment_label"),
        embedding=pl.lit(None).cast(pl.List(pl.Float32)),
        total_inviews=pl.col("total_inviews").cast(pl.Int32),
        total_pageviews=pl.col("total_pageviews").cast(pl.Int32),
        total_read_time=pl.col("total_read_time").cast(pl.Float32),
    ).select(ARTICLE_COLS)

    behaviors = pl.concat([
        pl.read_parquet(raw_dir / "train" / "behaviors.parquet"),
        pl.read_parquet(raw_dir / "validation" / "behaviors.parquet"),
    ])
    impressions = behaviors.select(
        dataset=pl.lit("ebnerd"),
        impression_id=pl.col("impression_id").cast(pl.Utf8),
        user_id=pl.col("user_id").cast(pl.Utf8),
        timestamp=pl.col("impression_time"),
        candidates=pl.col("article_ids_inview").cast(pl.List(pl.Utf8)),
        clicked=pl.col("article_ids_clicked").cast(pl.List(pl.Utf8)),
        device_type=pl.col("device_type").cast(pl.Utf8),
        # session_id is globally unique in EB-NeRD (verified: n_unique matches the
        # (user_id, session_id) group count), so it needs no per-user namespacing
        session_id=pl.col("session_id").cast(pl.Utf8),
        age=pl.col("age").cast(pl.Int8),
        gender=pl.col("gender").cast(pl.Int8),
        postcode=pl.col("postcode").cast(pl.Int8),
        is_subscriber=pl.col("is_subscriber").cast(pl.Boolean),
        read_time=pl.col("read_time").cast(pl.Float32),
        scroll_percentage=pl.col("scroll_percentage").cast(pl.Float32),
        next_read_time=pl.col("next_read_time").cast(pl.Float32),
        next_scroll_percentage=pl.col("next_scroll_percentage").cast(pl.Float32),
    ).select(IMPRESSION_COLS)

    history_raw = pl.concat([
        pl.read_parquet(raw_dir / "train" / "history.parquet"),
        pl.read_parquet(raw_dir / "validation" / "history.parquet"),
    ]).unique(subset=["user_id"], keep="last")  # validation's snapshot is the later/more complete one
    history = history_raw.select(
        dataset=pl.lit("ebnerd"),
        user_id=pl.col("user_id").cast(pl.Utf8),
        history_article_ids=pl.col("article_id_fixed").cast(pl.List(pl.Utf8)),
        history_timestamps=pl.col("impression_time_fixed"),
        history_length=pl.col("article_id_fixed").list.len(),
        last_history_time=pl.col("impression_time_fixed").list.max(),
        history_read_times=pl.col("read_time_fixed"),
        history_scroll_percentages=pl.col("scroll_percentage_fixed"),
    ).select(HISTORY_COLS)

    return {"articles": articles, "impressions": impressions, "history": history}


# ---------------------------------------------------------------------------
# MIND
# ---------------------------------------------------------------------------

def _mind_entity_labels(entities_json: str | None) -> list[str]:
    if not entities_json:
        return []
    try:
        parsed = json.loads(entities_json)
    except (json.JSONDecodeError, TypeError):
        return []
    return [e.get("Label", "") for e in parsed if e.get("Label")]


def _split_mind_impressions(s: str | None) -> tuple[list[str], list[str]]:
    """'-1'/'-0' labels are present in train/dev; absent (bare ids) in the unlabeled test tier."""
    if not s:
        return [], []
    candidates, clicked = [], []
    for tok in s.split():
        if "-" in tok:
            nid, label = tok.rsplit("-", 1)
            candidates.append(nid)
            if label == "1":
                clicked.append(nid)
        else:
            candidates.append(tok)
    return candidates, clicked


def _read_mind_news(news_dir: Path) -> pl.DataFrame:
    return pl.read_csv(
        news_dir / "news.tsv", separator="\t", quote_char=None, has_header=False,
        new_columns=MIND_NEWS_COLS,
        schema_overrides={"title_entities": pl.Utf8, "abstract_entities": pl.Utf8},
    )


def _read_mind_behaviors(behaviors_dir: Path) -> pl.DataFrame:
    return pl.read_csv(
        behaviors_dir / "behaviors.tsv", separator="\t", quote_char=None, has_header=False,
        new_columns=MIND_BEHAVIOR_COLS,
        schema_overrides={"impression_id": pl.Int64, "history": pl.Utf8, "impressions": pl.Utf8},
    )


def clean_mind(train_dir: Path, dev_dir: Path) -> dict[str, pl.DataFrame]:
    """train_dir / dev_dir e.g. data/MINDsmall_train/MINDsmall_train,
    data/MINDsmall_dev/MINDsmall_dev. Train and dev are concatenated for the same reason as
    EB-NeRD's train/validation -- Q1 does its own temporal split downstream."""
    train_dir, dev_dir = Path(train_dir), Path(dev_dir)

    news = pl.concat([_read_mind_news(train_dir), _read_mind_news(dev_dir)]).unique(
        subset=["news_id"], keep="first"
    )
    # entity extraction needs real JSON parsing, done per-column then zipped in Python
    title_entities = news["title_entities"].map_elements(_mind_entity_labels, return_dtype=pl.List(pl.Utf8))
    abstract_entities = news["abstract_entities"].map_elements(_mind_entity_labels, return_dtype=pl.List(pl.Utf8))
    entities_combined = [list(a) + list(b) for a, b in zip(title_entities.to_list(), abstract_entities.to_list())]

    articles = news.select(
        dataset=pl.lit("mind"),
        article_id=pl.col("news_id"),
        title=pl.col("title"),
        abstract=pl.col("abstract"),
        body=pl.lit(None).cast(pl.Utf8),
        category=pl.col("category"),
        entities=pl.Series("entities", entities_combined, dtype=pl.List(pl.Utf8)),
        published_time=pl.lit(None).cast(pl.Datetime),
        sentiment_score=pl.lit(None).cast(pl.Float32),
        sentiment_label=pl.lit(None).cast(pl.Utf8),
        embedding=pl.lit(None).cast(pl.List(pl.Float32)),
        # MIND ships no engagement counters at all -- null rather than 0, so "unknown" never
        # masquerades as "never viewed"
        total_inviews=pl.lit(None).cast(pl.Int32),
        total_pageviews=pl.lit(None).cast(pl.Int32),
        total_read_time=pl.lit(None).cast(pl.Float32),
    ).select(ARTICLE_COLS)

    behaviors = pl.concat([_read_mind_behaviors(train_dir), _read_mind_behaviors(dev_dir)])
    behaviors = behaviors.with_columns(
        timestamp=pl.col("time").str.strptime(pl.Datetime, "%m/%d/%Y %I:%M:%S %p"),
    )

    parsed = [_split_mind_impressions(s) for s in behaviors["impressions"].to_list()]
    candidates_col = pl.Series("candidates", [c for c, _ in parsed], dtype=pl.List(pl.Utf8))
    clicked_col = pl.Series("clicked", [k for _, k in parsed], dtype=pl.List(pl.Utf8))

    impressions = behaviors.select(
        dataset=pl.lit("mind"),
        impression_id=pl.col("impression_id").cast(pl.Utf8),
        user_id=pl.col("user_id"),
        timestamp=pl.col("timestamp"),
        candidates=candidates_col,
        clicked=clicked_col,
        device_type=pl.lit(None).cast(pl.Utf8),
        # MIND has no session id, no demographics and no dwell instrumentation. Nulls here are
        # load-bearing: behaviour.py branches on them to pick MIND's fallbacks (pseudo-sessions
        # from timestamp gaps, positional decay instead of time decay).
        session_id=pl.lit(None).cast(pl.Utf8),
        age=pl.lit(None).cast(pl.Int8),
        gender=pl.lit(None).cast(pl.Int8),
        postcode=pl.lit(None).cast(pl.Int8),
        is_subscriber=pl.lit(None).cast(pl.Boolean),
        read_time=pl.lit(None).cast(pl.Float32),
        scroll_percentage=pl.lit(None).cast(pl.Float32),
        next_read_time=pl.lit(None).cast(pl.Float32),
        next_scroll_percentage=pl.lit(None).cast(pl.Float32),
    ).select(IMPRESSION_COLS)

    # MIND's `history` is a fixed pre-collection-period snapshot, constant across all of a
    # user's impressions (verified empirically) -- unlike EB-NeRD it is NOT per-impression, so
    # one row per user is the right unit, and there's no per-item timestamp to store.
    history_raw = behaviors.select("user_id", "history").unique(subset=["user_id"], keep="first")
    history_lists = [
        (h.split() if h else []) for h in history_raw["history"].to_list()
    ]
    history = history_raw.select(
        dataset=pl.lit("mind"),
        user_id=pl.col("user_id"),
        history_article_ids=pl.Series("history_article_ids", history_lists, dtype=pl.List(pl.Utf8)),
        history_timestamps=pl.lit(None).cast(pl.List(pl.Datetime)),
        history_length=pl.Series("history_length", [len(h) for h in history_lists], dtype=pl.Int64),
        last_history_time=pl.lit(None).cast(pl.Datetime),
        history_read_times=pl.lit(None).cast(pl.List(pl.Float32)),
        history_scroll_percentages=pl.lit(None).cast(pl.List(pl.Float32)),
    ).select(HISTORY_COLS)

    return {"articles": articles, "impressions": impressions, "history": history}
