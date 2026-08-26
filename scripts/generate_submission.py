#!/usr/bin/env python3
"""Q5: generate Codabench submission files for the large, unlabeled test sets, re-ranking each
impression's own shown candidates with either BM25 (Q2) or embedding (Q3) retrieval -- same
scoring as Q4's official metrics, applied here since the large test set has no labels to score
offline against.

Engineered for scale (13.5M EB-NeRD / 2.37M MIND test impressions on this CPU-only machine):
- BM25Index.score_candidates does targeted (candidate x query-term) scoring only -- see its
  docstring in src/ire_a1/bm25.py -- never scanning documents outside the given candidate set.
  Benchmarked at ~0.01-0.4ms/impression against the real large-tier indexes, i.e. the full
  scoring pass is ~15 min per dataset, not the ~47/50 hours a naive full-corpus-scan approach
  would take at this scale. The embeddings path (ann.score_candidates) is cheaper still --
  plain cosine similarity against a handful of candidates, no index search involved.
- Each user's query/embedding is computed once and cached: EB-NeRD's test history is verified
  entirely pre-test-period (max history timestamp 2023-06-01 06:59:59, min test impression
  timestamp 2023-06-01 07:00:00), so a user's as-of history -- and therefore their
  query/embedding -- is identical across all of their own test impressions (16.76
  impressions/user on average). MIND's history has no timestamps and is a similarly fixed
  pre-period snapshot (verified empirically for the small tier in clean.py; same official MIND
  dataset construction for the large tier).
- Behaviors are streamed in row-group batches (EB-NeRD; MIND's 2.37M rows fit comfortably in
  memory as a single frame) so peak memory stays bounded regardless of the 13.5M-row EB-NeRD
  behaviors file, per the assignment's explicit memory-efficiency guidance for the large sets.

The embeddings path needs large-tier article embeddings (data/processed_large/<dataset>/
embeddings.parquet), computed the same way as Q3's small-tier ones -- see
scripts/compute_embeddings.py, run on Kaggle against data/processed_large/<dataset>/
articles.parquet (125,541 EB-NeRD / 120,961 MIND articles, extracted from the large-tier raw
files -- NOT the same catalog as Q3's small-tier embeddings.parquet).

Output format (both datasets, from the official MIND evaluate.py convention the PDF references):
    impression_id [rank_order]
where rank_order is a comma-separated 1-indexed permutation of the impression's own candidates,
in their *original* order (rank 1 = most likely to be clicked). The required filename *inside*
the zip is fixed per competition regardless of method (confirmed against each competition's own
Codabench guidelines): "predictions.txt" for EB-NeRD, "prediction.txt" for MIND.

Usage:
    python scripts/generate_submission.py --dataset ebnerd --method bm25
    python scripts/generate_submission.py --dataset mind --method embeddings
    python scripts/generate_submission.py --dataset ebnerd --method embeddings --limit 10000   # smoke test
"""
import argparse
import sys
import time
import zipfile
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ire_a1.ann import load_embedding_lookup, score_candidates as ann_score_candidates  # noqa: E402
from ire_a1.ann import user_embedding  # noqa: E402
from ire_a1.bm25 import BM25Index, build_article_corpus  # noqa: E402
from ire_a1.submission import format_submission_line, ranks_from_scores  # noqa: E402

EBNERD_RAW_DIR = Path("data/ebnerd_testset/ebnerd_testset")
MIND_RAW_DIR = Path("data/MINDlarge_test/MINDlarge_test")
NEWS_COLS = ["news_id", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities"]
BEHAVIOR_COLS = ["impression_id", "user_id", "time", "history", "impressions"]

REQUIRED_ARCNAME = {"ebnerd": "predictions.txt", "mind": "prediction.txt"}


def _progress(n_done: int, total: int, t0: float) -> None:
    elapsed = time.time() - t0
    rate = n_done / elapsed if elapsed > 0 else 0
    eta_min = (total - n_done) / rate / 60 if rate > 0 else float("inf")
    print(f"  {n_done:,}/{total:,} ({n_done / total:.1%}) -- {rate:.0f} imp/s -- ETA {eta_min:.1f} min", flush=True)


def _bm25_scorer(articles: pl.DataFrame, dataset_name: str):
    """Returns (build_user_repr, score) for the BM25 method -- repr is a query string."""
    print(f"Building BM25 index over {articles.height:,} {dataset_name} large-tier articles ...")
    t0 = time.time()
    doc_ids, texts = build_article_corpus(articles)
    index = BM25Index(doc_ids, texts)
    print(f"  built in {time.time() - t0:.1f}s")
    titles_by_id = dict(zip(articles["article_id"].to_list(), articles["title"].to_list()))

    def build_user_repr(recent_ids: list[str]) -> str:
        return " ".join(titles_by_id.get(a, "") or "" for a in recent_ids)

    def score(repr_: str, candidates: list[str]) -> list[float]:
        return index.score_candidates(repr_, candidates) if repr_ else [0.0] * len(candidates)

    return build_user_repr, score


def _embeddings_scorer(dataset: str, dataset_name: str):
    """Returns (build_user_repr, score) for the embeddings method -- repr is a mean-pooled
    vector (or None for cold-start, matching Q3's ann.user_embedding contract)."""
    emb_path = Path("data/processed_large") / dataset / "embeddings.parquet"
    if not emb_path.exists():
        raise SystemExit(
            f"ERROR: {emb_path} not found. Run scripts/compute_embeddings.py on Kaggle against "
            f"data/processed_large/{dataset}/articles.parquet first -- see this script's docstring."
        )
    embedding_lookup = load_embedding_lookup(str(emb_path))
    dim = next(iter(embedding_lookup.values())).shape[0]
    print(f"Loaded {len(embedding_lookup):,} {dataset_name} large-tier article embeddings (dim={dim})")

    def build_user_repr(recent_ids: list[str]):
        return user_embedding(recent_ids, embedding_lookup, dim)

    def score(repr_, candidates: list[str]) -> list[float]:
        return ann_score_candidates(repr_, candidates, embedding_lookup) if repr_ is not None else [0.0] * len(candidates)

    return build_user_repr, score


def run_ebnerd(args: argparse.Namespace) -> Path:
    articles_raw = pl.read_parquet(EBNERD_RAW_DIR / "articles.parquet")
    articles = articles_raw.select(
        article_id=pl.col("article_id").cast(pl.Utf8), title=pl.col("title"), abstract=pl.col("subtitle"),
    )
    build_user_repr, score = (
        _bm25_scorer(articles, "EB-NeRD") if args.method == "bm25" else _embeddings_scorer("ebnerd", "EB-NeRD")
    )

    print("Building per-user representation cache from test history ...")
    t0 = time.time()
    history = pl.read_parquet(EBNERD_RAW_DIR / "test" / "history.parquet")
    repr_by_user = {}
    for row in history.iter_rows(named=True):
        recent_ids = [str(a) for a in row["article_id_fixed"][-args.n_recent:]]
        repr_by_user[row["user_id"]] = build_user_repr(recent_ids)
    print(f"  {len(repr_by_user):,} users cached in {time.time() - t0:.1f}s")
    del history

    out_dir = args.out_dir or Path("data/submissions")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"predictions_{args.method}.txt"

    behaviors_path = EBNERD_RAW_DIR / "test" / "behaviors.parquet"
    pf = pq.ParquetFile(behaviors_path)
    total = pf.metadata.num_rows if args.limit is None else min(args.limit, pf.metadata.num_rows)
    print(f"Scoring {total:,} EB-NeRD test impressions (batched, batch_size={args.batch_size:,}) ...")

    n_done = 0
    t0 = time.time()
    with open(out_path, "w") as f:
        for batch in pf.iter_batches(batch_size=args.batch_size, columns=["impression_id", "article_ids_inview", "user_id"]):
            df = pl.from_arrow(batch)
            if args.limit and n_done + df.height > args.limit:
                df = df.head(args.limit - n_done)
            lines = []
            for row in df.iter_rows(named=True):
                repr_ = repr_by_user.get(row["user_id"])
                candidates = [str(a) for a in row["article_ids_inview"]]
                lines.append(format_submission_line(row["impression_id"], ranks_from_scores(score(repr_, candidates))))
            f.writelines(lines)
            n_done += df.height
            _progress(n_done, total, t0)
            if args.limit and n_done >= args.limit:
                break

    print(f"Done in {(time.time() - t0) / 60:.1f} min. Wrote {n_done:,} predictions to {out_path}")
    return out_path


def run_mind(args: argparse.Namespace) -> Path:
    news = pl.read_csv(
        MIND_RAW_DIR / "news.tsv", separator="\t", quote_char=None, has_header=False, new_columns=NEWS_COLS,
        schema_overrides={"title_entities": pl.Utf8, "abstract_entities": pl.Utf8},
    )
    articles = news.select(article_id=pl.col("news_id"), title=pl.col("title"), abstract=pl.col("abstract"))
    build_user_repr, score = (
        _bm25_scorer(articles, "MIND") if args.method == "bm25" else _embeddings_scorer("mind", "MIND")
    )

    print("Loading behaviors + building per-user representation cache ...")
    t0 = time.time()
    behaviors = pl.read_csv(
        MIND_RAW_DIR / "behaviors.tsv", separator="\t", quote_char=None, has_header=False, new_columns=BEHAVIOR_COLS,
        schema_overrides={"impression_id": pl.Int64, "history": pl.Utf8, "impressions": pl.Utf8},
    )
    if args.limit:
        behaviors = behaviors.head(args.limit)
    unique_users = behaviors.select("user_id", "history").unique(subset=["user_id"], keep="first")
    repr_by_user = {}
    for row in unique_users.iter_rows(named=True):
        hist_ids = (row["history"].split() if row["history"] else [])[-args.n_recent:]
        repr_by_user[row["user_id"]] = build_user_repr(hist_ids)
    print(f"  {len(repr_by_user):,} users cached, {behaviors.height:,} impressions loaded in {time.time() - t0:.1f}s")

    out_dir = args.out_dir or Path("data/submissions")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"mind_prediction_{args.method}.txt"

    total = behaviors.height
    print(f"Scoring {total:,} MIND test impressions ...")
    t0 = time.time()
    n_done = 0
    with open(out_path, "w") as f:
        lines = []
        for row in behaviors.iter_rows(named=True):
            repr_ = repr_by_user.get(row["user_id"])
            candidates = row["impressions"].split()
            lines.append(format_submission_line(row["impression_id"], ranks_from_scores(score(repr_, candidates))))
            n_done += 1
            if n_done % args.batch_size == 0:
                f.writelines(lines)
                lines = []
                _progress(n_done, total, t0)
        f.writelines(lines)

    print(f"Done in {(time.time() - t0) / 60:.1f} min. Wrote {n_done:,} predictions to {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["ebnerd", "mind"], required=True)
    parser.add_argument("--method", choices=["bm25", "embeddings"], default="bm25")
    parser.add_argument("--n-recent", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=200_000)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None, help="only score the first N impressions (smoke test)")
    args = parser.parse_args()

    out_path = run_ebnerd(args) if args.dataset == "ebnerd" else run_mind(args)

    # The name *inside* the zip is dictated by each competition's own submission guidelines, not
    # by our local working filename or method -- both confirmed directly against the two
    # competitions' actual Codabench "Submission Guidelines" pages (not assumed from the
    # reference notebooks, which got MIND's wrong -- see ai_usage_log.md): EB-NeRD requires
    # exactly "predictions.txt", MIND requires exactly "prediction.txt". The *outer* zip
    # filename is unconstrained by either competition, so it's free to encode the method.
    required_arcname = REQUIRED_ARCNAME[args.dataset]
    zip_path = out_path.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_path, arcname=required_arcname)
    print(f"Zipped to {zip_path} (containing {required_arcname}) -- upload this file to Codabench.")


if __name__ == "__main__":
    main()
