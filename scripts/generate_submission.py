#!/usr/bin/env python3
"""Q5: generate Codabench submission files for the large, unlabeled test sets, re-ranking each
impression's own shown candidates with our BM25 retrieval (same scoring as Q4's official
metrics, applied here since the large test set has no labels to score offline against).

Engineered for scale (13.5M EB-NeRD / 2.37M MIND test impressions on this CPU-only machine):
- BM25Index.score_candidates does targeted (candidate x query-term) scoring only -- see its
  docstring in src/ire_a1/bm25.py -- never scanning documents outside the given candidate set.
  Benchmarked at ~0.01-0.4ms/impression against the real large-tier indexes, i.e. the full
  scoring pass is ~15 min per dataset, not the ~47/50 hours a naive full-corpus-scan approach
  would take at this scale.
- Each user's query is computed once and cached: EB-NeRD's test history is verified entirely
  pre-test-period (max history timestamp 2023-06-01 06:59:59, min test impression timestamp
  2023-06-01 07:00:00), so a user's as-of history -- and therefore their query -- is identical
  across all of their own test impressions (16.76 impressions/user on average). MIND's history
  has no timestamps and is a similarly fixed pre-period snapshot (verified empirically for the
  small tier in clean.py; same official MIND dataset construction for the large tier).
- Behaviors are streamed in row-group batches (EB-NeRD; MIND's 2.37M rows fit comfortably in
  memory as a single frame) so peak memory stays bounded regardless of the 13.5M-row EB-NeRD
  behaviors file, per the assignment's explicit memory-efficiency guidance for the large sets.

Output format (both datasets, from the official MIND evaluate.py convention the PDF references):
    impression_id [rank_order]
where rank_order is a comma-separated 1-indexed permutation of the impression's own candidates,
in their *original* order (rank 1 = most likely to be clicked).

Usage:
    python scripts/generate_submission.py --dataset ebnerd
    python scripts/generate_submission.py --dataset mind
    python scripts/generate_submission.py --dataset ebnerd --limit 10000   # smoke test
"""
import argparse
import sys
import time
import zipfile
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ire_a1.bm25 import BM25Index, build_article_corpus  # noqa: E402
from ire_a1.submission import format_submission_line, ranks_from_scores  # noqa: E402

EBNERD_RAW_DIR = Path("data/ebnerd_testset/ebnerd_testset")
MIND_RAW_DIR = Path("data/MINDlarge_test/MINDlarge_test")
NEWS_COLS = ["news_id", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities"]
BEHAVIOR_COLS = ["impression_id", "user_id", "time", "history", "impressions"]


def _progress(n_done: int, total: int, t0: float) -> None:
    elapsed = time.time() - t0
    rate = n_done / elapsed if elapsed > 0 else 0
    eta_min = (total - n_done) / rate / 60 if rate > 0 else float("inf")
    print(f"  {n_done:,}/{total:,} ({n_done / total:.1%}) -- {rate:.0f} imp/s -- ETA {eta_min:.1f} min", flush=True)


def run_ebnerd(args: argparse.Namespace) -> Path:
    articles_raw = pl.read_parquet(EBNERD_RAW_DIR / "articles.parquet")
    articles = articles_raw.select(
        article_id=pl.col("article_id").cast(pl.Utf8), title=pl.col("title"), abstract=pl.col("subtitle"),
    )
    print(f"Building BM25 index over {articles.height:,} EB-NeRD large-tier articles ...")
    t0 = time.time()
    doc_ids, texts = build_article_corpus(articles)
    index = BM25Index(doc_ids, texts)
    print(f"  built in {time.time() - t0:.1f}s")
    titles_by_id = dict(zip(articles["article_id"].to_list(), articles["title"].to_list()))

    print("Building per-user query cache from test history ...")
    t0 = time.time()
    history = pl.read_parquet(EBNERD_RAW_DIR / "test" / "history.parquet")
    query_by_user: dict[int, str] = {}
    for row in history.iter_rows(named=True):
        recent_ids = [str(a) for a in row["article_id_fixed"][-args.n_recent:]]
        query_by_user[row["user_id"]] = " ".join(titles_by_id.get(a, "") or "" for a in recent_ids)
    print(f"  {len(query_by_user):,} users cached in {time.time() - t0:.1f}s")
    del history

    out_dir = args.out_dir or Path("data/submissions")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "predictions.txt"

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
                q = query_by_user.get(row["user_id"], "")
                candidates = [str(a) for a in row["article_ids_inview"]]
                scores = index.score_candidates(q, candidates) if q else [0.0] * len(candidates)
                lines.append(format_submission_line(row["impression_id"], ranks_from_scores(scores)))
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
    print(f"Building BM25 index over {articles.height:,} MIND large-tier articles ...")
    t0 = time.time()
    doc_ids, texts = build_article_corpus(articles)
    index = BM25Index(doc_ids, texts)
    print(f"  built in {time.time() - t0:.1f}s")
    titles_by_id = dict(zip(articles["article_id"].to_list(), articles["title"].to_list()))

    print("Loading behaviors + building per-user query cache ...")
    t0 = time.time()
    behaviors = pl.read_csv(
        MIND_RAW_DIR / "behaviors.tsv", separator="\t", quote_char=None, has_header=False, new_columns=BEHAVIOR_COLS,
        schema_overrides={"impression_id": pl.Int64, "history": pl.Utf8, "impressions": pl.Utf8},
    )
    if args.limit:
        behaviors = behaviors.head(args.limit)
    unique_users = behaviors.select("user_id", "history").unique(subset=["user_id"], keep="first")
    query_by_user: dict[str, str] = {}
    for row in unique_users.iter_rows(named=True):
        hist_ids = (row["history"].split() if row["history"] else [])[-args.n_recent:]
        query_by_user[row["user_id"]] = " ".join(titles_by_id.get(a, "") or "" for a in hist_ids)
    print(f"  {len(query_by_user):,} users cached, {behaviors.height:,} impressions loaded in {time.time() - t0:.1f}s")

    out_dir = args.out_dir or Path("data/submissions")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "mind_prediction.txt"

    total = behaviors.height
    print(f"Scoring {total:,} MIND test impressions ...")
    t0 = time.time()
    n_done = 0
    with open(out_path, "w") as f:
        lines = []
        for row in behaviors.iter_rows(named=True):
            q = query_by_user.get(row["user_id"], "")
            candidates = row["impressions"].split()
            scores = index.score_candidates(q, candidates) if q else [0.0] * len(candidates)
            lines.append(format_submission_line(row["impression_id"], ranks_from_scores(scores)))
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
    parser.add_argument("--n-recent", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=200_000)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None, help="only score the first N impressions (smoke test)")
    args = parser.parse_args()

    out_path = run_ebnerd(args) if args.dataset == "ebnerd" else run_mind(args)

    # The name *inside* the zip is dictated by each competition's own submission guidelines, not
    # by our local working filename -- both confirmed directly against the two competitions'
    # actual Codabench "Submission Guidelines" pages (not assumed from the reference notebooks,
    # which got MIND's wrong -- see ai_usage_log.md): EB-NeRD requires exactly "predictions.txt",
    # MIND requires exactly "prediction.txt" (no shared convention between them).
    required_arcname = {"ebnerd": "predictions.txt", "mind": "prediction.txt"}[args.dataset]
    zip_path = out_path.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_path, arcname=required_arcname)
    print(f"Zipped to {zip_path} (containing {required_arcname}) -- upload this file to Codabench.")


if __name__ == "__main__":
    main()
