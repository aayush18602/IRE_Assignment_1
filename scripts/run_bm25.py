#!/usr/bin/env python3
"""Q2: lexical candidate generation. Builds a BM25 inverted index over article title+abstract,
retrieves top-K candidates per impression using a query built from the user's recent click
history (as-of the impression's timestamp, never later -- no future-click leakage), and reports
recall@K for K in {50, 100, 200}.

Usage:
    python scripts/run_bm25.py --dataset ebnerd
    python scripts/run_bm25.py --dataset mind --k-values 50,100,200 --n-recent 10
    python scripts/run_bm25.py --dataset ebnerd --limit 2000   # quick smoke test
"""
import argparse
import json
import sys
import time
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ire_a1.bm25 import BM25Index, build_article_corpus  # noqa: E402
from ire_a1.candidate_eval import recall_at_k  # noqa: E402
from ire_a1.feature_store import recent_history_asof  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["ebnerd", "mind"], required=True)
    parser.add_argument("--processed-dir", default="data/processed", type=Path)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--k-values", default="50,100,200")
    parser.add_argument("--n-recent", type=int, default=10,
                         help="how many of the user's most recent (as-of impression time) clicked titles go into the query")
    parser.add_argument("--limit", type=int, default=None, help="only score the first N impressions (smoke test)")
    parser.add_argument("--out-dir", default=None, type=Path)
    args = parser.parse_args()

    k_values = sorted(int(k) for k in args.k_values.split(","))
    max_k = max(k_values)
    ds_dir = args.processed_dir / args.dataset
    out_dir = args.out_dir or ds_dir / "bm25"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.dataset} articles/history/{args.split} impressions ...")
    articles = pl.read_parquet(ds_dir / "articles.parquet")
    history = pl.read_parquet(ds_dir / "user_history.parquet")
    impressions = pl.read_parquet(ds_dir / f"impressions_{args.split}.parquet")
    if args.limit:
        impressions = impressions.head(args.limit)

    articles_title = dict(zip(articles["article_id"].to_list(), articles["title"].to_list()))
    history_by_user = {
        row["user_id"]: (row["history_article_ids"], row["history_timestamps"])
        for row in history.iter_rows(named=True)
    }

    print(f"Building BM25 index over {articles.height:,} articles ...")
    t0 = time.time()
    doc_ids, texts = build_article_corpus(articles)
    index = BM25Index(doc_ids, texts)
    print(f"  index built in {time.time() - t0:.1f}s, {len(index.postings):,} terms")

    recalls: dict[int, list[float]] = {k: [] for k in k_values}
    imp_ids, retrieved_col, scores_col = [], [], []
    n_cold_start = 0

    print(f"Scoring {impressions.height:,} impressions ...")
    t0 = time.time()
    for row in impressions.iter_rows(named=True):
        cutoff = row["timestamp"]
        clicked = row["clicked"]
        hist = history_by_user.get(row["user_id"])
        recent_ids = (
            recent_history_asof(hist[0], hist[1], cutoff, n_recent=args.n_recent) if hist else []
        )
        if not recent_ids:
            n_cold_start += 1

        query_text = " ".join(articles_title.get(aid, "") or "" for aid in recent_ids)
        retrieved = index.query(query_text, top_k=max_k) if query_text.strip() else []
        retrieved_ids = [aid for aid, _ in retrieved]

        for k in k_values:
            r = recall_at_k(retrieved_ids, clicked, k)
            if r is not None:
                recalls[k].append(r)

        imp_ids.append(row["impression_id"])
        retrieved_col.append(retrieved_ids)
        scores_col.append([s for _, s in retrieved])

    elapsed = time.time() - t0
    n = impressions.height
    print(f"Done in {elapsed:.1f}s ({n / max(elapsed, 1e-9):.0f} impressions/s)")
    print(f"Cold-start (no usable history as-of impression time): {n_cold_start:,} ({n_cold_start / n:.1%})")

    summary = {
        "dataset": args.dataset, "split": args.split, "n_impressions": n,
        "n_recent": args.n_recent, "n_cold_start": n_cold_start,
    }
    for k in k_values:
        vals = recalls[k]
        mean_r = sum(vals) / len(vals) if vals else 0.0
        summary[f"recall@{k}"] = mean_r
        print(f"  recall@{k}: {mean_r:.4f}  (n={len(vals):,})")

    (out_dir / f"recall_{args.split}.json").write_text(json.dumps(summary, indent=2))

    candidates_df = pl.DataFrame({
        "impression_id": imp_ids,
        "retrieved_article_ids": retrieved_col,
        "retrieved_scores": scores_col,
    })
    candidates_df.write_parquet(out_dir / f"candidates_{args.split}.parquet")
    print(f"Saved metrics + top-{max_k} candidates to {out_dir}/")


if __name__ == "__main__":
    main()
