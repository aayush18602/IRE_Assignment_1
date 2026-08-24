#!/usr/bin/env python3
"""Q4: offline evaluation harness. Runs AUC/MRR/nDCG@5/nDCG@10 (official-style: re-ranking
each impression's own shown candidates, not a full-catalog list) + diversity/novelty/coverage
(over the Q2/Q3 top-K recommended lists) + a cold-start-vs-warm slice, each with a bootstrap
95% CI, against either BM25 (Q2) or embedding (Q3) retrieval.

Usage:
    python scripts/run_eval.py --dataset ebnerd --method bm25
    python scripts/run_eval.py --dataset ebnerd --method embeddings
    python scripts/run_eval.py --dataset ebnerd --method embeddings \
        --embeddings-file embeddings_mpnet.parquet --candidates-dir data/processed/ebnerd/embeddings_eval_mpnet
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ire_a1.ann import load_embedding_lookup, score_candidates as ann_score_candidates  # noqa: E402
from ire_a1.ann import user_embedding  # noqa: E402
from ire_a1.bm25 import BM25Index, build_article_corpus  # noqa: E402
from ire_a1.eval import (  # noqa: E402
    bootstrap_ci,
    bootstrap_coverage_ci,
    intra_list_category_diversity,
    novelty,
    percentile_threshold,
    ranking_metrics,
)
from ire_a1.feature_store import recent_history_asof  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["ebnerd", "mind"], required=True)
    parser.add_argument("--method", choices=["bm25", "embeddings"], required=True)
    parser.add_argument("--processed-dir", default="data/processed", type=Path)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--embeddings-file", default="embeddings.parquet",
                         help="only used for --method embeddings")
    parser.add_argument("--candidates-dir", default=None, type=Path,
                         help="dir with candidates_<split>.parquet from run_bm25.py/run_embeddings.py "
                              "(default: <ds>/bm25 or <ds>/embeddings_eval); eval_<split>.json is written here too")
    parser.add_argument("--n-recent", type=int, default=10)
    parser.add_argument("--cold-percentile", type=float, default=25.0,
                         help="users at/below this percentile of the dataset's own history_length "
                              "distribution are 'cold', the rest 'warm' -- relative to each dataset's "
                              "own distribution rather than a fixed count, since some datasets (e.g. "
                              "EB-NeRD) are pre-filtered to only include users above a minimum history")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--limit", type=int, default=None, help="only score the first N impressions (smoke test)")
    args = parser.parse_args()

    ds_dir = args.processed_dir / args.dataset
    articles = pl.read_parquet(ds_dir / "articles.parquet")
    category_lookup = dict(zip(articles["article_id"].to_list(), articles["category"].to_list()))
    catalog_size = articles.height

    history = pl.read_parquet(ds_dir / "user_history.parquet")
    history_by_user = {
        row["user_id"]: (row["history_article_ids"], row["history_timestamps"])
        for row in history.iter_rows(named=True)
    }
    history_length_by_user = dict(zip(history["user_id"].to_list(), history["history_length"].to_list()))
    cold_threshold = percentile_threshold(history["history_length"].to_list(), args.cold_percentile)
    print(f"Cold/warm split: history_length < {cold_threshold:.1f} = cold "
          f"({args.cold_percentile:.0f}th percentile of this dataset's own distribution)")

    train_impressions = pl.read_parquet(ds_dir / "impressions_train.parquet")
    popularity: dict[str, int] = defaultdict(int)
    for clicked_list in train_impressions["clicked"].to_list():
        for a in clicked_list:
            popularity[a] += 1
    n_train_clicks = sum(popularity.values())
    print(f"Train popularity reference: {len(popularity):,} distinct clicked articles, {n_train_clicks:,} total clicks")

    print(f"Building {args.method} scorer ...")
    if args.method == "bm25":
        doc_ids, texts = build_article_corpus(articles)
        index = BM25Index(doc_ids, texts)
        articles_title = dict(zip(articles["article_id"].to_list(), articles["title"].to_list()))

        def score_fn(recent_ids: list[str], candidate_ids: list[str]) -> list[float]:
            query_text = " ".join(articles_title.get(a, "") or "" for a in recent_ids)
            return index.score_candidates(query_text, candidate_ids)

        default_candidates_dir = ds_dir / "bm25"
    else:
        embedding_lookup = load_embedding_lookup(str(ds_dir / args.embeddings_file))
        dim = next(iter(embedding_lookup.values())).shape[0]

        def score_fn(recent_ids: list[str], candidate_ids: list[str]) -> list[float]:
            u = user_embedding(recent_ids, embedding_lookup, dim)
            if u is None:
                return [0.0] * len(candidate_ids)
            return ann_score_candidates(u, candidate_ids, embedding_lookup)

        default_candidates_dir = ds_dir / "embeddings_eval"

    candidates_dir = args.candidates_dir or default_candidates_dir
    retrieved_df = pl.read_parquet(candidates_dir / f"candidates_{args.split}.parquet")
    retrieved_by_imp = dict(zip(retrieved_df["impression_id"].to_list(), retrieved_df["retrieved_article_ids"].to_list()))

    impressions = pl.read_parquet(ds_dir / f"impressions_{args.split}.parquet")
    if args.limit:
        impressions = impressions.head(args.limit)

    per_impression = []
    recommended_by_slice: dict[str, list[list[str]]] = defaultdict(list)

    print(f"Scoring {impressions.height:,} impressions ...")
    t0 = time.time()
    for row in impressions.iter_rows(named=True):
        uid, cutoff = row["user_id"], row["timestamp"]
        clicked_set = set(row["clicked"])
        candidate_ids = row["candidates"]
        hist = history_by_user.get(uid)
        recent_ids = recent_history_asof(hist[0], hist[1], cutoff, n_recent=args.n_recent) if hist else []

        scores = score_fn(recent_ids, candidate_ids)
        labels = [1 if c in clicked_set else 0 for c in candidate_ids]
        rm = ranking_metrics(scores, labels)

        hlen = history_length_by_user.get(uid, 0)
        slice_label = "cold" if hlen < cold_threshold else "warm"

        recommended = retrieved_by_imp.get(row["impression_id"], [])
        rm["diversity"] = intra_list_category_diversity(recommended, category_lookup)
        rm["novelty"] = novelty(recommended, popularity, n_train_clicks)
        rm["slice"] = slice_label

        per_impression.append(rm)
        recommended_by_slice[slice_label].append(recommended)
        recommended_by_slice["all"].append(recommended)

    print(f"Done in {time.time() - t0:.1f}s")

    def aggregate(rows: list[dict], key: str) -> dict:
        vals = [r[key] for r in rows if r[key] is not None]
        return bootstrap_ci(vals, n_boot=args.n_boot)

    result = {
        "dataset": args.dataset, "method": args.method, "split": args.split,
        "n_impressions": len(per_impression), "n_recent": args.n_recent,
        "cold_percentile": args.cold_percentile, "cold_threshold": cold_threshold,
    }
    metric_keys = ["auc", "mrr", "ndcg@5", "ndcg@10", "diversity", "novelty"]
    for slice_name in ["all", "cold", "warm"]:
        rows = per_impression if slice_name == "all" else [r for r in per_impression if r["slice"] == slice_name]
        metrics = {key: aggregate(rows, key) for key in metric_keys}
        metrics["coverage"] = bootstrap_coverage_ci(recommended_by_slice[slice_name], catalog_size, n_boot=args.n_boot)
        metrics["n_impressions"] = len(rows)
        result[slice_name] = metrics

        print(f"\n=== slice: {slice_name} (n={len(rows):,}) ===")
        for key in metric_keys + ["coverage"]:
            m = metrics[key]
            print(f"  {key:<10} {m['mean']:.4f}  95% CI [{m['ci_low']:.4f}, {m['ci_high']:.4f}]  (n={m['n']:,})")

    candidates_dir.mkdir(parents=True, exist_ok=True)
    out_path = candidates_dir / f"eval_{args.split}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
