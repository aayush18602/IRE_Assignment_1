#!/usr/bin/env python3
"""Q9 (Anti-Gaming): report metrics with and without a feature unavailable at serving time.

Our actual retrieval methods (Q2/Q3) never use leaky features -- the query is built only from
a user's *past* clicks (recent_history_asof, cutoff at the impression's own timestamp) and
candidates are scored on static title/abstract text. So there's no naturally-occurring "leaky
version" of our pipeline to compare against. This script builds one deliberately, to measure
and report the risk Q9 asks about: blend BM25's score with an article-popularity feature,
computed two ways --

- "safe": popularity counted from the TRAIN split only. Train entirely precedes val/test in our
  temporal_split (Q1), so this is available at serving time for any val/test impression.
- "leaky": popularity counted from train+val+test combined -- includes clicks from the very
  period being evaluated (test), which would NOT be knowable at real serving time. This is the
  violation Q9 is about: using future/contemporaneous information as if it were a static,
  known-in-advance feature.

Reports AUC/MRR/nDCG@5/nDCG@10 (Q4's official-style ranking metrics) for three variants: BM25
alone, BM25 + safe popularity, BM25 + leaky popularity. If the leaky variant's metrics come out
higher, that's the metric inflation Q9 warns about, demonstrated with real numbers rather than
just asserted.

Usage:
    python scripts/run_anti_gaming.py --dataset ebnerd
    python scripts/run_anti_gaming.py --dataset mind
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ire_a1.bm25 import BM25Index, build_article_corpus  # noqa: E402
from ire_a1.eval import article_popularity, bootstrap_ci, ranking_metrics  # noqa: E402
from ire_a1.feature_store import recent_history_asof  # noqa: E402

METRIC_KEYS = ["auc", "mrr", "ndcg@5", "ndcg@10"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["ebnerd", "mind"], required=True)
    parser.add_argument("--processed-dir", default="data/processed", type=Path)
    parser.add_argument("--n-recent", type=int, default=10)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--limit", type=int, default=None, help="only score the first N test impressions (smoke test)")
    args = parser.parse_args()

    ds_dir = args.processed_dir / args.dataset
    articles = pl.read_parquet(ds_dir / "articles.parquet")
    doc_ids, texts = build_article_corpus(articles)
    index = BM25Index(doc_ids, texts)
    titles_by_id = dict(zip(articles["article_id"].to_list(), articles["title"].to_list()))

    history = pl.read_parquet(ds_dir / "user_history.parquet")
    history_by_user = {
        row["user_id"]: (row["history_article_ids"], row["history_timestamps"])
        for row in history.iter_rows(named=True)
    }

    train = pl.read_parquet(ds_dir / "impressions_train.parquet")
    val = pl.read_parquet(ds_dir / "impressions_val.parquet")
    test = pl.read_parquet(ds_dir / "impressions_test.parquet")
    if args.limit:
        test = test.head(args.limit)

    safe_popularity = article_popularity(train["clicked"].to_list())
    leaky_popularity = article_popularity(
        train["clicked"].to_list() + val["clicked"].to_list() + test["clicked"].to_list()
    )
    print(f"Safe (train-only) popularity: {len(safe_popularity):,} distinct clicked articles")
    print(f"Leaky (train+val+test) popularity: {len(leaky_popularity):,} distinct clicked articles")

    variants: dict[str, dict[str, int] | None] = {
        "bm25_only": None,
        "bm25_plus_safe_popularity": safe_popularity,
        "bm25_plus_LEAKY_popularity": leaky_popularity,
    }
    per_variant: dict[str, dict[str, list[float]]] = {name: {k: [] for k in METRIC_KEYS} for name in variants}

    print(f"Scoring {test.height:,} {args.dataset} test impressions ...")
    for row in test.iter_rows(named=True):
        cutoff = row["timestamp"]
        clicked_set = set(row["clicked"])
        candidate_ids = row["candidates"]
        hist = history_by_user.get(row["user_id"])
        recent_ids = recent_history_asof(hist[0], hist[1], cutoff, n_recent=args.n_recent) if hist else []
        query_text = " ".join(titles_by_id.get(a, "") or "" for a in recent_ids)
        bm25_scores = np.array(
            index.score_candidates(query_text, candidate_ids) if query_text.strip() else [0.0] * len(candidate_ids)
        )
        labels = [1 if c in clicked_set else 0 for c in candidate_ids]

        for name, popularity in variants.items():
            if popularity is None:
                scores = bm25_scores
            else:
                pop_scores = np.array([np.log1p(popularity.get(c, 0)) for c in candidate_ids])
                scores = bm25_scores + pop_scores
            m = ranking_metrics(scores.tolist(), labels)
            for key in METRIC_KEYS:
                if m[key] is not None:
                    per_variant[name][key].append(m[key])

    result: dict = {"dataset": args.dataset, "n_impressions": test.height}
    print(f"\n=== {args.dataset}: BM25 alone vs. +safe popularity vs. +LEAKY popularity ===")
    for name in variants:
        result[name] = {}
        print(f"\n{name}:")
        for key in METRIC_KEYS:
            ci = bootstrap_ci(per_variant[name][key], n_boot=args.n_boot)
            result[name][key] = ci
            print(f"  {key:<8} {ci['mean']:.4f}  95% CI [{ci['ci_low']:.4f}, {ci['ci_high']:.4f}]  (n={ci['n']:,})")

    out_dir = Path("results") / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "anti_gaming.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
