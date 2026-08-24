#!/usr/bin/env python3
"""Q3: semantic candidate generation. Loads precomputed article embeddings (from
scripts/compute_embeddings.py, run on Kaggle -- see that script's docstring), builds a FAISS
flat (exact) ANN index, and for every impression in the chosen split retrieves top-K candidates
using the mean-pooled embedding of the user's recent click history (same leak-safe
recent_history_asof cutoff as Q2's BM25 query), reporting recall@K for K in {50, 100, 200} --
computed identically to Q2 (candidate_eval.recall_at_k) so the two are directly comparable.

Usage:
    python scripts/run_embeddings.py --dataset ebnerd
    python scripts/run_embeddings.py --dataset mind --limit 2000   # smoke test
"""
import argparse
import json
import sys
import time
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ire_a1.ann import ANNIndex, load_embedding_lookup, user_embedding  # noqa: E402
from ire_a1.candidate_eval import recall_at_k  # noqa: E402
from ire_a1.feature_store import recent_history_asof  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["ebnerd", "mind"], required=True)
    parser.add_argument("--processed-dir", default="data/processed", type=Path)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--k-values", default="50,100,200")
    parser.add_argument("--n-recent", type=int, default=10,
                         help="how many of the user's most recent (as-of impression time) clicked articles go into the user embedding")
    parser.add_argument("--limit", type=int, default=None, help="only score the first N impressions (smoke test)")
    parser.add_argument("--out-dir", default=None, type=Path)
    parser.add_argument("--embeddings-file", default="embeddings.parquet",
                         help="filename under data/processed/<dataset>/ to load -- use this to "
                              "evaluate a second embedding variant (e.g. embeddings_mpnet.parquet) "
                              "without overwriting the first one's results")
    args = parser.parse_args()

    k_values = sorted(int(k) for k in args.k_values.split(","))
    max_k = max(k_values)
    ds_dir = args.processed_dir / args.dataset
    out_dir = args.out_dir or ds_dir / "embeddings_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    emb_path = ds_dir / args.embeddings_file
    if not emb_path.exists():
        print(
            f"ERROR: {emb_path} not found. Run scripts/compute_embeddings.py --dataset "
            f"{args.dataset} first (on Kaggle -- see that script's docstring).",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loading {args.dataset} embeddings/history/{args.split} impressions ...")
    history = pl.read_parquet(ds_dir / "user_history.parquet")
    impressions = pl.read_parquet(ds_dir / f"impressions_{args.split}.parquet")
    if args.limit:
        impressions = impressions.head(args.limit)

    embedding_lookup = load_embedding_lookup(str(emb_path))
    doc_ids = list(embedding_lookup.keys())
    dim = next(iter(embedding_lookup.values())).shape[0]
    print(f"Building ANN index over {len(doc_ids):,} article embeddings (dim={dim}) ...")
    t0 = time.time()
    index = ANNIndex(doc_ids, [embedding_lookup[d] for d in doc_ids])
    print(f"  index built in {time.time() - t0:.1f}s")

    history_by_user = {
        row["user_id"]: (row["history_article_ids"], row["history_timestamps"])
        for row in history.iter_rows(named=True)
    }

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

        u_vec = user_embedding(recent_ids, embedding_lookup, dim) if recent_ids else None
        if u_vec is None:
            n_cold_start += 1
            retrieved = []
        else:
            retrieved = index.query(u_vec, top_k=max_k)
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
    print(f"Cold-start (no embedded history as-of impression time): {n_cold_start:,} ({n_cold_start / n:.1%})")

    summary = {
        "dataset": args.dataset, "split": args.split, "n_impressions": n,
        "n_recent": args.n_recent, "n_cold_start": n_cold_start, "embedding_dim": dim,
        "embeddings_file": args.embeddings_file,
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
