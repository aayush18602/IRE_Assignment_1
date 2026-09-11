#!/usr/bin/env python3
"""A2 Q2.1 taken literally: retrieve top-K from the full catalogue, then re-rank those K.

This is the pipeline the assignment describes word for word, and running it is how we justify
*not* shipping it. Recall@K is a hard ceiling on any two-stage system -- a re-ranker can reorder
the retrieved pool but cannot add to it, so an article the retriever missed is unrecoverable no
matter how good stage 2 is. A1 measured recall@200 at 2.01% (EB-NeRD) and 2.27% (MIND), which
caps the whole system at roughly 2%.

Reported two ways, because the difference is the point:

- **end-to-end** counts every impression, scoring 0 when the clicked article was never retrieved.
  This is the system's actual performance and it is dominated by the ceiling.
- **conditional** counts only impressions where stage 1 did retrieve the click. This isolates how
  well the re-ranker orders a pool that contains the answer, and it is *not* a system metric --
  quoting it alone would be the kind of favourable slicing Q9 exists to discourage.

Reuses the top-K lists A1 already wrote to data/processed/<ds>/bm25/candidates_<split>.parquet,
so no retrieval is recomputed here.

Usage:
    python scripts/run_two_stage.py --dataset ebnerd
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ire_a1.candidate_eval import recall_at_k  # noqa: E402
from ire_a1.eval import bootstrap_ci, ranking_metrics  # noqa: E402
from ire_a1.behaviour import session_features  # noqa: E402
from ire_a1.reranker import LambdaRanker, MatrixBuilder  # noqa: E402

METRICS = ["auc", "mrr", "ndcg@5", "ndcg@10"]
# A retrieval miss is a genuine zero for MRR and nDCG -- the system never surfaced the article, so
# the user's reciprocal rank really is 0. AUC is different: with no positive in the list it is
# *undefined*, not 0, and averaging a 0 in for it would manufacture a number rather than report
# one. So the end-to-end column covers only the metrics where a miss has a defined value.
END_TO_END_METRICS = ["mrr", "ndcg@5", "ndcg@10"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["ebnerd", "mind"], required=True)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--candidates-dir", default="bm25",
                        help="subdirectory holding A1's candidates_<split>.parquet")
    parser.add_argument("--embeddings-file", default="embeddings_mpnet.parquet")
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--k-values", default="50,100,200")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4000,
                        help="impressions per batch. 200 candidates each makes the row count 200x "
                             "the impression count, and materialising all 14.3M EB-NeRD rows at "
                             "once was an 8.3GB OOM kill that took the machine down")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    ds_dir = args.processed_dir / args.dataset
    k_values = [int(k) for k in args.k_values.split(",")]
    top_k = max(k_values)

    impressions = pl.read_parquet(ds_dir / f"impressions_{args.split}.parquet")
    retrieved = pl.read_parquet(ds_dir / args.candidates_dir / f"candidates_{args.split}.parquet")
    joined = impressions.join(retrieved, on="impression_id", how="inner")
    if joined.height == 0:
        raise SystemExit(
            f"no impression_id in {args.split} matches "
            f"{ds_dir / args.candidates_dir / f'candidates_{args.split}.parquet'}.\n"
            "That cache is keyed on impression ids, so it goes stale whenever cleaning changes "
            "them -- MIND's were namespaced by source file to stop train and dev colliding.\n"
            f"Regenerate it with: python scripts/run_bm25.py --dataset {args.dataset}"
        )
    if joined.height < impressions.height:
        print(f"  note: {impressions.height - joined.height:,} impressions have no cached "
              f"candidates and are excluded")
    if args.limit:
        joined = joined.head(args.limit)
    print(f"{args.dataset}: {joined.height:,} impressions with retrieved candidates")

    # --- the ceiling, before any re-ranking ------------------------------------------------
    clicked = joined["clicked"].to_list()
    retrieved_lists = joined["retrieved_article_ids"].to_list()
    recalls = {}
    for k in k_values:
        values = [r for r in (recall_at_k(ret, clk, k) for ret, clk in zip(retrieved_lists, clicked))
                  if r is not None]
        recalls[k] = float(np.mean(values)) if values else float("nan")
        print(f"  recall@{k:<4} {recalls[k]:.4f}")

    hit = np.array([bool(set(clk) & set(ret[:top_k])) for ret, clk in zip(retrieved_lists, clicked)])
    print(f"  the clicked article is in the retrieved top-{top_k} for "
          f"{hit.mean():.2%} of impressions -- stage 2 cannot recover the rest")

    # --- re-rank the retrieved pool ---------------------------------------------------------
    # A pseudo-impressions frame whose candidate column is the retrieved list rather than the
    # shown slate. Everything downstream is unchanged, which is the point: the same re-ranker,
    # pointed at a different stage-1.
    pseudo = joined.with_columns(
        candidates=pl.col("retrieved_article_ids").list.head(top_k)
    ).drop("retrieved_article_ids")          # retrieved_scores stays: it is the "before" row

    articles = pl.read_parquet(ds_dir / "articles.parquet")
    history = pl.read_parquet(ds_dir / "user_history.parquet")
    index_impressions = pl.concat(
        [pl.read_parquet(ds_dir / f"impressions_{s}.parquet") for s in ("train", "val", "test")])
    embeddings_path = ds_dir / args.embeddings_file
    builder = MatrixBuilder.build(articles=articles, history=history,
                                  index_impressions=index_impressions,
                                  embeddings_path=embeddings_path if embeddings_path.exists()
                                  else None)
    model = LambdaRanker.load(args.models_dir / args.dataset / "reranker.lgb")
    # Session features need the whole impression set to get session boundaries right, and they
    # are cheap (one row per impression), so compute them once up front and hand them to each
    # batch rather than letting a batch see only its own slice of a session.
    session_feats = session_features(pseudo)

    # Per-impression metric lists, accumulated across batches. Only these survive; the feature
    # matrix for a batch is dropped before the next one is built.
    per_method: dict[str, dict[str, dict[str, list[float]]]] = {
        name: {"end_to_end": {m: [] for m in END_TO_END_METRICS},
               "conditional": {m: [] for m in METRICS}}
        for name in ("bm25 top-K order", "reranker over top-K")}

    def accumulate(name: str, matrix, values: np.ndarray) -> None:
        offset = 0
        for size in matrix.groups:
            sl = slice(offset, offset + size)
            offset += size
            per = ranking_metrics(values[sl].tolist(), matrix.y[sl].tolist())
            bucket = per_method[name]
            for metric in METRICS:
                if per[metric] is not None:
                    bucket["conditional"][metric].append(per[metric])
                if metric in END_TO_END_METRICS:
                    bucket["end_to_end"][metric].append(
                        per[metric] if per[metric] is not None else 0.0)

    t0 = time.time()
    n_rows = 0
    for start in range(0, pseudo.height, args.batch_size):
        batch = pseudo.slice(start, args.batch_size)
        matrix = builder.transform(batch, session_feats=session_feats)
        n_rows += matrix.n_rows
        tie = "embed_cos" if "embed_cos" in matrix.feature_names else None
        accumulate("reranker over top-K", matrix, model.predict(matrix, tie_breaker=tie))
        # A1's own ordering of the same pool, as the "before" row
        bm25_scores = np.concatenate([np.asarray(s[:top_k], dtype=np.float64)
                                      for s in batch["retrieved_scores"].to_list()])
        accumulate("bm25 top-K order", matrix, bm25_scores)
        del matrix
        done = min(start + args.batch_size, pseudo.height)
        print(f"  {done:>7,}/{pseudo.height:,} impressions  ({time.time() - t0:5.0f}s)", flush=True)
    print(f"  scored {n_rows:,} (impression, retrieved-candidate) rows in {time.time() - t0:.0f}s")

    results = {
        name: {scope: {m: bootstrap_ci(vals, n_boot=args.n_boot) for m, vals in metrics.items()}
               for scope, metrics in scopes.items()}
        for name, scopes in per_method.items()}

    print(f"\n=== {args.dataset}: literal two-stage, top-{top_k} from the full catalogue ===")
    for scope, metrics in (("end_to_end", END_TO_END_METRICS), ("conditional", METRICS)):
        note = ("every impression; a retrieval miss scores 0. AUC is omitted because it is "
                "undefined, not 0, when no positive was retrieved" if scope == "end_to_end"
                else f"only the {hit.mean():.1%} where stage 1 retrieved the click -- NOT a "
                     "system metric")
        print(f"\n  {scope.replace('_', '-')}\n    ({note})")
        print(f"    {'method':<22}" + "".join(f"{m:>12}" for m in metrics))
        for name, payload in results.items():
            row = f"    {name:<22}"
            for metric in metrics:
                row += f"{payload[scope][metric]['mean']:>12.4f}"
            print(row)

    out_dir = args.results_dir / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": args.dataset, "split": args.split, "top_k": top_k,
        "n_impressions": joined.height, "recall": recalls,
        "hit_rate": float(hit.mean()), "metrics": results,
    }
    (out_dir / "two_stage_literal.json").write_text(json.dumps(payload, indent=2))
    print(f"\n-> {out_dir / 'two_stage_literal.json'}")


if __name__ == "__main__":
    main()
