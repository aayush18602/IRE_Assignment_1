#!/usr/bin/env python3
"""A2 Q1: measure every engineered feature, so the design note cites reproducible numbers.

Reports each feature's **per-impression** AUC alongside its pooled AUC. The gap between the two
is itself the finding: both leaderboards re-rank an impression's own slate, so a feature that is
constant within an impression cannot reorder anything, yet pooled AUC reports it as informative.
`slate_size` reads 0.3088 pooled and exactly 0.5000 per impression. Every feature claim in the
report should come from the per-impression column.

Usage:
    python scripts/run_feature_analysis.py --dataset ebnerd
    python scripts/run_feature_analysis.py --dataset mind --limit 20000   # smoke test

Writes results/<dataset>/feature_analysis.json and refreshes results/feature_analysis.md.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ire_a1.eval import batch_ranking_auc  # noqa: E402
from ire_a1.reranker import MatrixBuilder  # noqa: E402


def pooled_auc(labels: np.ndarray, values: np.ndarray) -> float:
    """Same rank identity as batch_ranking_auc, but over every row at once -- reported only to
    show how misleading it is relative to the per-impression figure."""
    ok = ~np.isnan(values)
    y, v = labels[ok], values[ok]
    n_pos = int(y.sum())
    if n_pos == 0 or n_pos == len(y):
        return float("nan")
    ranks = pl.Series(v).rank("average").to_numpy()
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * (len(y) - n_pos)))


def analyse(dataset: str, processed_dir: Path, split: str, limit: int | None,
            embeddings_file: str) -> dict:
    ds_dir = processed_dir / dataset
    articles = pl.read_parquet(ds_dir / "articles.parquet")
    history = pl.read_parquet(ds_dir / "user_history.parquet")
    # Index over every split, not just train: the counts are as-of filtered, and indexing train
    # alone leaves a gap wider than the trailing window (measured: every popularity feature drops
    # to exactly AUC 0.5000). See src/ire_a1/article_features.py.
    index_impressions = pl.concat(
        [pl.read_parquet(ds_dir / f"impressions_{s}.parquet") for s in ("train", "val", "test")]
    )
    scored = pl.read_parquet(ds_dir / f"impressions_{split}.parquet")
    if limit:
        scored = scored.head(limit)

    embeddings_path = ds_dir / embeddings_file
    if not embeddings_path.exists():
        print(f"  note: {embeddings_path.name} missing -- embedding similarity will be omitted")
        embeddings_path = None

    t0 = time.time()
    builder = MatrixBuilder.build(articles=articles, history=history,
                                  index_impressions=index_impressions,
                                  embeddings_path=embeddings_path)
    build_s = time.time() - t0

    t0 = time.time()
    matrix = builder.transform(scored)
    transform_s = time.time() - t0
    print(f"  built in {build_s:.1f}s, transformed {matrix.n_rows:,} rows in {transform_s:.1f}s "
          f"({matrix.n_rows / transform_s / 1000:.0f}K rows/s)")

    impression_ids = np.repeat(matrix.impression_ids, matrix.groups)
    safe = set(builder.submission_safe_feature_names)
    unstable = set(builder.split_unstable_feature_names)

    features = []
    for name in matrix.feature_names:
        values = matrix.column(name).astype(np.float64)
        stats = batch_ranking_auc(impression_ids, matrix.y, values)
        features.append({
            "feature": name,
            "per_impression_auc": stats["auc"],
            "pooled_auc": pooled_auc(matrix.y, values),
            "constant_frac": stats["constant_frac"],
            "nan_frac": float(np.isnan(values).mean()),
            "submission_safe": name in safe,
            "split_unstable": name in unstable,
        })
    features.sort(key=lambda f: -abs((f["per_impression_auc"] or 0.5) - 0.5))

    return {
        "dataset": dataset, "split": split,
        "n_impressions": matrix.n_impressions, "n_rows": matrix.n_rows,
        "n_features": len(matrix.feature_names),
        "positive_rate": float(matrix.y.mean()),
        "build_seconds": round(build_s, 2), "transform_seconds": round(transform_s, 2),
        "features": features,
    }


def _table(result: dict) -> str:
    lines = [
        f"### {result['dataset']} — {result['n_features']} features, "
        f"{result['n_rows']:,} rows over {result['n_impressions']:,} impressions "
        f"({result['positive_rate']:.2%} positive)",
        "",
        "| Feature | Per-impression AUC | Pooled AUC | Constant within impression | NaN | Submission-safe |",
        "|---|---|---|---|---|---|",
    ]
    for f in result["features"]:
        flag = " ⚠︎" if f["split_unstable"] else ""
        lines.append(
            f"| `{f['feature']}`{flag} | **{f['per_impression_auc']:.4f}** | "
            f"{f['pooled_auc']:.4f} | {f['constant_frac']:.0%} | {f['nan_frac']:.1%} | "
            f"{'yes' if f['submission_safe'] else 'no'} |"
        )
    return "\n".join(lines)


def write_markdown(results: list[dict], path: Path) -> None:
    header = """# Q1 feature analysis

Generated by `scripts/run_feature_analysis.py`. Every number here is measured on the test split.

**Read the per-impression column, not the pooled one.** Both leaderboards re-rank each
impression's own slate, so a feature that is constant within an impression cannot reorder
anything — pooled AUC still reports it as informative, which is how `slate_size` comes out at
0.3088 pooled and exactly 0.5000 per impression. A high *constant within impression* percentage
means the feature has no standalone ranking power; it can still earn its place as interaction
context for a GBDT, which groups by impression and is not fooled the way pooled AUC is.

An AUC below 0.5 is inverted, not useless — lower values of that feature predict clicks. A GBDT
does not care about direction.

⚠︎ marks features whose distribution shifts between train and test because of how the datasets
were assembled (see `HistoryFeatures.split_unstable_feature_names`).
"""
    body = "\n\n".join(_table(r) for r in results)
    path.write_text(f"{header}\n{body}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["ebnerd", "mind", "both"], default="both")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--embeddings-file", default="embeddings_mpnet.parquet")
    parser.add_argument("--limit", type=int, default=None,
                        help="only score the first N impressions (smoke test)")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    datasets = ["ebnerd", "mind"] if args.dataset == "both" else [args.dataset]
    results = []
    for dataset in datasets:
        print(f"Analysing {dataset} ({args.split}) ...")
        result = analyse(dataset, args.processed_dir, args.split, args.limit,
                         args.embeddings_file)
        out_dir = args.results_dir / dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "feature_analysis.json").write_text(json.dumps(result, indent=2))
        print(f"  -> {out_dir / 'feature_analysis.json'}")
        top = result["features"][0]
        print(f"  strongest: {top['feature']} at {top['per_impression_auc']:.4f} "
              f"(pooled {top['pooled_auc']:.4f})")
        results.append(result)

    if len(results) == len(datasets) and args.dataset == "both" and not args.limit:
        md = args.results_dir / "feature_analysis.md"
        write_markdown(results, md)
        print(f"\n-> {md}")


if __name__ == "__main__":
    main()
