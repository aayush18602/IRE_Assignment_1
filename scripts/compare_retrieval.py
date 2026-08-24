#!/usr/bin/env python3
"""Q3 step 5: compare BM25 (Q2) vs. one or more embedding-based (Q3) candidate retrieval runs
side by side.

Reads the recall_<split>.json summaries scripts/run_bm25.py and scripts/run_embeddings.py
already write, and prints a comparison table. Deeper slicing (cold-start vs. warm, head vs.
tail) is Q4's job (the offline evaluation harness); this just answers Q3's own "which works
better" question at the recall@K level, including each method's own cold-start rate (a user
with no click history produces an empty BM25 query and a None user embedding alike).

Usage:
    python scripts/compare_retrieval.py --dataset ebnerd
    # compare BM25 against two embedding variants at once (e.g. raw XLM-R vs. a fine-tuned
    # sentence-embedding model, each run separately via run_embeddings.py --out-dir ...):
    python scripts/compare_retrieval.py --dataset ebnerd \
        --variant "XLM-R=embeddings_eval" --variant "mpnet=embeddings_eval_mpnet"
"""
import argparse
import json
from pathlib import Path


def _parse_variant(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(f"--variant must be LABEL=DIR, got {spec!r}")
    label, dir_name = spec.split("=", 1)
    return label, dir_name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["ebnerd", "mind"], required=True)
    parser.add_argument("--processed-dir", default="data/processed", type=Path)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--variant", action="append", type=_parse_variant, dest="variants",
                         help="LABEL=DIR, e.g. mpnet=embeddings_eval_mpnet. Repeatable. "
                              "Default: a single 'Embeddings=embeddings_eval' variant.")
    args = parser.parse_args()
    variants = args.variants or [("Embeddings", "embeddings_eval")]

    ds_dir = args.processed_dir / args.dataset
    bm25_path = ds_dir / "bm25" / f"recall_{args.split}.json"
    if not bm25_path.exists():
        raise SystemExit(f"Missing {bm25_path}. Run scripts/run_bm25.py --dataset {args.dataset} first.")
    bm25 = json.loads(bm25_path.read_text())

    results = [("BM25", bm25)]
    for label, dir_name in variants:
        path = ds_dir / dir_name / f"recall_{args.split}.json"
        if not path.exists():
            raise SystemExit(
                f"Missing {path}. Run scripts/run_embeddings.py --dataset {args.dataset} "
                f"--out-dir {ds_dir / dir_name} first."
            )
        results.append((label, json.loads(path.read_text())))

    counts = {label: r["n_impressions"] for label, r in results}
    if len(set(counts.values())) > 1:
        print(f"WARNING: comparing runs over different impression counts ({counts}) -- "
              f"numbers below are not directly comparable until all are re-run consistently.")

    print(f"=== {args.dataset} / {args.split} split -- {' vs. '.join(l for l, _ in results)} ===")
    header = "metric".ljust(12) + "".join(label.rjust(14) for label, _ in results) + "winner".rjust(14)
    print(header)

    k_values = sorted(int(k.split("@")[1]) for k in bm25 if k.startswith("recall@"))
    for k in k_values:
        scores = [(label, r[f"recall@{k}"]) for label, r in results]
        best_label, best_score = max(scores, key=lambda ls: ls[1])
        winner = best_label if sum(1 for _, s in scores if s == best_score) == 1 else "tie"
        row = f"recall@{k}".ljust(12) + "".join(f"{s:.4f}".rjust(14) for _, s in scores) + winner.rjust(14)
        print(row)

    print()
    cold_start = "   ".join(f"{label}: {r['n_cold_start'] / r['n_impressions']:.1%}" for label, r in results)
    print(f"cold-start rate -- {cold_start}")


if __name__ == "__main__":
    main()
