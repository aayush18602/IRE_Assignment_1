#!/usr/bin/env python3
"""Q3 step 5: compare BM25 (Q2) vs. embedding-based (Q3) candidate retrieval side by side.

Reads the recall_<split>.json summaries both scripts/run_bm25.py and scripts/run_embeddings.py
already write, and prints a comparison table. Deeper slicing (cold-start vs. warm, head vs.
tail) is Q4's job (the offline evaluation harness); this just answers Q3's own "which works
better" question at the recall@K level, including each method's own cold-start rate (a user
with no click history produces an empty BM25 query and a None user embedding alike).

Usage:
    python scripts/compare_retrieval.py --dataset ebnerd
"""
import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["ebnerd", "mind"], required=True)
    parser.add_argument("--processed-dir", default="data/processed", type=Path)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    ds_dir = args.processed_dir / args.dataset
    bm25_path = ds_dir / "bm25" / f"recall_{args.split}.json"
    emb_path = ds_dir / "embeddings_eval" / f"recall_{args.split}.json"

    missing = [p for p in (bm25_path, emb_path) if not p.exists()]
    if missing:
        raise SystemExit(
            f"Missing result file(s): {missing}. Run scripts/run_bm25.py and "
            f"scripts/run_embeddings.py --dataset {args.dataset} first."
        )

    bm25 = json.loads(bm25_path.read_text())
    emb = json.loads(emb_path.read_text())

    if bm25["n_impressions"] != emb["n_impressions"]:
        print(
            f"WARNING: comparing runs over different impression counts "
            f"(bm25={bm25['n_impressions']}, embeddings={emb['n_impressions']}) -- "
            f"numbers below are not directly comparable until both are re-run consistently."
        )

    print(f"=== {args.dataset} / {args.split} split -- BM25 vs. embeddings ===")
    print(f"{'metric':<12}{'BM25':>10}{'Embeddings':>14}{'winner':>12}")
    k_values = sorted(int(k.split("@")[1]) for k in bm25 if k.startswith("recall@"))
    for k in k_values:
        b, e = bm25[f"recall@{k}"], emb[f"recall@{k}"]
        winner = "BM25" if b > e else ("Embeddings" if e > b else "tie")
        print(f"recall@{k:<5}{b:>10.4f}{e:>14.4f}{winner:>12}")

    print()
    print(f"cold-start rate -- BM25: {bm25['n_cold_start'] / bm25['n_impressions']:.1%}   "
          f"Embeddings: {emb['n_cold_start'] / emb['n_impressions']:.1%}")


if __name__ == "__main__":
    main()
