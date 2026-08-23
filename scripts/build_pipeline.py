#!/usr/bin/env python3
"""Q1: one-command rebuild -- raw files -> cleaned unified schema -> temporal split -> feature
store. Rerunning this script from scratch reproduces every output under --out-dir.

Usage:
    python scripts/build_pipeline.py                          # uses configs/pipeline.yaml
    python scripts/build_pipeline.py --ebnerd-tier large --ebnerd-raw-dir data/ebnerd_large
    python scripts/build_pipeline.py --skip-mind               # EB-NeRD only
"""
import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ire_a1.clean import clean_ebnerd, clean_mind  # noqa: E402
from ire_a1.feature_store import build_article_store, build_user_store  # noqa: E402
from ire_a1.split import temporal_split  # noqa: E402


def _write_dataset(name: str, tables: dict, out_dir: Path, test_frac: float, val_frac: float) -> None:
    ds_dir = out_dir / name
    ds_dir.mkdir(parents=True, exist_ok=True)

    article_store = build_article_store(tables["articles"])
    user_store = build_user_store(tables["history"])
    article_store.write_parquet(ds_dir / "articles.parquet")
    user_store.write_parquet(ds_dir / "user_history.parquet")
    print(f"  [{name}] articles: {article_store.height:,} rows, users: {user_store.height:,} rows")

    splits = temporal_split(tables["impressions"], test_frac=test_frac, val_frac=val_frac)
    for split_name, df in splits.items():
        df.write_parquet(ds_dir / f"impressions_{split_name}.parquet")
        t_min = df["timestamp"].min()
        t_max = df["timestamp"].max()
        print(f"  [{name}] {split_name}: {df.height:,} impressions ({t_min} -> {t_max})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/pipeline.yaml", type=Path)
    parser.add_argument("--ebnerd-raw-dir", type=Path)
    parser.add_argument("--mind-train-dir", type=Path)
    parser.add_argument("--mind-dev-dir", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--test-frac", type=float)
    parser.add_argument("--val-frac", type=float)
    parser.add_argument("--skip-ebnerd", action="store_true")
    parser.add_argument("--skip-mind", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    ebnerd_raw_dir = args.ebnerd_raw_dir or Path(cfg["ebnerd"]["raw_dir"])
    mind_train_dir = args.mind_train_dir or Path(cfg["mind"]["train_dir"])
    mind_dev_dir = args.mind_dev_dir or Path(cfg["mind"]["dev_dir"])
    out_dir = args.out_dir or Path(cfg["output_dir"])
    test_frac = args.test_frac if args.test_frac is not None else cfg["split"]["test_frac"]
    val_frac = args.val_frac if args.val_frac is not None else cfg["split"]["val_frac"]

    if not args.skip_ebnerd:
        print(f"Cleaning EB-NeRD from {ebnerd_raw_dir} ...")
        _write_dataset("ebnerd", clean_ebnerd(ebnerd_raw_dir), out_dir, test_frac, val_frac)

    if not args.skip_mind:
        print(f"Cleaning MIND from {mind_train_dir} + {mind_dev_dir} ...")
        _write_dataset("mind", clean_mind(mind_train_dir, mind_dev_dir), out_dir, test_frac, val_frac)

    print(f"Done. Feature store ready under {out_dir}/")


if __name__ == "__main__":
    main()
