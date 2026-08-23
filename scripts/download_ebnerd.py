#!/usr/bin/env python3
"""Download and extract EB-NeRD dataset bundles (Part 0, Q1 step 1).

Usage:
    python scripts/download_ebnerd.py --tier demo
    python scripts/download_ebnerd.py --tier small
    python scripts/download_ebnerd.py --tier large --include-embeddings   # run on Kaggle, multi-GB

Each bundle is downloaded as a zip, extracted into data/<name>/, and the zip is deleted
afterwards to save disk (data/ is git-ignored either way per Q8).
"""
import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

BASE = "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com"

BUNDLES = {
    "demo": [f"{BASE}/ebnerd_demo.zip"],
    "small": [f"{BASE}/ebnerd_small.zip"],
    "large": [
        f"{BASE}/ebnerd_large.zip",
        f"{BASE}/artifacts/articles_large_only.zip",
        f"{BASE}/ebnerd_testset.zip",
    ],
}

EMBEDDINGS = [
    f"{BASE}/artifacts/Ekstra_Bladet_word2vec.zip",
    f"{BASE}/artifacts/google_bert_base_multilingual_cased.zip",
]


def download(url: str, dest: Path) -> None:
    print(f"Downloading {url} -> {dest}")
    last_pct = -1

    def _hook(block_num, block_size, total_size):
        nonlocal last_pct
        if total_size <= 0:
            return
        done = block_num * block_size
        pct = min(100, done * 100 // total_size)
        if pct != last_pct and pct % 5 == 0:
            last_pct = pct
            print(f"  {pct:3d}% ({done / 1e6:.1f} / {total_size / 1e6:.1f} MB)")

    urllib.request.urlretrieve(url, dest, reporthook=_hook)


def _members(zf: zipfile.ZipFile):
    # Some upstream zips carry macOS resource-fork junk (__MACOSX/, ._*); skip it.
    return [m for m in zf.namelist() if "__MACOSX" not in m and not Path(m).name.startswith("._")]


def extract_and_clean(zip_path: Path, out_dir: Path) -> None:
    print(f"Extracting {zip_path.name} -> {out_dir}/")
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir, members=_members(zf))
    zip_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=["demo", "small", "large"], required=True)
    parser.add_argument("--out-dir", default="data", type=Path)
    parser.add_argument("--include-embeddings", action="store_true",
                         help="also fetch word2vec + multilingual BERT article embeddings (large tier only, optional)")
    parser.add_argument("--keep-zip", action="store_true", help="don't delete the zip after extracting")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    urls = list(BUNDLES[args.tier])
    if args.tier == "large" and args.include_embeddings:
        urls += EMBEDDINGS

    for url in urls:
        fname = url.rsplit("/", 1)[-1]
        zip_path = args.out_dir / fname
        try:
            download(url, zip_path)
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR downloading {url}: {exc}", file=sys.stderr)
            sys.exit(1)

        target_name = fname.removesuffix(".zip")
        target_dir = args.out_dir / target_name
        if args.keep_zip:
            print(f"Extracting {zip_path.name} -> {target_dir}/ (keeping zip)")
            target_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(target_dir, members=_members(zf))
        else:
            extract_and_clean(zip_path, target_dir)

    print(f"Done. EB-NeRD '{args.tier}' tier ready under {args.out_dir}/")


if __name__ == "__main__":
    main()
