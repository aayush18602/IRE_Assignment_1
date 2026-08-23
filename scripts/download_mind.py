#!/usr/bin/env python3
"""Download and extract MIND dataset bundles (Part 0, Q1 step 1).

The yjw1029/MIND repo on HuggingFace is *gated* -- you must, once, in a browser:
  1. Log in / sign up at https://huggingface.co
  2. Visit https://huggingface.co/datasets/yjw1029/MIND and accept the terms
  3. Create an access token at https://huggingface.co/settings/tokens (read access is enough)
  4. Either run `huggingface-cli login` once, or export HF_TOKEN=<token> before running this script

Usage:
    python scripts/download_mind.py --tier small          # train+dev, dev use
    python scripts/download_mind.py --tier large-test      # required for Codabench, ~large
    python scripts/download_mind.py --tier large-full      # train+dev+test, run on Kaggle
"""
import argparse
import sys
import zipfile
from pathlib import Path


def _members(zf: zipfile.ZipFile):
    # Some upstream zips carry macOS resource-fork junk (__MACOSX/, ._*); skip it.
    return [m for m in zf.namelist() if "__MACOSX" not in m and not Path(m).name.startswith("._")]

REPO_ID = "yjw1029/MIND"

FILES = {
    "small": ["MINDsmall_train.zip", "MINDsmall_dev.zip"],
    "large-test": ["MINDlarge_test.zip"],
    "large-full": ["MINDlarge_train.zip", "MINDlarge_dev.zip", "MINDlarge_test.zip"],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tier", choices=list(FILES), required=True)
    parser.add_argument("--out-dir", default="data", type=Path)
    parser.add_argument("--keep-zip", action="store_true", help="don't delete the zip after extracting")
    args = parser.parse_args()

    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import GatedRepoError, HfHubHTTPError
    except ImportError:
        print("huggingface_hub not installed. Run: pip install -r requirements.txt", file=sys.stderr)
        sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for fname in FILES[args.tier]:
        print(f"Fetching {fname} from {REPO_ID} ...")
        try:
            local_path = hf_hub_download(
                repo_id=REPO_ID, repo_type="dataset", filename=fname, local_dir=args.out_dir,
            )
        except (GatedRepoError, HfHubHTTPError) as exc:
            print(
                f"\nERROR: could not fetch {fname} ({exc}).\n"
                "This dataset is gated. See the docstring at the top of this script: "
                "accept the terms on the dataset page, create a token, and either run "
                "`huggingface-cli login` or export HF_TOKEN before retrying.",
                file=sys.stderr,
            )
            sys.exit(1)

        zip_path = Path(local_path)
        target_dir = args.out_dir / zip_path.stem
        print(f"Extracting {fname} -> {target_dir}/")
        target_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(target_dir, members=_members(zf))
        if not args.keep_zip:
            zip_path.unlink()

    print(f"Done. MIND '{args.tier}' tier ready under {args.out_dir}/")


if __name__ == "__main__":
    main()
