#!/usr/bin/env python3
"""Q3 step 1: compute our own article embeddings via BERT/XLM-RoBERTa (chosen over EB-NeRD's
provided embeddings). This is the one GPU-bound step in the assignment -- run it on Kaggle (or
any CUDA box); `requirements-gpu.txt` has the extra deps (torch, transformers) needed on top of
the base `requirements.txt`, not installed locally on this machine on purpose (see
ai_usage_log.md / README's compute-strategy notes).

Usage (on Kaggle, after `pip install -r requirements.txt -r requirements-gpu.txt` and copying
data/processed/<dataset>/articles.parquet there):
    python scripts/compute_embeddings.py --dataset ebnerd
    python scripts/compute_embeddings.py --dataset mind --model xlm-roberta-base

Default model is xlm-roberta-base for BOTH datasets (one multilingual model, not two
dataset-specific ones) -- it handles EB-NeRD's Danish text natively and MIND's English text
just as well, so the same embedding pipeline and dimensionality (768) applies to both, keeping
Q3's downstream ANN/eval code dataset-agnostic like everything else in this pipeline.

Output: data/processed/<dataset>/embeddings.parquet with columns article_id, embedding (list[f32]).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def _load_model(model_name: str, device: str):
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "torch/transformers not installed. This script needs requirements-gpu.txt "
            "(pip install -r requirements.txt -r requirements-gpu.txt) -- meant to run on "
            "Kaggle/Colab, not this local machine (no working CUDA driver here)."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    return torch, tokenizer, model


def _mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def compute_embeddings(
    texts: list[str], model_name: str, batch_size: int, max_length: int, device: str | None
) -> np.ndarray:
    torch, tokenizer, model = _load_model(model_name, device or ("cuda" if _cuda_available() else "cpu"))
    device = next(model.parameters()).device

    all_vectors = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = tokenizer(
                batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
            ).to(device)
            outputs = model(**inputs)
            pooled = _mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
            all_vectors.append(pooled.cpu().numpy())
            if (i // batch_size) % 20 == 0:
                print(f"  {min(i + batch_size, len(texts)):,} / {len(texts):,}", flush=True)
    return np.concatenate(all_vectors, axis=0).astype(np.float32)


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["ebnerd", "mind"], required=True)
    parser.add_argument("--processed-dir", default="data/processed", type=Path)
    parser.add_argument("--model", default="xlm-roberta-base")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=64,
                         help="tokens; title+abstract, not full body, so 64 covers most articles")
    parser.add_argument("--device", default=None, help="cuda | cpu | mps (default: auto-detect)")
    parser.add_argument("--limit", type=int, default=None, help="only embed the first N articles (smoke test)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    ds_dir = args.processed_dir / args.dataset
    articles = pl.read_parquet(ds_dir / "articles.parquet")
    if args.limit:
        articles = articles.head(args.limit)

    texts = (
        (articles["title"].fill_null("") + " " + articles["abstract"].fill_null("")).to_list()
    )
    article_ids = articles["article_id"].to_list()

    print(f"Computing {args.model} embeddings for {len(texts):,} {args.dataset} articles ...")
    t0 = time.time()
    vectors = compute_embeddings(texts, args.model, args.batch_size, args.max_length, args.device)
    print(f"Done in {time.time() - t0:.1f}s, embedding dim = {vectors.shape[1]}")

    out_path = args.out or ds_dir / "embeddings.parquet"
    pl.DataFrame({"article_id": article_ids, "embedding": vectors.tolist()}).write_parquet(out_path)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
