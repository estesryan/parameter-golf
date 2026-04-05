"""
python prepare_fineweb_byte.py \
  --input_glob "./raw_fineweb/*.jsonl" \
  --output_dir "./data/datasets/fineweb10B_byte" \
  --text_key text \
  --val_files 1 \
  --shard_size 50000000
"""

from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np

SHARD_MAGIC = 20240520
SHARD_VERSION = 1
HEADER_INTS = 256
HEADER_DTYPE = np.dtype("<i4")
TOKEN_DTYPE = np.dtype("<u2")


def encode_text_to_uint16_bytes(text: str) -> np.ndarray:
    b = text.encode("utf-8", errors="ignore")
    return np.frombuffer(b, dtype=np.uint8).astype(np.uint16)


def iter_jsonl_texts(path: Path, text_key: str):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get(text_key, "")
            if text:
                yield text


def load_file_tokens(args: tuple[Path, str]) -> np.ndarray:
    path, text_key = args
    chunks: list[np.ndarray] = []
    for text in iter_jsonl_texts(path, text_key):
        toks = encode_text_to_uint16_bytes(text)
        if toks.size:
            chunks.append(toks)
    if not chunks:
        return np.empty((0,), dtype=np.uint16)
    return np.concatenate(chunks)


def write_shard(path: Path, tokens: np.ndarray) -> None:
    header = np.zeros((HEADER_INTS,), dtype=HEADER_DTYPE)
    header[0] = SHARD_MAGIC
    header[1] = SHARD_VERSION
    header[2] = int(tokens.size)

    with path.open("wb") as f:
        header.tofile(f)
        tokens.astype(TOKEN_DTYPE, copy=False).tofile(f)


def shard_tokens(tokens: np.ndarray, shard_size: int):
    start = 0
    while start < tokens.size:
        end = min(start + shard_size, tokens.size)
        yield tokens[start:end]
        start = end


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_glob", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--text_key", default="text")
    ap.add_argument("--val_files", type=int, default=1)
    ap.add_argument("--shard_size", type=int, default=50_000_000)
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    args = ap.parse_args()

    files = [Path(p) for p in sorted(glob.glob(args.input_glob))]
    if not files:
        raise FileNotFoundError(f"No files matched: {args.input_glob}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with mp.Pool(args.workers) as pool:
        token_arrays = pool.map(load_file_tokens, [(p, args.text_key) for p in files])

    if not token_arrays:
        raise RuntimeError("No token arrays produced")

    all_tokens = np.concatenate(token_arrays)
    if all_tokens.size == 0:
        raise RuntimeError("No byte tokens produced")

    shard_list = list(shard_tokens(all_tokens, args.shard_size))

    for i, shard in enumerate(shard_list):
        split = "val" if i < args.val_files else "train"
        split_idx = i if split == "val" else i - args.val_files
        out_path = out_dir / f"fineweb_{split}_{split_idx:06d}.bin"
        write_shard(out_path, shard)
        print(f"wrote {out_path} tokens={shard.size}")


if __name__ == "__main__":
    main()