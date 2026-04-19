import glob
from collections import Counter
from pathlib import Path

import numpy as np
import sentencepiece as spm

# ===== CONFIG =====
DATA_GLOB = "./data/datasets/fineweb10B_sp4096/fineweb_val_*.bin"
TOKENIZER_PATH = "./data/tokenizers/fineweb_4096_bpe.model"
MAX_TOKENS = 20_000_000
TOP_K = 30

# Saved baseline from your old stats
BASELINE_TOKENS_PER_BYTE = 0.302224
BASELINE_BYTES_PER_TOKEN = 3.308807
BASELINE_ACTIVE_VOCAB = 3951

# Shard format
HEADER_INTS = 256
MAGIC = 20240520
VERSION = 1

files = [Path(p) for p in sorted(glob.glob(DATA_GLOB))]
if not files:
    raise FileNotFoundError(f"No shard files matched: {DATA_GLOB}")

sp = spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)
V = int(sp.vocab_size())


def load_shard(path: Path) -> np.ndarray:
    header = np.fromfile(path, dtype="<i4", count=HEADER_INTS)
    if header.size != HEADER_INTS:
        raise ValueError(f"{path} has incomplete header")
    if int(header[0]) != MAGIC:
        raise ValueError(f"{path} has bad magic: {int(header[0])}")
    if int(header[1]) != VERSION:
        raise ValueError(f"{path} has bad version: {int(header[1])}")

    n = int(header[2])
    arr = np.fromfile(
        path,
        dtype="<u2",
        count=n,
        offset=HEADER_INTS * np.dtype("<i4").itemsize,
    )
    if arr.size != n:
        raise ValueError(f"{path} expected {n} tokens, found {arr.size}")
    return arr.astype(np.int32, copy=False)


# Precompute byte contribution per token piece.
# This matches your earlier shard-based accounting style.
base_bytes = np.zeros(V, dtype=np.int64)
leading_space = np.zeros(V, dtype=np.bool_)

for i in range(V):
    piece = sp.id_to_piece(i)
    if sp.is_byte(i):
        base_bytes[i] = 1
        continue
    if piece.startswith("▁"):
        leading_space[i] = True
        piece = piece[1:]
    base_bytes[i] = len(piece.encode("utf-8"))


freq = np.zeros(V, dtype=np.int64)
byte_total = np.zeros(V, dtype=np.int64)
boundary_freq = np.zeros(V, dtype=np.int64)

scanned = 0
for fp in files:
    toks = load_shard(fp)

    remaining = MAX_TOKENS - scanned
    if remaining <= 0:
        break
    toks = toks[:remaining]

    if toks.size == 0:
        continue

    # Token frequency
    freq += np.bincount(toks, minlength=V)

    # First token in shard as a boundary proxy
    boundary_freq[toks[0]] += 1

    # Byte accounting from token pieces on positions 1..end
    if toks.size >= 2:
        curr = toks[1:]
        extra_space = leading_space[curr].astype(np.int64)
        tok_bytes = base_bytes[curr] + extra_space
        sums = np.bincount(curr, weights=tok_bytes, minlength=V)
        byte_total[: len(sums)] += sums.astype(np.int64)

    scanned += toks.size
    print(f"scanned {scanned:,} tokens")
    if scanned >= MAX_TOKENS:
        break

total_tokens = int(freq.sum())
total_bytes = int(byte_total.sum())
tokens_per_byte = total_tokens / max(total_bytes, 1)
bytes_per_token = total_bytes / max(total_tokens, 1)
active_vocab = int((freq > 0).sum())

print("\n=== BASIC METRICS ===")
print(f"total_tokens     {total_tokens}")
print(f"total_bytes      {total_bytes}")
print(f"tokens_per_byte  {tokens_per_byte:.6f}")
print(f"bytes_per_token  {bytes_per_token:.6f}")
print(f"active_vocab     {active_vocab}")

print("\n=== TOP TOKENS ===")
top_ids = np.argsort(-freq)[:TOP_K]
for tok_id in top_ids:
    if freq[tok_id] == 0:
        continue
    print(f"{int(freq[tok_id]):10d}  {tok_id:6d}  {sp.id_to_piece(int(tok_id))}")

print("\n=== TOP BOUNDARY TOKENS ===")
top_boundary_ids = np.argsort(-boundary_freq)[:TOP_K]
for tok_id in top_boundary_ids:
    if boundary_freq[tok_id] == 0:
        continue
    print(f"{int(boundary_freq[tok_id]):10d}  {tok_id:6d}  {sp.id_to_piece(int(tok_id))}")

print("\n=== TOP TOKENS BY BYTE MASS ===")
top_byte_ids = np.argsort(-byte_total)[:TOP_K]
for tok_id in top_byte_ids:
    if byte_total[tok_id] == 0:
        continue
    print(
        f"{int(byte_total[tok_id]):10d}  "
        f"{int(freq[tok_id]):10d}  "
        f"{tok_id:6d}  "
        f"{sp.id_to_piece(int(tok_id))}"
    )

print("\n=== QUICK CHECK VS BASELINE ===")
print(f"baseline tokens_per_byte  {BASELINE_TOKENS_PER_BYTE:.6f}")
print(f"baseline bytes_per_token  {BASELINE_BYTES_PER_TOKEN:.6f}")
print(f"baseline active_vocab     {BASELINE_ACTIVE_VOCAB}")

delta_tpb = tokens_per_byte - BASELINE_TOKENS_PER_BYTE
delta_bpt = bytes_per_token - BASELINE_BYTES_PER_TOKEN
delta_vocab = active_vocab - BASELINE_ACTIVE_VOCAB

print(f"\ndelta tokens_per_byte  {delta_tpb:+.6f}")
print(f"delta bytes_per_token  {delta_bpt:+.6f}")
print(f"delta active_vocab     {delta_vocab:+d}")

if tokens_per_byte < BASELINE_TOKENS_PER_BYTE:
    print("✔ Compression improved")
else:
    print("✖ Compression did not improve")

if bytes_per_token > BASELINE_BYTES_PER_TOKEN:
    print("✔ Bytes/token improved")
else:
    print("✖ Bytes/token did not improve")