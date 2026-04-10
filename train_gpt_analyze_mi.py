"""
Analyze token-level mutual information (MI) across lags in FineWeb training shards.

Loads the first 5M tokens from up to 3 training shards and computes pairwise MI
at lags 1-32. 

The motivation: early experiments showed suspiciously strong local
structure in the loss curves, raising the question of whether the competition
reward is dominated by short-range pattern matching rather than genuine language
understanding. Rather than guessing at an architecture, this script lets the data
answer directly - if MI decays sharply after a few lags, a small local kernel
captures most of the signal and long-range attention is wasted capacity. The MI
scores are used to score and recommend sparse attention window patterns
(contiguous k=3/5/7 vs. dilated 1-2-4 / 1-2-4-8) that match the actual
statistical structure of the corpus.
"""

import glob
import numpy as np
import os

# Configure via env var if needed
data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
DATA = os.path.join(data_path, "fineweb_train_*.bin")
HEADER = 256 * 4

def load_shard(f):
    h = np.fromfile(f, dtype="<i4", count=256)
    if len(h) != 256:
        raise ValueError(f"Bad header: {f}")
    n = int(h[2])
    return np.fromfile(f, dtype="<u2", count=n, offset=HEADER).astype(np.int32)

files = sorted(glob.glob(DATA))[:3]
if not files:
    raise SystemExit(
        f"No shards found.\n"
        f"Tried: {DATA}\n"
        f"CWD: {os.getcwd()}\n"
        f"Hint: set DATA_PATH or run from repo root."
    )

tokens = np.concatenate([load_shard(f) for f in files])[:5_000_000]

def mi_lag(x, lag, V=1024):
    a = x[:-lag]
    b = x[lag:]
    joint = np.zeros((V, V), dtype=np.int64)
    np.add.at(joint, (a, b), 1)
    p = joint / joint.sum()
    pa = p.sum(1)
    pb = p.sum(0)
    nz = p > 0
    i, j = np.nonzero(nz)
    return np.sum(p[i, j] * np.log2(p[i, j] / (pa[i] * pb[j])))

mis = [mi_lag(tokens, k) for k in range(1, 17)]

print("MI lags 1-32:")
mis = [mi_lag(tokens, k) for k in range(1, 33)]
for i, m in enumerate(mis, 1):
    print(i, round(float(m), 4))

def window_score(lags):
    return sum(mis[k - 1] for k in lags)

windows = {
    "k3"        : [1, 2, 3],
    "k5"        : [1, 2, 3, 4, 5],
    "k7"        : [1, 2, 3, 4, 5, 6, 7],
    "dilated_1_2_4": [1, 2, 4],
    "dilated_1_2_4_8": [1, 2, 4, 8],
    "mid_2_4_8" : [2, 4, 8],
    "long_4_8_16": [4, 8, 16],
}

print("\nWindow scores:")
for name, lags in windows.items():
    print(f"{name:14s} {round(float(window_score(lags)), 4)}   lags={lags}")

print("\nIncremental gains:")
for k in [3, 5, 7]:
    vals = mis[:k]
    inc = [vals[0]] + [vals[i] - vals[i+1] for i in range(len(vals)-1)]
    print(f"k={k}: {[round(float(x), 4) for x in inc]}")

print("\nKernel recommendation:")
k3 = window_score([1,2,3])
k5 = window_score([1,2,3,4,5])
k7 = window_score([1,2,3,4,5,6,7])
d124 = window_score([1,2,4])
d1248 = window_score([1,2,4,8])

if k5 > 1.15 * k3 and k7 < 1.08 * k5:
    print("Use contiguous kernel 5")
elif k7 > 1.10 * k5:
    print("Use contiguous kernel 7")
elif d124 > 0.95 * k5 or d1248 > k5:
    print("Use local + dilated mixer")
else:
    print("Use contiguous kernel 3 or 5")