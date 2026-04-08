import glob
import numpy as np
import os

# Configure via env var if needed
data_path = os.environ.get("DATA_PATH", r".\data\datasets\fineweb10B_sp1024")
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

print("MI lags 1–16:")
for i, m in enumerate(mis, 1):
    print(i, round(float(m), 4))

short = sum(mis[:4])
mid   = sum(mis[4:8])
long  = sum(mis[8:16])

print("\nSummary:")
print("short(1–4):", round(float(short), 4))
print("mid(5–8):  ", round(float(mid), 4))
print("long(9–16):", round(float(long), 4))

print("\nRecommendation:")
if short > 3 * mid:
    print("→ small local mixer (kernel 3–5)")
elif mid > 0.3 * short:
    print("→ local + dilated mixer")
else:
    print("→ local weak → focus on long-range state")
