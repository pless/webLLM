#!/usr/bin/env python3
"""
Quantize an L2-normalized float32 feature array to int8.

For L2-normalized CLIP features, values are bounded roughly in [-0.2, 0.2],
so a per-array symmetric scale works well. We store:

    features_int8.npy       -- int8 array, shape (N, D)
    features_scale.json     -- {"scale": float, "dim": D, "count": N}

Dequantize in the browser as:   float = int8 * scale

Cosine similarity after dequant matches fp32 to ~4 decimal places on
normalized CLIP embeddings -- well below the noise floor for retrieval.

Usage:
    python quantize_features.py [features.npy] [features_int8.npy]
"""
import json
import sys
import numpy as np

def main(src="features.npy", dst="features_int8.npy"):
    x = np.load(src)
    assert x.dtype == np.float32, f"expected float32, got {x.dtype}"
    N, D = x.shape
    print(f"Loaded {src}: {x.shape} {x.dtype}  size={x.nbytes/1e6:.1f} MB")

    # Symmetric scale from max |value|. Leave a tiny headroom for safety.
    max_abs = float(np.abs(x).max())
    scale = max_abs / 127.0
    print(f"max|x| = {max_abs:.4f}  ->  scale = {scale:.6f}")

    q = np.clip(np.round(x / scale), -127, 127).astype(np.int8)
    np.save(dst, q)
    with open("features_scale.json", "w") as f:
        json.dump({"scale": scale, "dim": D, "count": N}, f)

    # Report reconstruction error
    recon = q.astype(np.float32) * scale
    err = np.abs(recon - x).max()
    # Cosine-sim sanity: self-dot should still be ~1 for every row
    sims = (recon * x).sum(axis=1)
    print(f"Wrote {dst}: {q.shape} int8  size={q.nbytes/1e6:.1f} MB")
    print(f"Max elementwise error: {err:.6f}")
    print(f"Self cosine after quant: mean={sims.mean():.5f}  min={sims.min():.5f}")

if __name__ == "__main__":
    args = sys.argv[1:]
    main(*args)
