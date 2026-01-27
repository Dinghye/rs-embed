from __future__ import annotations
import numpy as np

def pool_chw_to_vec(emb_chw: np.ndarray, method: str = "mean") -> np.ndarray:
    if emb_chw.ndim != 3:
        raise ValueError(f"Expected [C,H,W], got {emb_chw.shape}")
    C, H, W = emb_chw.shape
    X = emb_chw.reshape(C, H * W)
    if method == "mean":
        v = np.nanmean(X, axis=1)
    elif method == "max":
        v = np.nanmax(X, axis=1)
    else:
        raise ValueError(f"Unknown pooling: {method}")
    return v.astype(np.float32)