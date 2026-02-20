import numpy as np
import matplotlib.pyplot as plt

def _to_dhw(arr):
    if hasattr(arr, "values"):  # xarray
        arr = arr.values
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr[None, ...].astype(np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected 2D/3D array, got {arr.shape}")
    # HWD -> DHW if last dim looks like D
    if arr.shape[-1] in (32, 64, 128, 256, 512, 768, 1024) and arr.shape[0] not in (32, 64, 128, 256, 512, 768, 1024):
        arr = np.moveaxis(arr, -1, 0)
    return arr.astype(np.float32)

def _robust_scale01(x, lo=2.0, hi=98.0, eps=1e-8):
    """Scale array to [0,1] with percentile clipping."""
    a = np.percentile(x, lo)
    b = np.percentile(x, hi)
    y = np.clip((x - a) / (b - a + eps), 0.0, 1.0)
    return y

def _stabilize_pca_sign(components: np.ndarray) -> np.ndarray:
    """
    Make PCA component signs deterministic.
    For each component row, enforce the element with max abs value to be positive.
    """
    comps = np.asarray(components, dtype=np.float32).copy()
    for i in range(comps.shape[0]):
        row = comps[i]
        if row.size == 0:
            continue
        j = int(np.argmax(np.abs(row)))
        if row[j] < 0:
            comps[i] = -row
    return comps

def fit_pca_rgb(
    emb,
    *,
    n_samples=100_000,
    seed=0,
    center=True,
):
    """
    Fit PCA on pixels of a (D,H,W) grid and return a dict with components for reuse.
    No sklearn dependency (uses SVD).
    """
    data = getattr(emb, "data", emb)
    dhw = _to_dhw(data)
    D, H, W = dhw.shape

    X = dhw.reshape(D, H * W).T  # [N, D]
    N = X.shape[0]

    rng = np.random.default_rng(seed)
    if n_samples is not None and N > n_samples:
        idx = rng.choice(N, size=int(n_samples), replace=False)
        Xs = X[idx]
    else:
        Xs = X

    # center
    mean = Xs.mean(axis=0) if center else np.zeros((D,), dtype=np.float32)
    Xc = Xs - mean

    # SVD for PCA
    # Xc = U S Vt, rows are samples
    # PCs are rows of Vt
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = _stabilize_pca_sign(Vt[:3].astype(np.float32))  # [3, D]

    return {
        "mean": mean.astype(np.float32),
        "components": comps,
        "center": bool(center),
    }

def transform_pca_rgb(
    emb,
    pca,
    *,
    robust_lo=2.0,
    robust_hi=98.0,
):
    """
    Apply fitted PCA to emb, return rgb in [0,1] as (H,W,3).
    """
    data = getattr(emb, "data", emb)
    dhw = _to_dhw(data)
    D, H, W = dhw.shape

    X = dhw.reshape(D, H * W).T  # [N, D]
    mean = pca["mean"]
    comps = pca["components"]  # [3,D]

    Xc = X - mean if pca.get("center", True) else X
    Y = Xc @ comps.T  # [N,3]

    # robust scale each channel to [0,1]
    rgb = np.zeros_like(Y, dtype=np.float32)
    for k in range(3):
        rgb[:, k] = _robust_scale01(Y[:, k], lo=robust_lo, hi=robust_hi)

    return rgb.reshape(H, W, 3)

def plot_embedding_pseudocolor(
    emb,
    *,
    title=None,
    pca=None,
    n_samples=100_000,
    seed=0,
    robust_lo=2.0,
    robust_hi=98.0,
    figsize=(6, 5),
    show_colorbars=False,
):
    """
    Plot PCA pseudocolor image. If pca is None, fit PCA on this embedding.
    Returns fitted pca for reuse across images.
    """
    meta = getattr(emb, "meta", {})
    if title is None:
        title = meta.get("model", "embedding PCA")

    if pca is None:
        pca = fit_pca_rgb(emb, n_samples=n_samples, seed=seed)

    rgb = transform_pca_rgb(emb, pca, robust_lo=robust_lo, robust_hi=robust_hi)
    # rgb = np.flipud(rgb)

    plt.figure(figsize=figsize)
    plt.imshow(rgb)
    plt.title(title)
    plt.axis("off")
    plt.show()

    if show_colorbars:
        # Optional: also show the three PCA channels as grayscale for debugging
        data = getattr(emb, "data", emb)
        dhw = _to_dhw(data)
        D, H, W = dhw.shape
        X = dhw.reshape(D, H * W).T
        Xc = X - pca["mean"]
        Y = Xc @ pca["components"].T
        for k in range(3):
            img = _robust_scale01(Y[:, k], lo=robust_lo, hi=robust_hi).reshape(H, W)
            plt.figure(figsize=figsize)
            plt.imshow(img)
            plt.title(f"{title} | PC{k+1}")
            plt.axis("off")
            plt.show()
    plt.savefig(f"{title.replace(' ','_')}_pca.png")
    return pca

def percentile_stretch(rgb_hwc: np.ndarray, p_low=1.0, p_high=99.0, gamma=1.0) -> np.ndarray:
    """Per-channel percentile stretch to [0,1]."""
    rgb = rgb_hwc.astype(np.float32)
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=0.0, neginf=0.0)

    out = np.empty_like(rgb)
    for c in range(3):
        lo = np.percentile(rgb[..., c], p_low)
        hi = np.percentile(rgb[..., c], p_high)
        if hi <= lo + 1e-6:
            out[..., c] = 0.0
        else:
            out[..., c] = (rgb[..., c] - lo) / (hi - lo)
    out = np.clip(out, 0.0, 1.0)
    if gamma != 1.0:
        out = out ** (1.0 / gamma)
    return out


def show_input_chw(x_chw: np.ndarray, title: str, rgb_idx=(0, 1, 2), p_low=1, p_high=99):
    """Visualize CHW input. If C>=3 show RGB via indices; else show grayscale."""
    if x_chw.ndim != 3:
        print(f"Skip {title}: expected CHW, got shape={x_chw.shape}")
        return

    c, h, w = x_chw.shape
    print(f"{title:40s} shape={x_chw.shape} dtype={x_chw.dtype} min={x_chw.min():.3g} max={x_chw.max():.3g}")

    plt.figure(figsize=(6, 6))
    if c >= 3:
        r, g, b = rgb_idx
        if max(rgb_idx) >= c:
            raise ValueError(f"rgb_idx={rgb_idx} out of range for C={c}")

        rgb = np.stack([x_chw[r], x_chw[g], x_chw[b]], axis=-1)  # HWC
        rgb = percentile_stretch(rgb, p_low=p_low, p_high=p_high, gamma=1.0)
        plt.imshow(rgb)
    else:
        plt.imshow(x_chw[0], cmap="gray")

    plt.title(f"{title}  (C={c}, H={h}, W={w})")
    plt.axis("off")
    plt.show()
