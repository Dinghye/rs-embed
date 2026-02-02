from __future__ import annotations

"""Lightweight, on-the-fly input image inspection.

This module is intentionally dependency-light (numpy only by default).
It is used to sanity-check patches downloaded from Google Earth Engine (GEE)
right before they are fed into on-the-fly embedders.

You can enable checks in two ways:

1) Per-request via SensorSpec:
   SensorSpec(..., check_input=True)

2) Globally via environment variables:
   RS_EMBED_CHECK_INPUT=1
   RS_EMBED_CHECK_RAISE=1
   RS_EMBED_CHECK_SAVE_DIR=/tmp/rs_embed_checks

When enabled, we return a report dict that can be attached into embedding meta.
If `check_raise` is enabled and issues are detected, embedders may raise.
"""

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np


def _env_flag(name: str, default: str = "0") -> bool:
    v = os.environ.get(name, default)
    return str(v).strip().lower() not in ("", "0", "false", "no", "off")


def checks_enabled(sensor: Any = None) -> bool:
    """Return True if input checks should run."""
    if _env_flag("RS_EMBED_CHECK_INPUT", "0"):
        return True
    return bool(getattr(sensor, "check_input", False))


def checks_should_raise(sensor: Any = None) -> bool:
    """Return True if embedders should raise on detected issues."""
    if _env_flag("RS_EMBED_CHECK_RAISE", "1"):
        return True
    return bool(getattr(sensor, "check_raise", True))


def checks_save_dir(sensor: Any = None) -> Optional[str]:
    """Optional directory to save quicklooks/stat dumps."""
    d = os.environ.get("RS_EMBED_CHECK_SAVE_DIR")
    if d:
        return str(d)
    return getattr(sensor, "check_save_dir", None)


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def inspect_chw(
    x_chw: np.ndarray,
    *,
    name: str = "input",
    expected_channels: Optional[int] = None,
    value_range: Optional[Tuple[float, float]] = None,
    fill_value: Optional[float] = None,
    max_pixels_for_full_stats: int = 1_500_000,
    quantiles: Tuple[float, ...] = (0.01, 0.5, 0.99),
    hist_bins: int = 32,
    hist_clip_range: Optional[Tuple[float, float]] = None,
    max_bands_for_hist: int = 16,
) -> Dict[str, Any]:
    """Inspect a CHW numpy array and return a compact report.

    Parameters
    ----------
    x_chw:
        Input array with shape [C,H,W].
    expected_channels:
        If set, we flag a mismatch.
    value_range:
        If set, we compute fraction of values outside [lo, hi].
    fill_value:
        If set, we compute fraction of pixels equal to fill_value.
    max_pixels_for_full_stats:
        To keep inspection cheap, if C*H*W exceeds this value we downsample
        (strided sampling) for per-band stats.
    """
    report: Dict[str, Any] = {
        "name": name,
        "ok": True,
        "issues": [],
        "shape": tuple(int(i) for i in getattr(x_chw, "shape", ())),
        "dtype": str(getattr(x_chw, "dtype", None)),
    }

    # Basic shape checks
    if not isinstance(x_chw, np.ndarray):
        report["ok"] = False
        report["issues"].append(f"{name}: not a numpy array")
        return report

    if x_chw.ndim != 3:
        report["ok"] = False
        report["issues"].append(f"{name}: expected CHW with ndim=3, got ndim={x_chw.ndim}")
        return report

    c, h, w = (int(x_chw.shape[0]), int(x_chw.shape[1]), int(x_chw.shape[2]))
    if expected_channels is not None and c != int(expected_channels):
        report["ok"] = False
        report["issues"].append(f"{name}: channel mismatch (C={c}, expected {expected_channels})")

    if h <= 0 or w <= 0:
        report["ok"] = False
        report["issues"].append(f"{name}: non-positive H/W ({h},{w})")
        return report

    # Downsample if huge (strided sampling across spatial dims)
    x = x_chw
    total = int(c) * int(h) * int(w)
    if total > int(max_pixels_for_full_stats):
        stride = int(np.ceil(np.sqrt(total / max_pixels_for_full_stats)))
        stride = max(stride, 1)
        x = x_chw[:, ::stride, ::stride]
        report["downsample_stride"] = stride

    xf = x.astype(np.float32, copy=False)

    finite = np.isfinite(xf)
    finite_frac = float(finite.mean())
    report["finite_frac"] = finite_frac
    if finite_frac < 0.999:
        report["ok"] = False
        n_bad = int((~finite).sum())
        report["issues"].append(f"{name}: contains NaN/Inf (count≈{n_bad} on sampled data)")

    # Replace non-finite for stats
    xf2 = np.where(finite, xf, np.nan)

    # Per-band stats
    bmin = np.nanmin(xf2, axis=(1, 2))
    bmax = np.nanmax(xf2, axis=(1, 2))
    bmean = np.nanmean(xf2, axis=(1, 2))
    bstd = np.nanstd(xf2, axis=(1, 2))

    report["band_min"] = [float(v) for v in bmin]
    report["band_max"] = [float(v) for v in bmax]
    report["band_mean"] = [float(v) for v in bmean]
    report["band_std"] = [float(v) for v in bstd]

    # Robust distribution summaries (useful for catching saturation, empty patches, or scaling issues)
    if quantiles:
        try:
            qs = tuple(float(q) for q in quantiles)
            qv = np.nanquantile(xf2, qs, axis=(1, 2))  # [Q, C]
            # Store as a dict of per-band lists, keyed by quantile
            report["band_quantiles"] = {
                f"p{int(round(q*100)):02d}": [float(v) for v in qv[i]] for i, q in enumerate(qs)
            }
        except Exception as e:
            # Quantiles are non-critical; keep running.
            report.setdefault("warnings", [])
            report["warnings"].append(f"{name}: failed to compute quantiles: {e!r}")

    # Optional per-band histograms (kept compact; mainly for debugging)
    if int(hist_bins) > 0 and c <= int(max_bands_for_hist):
        try:
            # Clip hist range if requested; otherwise infer from robust quantiles when possible.
            hr = None
            if hist_clip_range is not None:
                hr = (float(hist_clip_range[0]), float(hist_clip_range[1]))
            elif "band_quantiles" in report:
                # Use p01..p99 if present to avoid extreme outliers dominating bins
                qd = report["band_quantiles"]
                # fall back gracefully if keys differ
                lo_key = next((k for k in ("p01", "p00") if k in qd), None)
                hi_key = next((k for k in ("p99", "p100") if k in qd), None)
                if lo_key and hi_key:
                    lo = float(np.nanmin(qd[lo_key]))
                    hi = float(np.nanmax(qd[hi_key]))
                    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                        hr = (lo, hi)

            # Compute shared bin edges across bands for easy comparison
            if hr is None:
                glo = float(np.nanmin(bmin))
                ghi = float(np.nanmax(bmax))
                if np.isfinite(glo) and np.isfinite(ghi) and ghi > glo:
                    hr = (glo, ghi)

            if hr is not None:
                edges = np.linspace(hr[0], hr[1], int(hist_bins) + 1, dtype=np.float32)
                counts = []
                for bi in range(c):
                    v = xf2[bi].ravel()
                    v = v[np.isfinite(v)]
                    if v.size == 0:
                        counts.append([0] * int(hist_bins))
                        continue
                    h, _ = np.histogram(v, bins=edges)
                    counts.append([int(x) for x in h.tolist()])

                report["hist"] = {
                    "bins": [float(x) for x in edges.tolist()],
                    "counts": counts,
                    "range": [float(hr[0]), float(hr[1])],
                }
        except Exception as e:
            report.setdefault("warnings", [])
            report["warnings"].append(f"{name}: failed to compute histogram: {e!r}")

    # Per-band quantiles (cheap + very diagnostic)
    qs = tuple(float(q) for q in quantiles) if quantiles else ()
    if qs:
        try:
            qv = np.nanquantile(xf2, qs, axis=(1, 2))  # [Q, C]
            for qi, q in enumerate(qs):
                key = f"band_p{int(round(q * 100)):02d}"
                report[key] = [float(v) for v in qv[qi]]
            report["quantiles"] = list(qs)
        except Exception as e:
            # Don't fail inspection due to quantile issues
            report.setdefault("warnings", []).append(f"{name}: quantiles failed: {e!r}")

    # Per-band histograms (optional; keep compact)
    # We store one shared set of bin edges + per-band counts.
    if int(hist_bins) > 0 and c <= int(max_bands_for_hist):
        try:
            # Determine histogram range
            if hist_clip_range is not None:
                h_lo, h_hi = float(hist_clip_range[0]), float(hist_clip_range[1])
            elif value_range is not None:
                h_lo, h_hi = float(value_range[0]), float(value_range[1])
            else:
                # Robust range based on quantiles if available
                if "band_p01" in report and "band_p99" in report:
                    h_lo = float(np.nanmin(np.array(report["band_p01"], dtype=np.float32)))
                    h_hi = float(np.nanmax(np.array(report["band_p99"], dtype=np.float32)))
                else:
                    h_lo = float(np.nanmin(bmin))
                    h_hi = float(np.nanmax(bmax))

            if not np.isfinite(h_lo) or not np.isfinite(h_hi) or (h_hi <= h_lo):
                raise ValueError(f"bad hist range ({h_lo},{h_hi})")

            edges = np.linspace(h_lo, h_hi, int(hist_bins) + 1, dtype=np.float32)
            counts = []
            for bi in range(c):
                v = xf2[bi].ravel()
                v = v[np.isfinite(v)]
                if v.size == 0:
                    counts.append([0] * int(hist_bins))
                    continue
                h, _ = np.histogram(v, bins=edges)
                counts.append([int(x) for x in h])

            report["hist_bins"] = [float(x) for x in edges]
            report["band_hist"] = counts
            report["hist_range"] = [float(h_lo), float(h_hi)]
        except Exception as e:
            report.setdefault("warnings", []).append(f"{name}: histogram failed: {e!r}")

    # Constant / near-constant bands are often a sign of empty ROI, fill, or bad reprojection
    const = (bstd < 1e-6) | (~np.isfinite(bstd))
    if bool(const.any()):
        report["ok"] = False
        idx = np.where(const)[0].tolist()
        report["issues"].append(f"{name}: near-constant bands at indices {idx}")

    # Value range checks
    if value_range is not None:
        lo, hi = float(value_range[0]), float(value_range[1])
        outside = (xf2 < lo) | (xf2 > hi)
        outside_frac = float(np.nanmean(outside))
        report["outside_range_frac"] = outside_frac
        if outside_frac > 0.001:
            report["ok"] = False
            report["issues"].append(
                f"{name}: values outside range [{lo},{hi}] (frac≈{outside_frac:.4f} on sampled data)"
            )

    # Fill ratio checks
    fv = _safe_float(fill_value)
    if fv is not None:
        # equality on float is OK here because fill_value is often 0.0
        fill_mask = (xf2 == fv)
        fill_frac = float(np.nanmean(fill_mask))
        report["fill_value"] = fv
        report["fill_frac"] = fill_frac
        if fill_frac > 0.98:
            report["ok"] = False
            report["issues"].append(
                f"{name}: almost all pixels are fill_value={fv} (frac≈{fill_frac:.4f} on sampled data)"
            )

    return report


def maybe_inspect_chw(
    x_chw: np.ndarray,
    *,
    sensor: Any = None,
    name: str = "input",
    expected_channels: Optional[int] = None,
    value_range: Optional[Tuple[float, float]] = None,
    fill_value: Optional[float] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Run inspect_chw if enabled; optionally attach report to meta.

    Returns the report dict (or None if checks disabled).
    """
    if not checks_enabled(sensor):
        return None

    report = inspect_chw(
        x_chw,
        name=name,
        expected_channels=expected_channels,
        value_range=value_range,
        fill_value=fill_value,
    )

    if meta is not None:
        # Avoid huge meta: keep a single report object under a stable key.
        meta.setdefault("input_checks", {})
        meta["input_checks"][name] = report

        # Store the inspection config for reproducibility
        meta.setdefault("input_checks_config", {})
        meta["input_checks_config"].update(
            {
                "enabled": True,
                "raise": checks_should_raise(sensor),
                "save_dir": checks_save_dir(sensor),
            }
        )

    return report


def save_quicklook_rgb(
    x_chw: np.ndarray,
    *,
    path: str,
    bands: Tuple[int, int, int] = (0, 1, 2),
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> None:
    """Save a simple RGB quicklook PNG/JPG using matplotlib.

    This is optional and only used when a save_dir is configured.
    """
    import matplotlib.pyplot as plt

    if x_chw.ndim != 3:
        raise ValueError(f"Expected CHW, got {x_chw.shape}")
    c, h, w = x_chw.shape
    r, g, b = bands
    if max(r, g, b) >= c:
        raise ValueError(f"bands={bands} out of range for C={c}")

    rgb = np.stack([x_chw[r], x_chw[g], x_chw[b]], axis=-1).astype(np.float32)
    rgb = np.clip((rgb - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.figure()
    plt.imshow(rgb)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close()
