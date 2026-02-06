"""Format-specific writers for embedding export.

Each writer persists:
  - arrays   : Dict[str, np.ndarray]  — named arrays (inputs / embeddings)
  - manifest : Dict[str, Any]         — JSON-serializable metadata

Supported formats
-----------------
- **npz**    – ``numpy.savez_compressed`` + sidecar ``.json`` manifest.
- **netcdf** – CF-flavored NetCDF file with named dimensions + global attrs.
               Requires one of: ``netCDF4``, ``h5netcdf``, or ``scipy``.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Tuple

import numpy as np

# ── format → file extension mapping ────────────────────────────────

_FORMAT_EXT: Dict[str, str] = {
    "npz": ".npz",
    "netcdf": ".nc",
}

SUPPORTED_FORMATS = tuple(_FORMAT_EXT.keys())


def get_extension(fmt: str) -> str:
    """Return the canonical file extension (including dot) for *fmt*."""
    try:
        return _FORMAT_EXT[fmt]
    except KeyError:
        raise ValueError(f"Unknown format {fmt!r}. Supported: {SUPPORTED_FORMATS}")


# ── public dispatcher ──────────────────────────────────────────────

def write_arrays(
    *,
    fmt: str,
    out_path: str,
    arrays: Dict[str, np.ndarray],
    manifest: Dict[str, Any],
    save_manifest: bool,
) -> Dict[str, Any]:
    """Persist *arrays* + *manifest* in the requested format.

    Returns an updated copy of *manifest* with format-specific path keys.
    """
    if fmt == "npz":
        return _write_npz(out_path, arrays, manifest, save_manifest)
    if fmt == "netcdf":
        return _write_netcdf(out_path, arrays, manifest, save_manifest)
    raise ValueError(f"Unknown format {fmt!r}. Supported: {SUPPORTED_FORMATS}")


# ── NPZ writer ────────────────────────────────────────────────────

def _write_npz(
    out_path: str,
    arrays: Dict[str, np.ndarray],
    manifest: Dict[str, Any],
    save_manifest: bool,
) -> Dict[str, Any]:
    if not out_path.endswith(".npz"):
        out_path += ".npz"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    np.savez_compressed(out_path, **arrays)

    if save_manifest:
        json_path = os.path.splitext(out_path)[0] + ".json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        manifest["manifest_path"] = json_path

    manifest["npz_path"] = out_path
    manifest["npz_keys"] = sorted(arrays.keys())
    return manifest


# ── NetCDF writer ──────────────────────────────────────────────────

def _write_netcdf(
    out_path: str,
    arrays: Dict[str, np.ndarray],
    manifest: Dict[str, Any],
    save_manifest: bool,
) -> Dict[str, Any]:
    import xarray as xr

    if not out_path.endswith(".nc"):
        out_path += ".nc"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # Build xarray Dataset with semantically-named dimensions.
    data_vars: Dict[str, xr.DataArray] = {}
    for key, arr in arrays.items():
        dims = _infer_dims(key, arr)
        data_vars[key] = xr.DataArray(data=arr, dims=dims)

    ds = xr.Dataset(data_vars)

    # Embed useful global attributes (CF-like).
    ds.attrs["Conventions"] = "CF-1.8"
    ds.attrs["history"] = "Created by rs-embed export_batch (format=netcdf)"
    for attr in ("created_at", "backend", "device"):
        val = manifest.get(attr)
        if val is not None:
            ds.attrs[attr] = str(val)
    if "n_items" in manifest:
        ds.attrs["n_items"] = int(manifest["n_items"])

    engine = _pick_engine()
    ds.to_netcdf(out_path, engine=engine)

    if save_manifest:
        json_path = os.path.splitext(out_path)[0] + ".json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        manifest["manifest_path"] = json_path

    manifest["nc_path"] = out_path
    manifest["nc_variables"] = sorted(arrays.keys())
    return manifest


# ── dimension inference ────────────────────────────────────────────

def _infer_dims(key: str, arr: np.ndarray) -> Tuple[str, ...]:
    """Map array key + shape to semantically named NetCDF dimensions.

    Convention used by the NPZ export:
        input_chw__<model>            → (band, y, x)
        inputs_bchw__<model>          → (point, band, y, x)
        embedding__<model>            → (dim,)          for pooled
        embedding__<model>            → (band, y, x)    for grid
        embeddings__<model>           → (point, dim)    for pooled batch
        embeddings__<model>           → (point, band, y, x) for grid batch
    """
    ndim = arr.ndim

    if "bchw" in key:
        if ndim == 4:
            return ("point", "band", "y", "x")

    if "chw" in key:
        if ndim == 3:
            return ("band", "y", "x")

    if "embeddings" in key:
        if ndim == 2:
            return ("point", "dim")
        if ndim == 4:
            return ("point", "band", "y", "x")

    if "embedding" in key:
        if ndim == 1:
            return ("dim",)
        if ndim == 3:
            return ("band", "y", "x")

    # Fallback: generic numbered dimensions.
    return tuple(f"d{i}" for i in range(ndim))


# ── engine selection ───────────────────────────────────────────────

def _pick_engine() -> str:
    """Return the best available xarray NetCDF engine."""
    for engine, pkg in [("netcdf4", "netCDF4"), ("h5netcdf", "h5netcdf"), ("scipy", "scipy")]:
        try:
            __import__(pkg)
            return engine
        except ImportError:
            continue
    raise ImportError(
        "No NetCDF engine available. Install one of: netCDF4, h5netcdf, or scipy.\n"
        "  pip install netCDF4     (recommended)\n"
        "  pip install h5netcdf\n"
        "  pip install scipy"
    )
