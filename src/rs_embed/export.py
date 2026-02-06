from __future__ import annotations

"""Export utilities.

This module provides a high-level helper to export, for a given spatial/temporal
query and one or more model IDs:

1) The raw GEE patch used as model input (saved as CHW numpy arrays),
2) One embedding per model (pooled vectors or grids),
3) A JSON manifest describing inputs, embeddings, and all relevant metadata.

Artifacts
---------
- <name>.npz  : arrays (inputs + embeddings)
- <name>.json : human-readable manifest (serializable)

Notes
-----
- For on-the-fly models, the raw input patch is fetched via `inspect_gee_patch(..., return_array=True)`.
- Embeddings are computed via `get_embedding(...)`.
- Inputs are saved in *raw provider units* (e.g., Sentinel-2 SR typically 0..10000).
"""

import datetime as _dt
import hashlib
import json
import os
import re
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from .api import get_embedding, _get_embedder_bundle_cached, _sensor_key
from .core.embedding import Embedding
from .core.registry import get_embedder_cls
from .core.specs import OutputSpec, SensorSpec, SpatialSpec, TemporalSpec
from .inspect import inspect_gee_patch


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _utc_ts() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _sanitize_key(s: str) -> str:
    """Safe npz key: [A-Za-z0-9_]."""
    s = re.sub(r"[^0-9a-zA-Z_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "item"


def _sha1(arr: np.ndarray, max_bytes: int = 2_000_000) -> str:
    """Hash an array (best-effort) without huge memory spikes."""
    h = hashlib.sha1()
    # Hash dtype/shape too
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(str(arr.shape).encode("utf-8"))
    # Hash up to max_bytes of content (deterministic)
    b = arr.tobytes(order="C")
    if len(b) > max_bytes:
        b = b[:max_bytes]
    h.update(b)
    return h.hexdigest()


def _jsonable(obj: Any) -> Any:
    """Convert common non-JSON types into JSON-safe objects."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    # numpy scalars
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    # numpy arrays: summarize rather than inline
    if isinstance(obj, np.ndarray):
        return {
            "__ndarray__": True,
            "dtype": str(obj.dtype),
            "shape": list(obj.shape),
            "min": float(np.nanmin(obj)) if obj.size else None,
            "max": float(np.nanmax(obj)) if obj.size else None,
        }
    # dataclasses
    if is_dataclass(obj):
        return _jsonable(asdict(obj))
    # xarray
    try:
        import xarray as xr

        if isinstance(obj, xr.DataArray):
            return {
                "__xarray__": True,
                "dtype": str(obj.dtype),
                "shape": list(obj.shape),
                "dims": list(obj.dims),
            }
    except Exception:
        pass

    # Fallback: repr
    return repr(obj)


def _default_sensor_for_model(model_id: str) -> Optional[SensorSpec]:
    """Best-effort default SensorSpec derived from the embedder's `describe()`.

    This is only used for *on-the-fly* models when the caller didn't provide
    an explicit SensorSpec override.

    Returns None for precomputed models.
    """
    cls = get_embedder_cls(model_id)
    desc = cls().describe()
    typ = str(desc.get("type", "")).lower()
    if "precomputed" in typ:
        return None

    # Common structure: desc['inputs']
    inputs = desc.get("inputs")
    defaults = desc.get("defaults", {}) or {}

    def _mk(collection: str, bands: Iterable[str]) -> SensorSpec:
        return SensorSpec(
            collection=str(collection),
            bands=tuple(str(b) for b in bands),
            scale_m=int(defaults.get("scale_m", 10)),
            cloudy_pct=int(defaults.get("cloudy_pct", 30)),
            composite=str(defaults.get("composite", "median")),
            fill_value=float(defaults.get("fill_value", 0.0)),
        )

    # Case 1: {'collection': ..., 'bands': ...}
    if isinstance(inputs, dict) and "collection" in inputs and "bands" in inputs:
        return _mk(inputs["collection"], inputs["bands"])

    # Case 2: modality dict, choose S2 by default if present
    if isinstance(inputs, dict) and "s2_sr" in inputs:
        s2 = inputs["s2_sr"]
        if isinstance(s2, dict) and "collection" in s2 and "bands" in s2:
            return _mk(s2["collection"], s2["bands"])

    # Case 3: Prithvi-style: 'input_bands'
    if "input_bands" in desc:
        bands = desc["input_bands"]
        # assume Sentinel-2 SR unless specified elsewhere
        collection = "COPERNICUS/S2_SR_HARMONIZED"
        return _mk(collection, bands)

    # If we cannot infer, return None (caller may choose to still export embeddings)
    return None


def _embedding_to_numpy(emb: Embedding) -> np.ndarray:
    if isinstance(emb.data, np.ndarray):
        return emb.data.astype(np.float32, copy=False)
    # xarray
    try:
        import xarray as xr

        if isinstance(emb.data, xr.DataArray):
            return np.asarray(emb.data.values, dtype=np.float32)
    except Exception:
        pass
    return np.asarray(emb.data, dtype=np.float32)


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def export_npz(
    *,
    spatial: SpatialSpec,
    temporal: Optional[TemporalSpec],
    models: List[str],
    out_path: str,
    backend: str = "gee",
    device: str = "auto",
    output: OutputSpec = OutputSpec.pooled(),
    sensor: Optional[SensorSpec] = None,
    per_model_sensors: Optional[Dict[str, SensorSpec]] = None,
    save_inputs: bool = True,
    save_embeddings: bool = True,
    save_manifest: bool = True,
    fail_on_bad_input: bool = False,
) -> Dict[str, Any]:
    """Export inputs + embeddings for one spatial/temporal query as a single `.npz`.

    This is a thin wrapper over :func:`rs_embed.api.export_batch` (format='npz') and
    benefits from all the same performance improvements (embedder caching and
    avoiding duplicate GEE downloads when saving inputs + embeddings).
    """
    from .api import export_batch

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if not out_path.endswith(".npz"):
        out_path = out_path + ".npz"

    return export_batch(
        spatials=[spatial],
        temporal=temporal,
        models=models,
        out_path=out_path,
        backend=backend,
        device=device,
        output=output,
        sensor=sensor,
        per_model_sensors=per_model_sensors,
        format="npz",
        save_inputs=save_inputs,
        save_embeddings=save_embeddings,
        save_manifest=save_manifest,
        fail_on_bad_input=fail_on_bad_input,
    )

