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

from .api import get_embedding
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
    """Export inputs + embeddings for one spatial/temporal query.

    Parameters
    ----------
    spatial, temporal
        Query area and time window.
    models
        List of model IDs to run.
    out_path
        Path to the `.npz` file to write. A `.json` manifest will be written
        next to it if `save_manifest=True`.
    sensor
        Optional SensorSpec override applied to *all* models (both for
        input fetching and embedding computation). If you pass this, make sure
        it matches each model's expectations.
    per_model_sensors
        Optional per-model SensorSpec overrides. Overrides `sensor` for the
        specified model IDs.
    save_inputs
        If True, saves raw GEE patches (CHW) for on-the-fly models.
    save_embeddings
        If True, saves embedding arrays for each model.
    fail_on_bad_input
        If True, raise if the input inspection report returns ok=False.

    Returns
    -------
    dict
        The JSON-serializable manifest (also written to disk if enabled).
    """

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if not out_path.endswith(".npz"):
        out_path = out_path + ".npz"

    per_model_sensors = per_model_sensors or {}

    arrays: Dict[str, np.ndarray] = {}
    manifest: Dict[str, Any] = {
        "created_at": _utc_ts(),
        "backend": backend,
        "device": device,
        "models": [],
        "spatial": _jsonable(spatial),
        "temporal": _jsonable(temporal),
        "output": _jsonable(output),
    }

    # Best-effort package version
    try:
        from importlib.metadata import version

        manifest["package_version"] = version("rs-embed")
    except Exception:
        manifest["package_version"] = None

    # Cache raw inputs so multiple models sharing the same SensorSpec do not re-download.
    _input_cache: Dict[str, Tuple[np.ndarray, Dict[str, Any]]] = {}

    for model_id in models:
        model_entry: Dict[str, Any] = {"model": model_id}

        # Resolve sensor for this model
        m_sensor = per_model_sensors.get(model_id) or sensor or _default_sensor_for_model(model_id)
        model_entry["sensor"] = _jsonable(m_sensor)

        # 1) Save raw input patch (on-the-fly only)
        input_report = None
        input_key = None
        if save_inputs and backend.lower() == "gee" and m_sensor is not None:
            # Use inspect_gee_patch to fetch raw patch and run checks (cached by query + SensorSpec).
            cache_obj = {
                'backend': 'gee',
                'spatial': _jsonable(spatial),
                'temporal': _jsonable(temporal),
                'sensor': _jsonable(m_sensor),
            }
            cache_key = json.dumps(cache_obj, ensure_ascii=False, sort_keys=True)
            if cache_key in _input_cache:
                x_chw, input_report = _input_cache[cache_key]
            else:
                insp = inspect_gee_patch(
                    spatial=spatial,
                    temporal=temporal,
                    sensor=m_sensor,
                    backend='gee',
                    name=f'input_{_sanitize_key(model_id)}',
                    value_range=None,
                    return_array=True,
                )
                x_chw = insp.pop('array_chw')
                input_report = insp
                _input_cache[cache_key] = (np.asarray(x_chw, dtype=np.float32), input_report)

            input_key = f"input_chw__{_sanitize_key(model_id)}"
            arrays[input_key] = np.asarray(x_chw, dtype=np.float32)

            model_entry["input"] = {
                "npz_key": input_key,
                "dtype": str(arrays[input_key].dtype),
                "shape": list(arrays[input_key].shape),
                "sha1": _sha1(arrays[input_key]),
                "inspection": _jsonable(input_report),
            }

            if fail_on_bad_input and input_report is not None and (not bool(input_report.get("ok", True))):
                issues = (input_report.get("report", {}) or {}).get("issues", [])
                raise RuntimeError(f"Input inspection failed for model={model_id}: {issues}")

        else:
            model_entry["input"] = None

        # 2) Save embeddings
        if save_embeddings:
            emb = get_embedding(
                model_id,
                spatial=spatial,
                temporal=temporal,
                sensor=m_sensor,
                output=output,
                backend=backend,
                device=device,
            )
            e_np = _embedding_to_numpy(emb)
            emb_key = f"embedding__{_sanitize_key(model_id)}"
            arrays[emb_key] = e_np

            model_entry["embedding"] = {
                "npz_key": emb_key,
                "dtype": str(e_np.dtype),
                "shape": list(e_np.shape),
                "sha1": _sha1(e_np),
            }
            # Store meta (sanitized)
            model_entry["meta"] = _jsonable(emb.meta)

        else:
            model_entry["embedding"] = None
            model_entry["meta"] = None

        # Include describe() for transparency
        try:
            cls = get_embedder_cls(model_id)
            model_entry["describe"] = _jsonable(cls().describe())
        except Exception as e:
            model_entry["describe"] = {"error": repr(e)}

        manifest["models"].append(model_entry)

    # Write npz
    np.savez_compressed(out_path, **arrays)

    # Write manifest
    if save_manifest:
        json_path = os.path.splitext(out_path)[0] + ".json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(_jsonable(manifest), f, ensure_ascii=False, indent=2)
        manifest["manifest_path"] = json_path

    manifest["npz_path"] = out_path
    manifest["npz_keys"] = sorted(list(arrays.keys()))

    return _jsonable(manifest)


# Backwards-compatible alias (internal)
export_to_npz = export_npz
