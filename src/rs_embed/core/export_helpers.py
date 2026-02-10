from __future__ import annotations

import datetime as _dt
import hashlib
import json
import re
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Iterable, Optional

import numpy as np

from .embedding import Embedding
from .registry import get_embedder_cls
from .specs import SensorSpec


def utc_ts() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def sanitize_key(s: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "item"


def sha1(arr: np.ndarray, max_bytes: int = 2_000_000) -> str:
    h = hashlib.sha1()
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(str(arr.shape).encode("utf-8"))
    b = arr.tobytes(order="C")
    if len(b) > max_bytes:
        b = b[:max_bytes]
    h.update(b)
    return h.hexdigest()


def jsonable(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return {
            "__ndarray__": True,
            "dtype": str(obj.dtype),
            "shape": list(obj.shape),
            "min": float(np.nanmin(obj)) if obj.size else None,
            "max": float(np.nanmax(obj)) if obj.size else None,
        }
    if is_dataclass(obj):
        return jsonable(asdict(obj))
    try:
        import xarray as xr
        if isinstance(obj, xr.DataArray):
            return {"__xarray__": True, "dtype": str(obj.dtype), "shape": list(obj.shape), "dims": list(obj.dims)}
    except Exception:
        pass
    return repr(obj)


def embedding_to_numpy(emb: Embedding) -> np.ndarray:
    if isinstance(emb.data, np.ndarray):
        return emb.data.astype(np.float32, copy=False)
    try:
        import xarray as xr
        if isinstance(emb.data, xr.DataArray):
            return np.asarray(emb.data.values, dtype=np.float32)
    except Exception:
        pass
    return np.asarray(emb.data, dtype=np.float32)


def default_sensor_for_model(model_id: str) -> Optional[SensorSpec]:
    cls = get_embedder_cls(model_id)
    try:
        desc = cls().describe() or {}
    except Exception:
        desc = {}

    typ = str(desc.get("type", "")).lower()
    if "precomputed" in typ:
        return None

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

    if isinstance(inputs, dict) and "collection" in inputs and "bands" in inputs:
        return _mk(inputs["collection"], inputs["bands"])
    if isinstance(inputs, dict) and "s2_sr" in inputs:
        s2 = inputs["s2_sr"]
        if isinstance(s2, dict) and "collection" in s2 and "bands" in s2:
            return _mk(s2["collection"], s2["bands"])
    if "input_bands" in desc:
        return _mk("COPERNICUS/S2_SR_HARMONIZED", desc["input_bands"])

    return None


def sensor_cache_key(sensor: SensorSpec) -> str:
    obj = {
        "collection": sensor.collection,
        "bands": list(sensor.bands),
        "scale_m": int(sensor.scale_m),
        "cloudy_pct": int(sensor.cloudy_pct),
        "fill_value": float(sensor.fill_value),
        "composite": str(sensor.composite),
    }
    data = json.dumps(obj, sort_keys=True).encode("utf-8")
    return sanitize_key(hashlib.sha1(data).hexdigest()[:12])
