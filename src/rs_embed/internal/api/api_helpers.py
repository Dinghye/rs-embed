from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from ...core.errors import ModelError
from ...core.export_helpers import jsonable as _jsonable
from ...core.gee_image import build_gee_image as _build_gee_image
from ...core.specs import SensorSpec, SpatialSpec, TemporalSpec
from ...providers.gee import GEEProvider


def normalize_model_name(model: str) -> str:
    return str(model).strip().lower()


def normalize_backend_name(backend: str) -> str:
    return str(backend).strip().lower()


def normalize_device_name(device: Optional[str]) -> str:
    if device is None:
        return "auto"
    dev = str(device).strip().lower()
    return dev or "auto"


def normalize_input_chw(
    x_chw: np.ndarray,
    *,
    expected_channels: Optional[int] = None,
    name: str = "input_chw",
) -> np.ndarray:
    x = np.asarray(x_chw, dtype=np.float32)
    if x.ndim != 3:
        raise ModelError(f"{name} must be CHW with ndim=3, got shape={getattr(x, 'shape', None)}")
    if expected_channels is not None and int(x.shape[0]) != int(expected_channels):
        raise ModelError(
            f"{name} channel mismatch: got C={int(x.shape[0])}, expected C={int(expected_channels)}"
        )
    return x


def fetch_gee_patch_raw(
    provider: GEEProvider,
    *,
    spatial: SpatialSpec,
    temporal: Optional[TemporalSpec],
    sensor: SensorSpec,
) -> np.ndarray:
    region = provider.get_region_3857(spatial)
    img = _build_gee_image(sensor=sensor, temporal=temporal, region=region)
    x = provider.fetch_array_chw(
        image=img,
        bands=sensor.bands,
        region=region,
        scale_m=int(sensor.scale_m),
        fill_value=float(sensor.fill_value),
        collection=sensor.collection,
    )
    return normalize_input_chw(
        x,
        expected_channels=len(sensor.bands),
        name=f"gee_input[{sensor.collection}]",
    )


def inspect_input_raw(x_chw: np.ndarray, *, sensor: SensorSpec, name: str) -> Dict[str, Any]:
    from ...core.input_checks import inspect_chw

    x = normalize_input_chw(
        x_chw,
        expected_channels=len(sensor.bands),
        name=name,
    )
    rep = inspect_chw(
        x,
        name=name,
        expected_channels=len(sensor.bands),
        value_range=None,
        fill_value=float(sensor.fill_value),
    )
    return {"ok": bool(rep.get("ok", False)), "report": rep, "sensor": _jsonable(sensor)}
