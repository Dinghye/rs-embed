from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from ...core.errors import ModelError
from ...core.export_helpers import jsonable as _jsonable
from ...core.specs import SensorSpec, SpatialSpec, TemporalSpec
from ...providers.base import ProviderBase


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


def fetch_provider_patch_raw(
    provider: ProviderBase,
    *,
    spatial: SpatialSpec,
    temporal: Optional[TemporalSpec],
    sensor: SensorSpec,
) -> np.ndarray:
    x = provider.fetch_sensor_patch_chw(
        spatial=spatial,
        temporal=temporal,
        sensor=sensor,
    )
    return normalize_input_chw(
        x,
        expected_channels=len(sensor.bands),
        name=f"gee_input[{sensor.collection}]",
    )


# Backwards-compatible alias kept for existing imports/tests.
fetch_gee_patch_raw = fetch_provider_patch_raw


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
