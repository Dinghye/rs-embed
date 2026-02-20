from __future__ import annotations

"""Utilities for inspecting raw patches downloaded from providers.

This module is intentionally model-free: it lets you sanity-check the *input imagery*
you would feed into an on-the-fly embedder, without running the model.

Currently we support Google Earth Engine (backend="gee").
"""

from dataclasses import asdict
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .core.errors import ProviderError
from .core.gee_image import build_gee_image as _build_gee_image
from .core.input_checks import inspect_chw, checks_save_dir, save_quicklook_rgb
from .core.specs import SensorSpec, SpatialSpec, TemporalSpec
from .providers.gee import GEEProvider


def inspect_gee_patch(
    *,
    spatial: SpatialSpec,
    temporal: Optional[TemporalSpec] = None,
    sensor: SensorSpec,
    backend: str = "gee",
    name: str = "gee_patch",
    value_range: Optional[Tuple[float, float]] = None,
    return_array: bool = False,
) -> Dict[str, Any]:
    """Download a patch from GEE and return an input inspection report.

    This does **not** run any embedding model.

    Returns
    -------
    dict
        A JSON-serializable report. If `return_array=True`, the report also
        includes a non-serializable `array_chw` entry with the numpy array.
    """

    if backend != "gee":
        raise ProviderError(f"inspect_gee_patch currently only supports backend='gee', got {backend!r}")

    provider = GEEProvider(auto_auth=True)
    provider.ensure_ready()

    region = provider.get_region_3857(spatial)
    img = _build_gee_image(sensor=sensor, temporal=temporal, region=region)

    x_chw = provider.fetch_array_chw(
        image=img,
        bands=sensor.bands,
        region=region,
        scale_m=int(sensor.scale_m),
        fill_value=float(sensor.fill_value),
        collection=sensor.collection,
    )

    report = inspect_chw(
        x_chw,
        name=name,
        expected_channels=len(sensor.bands),
        value_range=value_range,
        fill_value=sensor.fill_value,
    )

    # Save quicklook if requested (best-effort)
    artifacts: Dict[str, Any] = {}
    save_dir = checks_save_dir(sensor)
    if save_dir and x_chw.ndim == 3 and x_chw.shape[0] >= 3:
        try:
            import os
            import datetime as _dt

            ts = _dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            path = os.path.join(save_dir, f"{name}_{ts}.png")
            save_quicklook_rgb(x_chw, path=path, bands=(0, 1, 2))
            artifacts["quicklook_rgb"] = path
        except Exception as e:
            artifacts["quicklook_rgb_error"] = repr(e)

    out: Dict[str, Any] = {
        "ok": bool(report.get("ok", False)),
        "report": report,
        "sensor": asdict(sensor),
        "temporal": asdict(temporal) if temporal is not None else None,
        "backend": backend,
        "artifacts": artifacts or None,
    }
    if return_array:
        out["array_chw"] = x_chw
    return out
