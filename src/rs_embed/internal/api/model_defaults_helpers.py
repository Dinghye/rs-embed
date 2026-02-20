from __future__ import annotations

from typing import Iterable, Optional

from ...core.registry import get_embedder_cls
from ...core.specs import SensorSpec


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
