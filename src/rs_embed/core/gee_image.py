from __future__ import annotations

from typing import Any, Optional, Tuple

from .errors import ModelError
from .specs import SensorSpec, TemporalSpec


def build_gee_image(
    *,
    sensor: SensorSpec,
    temporal: Optional[TemporalSpec],
    region: Optional[Any] = None,
) -> Any:
    """Build an ee.Image from SensorSpec and TemporalSpec.

    If `sensor.collection` is an ImageCollection, we filter and composite it.
    Otherwise, we fall back to an ee.Image asset id.
    """
    import ee

    temporal_range: Optional[Tuple[str, str]] = None
    if temporal is not None:
        temporal.validate()
        if temporal.mode == "range":
            temporal_range = (temporal.start, temporal.end)
        elif temporal.mode == "year":
            y = int(temporal.year)
            temporal_range = (f"{y}-01-01", f"{y+1}-01-01")
        else:
            raise ModelError(f"Unknown TemporalSpec mode: {temporal.mode}")

    try:
        ic = ee.ImageCollection(sensor.collection)
        if region is not None:
            ic = ic.filterBounds(region)
        if temporal_range is not None:
            ic = ic.filterDate(temporal_range[0], temporal_range[1])
        if sensor.cloudy_pct is not None:
            try:
                ic = ic.filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", int(sensor.cloudy_pct)))
            except Exception:
                pass

        if sensor.composite == "median":
            img = ic.median()
        elif sensor.composite == "mosaic":
            img = ic.mosaic()
        else:
            img = ic.median()
    except Exception:
        img = ee.Image(sensor.collection)

    return img
