from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
from pyproj import Transformer
from shapely.geometry import box, Point

from ..core.errors import ProviderError
from ..core.specs import BBox, PointBuffer, SpatialSpec
from .base import ProviderBase

class GEEProvider(ProviderBase):
    name = "gee"

    def __init__(self, auto_auth: bool = True):
        self.auto_auth = auto_auth

    def ensure_ready(self) -> None:
        try:
            import ee
            ee.Initialize()
        except Exception:
            if not self.auto_auth:
                raise ProviderError("Earth Engine not initialized. Run `earthengine authenticate` and try again.")
            try:
                import geemap
                geemap.ee_initialize()
            except Exception as e:
                raise ProviderError(f"Failed to initialize GEE: {e!r}")

    def _to_ee_region_3857(self, spatial: SpatialSpec):
        import ee
        to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

        if isinstance(spatial, PointBuffer):
            spatial.validate()
            x, y = to_3857.transform(spatial.lon, spatial.lat)
            half = spatial.buffer_m
            minx, miny, maxx, maxy = x - half, y - half, x + half, y + half
            return ee.Geometry.Rectangle([minx, miny, maxx, maxy], proj="EPSG:3857", geodesic=False)

        if isinstance(spatial, BBox):
            spatial.validate()
            # bbox -> take center & compute approx half-size in 3857
            # v0.1: simplest approach: transform corners and build rectangle
            minx, miny = to_3857.transform(spatial.minlon, spatial.minlat)
            maxx, maxy = to_3857.transform(spatial.maxlon, spatial.maxlat)
            return ee.Geometry.Rectangle([minx, miny, maxx, maxy], proj="EPSG:3857", geodesic=False)

        raise ProviderError(f"Unsupported spatial type: {type(spatial)}")

    def fetch_array_chw(self, *, image: Any, bands: Tuple[str, ...], region: Any,
                        scale_m: int, fill_value: float) -> np.ndarray:
        """Download a rectangular patch as a CHW numpy array.

        Notes
        -----
        `ee.Image.sampleRectangle` does **not** accept a `scale` argument.
        Without explicitly setting a projection/scale on the image, Earth Engine
        may return a *single aggregated pixel* (often yielding arrays of shape
        (C, 1, 1)), which then fails our input checks.

        To make the output resolution deterministic, we reproject the image to
        EPSG:3857 at the requested `scale_m` before sampling.
        """

        import ee

        # Force deterministic pixel grid.
        proj = ee.Projection("EPSG:3857").atScale(int(scale_m))
        img = image.reproject(proj).clip(region)

        rect = img.sampleRectangle(region=region, defaultValue=fill_value).getInfo()
        props = rect["properties"]
        arrs = []
        for b in bands:
            if b not in props:
                raise ProviderError(f"Band '{b}' not in sampled properties.")
            arrs.append(np.array(props[b], dtype=np.float32))
        x_chw = np.stack(arrs, axis=0)
        return x_chw

    def get_region_3857(self, spatial: SpatialSpec):
        self.ensure_ready()
        return self._to_ee_region_3857(spatial)