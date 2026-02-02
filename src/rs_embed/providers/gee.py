
from __future__ import annotations

from typing import Any, Tuple

import numpy as np
from pyproj import Transformer

from ..core.errors import ProviderError
from ..core.specs import BBox, PointBuffer, SpatialSpec
from .base import ProviderBase


_ALIAS_S2 = {
    "BLUE": "B2",
    "GREEN": "B3",
    "RED": "B4",

    # NIR
    "NIR": "B8",
    "NIR_BROAD": "B8",
    "NIR_WIDE": "B8",

    # Narrow NIR (S2 band B8A)
    "NIR_NARROW": "B8A",
    "NIRN": "B8A",
    "NIRNARROW": "B8A",
    "NIR_N": "B8A",

    # Red edge (optional but common)
    "RE1": "B5",
    "RED_EDGE_1": "B5",
    "RE2": "B6",
    "RED_EDGE_2": "B6",
    "RE3": "B7",
    "RED_EDGE_3": "B7",
    "RE4": "B8A",
    "RED_EDGE_4": "B8A",

    # SWIR
    "SWIR1": "B11",
    "SWIR_1": "B11",
    "SWIR2": "B12",
    "SWIR_2": "B12",
}
_ALIAS_LS89_SR = {
    "BLUE": "SR_B2",
    "GREEN": "SR_B3",
    "RED": "SR_B4",
    "NIR": "SR_B5",
    "SWIR1": "SR_B6",
    "SWIR2": "SR_B7",
}

_ALIAS_LS457_SR = {
    "BLUE": "SR_B1",
    "GREEN": "SR_B2",
    "RED": "SR_B3",
    "NIR": "SR_B4",
    "SWIR1": "SR_B5",
    "SWIR2": "SR_B7",
}


def _resolve_band_aliases(collection: str, bands: Tuple[str, ...]) -> Tuple[str, ...]:
    """Resolve semantic band aliases to real band names based on collection id."""
    if not bands:
        return bands

    c = (collection or "").upper()
    # Sentinel-2 (SR/TOA/HARMONIZED etc.)
    if "COPERNICUS/S2" in c:
        amap = _ALIAS_S2
    # Landsat Collection 2 L2 SR (typical ids)
    elif "LANDSAT/LC08/C02/T1_L2" in c or "LANDSAT/LC09/C02/T1_L2" in c:
        amap = _ALIAS_LS89_SR
    elif "LANDSAT/LE07/C02/T1_L2" in c or "LANDSAT/LT05/C02/T1_L2" in c or "LANDSAT/LT04/C02/T1_L2" in c:
        amap = _ALIAS_LS457_SR
    else:
        # Unknown collection: do not map
        amap = {}

    out = []
    for b in bands:
        key = (b or "").upper()
        out.append(amap.get(key, b))
    return tuple(out)


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
                raise ProviderError(
                    "Earth Engine not initialized. Run `earthengine authenticate` and try again."
                )
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
            return ee.Geometry.Rectangle(
                [minx, miny, maxx, maxy], proj="EPSG:3857", geodesic=False
            )

        if isinstance(spatial, BBox):
            spatial.validate()
            minx, miny = to_3857.transform(spatial.minlon, spatial.minlat)
            maxx, maxy = to_3857.transform(spatial.maxlon, spatial.maxlat)
            return ee.Geometry.Rectangle(
                [minx, miny, maxx, maxy], proj="EPSG:3857", geodesic=False
            )

        raise ProviderError(f"Unsupported spatial type: {type(spatial)}")

    def get_region_3857(self, spatial: SpatialSpec):
        self.ensure_ready()
        return self._to_ee_region_3857(spatial)

    def fetch_array_chw(
        self,
        *,
        image: Any,
        bands: Tuple[str, ...],
        region: Any,
        scale_m: int,
        fill_value: float,
        collection: str | None = None,
    ) -> np.ndarray:
        """Download a rectangular patch as CHW array.

        - Resolves band aliases like BLUE/GREEN/RED -> B2/B3/B4 (S2) etc.
        - Forces deterministic pixel grid by reprojecting to EPSG:3857 at `scale_m`
          before sampleRectangle (prevents accidental (C,1,1)).
        """
        import ee

        # 1) Resolve aliases using collection hint when provided
        if collection:
            resolved = _resolve_band_aliases(collection, bands)
        else:
            # No collection hint: best effort — resolve nothing
            resolved = bands

        # 2) Select resolved bands (this will error at compute-time if typo)
        img = image.select(list(resolved))

        # 3) Force pixel grid at desired scale
        proj = ee.Projection("EPSG:3857").atScale(int(scale_m))
        img = img.reproject(proj).clip(region)

        # 4) Sample and build CHW
        rect = img.sampleRectangle(region=region, defaultValue=fill_value).getInfo()
        props = rect.get("properties", {})

        arrs = []
        missing = []
        for b in resolved:
            if b not in props:
                missing.append(b)
            else:
                arrs.append(np.array(props[b], dtype=np.float32))

        if missing:
            avail = sorted(list(props.keys()))
            raise ProviderError(
                f"Band(s) {missing} not in sampled properties. "
                f"Requested={resolved}. Available bands={avail}"
            )

        return np.stack(arrs, axis=0)