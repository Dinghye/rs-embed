from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..core.registry import register
from ..core.embedding import Embedding
from ..core.errors import ModelError
from ..core.specs import BBox, PointBuffer, SpatialSpec, TemporalSpec, SensorSpec, OutputSpec
from .base import EmbedderBase



def _parse_affine(transform) -> Tuple[float, float, float, float, float, float]:
    """
    Accepts:
      - rasterio.Affine
      - tuple/list of 6
      - string like 'Affine(a, b, c, d, e, f)'
    Returns (a,b,c,d,e,f) where:
      x = a*col + b*row + c
      y = d*col + e*row + f
    """
    # rasterio.Affine has attributes a,b,c,d,e,f
    for attr in ("a", "b", "c", "d", "e", "f"):
        if not hasattr(transform, attr):
            break
    else:
        return (float(transform.a), float(transform.b), float(transform.c),
                float(transform.d), float(transform.e), float(transform.f))

    if isinstance(transform, (tuple, list)) and len(transform) == 6:
        a, b, c, d, e, f = transform
        return (float(a), float(b), float(c), float(d), float(e), float(f))

    s = str(transform)
    if "Affine" in s:
        # very defensive parse
        nums = []
        cur = ""
        for ch in s:
            if ch in "0123456789.-eE":
                cur += ch
            else:
                if cur:
                    try:
                        nums.append(float(cur))
                    except Exception:
                        pass
                    cur = ""
        if cur:
            try:
                nums.append(float(cur))
            except Exception:
                pass
        if len(nums) >= 6:
            return tuple(nums[:6])  # type: ignore

    raise ModelError(f"Cannot parse affine transform: {transform}")


def _affine_inv(a, b, c, d, e, f):
    """
    Invert 2x2 + translation:
      x = a*col + b*row + c
      y = d*col + e*row + f
    Return coefficients for:
      col = ai*x + bi*y + ci
      row = di*x + ei*y + fi
    """
    det = a * e - b * d
    if abs(det) < 1e-12:
        raise ModelError("Non-invertible affine transform.")
    ai =  e / det
    bi = -b / det
    di = -d / det
    ei =  a / det
    ci = -(ai * c + bi * f)
    fi = -(di * c + ei * f)
    return ai, bi, ci, di, ei, fi


def _world_to_pixel(transform, x, y) -> Tuple[float, float]:
    """
    Map world coordinates (x,y) -> fractional pixel (col,row).
    """
    a, b, c, d, e, f = _parse_affine(transform)
    ai, bi, ci, di, ei, fi = _affine_inv(a, b, c, d, e, f)
    col = ai * x + bi * y + ci
    row = di * x + ei * y + fi
    return col, row


def _clip_window(x0, y0, x1, y1, w, h) -> Tuple[int, int, int, int]:
    """
    Clamp to image bounds and ensure non-empty window.
    """
    x0 = max(0, min(w, x0))
    x1 = max(0, min(w, x1))
    y0 = max(0, min(h, y0))
    y1 = max(0, min(h, y1))
    if x1 <= x0 or y1 <= y0:
        raise ModelError("ROI does not overlap the selected Tessera tile embedding.")
    return x0, y0, x1, y1


def _crop_chw_to_bbox(
    chw: np.ndarray,
    *,
    bbox_4326: BBox,
    tile_crs: str,
    tile_transform,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Crop CHW embedding grid to requested bbox.
    Supports projected CRS via pyproj.
    """
    d, h, w = chw.shape
    crs_str = str(tile_crs)

    # 1) bbox corners in EPSG:4326
    lon_min, lat_min, lon_max, lat_max = bbox_4326.minlon, bbox_4326.minlat, bbox_4326.maxlon, bbox_4326.maxlat

    # 2) reproject bbox to tile CRS if needed
    if crs_str.upper() not in ("EPSG:4326", "WGS84", "CRS:84"):
        try:
            from pyproj import Transformer
        except Exception as e:
            raise ModelError(
                f"Tessera tile CRS is {crs_str}; crop needs pyproj. Install: pip install pyproj"
            ) from e

        tfm = Transformer.from_crs("EPSG:4326", crs_str, always_xy=True)
        x0, y0 = tfm.transform(lon_min, lat_min)
        x1, y1 = tfm.transform(lon_max, lat_max)

        # bbox in projected coords
        x_min, x_max = (min(x0, x1), max(x0, x1))
        y_min, y_max = (min(y0, y1), max(y0, y1))
    else:
        # already lon/lat
        x_min, x_max = lon_min, lon_max
        y_min, y_max = lat_min, lat_max

    # 3) map bbox corners to pixel coordinates (col,row)
    # top-left uses (x_min, y_max), bottom-right uses (x_max, y_min)
    col0, row0 = _world_to_pixel(tile_transform, x_min, y_max)
    col1, row1 = _world_to_pixel(tile_transform, x_max, y_min)

    x0i = int(np.floor(min(col0, col1)))
    x1i = int(np.ceil(max(col0, col1)))
    y0i = int(np.floor(min(row0, row1)))
    y1i = int(np.ceil(max(row0, row1)))

    x0i, y0i, x1i, y1i = _clip_window(x0i, y0i, x1i, y1i, w, h)

    cropped = chw[:, y0i:y1i, x0i:x1i].astype(np.float32)
    info = {
        "crop_px_window": (x0i, y0i, x1i, y1i),
        "crop_hw": (int(y1i - y0i), int(x1i - x0i)),
        "tile_crs": crs_str,
    }
    return cropped, info

def _buffer_m_to_deg(lat: float, buffer_m: float) -> Tuple[float, float]:
    import math
    m_per_deg_lat = 111_320.0
    dlat = buffer_m / m_per_deg_lat
    cos_lat = max(1e-6, math.cos(math.radians(lat)))
    dlon = buffer_m / (m_per_deg_lat * cos_lat)
    return dlon, dlat


def _spatial_to_bbox_4326(spatial: SpatialSpec) -> BBox:
    if isinstance(spatial, BBox):
        spatial.validate()
        return spatial
    if isinstance(spatial, PointBuffer):
        spatial.validate()
        dlon, dlat = _buffer_m_to_deg(spatial.lat, spatial.buffer_m)
        return BBox(
            minlon=spatial.lon - dlon,
            minlat=spatial.lat - dlat,
            maxlon=spatial.lon + dlon,
            maxlat=spatial.lat + dlat,
            crs="EPSG:4326",
        )
    raise ModelError(f"Unsupported SpatialSpec type: {type(spatial)}")


def _bbox_center(b: BBox) -> Tuple[float, float]:
    return (b.minlon + b.maxlon) / 2, (b.minlat + b.maxlat) / 2


def _preferred_year(temporal: Optional[TemporalSpec], default_year: int = 2021) -> int:
    if temporal is None:
        return default_year
    temporal.validate()
    if temporal.mode == "year" and temporal.year is not None:
        return int(temporal.year)
    if temporal.mode == "range" and temporal.start:
        return int(str(temporal.start)[:4])
    return default_year


def _pool_chw(chw: np.ndarray, pooling: str) -> np.ndarray:
    if pooling == "mean":
        return chw.mean(axis=(1, 2)).astype(np.float32)
    if pooling == "max":
        return chw.max(axis=(1, 2)).astype(np.float32)
    raise ModelError(f"Unknown pooling='{pooling}' (expected 'mean' or 'max').")


def _pick_nearest_tile(tiles, lon: float, lat: float):
    """
    tiles are whatever geotessera returns; we try to read lon/lat/year from attributes or tuple positions.
    Returns (year, tile_lon, tile_lat)
    """
    best = None
    best_d = 1e18
    for t in tiles:
        t_lon = getattr(t, "lon", None)
        t_lat = getattr(t, "lat", None)
        t_year = getattr(t, "year", None)
        if t_lon is None or t_lat is None or t_year is None:
            # fallback: assume tuple-like
            try:
                t_year, t_lon, t_lat = int(t[0]), float(t[1]), float(t[2])
            except Exception:
                continue
        d = (float(t_lon) - lon) ** 2 + (float(t_lat) - lat) ** 2
        if d < best_d:
            best_d = d
            best = (int(t_year), float(t_lon), float(t_lat))
    return best


def _fetch_tile_embedding(
    gt,
    bbox: BBox,
    lon: float,
    lat: float,
    preferred_year: int,
    *,
    refresh: bool,
    fallback_years=(2024, 2023, 2022, 2021, 2020, 2019, 2018, 2017),
):
    years = [preferred_year] + [y for y in fallback_years if y != preferred_year]
    last_err = None

    for y in years:
        try:
            tiles = gt.registry.load_blocks_for_region(
                bounds=(bbox.minlon, bbox.minlat, bbox.maxlon, bbox.maxlat),
                year=int(y),
            )
        except Exception as e:
            last_err = e
            continue

        if not tiles:
            continue

        pick = _pick_nearest_tile(tiles, lon, lat)
        if pick is None:
            continue

        tile_year, tile_lon, tile_lat = pick
        try:
            emb, crs, transform = gt.fetch_embedding(
                lon=tile_lon, lat=tile_lat, year=tile_year, refresh=refresh
            )
            arr = np.array(emb)

            # normalize shape to CHW
            if arr.ndim == 3 and arr.shape[-1] in (64, 128, 256, 512, 768, 1024):
                chw = np.moveaxis(arr, -1, 0).astype(np.float32)  # HWD -> DHW
            elif arr.ndim == 3 and arr.shape[0] in (64, 128, 256, 512, 768, 1024):
                chw = arr.astype(np.float32)  # DHW already
            else:
                raise ValueError(f"Unexpected Tessera embedding shape: {arr.shape}")

            meta = {
                "used_year": tile_year,
                "tile_lon": tile_lon,
                "tile_lat": tile_lat,
                "crs": crs,               # keep object
                "transform": transform,   # keep object (usually Affine)
                # "crs": str(crs),
                # "transform": str(transform),
            }
            return chw, meta

        except Exception as e:
            last_err = e
            continue

    raise ModelError(f"Failed to fetch Tessera embedding for ROI; last error: {last_err}")


@register("tessera")
class TesseraEmbedder(EmbedderBase):
    """
    Precomputed embeddings via geotessera.

    Output:
      - OutputSpec.pooled(): (D,)
      - OutputSpec.grid():   xarray.DataArray (d,y,x) from CHW
    """

    def describe(self) -> Dict[str, Any]:
        return {
            "type": "precomputed",
            "backend": ["local", "auto"],
            "inputs": {"spatial": "BBox or PointBuffer (EPSG:4326)"},
            "temporal": {"mode": "year or range(start-year used)"},
            "output": ["pooled", "grid"],
            "defaults": {
                "cache_dir_env": "RS_EMBED_TESSERA_CACHE",
                "cache_dir_default": "data/tessera_cache",
                "refresh": False,
                "fallback_years": [2024, 2023, 2022, 2021, 2020, 2019, 2018, 2017],
            },
        }

    def get_embedding(
        self,
        *,
        spatial: SpatialSpec,
        temporal: Optional[TemporalSpec],
        sensor: Optional[SensorSpec],
        output: OutputSpec,
        backend: str,
        device: str = "auto",
    ) -> Embedding:
        if backend.lower() not in ("local", "auto"):
            raise ModelError("tessera is precomputed/local; use backend='local' or 'auto'.")

        try:
            from geotessera import GeoTessera
        except Exception as e:
            raise ModelError("Tessera requires geotessera. Install: pip install geotessera") from e

        bbox = _spatial_to_bbox_4326(spatial)
        lon, lat = _bbox_center(bbox)
        pref_year = _preferred_year(temporal, default_year=2021)

        # cache_dir: env var override OR (optional) sensor.collection override
        cache_dir = os.environ.get("RS_EMBED_TESSERA_CACHE", "data/tessera_cache")
        if sensor and isinstance(sensor.collection, str):
            # convention: collection="cache:/path/to/tessera_cache"
            if sensor.collection.startswith("cache:"):
                cache_dir = sensor.collection.replace("cache:", "", 1).strip()

        os.makedirs(cache_dir, exist_ok=True)

        # v0.1: keep it simple; notebook often used verify_hashes=False
        gt = GeoTessera(cache_dir=cache_dir, verify_hashes=False)

        refresh = False
        fallback_years = (2024, 2023, 2022, 2021, 2020, 2019, 2018, 2017)

        chw, extra = _fetch_tile_embedding(
            gt,
            bbox=bbox,
            lon=lon,
            lat=lat,
            preferred_year=pref_year,
            refresh=refresh,
            fallback_years=fallback_years,
        )
        
        # --- NEW: crop to requested bbox (optional, but recommended) ---
        do_crop = True  # v0.1: default True (you can make this configurable)
        if do_crop:
            try:
                chw, crop_meta = _crop_chw_to_bbox(
                    chw,
                    bbox_4326=bbox,
                    tile_crs=extra.get("crs"),
                    tile_transform=extra.get("transform"),
                )
                extra.update(crop_meta)
            except ModelError as e:
                import logging
                logging.warning(f"Tessera crop skipped: {e}")

        meta = {
            "model": self.model_name,
            "type": "precomputed",
            "source": "geotessera.GeoTessera",
            "cache_dir": cache_dir,
            "bbox_4326": (bbox.minlon, bbox.minlat, bbox.maxlon, bbox.maxlat),
            "center_lonlat": (lon, lat),
            "preferred_year": pref_year,
            "chw_shape": tuple(chw.shape),
            **extra,
        }

        if output.mode == "pooled":
            vec = _pool_chw(chw, output.pooling)
            meta["pooling"] = f"{output.pooling}_hw"
            return Embedding(data=vec, meta=meta)

        if output.mode == "grid":
            try:
                import xarray as xr
            except Exception as e:
                raise ModelError("grid output requires xarray. Install: pip install xarray") from e

            da = xr.DataArray(
                chw,
                dims=("d", "y", "x"),
                coords={
                    "d": np.arange(chw.shape[0]),
                    "y": np.arange(chw.shape[1]),
                    "x": np.arange(chw.shape[2]),
                },
                name="embedding",
                attrs=meta,
            )
            return Embedding(data=da, meta=meta)

        raise ModelError(f"Unknown output mode: {output.mode}")