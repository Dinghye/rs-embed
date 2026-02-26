from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Optional, Tuple, List

import numpy as np

from ..core.registry import register
from ..core.embedding import Embedding
from ..core.errors import ModelError
from ..core.specs import BBox, PointBuffer, SpatialSpec, TemporalSpec, SensorSpec, OutputSpec
from .base import EmbedderBase


def _buffer_m_to_deg(lat: float, buffer_m: float) -> Tuple[float, float]:
    import math
    m_per_deg_lat = 111_320.0
    dlat = buffer_m / m_per_deg_lat
    cos_lat = max(1e-6, math.cos(math.radians(lat)))
    dlon = buffer_m / (m_per_deg_lat * cos_lat)
    return dlon, dlat


def _to_bbox_4326(spatial: SpatialSpec) -> BBox:
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
    raise ModelError(f"Unsupported SpatialSpec: {type(spatial)}")


def _year_from_temporal(temporal: Optional[TemporalSpec], default_year: int = 2021) -> int:
    if temporal is None:
        return default_year
    temporal.validate()
    if temporal.mode == "year" and temporal.year is not None:
        return int(temporal.year)
    if temporal.mode == "range" and temporal.start:
        return int(str(temporal.start)[:4])
    return default_year


def _pool(chw: np.ndarray, pooling: str) -> np.ndarray:
    if pooling == "mean":
        return chw.mean(axis=(1, 2)).astype(np.float32)
    if pooling == "max":
        return chw.max(axis=(1, 2)).astype(np.float32)
    raise ModelError(f"Unknown pooling: {pooling}")


def _to_hwc(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim != 3:
        raise ModelError(f"Unexpected embedding ndim={a.ndim}, shape={a.shape}")
    # geotessera：HWC (H, W, D)
    if a.shape[-1] in (64, 128, 256, 512, 768, 1024):
        return a.astype(np.float32)
    # also support CHW
    if a.shape[0] in (64, 128, 256, 512, 768, 1024):
        return np.moveaxis(a, 0, -1).astype(np.float32)
    raise ModelError(f"Unexpected embedding shape: {a.shape}")


def _assert_north_up(transform):
    # 期望：无旋转 b=d=0，且 a>0, e<0
    b = float(getattr(transform, "b", 0.0))
    d = float(getattr(transform, "d", 0.0))
    if abs(b) > 1e-12 or abs(d) > 1e-12:
        raise ModelError("Tile transform has rotation/shear; mosaic+crop requires north-up (b=d=0).")


def _tile_bounds(transform, w: int, h: int) -> Tuple[float, float, float, float]:
    # (left, bottom, right, top) in tile CRS
    x0, y0 = transform * (0, 0)      # top-left
    x1, y1 = transform * (w, h)      # bottom-right (for north-up, y decreases)
    left, right = (min(x0, x1), max(x0, x1))
    bottom, top = (min(y0, y1), max(y0, y1))
    return left, bottom, right, top


def _reproject_bbox_4326_to(tile_crs_str: str, bbox: BBox) -> Tuple[float, float, float, float]:
    # returns (xmin, ymin, xmax, ymax) in tile CRS
    if str(tile_crs_str).upper() in ("EPSG:4326", "WGS84", "CRS:84"):
        return bbox.minlon, bbox.minlat, bbox.maxlon, bbox.maxlat

    try:
        from pyproj import Transformer
    except Exception as e:
        raise ModelError(f"Need pyproj for CRS={tile_crs_str}. Install: pip install pyproj") from e

    tfm = Transformer.from_crs("EPSG:4326", str(tile_crs_str), always_xy=True)
    x0, y0 = tfm.transform(bbox.minlon, bbox.minlat)
    x1, y1 = tfm.transform(bbox.maxlon, bbox.maxlat)
    return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)


def _mosaic_and_crop_strict_roi(
    tiles_rows: List[Tuple[int, float, float, np.ndarray, Any, Any]],
    bbox_4326: BBox,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    tiles_rows: list of (year, tile_lon, tile_lat, embedding_array, crs, transform)
    Return cropped CHW + meta.
    """
    # normalize + collect tile meta
    hwc_list = []
    crs0 = None
    a0 = e0 = None

    bounds_list = []
    for year, tlon, tlat, emb, crs, transform in tiles_rows:
        _assert_north_up(transform)
        hwc = _to_hwc(emb)
        h, w, d = hwc.shape
        left, bottom, right, top = _tile_bounds(transform, w, h)

        if crs0 is None:
            crs0 = crs
            a0 = float(transform.a)
            e0 = float(transform.e)
        else:
            if str(crs) != str(crs0):
                raise ModelError("Tiles have different CRS; cannot mosaic.")
            if abs(float(transform.a) - a0) > 1e-12 or abs(float(transform.e) - e0) > 1e-12:
                raise ModelError("Tiles have different resolution; cannot mosaic without resampling.")

        hwc_list.append((hwc, transform, (left, bottom, right, top)))
        bounds_list.append((left, bottom, right, top))

    if crs0 is None:
        raise ModelError("No tiles fetched; cannot mosaic.")

    # global mosaic bounds
    left = min(b[0] for b in bounds_list)
    bottom = min(b[1] for b in bounds_list)
    right = max(b[2] for b in bounds_list)
    top = max(b[3] for b in bounds_list)

    px_w = float(a0)                 # >0
    px_h = abs(float(e0))            # >0 (since e<0)

    mosaic_w = int(np.ceil((right - left) / px_w))
    mosaic_h = int(np.ceil((top - bottom) / px_h))

    d = hwc_list[0][0].shape[-1]
    mosaic = np.zeros((mosaic_h, mosaic_w, d), dtype=np.float32)

    # global transform (north-up)
    # x = left + col*px_w, y = top - row*px_h
    # Affine(a, b, c, d, e, f) = (px_w, 0, left, 0, -px_h, top)
    from affine import Affine
    global_transform = Affine(px_w, 0.0, left, 0.0, -px_h, top)

    # paste tiles into mosaic
    for hwc, transform, (t_left, t_bottom, t_right, t_top) in hwc_list:
        h, w, _ = hwc.shape
        x_off = int(round((t_left - left) / px_w))
        y_off = int(round((top - t_top) / px_h))
        mosaic[y_off:y_off + h, x_off:x_off + w, :] = hwc

    # crop window for ROI in tile CRS
    xmin, ymin, xmax, ymax = _reproject_bbox_4326_to(str(crs0), bbox_4326)
    inv = ~global_transform
    c0, r0 = inv * (xmin, ymax)  # top-left
    c1, r1 = inv * (xmax, ymin)  # bottom-right

    x0 = int(np.floor(min(c0, c1)))
    x1 = int(np.ceil(max(c0, c1)))
    y0 = int(np.floor(min(r0, r1)))
    y1 = int(np.ceil(max(r0, r1)))

    # clip
    x0 = max(0, min(mosaic_w, x0))
    x1 = max(0, min(mosaic_w, x1))
    y0 = max(0, min(mosaic_h, y0))
    y1 = max(0, min(mosaic_h, y1))
    if x1 <= x0 or y1 <= y0:
        raise ModelError("ROI does not overlap fetched tessera tiles.")

    cropped_hwc = mosaic[y0:y1, x0:x1, :]
    chw = np.moveaxis(cropped_hwc, -1, 0).astype(np.float32)

    meta = {
        "tile_crs": str(crs0),
        "mosaic_hw": (mosaic_h, mosaic_w),
        "crop_px_window": (x0, y0, x1, y1),
        "crop_hw": (y1 - y0, x1 - x0),
        "global_transform": global_transform,
    }
    return chw, meta


@register("tessera")
class TesseraEmbedder(EmbedderBase):
    DEFAULT_BATCH_WORKERS = 4

    def describe(self) -> Dict[str, Any]:
        return {
            "type": "precomputed",
            "backend": ["local", "auto"],
            "inputs": {"spatial": "BBox or PointBuffer (EPSG:4326)"},
            "temporal": {"mode": "year_or_range", "default_year": 2021},
            "output": ["pooled", "grid"],
            "source": "geotessera.GeoTessera",
            "defaults": {
                "cache_dir_env": "RS_EMBED_TESSERA_CACHE",
            },
            "notes": [
                "Precomputed GeoTessera tiles are local-cache backed (no provider backend).",
                "TemporalSpec.range uses the start year for tile lookup in v0.1.",
            ],
        }

    def __init__(self) -> None:
        # Cache GeoTessera instances per cache_dir to avoid repeated index scans.
        self._gt_cache: Dict[str, Any] = {}

    def _get_gt(self, cache_dir: str):
        if cache_dir not in self._gt_cache:
            from geotessera import GeoTessera
            if cache_dir:
                self._gt_cache[cache_dir] = GeoTessera(cache_dir=cache_dir)
            else:
                self._gt_cache[cache_dir] = GeoTessera()
        return self._gt_cache[cache_dir]

    @staticmethod
    def _resolve_batch_workers(n_items: int) -> int:
        v = int(os.environ.get("RS_EMBED_TESSERA_BATCH_WORKERS", str(TesseraEmbedder.DEFAULT_BATCH_WORKERS)))
        return max(1, min(int(n_items), v))

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
            raise ModelError("tessera is precomputed; use backend='local' or 'auto'.")

        bbox = _to_bbox_4326(spatial)
        year = _year_from_temporal(temporal, default_year=2021)

        cache_dir = os.environ.get("RS_EMBED_TESSERA_CACHE")
        if sensor and isinstance(sensor.collection, str) and sensor.collection.startswith("cache:"):
            cache_dir = sensor.collection.replace("cache:", "", 1).strip() or cache_dir

        # GeoTessera cache keyed by cache_dir path (empty string for default).
        cache_key = cache_dir or ""
        gt = self._get_gt(cache_key)

        bounds = (bbox.minlon, bbox.minlat, bbox.maxlon, bbox.maxlat)
        tiles = gt.registry.load_blocks_for_region(bounds=bounds, year=int(year))
        if not tiles:
            raise ModelError(f"No tessera tiles for bounds={bounds}, year={year}")

        rows = list(gt.fetch_embeddings(tiles))
        chw, crop_meta = _mosaic_and_crop_strict_roi(rows, bbox_4326=bbox)

        meta = {
            "model": self.model_name,
            "type": "precomputed",
            "source": "geotessera.GeoTessera",
            "cache_dir": cache_dir,
            "bbox_4326": bounds,
            "preferred_year": year,
            "chw_shape": tuple(chw.shape),
            **crop_meta,
        }

        if output.mode == "pooled":
            vec = _pool(chw, output.pooling)
            meta["pooling"] = f"{output.pooling}_hw"
            return Embedding(data=vec, meta=meta)

        if output.mode == "grid":
            try:
                import xarray as xr
            except Exception as e:
                raise ModelError("grid output requires xarray: pip install xarray") from e

            da = xr.DataArray(
                chw,
                dims=("d", "y", "x"),
                coords={"d": np.arange(chw.shape[0]), "y": np.arange(chw.shape[1]), "x": np.arange(chw.shape[2])},
                name="embedding",
                attrs=meta,
            )
            return Embedding(data=da, meta=meta)

        raise ModelError(f"Unknown output mode: {output.mode}")

    def get_embeddings_batch(
        self,
        *,
        spatials: list[SpatialSpec],
        temporal: Optional[TemporalSpec] = None,
        sensor: Optional[SensorSpec] = None,
        output: OutputSpec = OutputSpec.pooled(),
        backend: str = "local",
        device: str = "auto",
    ) -> list[Embedding]:
        if not spatials:
            return []

        n = len(spatials)
        out: List[Optional[Embedding]] = [None] * n

        def _one(i: int, sp: SpatialSpec) -> Tuple[int, Embedding]:
            emb = self.get_embedding(
                spatial=sp,
                temporal=temporal,
                sensor=sensor,
                output=output,
                backend=backend,
                device=device,
            )
            return i, emb

        mw = self._resolve_batch_workers(n)
        if mw == 1:
            for i, sp in enumerate(spatials):
                _, emb = _one(i, sp)
                out[i] = emb
        else:
            with ThreadPoolExecutor(max_workers=mw) as ex:
                futs = [ex.submit(_one, i, sp) for i, sp in enumerate(spatials)]
                for fut in as_completed(futs):
                    i, emb = fut.result()
                    out[i] = emb

        if any(e is None for e in out):
            raise ModelError("tessera batch failed to produce all outputs.")
        return [e for e in out if e is not None]
