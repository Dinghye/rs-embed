from __future__ import annotations

import os
import math
from dataclasses import asdict
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..core.embedding import Embedding
from ..core.errors import ModelError
from ..core.registry import register
from ..core.specs import SpatialSpec, TemporalSpec, SensorSpec, OutputSpec
from ..providers.gee import GEEProvider
from .base import EmbedderBase

from ._vit_mae_utils import (
    ensure_torch,
    pool_from_tokens,
    tokens_to_grid_dhw,
    base_meta,
    temporal_to_range,
)


# -------------------------
# GEE: Sentinel-2 -> Prithvi 6-band (CHW float32 in [0,1])
# -------------------------
PRITHVI_S2_BANDS_SRC = ["B2", "B3", "B4", "B8", "B11", "B12"]
PRITHVI_S2_BANDS_DST = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]


def _fetch_s2_prithvi6_chw(
    provider: GEEProvider,
    spatial: SpatialSpec,
    temporal: TemporalSpec,
    *,
    scale_m: int = 30,
    cloudy_pct: int = 30,
    composite: str = "median",
    fill_value: float = 0.0,
) -> np.ndarray:
    """
    Returns CHW float32 [6,H,W] normalized to [0,1] from S2 SR (scaled by 1/10000).
    Uses provider.get_region_3857(spatial) to define the sampling rectangle.
    """
    import ee  # lazy

    region = provider.get_region_3857(spatial)

    col = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(temporal.start, temporal.end)
        .filterBounds(region)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloudy_pct))
    )

    if composite == "median":
        img = col.median()
    elif composite == "mosaic":
        img = col.mosaic()
    else:
        raise ModelError(f"Unknown composite='{composite}'. Use 'median' or 'mosaic'.")

    img = (
        img.select(PRITHVI_S2_BANDS_SRC, PRITHVI_S2_BANDS_DST)
        .reproject(crs="EPSG:3857", scale=scale_m)
    )

    rect = img.sampleRectangle(region=region, defaultValue=float(fill_value)).getInfo()
    props = rect["properties"]

    arrs = [np.array(props[b], dtype=np.float32) for b in PRITHVI_S2_BANDS_DST]  # HxW each
    x_chw = np.stack(arrs, axis=0)  # [6,H,W]

    # S2 SR scaling: 0..10000
    x_chw = x_chw / 10000.0
    x_chw = np.clip(x_chw, 0.0, 1.0)
    x_chw = np.nan_to_num(x_chw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return x_chw

def _pad_chw_to_multiple(x_chw: np.ndarray, mult: int = 16, value: float = 0.0) -> np.ndarray:
    """
    Pad CHW to make H and W divisible by mult.
    Pads on bottom and right only.
    """
    if x_chw.ndim != 3:
        raise ModelError(f"Expected CHW, got {x_chw.shape}")
    c, h, w = x_chw.shape
    nh = int(math.ceil(h / mult) * mult)
    nw = int(math.ceil(w / mult) * mult)
    if nh == h and nw == w:
        return x_chw
    out = np.full((c, nh, nw), float(value), dtype=np.float32)
    out[:, :h, :w] = x_chw.astype(np.float32)
    return out

# -------------------------
# Prithvi model loading (TerraTorch)
# -------------------------
def _load_prithvi(
    model_key: str,
    *,
    pretrained: bool,
    bands: Tuple[str, ...],
    num_frames: int,
    coords_encoding: Tuple[str, ...],
    device: str = "auto",
):
    ensure_torch()
    import torch

    try:
        from terratorch.registry import BACKBONE_REGISTRY
    except Exception as e:
        raise ModelError("Prithvi requires terratorch. Install: pip install terratorch") from e

    dev = "cuda" if (device == "auto" and torch.cuda.is_available()) else ("cpu" if device == "auto" else device)

    try:
        m = BACKBONE_REGISTRY.build(
            model_key,
            pretrained=bool(pretrained),
            bands=list(bands),
            num_frames=int(num_frames),
            coords_encoding=list(coords_encoding),
        )
    except Exception as e:
        raise ModelError(
            f"Failed to build Prithvi backbone '{model_key}' via TerraTorch BACKBONE_REGISTRY. "
            f"Original error: {e}"
        ) from e

    m = m.to(dev).eval()
    return m, dev


def _mid_date_str(start: str, end: str) -> str:
    # robust mid-date for coords (Prithvi uses year + day-of-year)
    import pandas as pd

    s = pd.to_datetime(start)
    e = pd.to_datetime(end)
    mid = s + (e - s) / 2
    return mid.strftime("%Y-%m-%d")


def _prithvi_forward_tokens(
    model,
    x_chw: np.ndarray,
    *,
    lon: float,
    lat: float,
    date_str: str,
    device: str,
) -> np.ndarray:
    """
    Run Prithvi forward and return token sequence [N,D] (may include CLS).
    TerraTorch Prithvi commonly returns tokens as the last element in tuple/list.
    """
    ensure_torch()
    import torch
    import pandas as pd

    if x_chw.ndim != 3 or x_chw.shape[0] != 6:
        raise ModelError(f"Prithvi expects 6-band CHW, got {x_chw.shape}")

    x = torch.from_numpy(x_chw).unsqueeze(0).to(device)  # [1,6,H,W]

    d = pd.to_datetime(date_str)
    temporal_coords = torch.tensor([[[float(d.year), float(d.dayofyear - 1)]]], dtype=torch.float32, device=device)  # [1,1,2]
    location_coords = torch.tensor([[float(lon), float(lat)]], dtype=torch.float32, device=device)  # [1,2]

    with torch.no_grad():
        out = model(x, temporal_coords=temporal_coords, location_coords=location_coords)

    # normalize output -> tokens
    tokens = None
    if isinstance(out, (tuple, list)):
        # notebook assumed last is tokens
        tokens = out[-1]
    elif hasattr(out, "last_hidden_state"):
        tokens = out.last_hidden_state
    elif isinstance(out, dict):
        tokens = out.get("tokens") or out.get("last_hidden_state") or out.get("hidden_states")
        if isinstance(tokens, (tuple, list)):
            tokens = tokens[-1]
    else:
        tokens = out

    if tokens is None:
        raise ModelError("Prithvi forward did not return tokens.")

    if hasattr(tokens, "ndim") and tokens.ndim == 3:
        # [B,N,D]
        return tokens[0].detach().float().cpu().numpy().astype(np.float32)

    raise ModelError(f"Unexpected Prithvi tokens shape/type: {type(tokens)} {getattr(tokens, 'shape', None)}")


# -------------------------
# Embedder
# -------------------------
@register("prithvi_eo_v2_s2_6b")
class PrithviEOV2S2_6B_Embedder(EmbedderBase):
    """
    Prithvi-EO v2 (TerraTorch) on-the-fly embeddings from Sentinel-2 6-band patch.

    Inputs:
      - spatial: BBox/PointBuffer (EPSG:4326)
      - temporal: range/year (year->full year)
      - sensor: controls GEE fetch (scale/cloudy/composite)

    Outputs (aligned with _vit_mae_utils):
      - pooled: patch-token mean/max (exclude CLS if present)
      - grid: token map [D,H,W] (exclude CLS if present)
    """

    DEFAULT_MODEL_KEY = "prithvi_eo_v2_100_tl"
    DEFAULT_IMAGE_SCALE_M = 30  # notebook used 30m
    DEFAULT_CLOUDY_PCT = 30
    DEFAULT_COMPOSITE = "median"

    def describe(self) -> Dict[str, Any]:
        return {
            "type": "onthefly",
            "backend": ["gee"],
            "model_key_default": self.DEFAULT_MODEL_KEY,
            "input_bands": PRITHVI_S2_BANDS_DST,
            "output": ["pooled", "grid"],
            "notes": [
                "Uses TerraTorch BACKBONE_REGISTRY.build(...)",
                "Requires temporal_coords (year, dayofyear-1) and location_coords (lon, lat).",
            ],
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
        if backend.lower() not in ("gee", "auto"):
            raise ModelError("prithvi_eo_v2_s2_6b expects backend='gee' (or 'auto').")

        # Defaults for Prithvi inputs
        if sensor is None:
            sensor = SensorSpec(
                collection="COPERNICUS/S2_SR_HARMONIZED",
                bands=tuple(PRITHVI_S2_BANDS_DST),
                scale_m=self.DEFAULT_IMAGE_SCALE_M,
                cloudy_pct=self.DEFAULT_CLOUDY_PCT,
                composite=self.DEFAULT_COMPOSITE,
                fill_value=0.0,
            )

        t = temporal_to_range(temporal)  # normalize to range

        # Load model
        model_key = os.environ.get("RS_EMBED_PRITHVI_KEY", self.DEFAULT_MODEL_KEY)
        pretrained = os.environ.get("RS_EMBED_PRITHVI_PRETRAINED", "1").strip() not in ("0", "false", "False")
        coords_encoding = ("time", "location")
        num_frames = 1

        model, dev = _load_prithvi(
            model_key,
            pretrained=pretrained,
            bands=tuple(PRITHVI_S2_BANDS_DST),
            num_frames=num_frames,
            coords_encoding=coords_encoding,
            device=device,
        )

        # Fetch S2 6-band patch from GEE
        provider = GEEProvider(auto_auth=True)
        provider.ensure_ready()

        x_chw = _fetch_s2_prithvi6_chw(
            provider,
            spatial=spatial,
            temporal=t,
            scale_m=int(sensor.scale_m),
            cloudy_pct=int(sensor.cloudy_pct),
            composite=str(sensor.composite),
            fill_value=float(sensor.fill_value),
        )
        # Prithvi patch_size usually 16; pad to avoid border being ignored
        patch_mult = int(os.environ.get("RS_EMBED_PRITHVI_PATCH_MULT", "16"))
        x_chw = _pad_chw_to_multiple(x_chw, mult=patch_mult, value=float(sensor.fill_value))

        # coords: use temporal mid-date and ROI center (provider can compute center from region; v0.1: derive from SpatialSpec)
        # We keep it simple: for BBox/PointBuffer in EPSG:4326, we can derive center lon/lat.
        from ..core.specs import BBox, PointBuffer  # local import to avoid cycles

        if isinstance(spatial, BBox):
            spatial.validate()
            lon = (spatial.minlon + spatial.maxlon) / 2
            lat = (spatial.minlat + spatial.maxlat) / 2
        elif isinstance(spatial, PointBuffer):
            spatial.validate()
            lon = spatial.lon
            lat = spatial.lat
        else:
            raise ModelError(f"Unsupported SpatialSpec: {type(spatial)}")

        date_str = _mid_date_str(t.start, t.end)

        tokens = _prithvi_forward_tokens(
            model,
            x_chw,
            lon=lon,
            lat=lat,
            date_str=date_str,
            device=dev,
        )  # [N,D] (maybe includes CLS)

        meta = base_meta(
            model_name=self.model_name,
            hf_id=model_key,  # for terratorch we store model key here
            backend="gee",
            image_size=int(x_chw.shape[-1]),  # not fixed 224; depends on ROI/scale
            sensor=sensor,
            extra={
                "temporal_range": (t.start, t.end),
                "coords_date": date_str,
                "coords_lonlat": (float(lon), float(lat)),
                "tokens_shape": tuple(tokens.shape),
                "model_key": model_key,
                "pretrained": bool(pretrained),
                "coords_encoding": coords_encoding,
                "num_frames": num_frames,
                "input_hw":(int(x_chw.shape[1]), int(x_chw.shape[2])),
                "patch_mult": patch_mult
            },
        )

        if output.mode == "pooled":
            vec, cls_removed = pool_from_tokens(tokens, output.pooling)
            meta.update({"pooling": f"patch_{output.pooling}", "cls_removed": bool(cls_removed)})
            return Embedding(data=vec, meta=meta)

        if output.mode == "grid":
            grid, (h, w), cls_removed = tokens_to_grid_dhw(tokens)
            meta.update({"grid_hw": (h, w), "grid_kind": "patch_tokens", "cls_removed": bool(cls_removed)})

            try:
                import xarray as xr
            except Exception as e:
                raise ModelError("grid output requires xarray. Install: pip install xarray") from e

            da = xr.DataArray(
                grid,
                dims=("d", "y", "x"),
                coords={"d": np.arange(grid.shape[0]), "y": np.arange(h), "x": np.arange(w)},
                name="embedding",
                attrs=meta,
            )
            return Embedding(data=da, meta=meta)

        raise ModelError(f"Unknown output mode: {output.mode}")