# src/rs_embed/embedders/onthefly_dofa.py
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import xarray as xr

from ..core.registry import register
from ..core.embedding import Embedding
from ..core.errors import ModelError
from ..core.specs import SpatialSpec, TemporalSpec, SensorSpec, OutputSpec
from ..providers.gee import GEEProvider
from .base import EmbedderBase


# -----------------------------
# Defaults: Sentinel-2 SR (12 bands)
# -----------------------------
_S2_SR_12_BANDS = [
    "B1", "B2", "B3", "B4", "B5", "B6",
    "B7", "B8", "B8A", "B9", "B11", "B12",
]

# Sentinel-2 MSI band central wavelengths (µm)
_S2_WAVELENGTHS_UM = {
    "B1": 0.443,
    "B2": 0.490,
    "B3": 0.560,
    "B4": 0.665,
    "B5": 0.705,
    "B6": 0.740,
    "B7": 0.783,
    "B8": 0.842,
    "B8A": 0.865,
    "B9": 0.945,
    "B11": 1.610,
    "B12": 2.190,
}


# -----------------------------
# Small utils
# -----------------------------
def _auto_device(device: str) -> str:
    import torch

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _infer_wavelengths_um(bands: List[str]) -> Optional[List[float]]:
    wv = []
    for b in bands:
        if b not in _S2_WAVELENGTHS_UM:
            return None
        wv.append(float(_S2_WAVELENGTHS_UM[b]))
    return wv


def _resize_chw(
    x_chw: np.ndarray,
    *,
    size: int = 224,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    CHW float32 -> CHW float32 resized to (size,size) (bilinear), no crop/pad.
    """
    import torch
    import torch.nn.functional as F

    if x_chw.ndim != 3:
        raise ModelError(f"Expected CHW, got shape={x_chw.shape}")
    c, h, w = x_chw.shape
    info = {"orig_hw": (int(h), int(w)), "target_hw": (int(size), int(size)), "mode": "bilinear"}

    x = torch.from_numpy(x_chw.astype(np.float32, copy=False)).unsqueeze(0)  # [1,C,H,W]
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    y = x[0].cpu().numpy().astype(np.float32)
    return y, info


# -----------------------------
# GEE fetch (generic SR scaling /10000)
# -----------------------------
def _fetch_gee_multiband_sr_chw(
    provider: GEEProvider,
    spatial: SpatialSpec,
    temporal: TemporalSpec,
    *,
    collection: str,
    bands: List[str],
    scale_m: int = 10,
    cloudy_pct: int = 30,
    composite: str = "median",
    default_value: float = 0.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    import ee

    region = provider.get_region_3857(spatial)

    col = (
        ee.ImageCollection(collection)
        .filterDate(temporal.start, temporal.end)
        .filterBounds(region)
    )

    cloud_filter_applied = False
    try:
        col = col.filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloudy_pct))
        cloud_filter_applied = True
    except Exception:
        pass

    n_img = None
    t_min = None
    t_max = None
    try:
        n_img = int(col.size().getInfo())
        t_min = int(ee.Number(col.aggregate_min("system:time_start")).getInfo())
        t_max = int(ee.Number(col.aggregate_max("system:time_start")).getInfo())
    except Exception:
        pass

    if composite == "median":
        img = col.median()
    elif composite == "mosaic":
        img = col.mosaic()
    else:
        raise ModelError(f"Unknown composite='{composite}'. Use 'median' or 'mosaic'.")

    img = img.select(bands).reproject(crs="EPSG:3857", scale=scale_m)

    rect = img.sampleRectangle(region=region, defaultValue=default_value).getInfo()
    props = rect["properties"]

    stack = []
    for b in bands:
        arr = np.array(props[b], dtype=np.float32) / 10000.0
        stack.append(arr)

    x = np.clip(np.stack(stack, axis=0), 0.0, 1.0).astype(np.float32)

    meta: Dict[str, Any] = {
        "gee_collection": collection,
        "gee_bands": list(bands),
        "gee_scale_m": int(scale_m),
        "gee_cloudy_pct": int(cloudy_pct),
        "gee_cloud_filter_applied": bool(cloud_filter_applied),
        "gee_composite": str(composite),
        "gee_n_images": n_img,
        "gee_time_start_ms": t_min,
        "gee_time_end_ms": t_max,
        "raw_chw_shape": tuple(x.shape),
        "region_crs": "EPSG:3857",
    }
    return x, meta


# -----------------------------
# DOFA model + forward adapters
# -----------------------------
def _load_dofa_model(
    *,
    variant: str = "base",
    device: str = "auto",
) -> Tuple[Any, Dict[str, Any]]:
    try:
        import torch
        from torchgeo.models import (
            DOFABase16_Weights,
            DOFALarge16_Weights,
            dofa_base_patch16_224,
            dofa_large_patch16_224,
        )
    except Exception as e:
        raise ModelError("DOFA requires torchgeo. Install: pip install torchgeo") from e

    dev = _auto_device(device)
    variant_l = str(variant).lower().strip()

    if variant_l == "base":
        weights = DOFABase16_Weights.DOFA_MAE
        model = dofa_base_patch16_224(weights=weights)
        weight_url = weights.url
        weight_meta = weights.meta
    elif variant_l == "large":
        weights = DOFALarge16_Weights.DOFA_MAE
        model = dofa_large_patch16_224(weights=weights)
        weight_url = weights.url
        weight_meta = weights.meta
    else:
        raise ModelError("DOFA variant must be 'base' or 'large'.")

    model = model.to(dev).eval()

    # sanity
    p0 = None
    for _, p in model.named_parameters():
        if p is not None and p.numel() > 0:
            p0 = p.detach()
            break
    if p0 is None:
        raise ModelError("DOFA model has no parameters; unexpected.")
    if not torch.isfinite(p0).all():
        raise ModelError("DOFA parameters contain NaN/Inf; weight load likely failed.")

    meta = {
        "variant": variant_l,
        "weights_url": str(weight_url),
        "weights_meta": dict(weight_meta) if isinstance(weight_meta, dict) else str(weight_meta),
        "device_resolved": dev,
        "img_size": int(getattr(model, "img_size", 224)),
        "patch_size": int(getattr(model, "patch_size", 16)),
        "embed_dim": int(getattr(model, "embed_dim", -1)),
        "global_pool": bool(getattr(model, "global_pool", True)),
    }
    return model, meta


def _dofa_forward_tokens_and_pooled(
    model,
    x_bchw: np.ndarray,
    wavelengths_um: List[float],
    *,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Returns:
      patch_tokens: [N, D] (no CLS)
      pooled:       [D]
    """
    import torch

    dev = _auto_device(device)
    x = torch.from_numpy(x_bchw).to(dev)
    if x.dtype != torch.float32:
        x = x.float()

    wavelist = torch.tensor(wavelengths_um, device=dev).float()

    with torch.no_grad():
        # Patch embedding
        xtok, _ = model.patch_embed(x, wavelist)  # [B, N, D]
        # Pos embed (skip cls position)
        xtok = xtok + model.pos_embed[:, 1:, :]
        # Prepend CLS
        cls_token = model.cls_token + model.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(xtok.shape[0], -1, -1)
        xseq = torch.cat((cls_tokens, xtok), dim=1)  # [B, 1+N, D]

        # Transformer blocks
        for blk in model.blocks:
            xseq = blk(xseq)

        # pooled to match torchgeo logic
        if getattr(model, "global_pool", True):
            pooled_t = xseq[:, 1:, :].mean(dim=1)
            pooled_t = model.fc_norm(pooled_t)
            pooled = pooled_t[0].detach().float().cpu().numpy().astype(np.float32)
            norm_applied = "fc_norm(global_pool_mean)"
        else:
            xseq = model.norm(xseq)
            pooled = xseq[:, 0][0].detach().float().cpu().numpy().astype(np.float32)
            norm_applied = "norm(cls)"

        patch_tokens = xseq[:, 1:, :][0].detach().float().cpu().numpy().astype(np.float32)  # [N,D]

    n, d = patch_tokens.shape
    side = int(round(math.sqrt(n)))
    extra = {
        "token_count": int(n),
        "token_dim": int(d),
        "token_grid_side": int(side) if side * side == n else None,
        "tokens_include_cls": False,
        "pooled_norm": norm_applied,
    }
    return patch_tokens, pooled, extra


# -----------------------------
# Embedder
# -----------------------------
@register("dofa")
class DOFAEmbedder(EmbedderBase):
    """
    DOFA (TorchGeo) embeddings.

    - backend="gee": ROI -> S2 SR -> resize to 224 -> DOFA -> pooled/grid
    - backend="tensor": sensor.data (CHW/BCHW) -> resize to 224 -> DOFA

    Output:
      - OutputSpec.pooled(): (D,)
      - OutputSpec.grid():   (D, Ht, Wt) token grid, usually 14x14 for 224/patch16
    """

    def describe(self) -> Dict[str, Any]:
        return {
            "type": "on_the_fly",
            "backend": ["gee", "tensor"],
            "inputs": {
                "gee_default": {
                    "collection": "COPERNICUS/S2_SR_HARMONIZED",
                    "bands": _S2_SR_12_BANDS,
                    "wavelengths_um": "auto for S2 bands",
                }
            },
            "temporal": {"mode": "range"},
            "output": ["pooled", "grid"],
            "defaults": {
                "variant": "base",
                "image_size": 224,
                "scale_m": 10,
                "cloudy_pct": 30,
                "composite": "median",
                "preprocess": "resize_to_224_bilinear",
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
        backend_l = backend.lower().strip()
        variant = getattr(sensor, "variant", "base") if sensor else "base"
        image_size = 224

        # For optional on-the-fly input inspection
        check_meta: Dict[str, Any] = {}

        # -----------------
        # Build input + wavelengths
        # -----------------
        if backend_l == "tensor":
            if sensor is None or not hasattr(sensor, "data"):
                raise ModelError("backend='tensor' requires sensor.data as CHW or BCHW.")
            x = sensor.data
            try:
                import torch
                if torch.is_tensor(x):
                    x = x.detach().cpu().numpy()
            except Exception:
                pass

            x = np.asarray(x)
            if x.ndim == 3:
                x_chw = x.astype(np.float32)
                x_chw, resize_meta = _resize_chw(x_chw, size=image_size)
                x_bchw = x_chw[None, ...]
            elif x.ndim == 4:
                if x.shape[0] != 1:
                    raise ModelError("v0.1: tensor backend expects B=1.")
                x_chw = x[0].astype(np.float32)
                x_chw, resize_meta = _resize_chw(x_chw, size=image_size)
                x_bchw = x_chw[None, ...]
            else:
                raise ModelError(f"Expected CHW or BCHW, got {x.shape}")

            wavelengths_um = getattr(sensor, "wavelengths", None)
            if wavelengths_um is None:
                bands = list(getattr(sensor, "bands", [])) if hasattr(sensor, "bands") else []
                if bands:
                    wavelengths_um = _infer_wavelengths_um(bands)
            if wavelengths_um is None:
                raise ModelError(
                    "DOFA requires wavelengths (µm) per channel. "
                    "Provide sensor.wavelengths=[...] or (for S2) provide sensor.bands to infer."
                )
            wavelengths_um = [float(v) for v in wavelengths_um]

            gee_meta = {"backend_tensor": True}

        elif backend_l == "gee":
            if temporal is None:
                raise ModelError("dofa backend='gee' requires TemporalSpec.range(start,end).")
            temporal.validate()
            if temporal.mode != "range":
                raise ModelError("dofa backend='gee' requires TemporalSpec.range in v0.1.")

            # overrides
            collection = getattr(sensor, "collection", "COPERNICUS/S2_SR_HARMONIZED") if sensor else "COPERNICUS/S2_SR_HARMONIZED"
            bands = list(getattr(sensor, "bands", _S2_SR_12_BANDS)) if sensor else list(_S2_SR_12_BANDS)
            scale_m = int(getattr(sensor, "scale_m", 10)) if sensor else 10
            cloudy_pct = int(getattr(sensor, "cloudy_pct", 30)) if sensor else 30
            composite = str(getattr(sensor, "composite", "median")) if sensor else "median"

            wavelengths_um = getattr(sensor, "wavelengths", None) if sensor else None
            if wavelengths_um is None:
                wavelengths_um = _infer_wavelengths_um(bands)
            if wavelengths_um is None:
                raise ModelError(
                    f"Cannot infer wavelengths for bands={bands}. Provide sensor.wavelengths explicitly (µm)."
                )
            wavelengths_um = [float(v) for v in wavelengths_um]

            provider = GEEProvider(auto_auth=True)
            provider.ensure_ready()

            x_chw, gee_meta = _fetch_gee_multiband_sr_chw(
                provider,
                spatial,
                temporal,
                collection=str(collection),
                bands=bands,
                scale_m=scale_m,
                cloudy_pct=cloudy_pct,
                composite=composite,
                default_value=0.0,
            )

            # Optional: inspect on-the-fly GEE input
            from ..core.input_checks import maybe_inspect_chw, checks_should_raise
            check_meta.clear()
            report = maybe_inspect_chw(
                x_chw,
                sensor=sensor,
                name="gee_multiband_sr_chw",
                expected_channels=len(bands),
                value_range=(0.0, 1.0),
                fill_value=0.0,
                meta=check_meta,
            )
            if report is not None and (not report.get("ok", True)) and checks_should_raise(sensor):
                raise ModelError("GEE input inspection failed: " + "; ".join(report.get("issues", [])))

            x_chw, resize_meta = _resize_chw(x_chw, size=image_size)
            x_bchw = x_chw[None, ...].astype(np.float32)

        else:
            raise ModelError("dofa supports backend='gee' or 'tensor' only.")

        c = int(x_bchw.shape[1])
        if len(wavelengths_um) != c:
            raise ModelError(f"wavelengths length={len(wavelengths_um)} must equal channels C={c}.")

        # -----------------
        # Model + forward
        # -----------------
        model, mmeta = _load_dofa_model(variant=variant, device=device)
        tokens, pooled, tmeta = _dofa_forward_tokens_and_pooled(
            model, x_bchw, wavelengths_um=wavelengths_um, device=device
        )

        base_meta: Dict[str, Any] = {
            "model": self.model_name,
            "type": "on_the_fly",
            "backend": backend_l,
            "variant": str(variant),
            "output_mode": output.mode,
            "device": str(device),
            "preprocess": {"strategy": "resize_to_224_bilinear", "resize_meta": resize_meta},
            "input_channels": int(c),
            "wavelengths_um": list(map(float, wavelengths_um)),
            "input_size_hw": (int(x_bchw.shape[2]), int(x_bchw.shape[3])),
            "token_meta": tmeta,
            **check_meta,
            **mmeta,
            **gee_meta,
        }

        if output.mode == "pooled":
            base_meta["pooled_shape"] = tuple(pooled.shape)
            return Embedding(data=pooled.astype(np.float32), meta=base_meta)

        if output.mode == "grid":
            n, d = tokens.shape
            side = int(round(math.sqrt(n)))
            if side * side != n:
                raise ModelError(f"DOFA tokens N={n} not square; cannot reshape to grid.")
            grid = tokens.reshape(side, side, d).transpose(2, 0, 1).astype(np.float32)

            meta = {
                **base_meta,
                "grid_type": "vit_patch_tokens",
                "grid_shape": tuple(grid.shape),
                "grid_hw_tokens": (int(side), int(side)),
                "patch_size": int(getattr(model, "patch_size", 16)),
            }

            da = xr.DataArray(
                grid,
                dims=("d", "y", "x"),
                coords={
                    "d": np.arange(grid.shape[0]),
                    "y": np.arange(grid.shape[1]),
                    "x": np.arange(grid.shape[2]),
                },
                name="embedding",
                attrs=meta,
            )
            return Embedding(data=da, meta=meta)

        raise ModelError(f"Unknown output mode: {output.mode}")