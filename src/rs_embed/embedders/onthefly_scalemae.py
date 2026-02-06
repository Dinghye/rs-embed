from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple, Tuple

import numpy as np

from ..core.embedding import Embedding
from ..core.errors import ModelError
from ..core.registry import register
from ..core.specs import SpatialSpec, TemporalSpec, SensorSpec, OutputSpec
from .base import EmbedderBase

from ._vit_mae_utils import (
    fetch_s2_rgb_u8_from_gee,
    temporal_to_range,
    pool_from_tokens,
    tokens_to_grid_dhw,
    base_meta,
    ensure_torch,
    rgb_u8_to_tensor_clipnorm,
)



def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


@lru_cache(maxsize=8)
def _load_scalemae_cached(model_id: str, dev: str):
    ensure_torch()
    import torch

    try:
        from rshf.scalemae import ScaleMAE  # type: ignore
    except Exception as e:
        raise ModelError("ScaleMAE requires rshf with rshf.scalemae.ScaleMAE. Try: pip install -U rshf") from e

    model = ScaleMAE.from_pretrained(model_id)
    try:
        model = model.to(dev).eval()
    except Exception:
        pass

    meta = {"model_id": model_id, "device": dev}
    return model, meta

def _load_scalemae(model_id: str, device: str = "auto"):
    dev = _resolve_device(device)
    return _load_scalemae_cached(model_id, dev)


def _infer_patch_size(model) -> int:
    """
    Try best-effort to infer ViT patch size from common attributes.
    """
    # common: model.patch_size
    ps = getattr(model, "patch_size", None)
    if isinstance(ps, (int, float)):
        return int(ps)

    # common: model.patch_embed.patch_size (int or tuple)
    pe = getattr(model, "patch_embed", None)
    if pe is not None:
        ps2 = getattr(pe, "patch_size", None)
        if isinstance(ps2, (int, float)):
            return int(ps2)
        if isinstance(ps2, (tuple, list)) and len(ps2) >= 1:
            return int(ps2[0])

    # some timm variants: model.patch_embed.patch_size[0]
    # fallback: ViT-L/16 is common for ScaleMAE
    return 16


def _call_with_patch_size(fn, x, *, patch_size: int, input_res):
    """
    Call forward/forward_features with compatible signature across rshf versions.
    Tries kwargs then positional.
    """
    import inspect

    sig = None
    try:
        sig = inspect.signature(fn)
    except Exception:
        sig = None

    # Try kwargs if signature seems to accept them
    if sig is not None:
        params = sig.parameters
        kw = {}
        if "patch_size" in params:
            kw["patch_size"] = patch_size
        if "input_res" in params:
            kw["input_res"] = input_res
        try:
            return fn(x, **kw)
        except TypeError:
            pass

    # Positional fallbacks (your current error shows patch_size is positional)
    # Common patterns:
    #   fn(x, patch_size, input_res)
    #   fn(x, patch_size=..., input_res=...)
    #   fn(x, input_res, patch_size)  (rare)
    try:
        return fn(x, patch_size, input_res)
    except TypeError:
        try:
            return fn(x, patch_size=patch_size, input_res=input_res)
        except TypeError:
            try:
                return fn(x, input_res, patch_size)
            except TypeError as e:
                raise ModelError(f"ScaleMAE call failed even with patch_size/input_res: {e}") from e


def _scalemae_forward_tokens_or_vec(
    model,
    rgb_u8: np.ndarray,
    *,
    image_size: int,
    device: str,
    input_res_m: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Your rshf ScaleMAE requires:
      - input_res: 1D tensor
      - patch_size: required positional argument in forward() (and sometimes in forward_features()).

    Prefer forward_features() to get tokens [N,D]; fallback to forward().
    """
    ensure_torch()
    import torch

    x = rgb_u8_to_tensor_clipnorm(rgb_u8, image_size).to(device)  # [B,3,H,W]
    input_res = torch.tensor([float(input_res_m)], dtype=torch.float32, device=device)  # 1D
    patch_size = _infer_patch_size(model)

    with torch.no_grad():
        ff = getattr(model, "forward_features", None)
        if callable(ff):
            out = _call_with_patch_size(ff, x, patch_size=patch_size, input_res=input_res)
            out0 = out[0] if isinstance(out, (tuple, list)) else out

            if hasattr(out0, "ndim") and out0.ndim == 3:  # [B,N,D]
                toks = out0
                return toks[0].detach().float().cpu().numpy().astype(np.float32), {
                    "tokens_kind": "tokens_forward_features",
                    "input_res_m": float(input_res_m),
                    "used_patch_size": int(patch_size),
                    "tokens_shape": tuple(toks.shape),
                }

            if hasattr(out0, "ndim") and out0.ndim == 2:  # [B,D]
                v = out0
                return v[0].detach().float().cpu().numpy().astype(np.float32), {
                    "tokens_kind": "pooled_forward_features",
                    "input_res_m": float(input_res_m),
                    "used_patch_size": int(patch_size),
                    "vec_shape": tuple(v.shape),
                }

            if hasattr(out0, "ndim") and out0.ndim == 4:  # [B,C,H,W] -> tokens
                b, c, h, w = out0.shape
                toks = out0.permute(0, 2, 3, 1).reshape(b, h * w, c)
                return toks[0].detach().float().cpu().numpy().astype(np.float32), {
                    "tokens_kind": "tokens_from_feature_map",
                    "input_res_m": float(input_res_m),
                    "used_patch_size": int(patch_size),
                    "feature_map_hw": (int(h), int(w)),
                    "tokens_shape": tuple(toks.shape),
                }

            raise ModelError(f"ScaleMAE forward_features returned unsupported: {type(out0)} {getattr(out0,'shape',None)}")

        # fallback: forward (must pass patch_size + input_res)
        out = _call_with_patch_size(model, x, patch_size=patch_size, input_res=input_res)
        out0 = out[0] if isinstance(out, (tuple, list)) else out

        if hasattr(out0, "ndim") and out0.ndim == 3:
            return out0[0].detach().float().cpu().numpy().astype(np.float32), {
                "tokens_kind": "tokens_forward",
                "input_res_m": float(input_res_m),
                "used_patch_size": int(patch_size),
                "tokens_shape": tuple(out0.shape),
            }
        if hasattr(out0, "ndim") and out0.ndim == 2:
            return out0[0].detach().float().cpu().numpy().astype(np.float32), {
                "tokens_kind": "pooled_forward",
                "input_res_m": float(input_res_m),
                "used_patch_size": int(patch_size),
                "vec_shape": tuple(out0.shape),
            }

        raise ModelError("ScaleMAE: cannot obtain tokens or pooled vector from this model.")

@register("scalemae_rgb")
class ScaleMAERGBEmbedder(EmbedderBase):
    """
    ScaleMAE on-the-fly embedding from Sentinel-2 RGB patch (GEE).

    Strategy aligned via _vit_mae_utils:
      - pooled: pool patch tokens by OutputSpec.pooling (exclude CLS if present)
      - grid: patch token grid (exclude CLS if present)
      - scale: uses sensor.scale_m as input_res_m (required by your rshf)
    """

    DEFAULT_MODEL_ID = "MVRL/scalemae-vitlarge-800"
    DEFAULT_IMAGE_SIZE = 224

    def describe(self) -> Dict[str, Any]:
        return {
            "type": "onthefly",
            "backend": ["gee"],
            "model_id_default": self.DEFAULT_MODEL_ID,
            "image_size": self.DEFAULT_IMAGE_SIZE,
            "output": ["pooled", "grid"],
        }

    
    def __init__(self) -> None:
        self._provider: Optional[Any] = None

    def _get_provider(self):
        if self._provider is None:
            from ..providers.gee import GEEProvider
            p = GEEProvider(auto_auth=True)
            p.ensure_ready()
            self._provider = p
        return self._provider

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
                raise ModelError("scalemae_rgb expects backend='gee' (or 'auto').")

            if sensor is None:
                sensor = SensorSpec(
                    collection="COPERNICUS/S2_SR_HARMONIZED",
                    bands=("B4", "B3", "B2"),
                    scale_m=10,
                    cloudy_pct=30,
                    composite="median",
                )

            model_id = os.environ.get("RS_EMBED_SCALEMAE_ID", self.DEFAULT_MODEL_ID)
            image_size = int(os.environ.get("RS_EMBED_SCALEMAE_IMG", str(self.DEFAULT_IMAGE_SIZE)))

            t = temporal_to_range(temporal)
            rgb_u8 = fetch_s2_rgb_u8_from_gee(
                spatial=spatial,
                temporal=t,
                sensor=sensor,
                out_size=image_size,
                provider=self._get_provider(),
            )

            model, wmeta = _load_scalemae(model_id=model_id, device=device)
            dev = wmeta.get("device", device)
            out, extra = _scalemae_forward_tokens_or_vec(
                model,
                rgb_u8,
                image_size=image_size,
                device=dev,
                input_res_m=float(sensor.scale_m),
            )
            
            meta = base_meta(
                model_name=self.model_name,
                hf_id=model_id,
                backend="gee",
                image_size=image_size,
                sensor=sensor,
                temporal=t,
                source=sensor.collection,
                extra={"used_scale_m": float(sensor.scale_m), **extra, "out_shape": tuple(out.shape)},
            )

            if output.mode == "pooled":
                if out.ndim == 2:
                    vec, cls_removed = pool_from_tokens(out, output.pooling)
                    meta.update({"pooling": f"patch_{output.pooling}", "cls_removed": bool(cls_removed)})
                    return Embedding(data=vec, meta=meta)

                if out.ndim == 1:
                    meta.update({"pooling": "model_pooled", "cls_removed": False})
                    return Embedding(data=out.astype(np.float32), meta=meta)

                raise ModelError(f"Unexpected shape for pooled: {out.shape}")

            if output.mode == "grid":
                if out.ndim != 2:
                    raise ModelError(
                        "grid output requires token sequence [N,D]. "
                        f"Got {out.shape} (tokens_kind={meta.get('tokens_kind')})."
                    )

                grid, (h, w), cls_removed = tokens_to_grid_dhw(out)
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
