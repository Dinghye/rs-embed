from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np

from ..core.embedding import Embedding
from ..core.errors import ModelError
from ..core.registry import register
from ..core.specs import SpatialSpec, TemporalSpec, SensorSpec, OutputSpec
from .base import EmbedderBase

from ._vit_mae_utils import (
    fetch_s2_rgb_u8_from_gee,
    pool_from_tokens,
    tokens_to_grid_dhw,
    base_meta,
    temporal_to_range,
    ensure_torch,
    maybe_use_model_transform,
    rgb_u8_to_tensor_clipnorm,
)


def _load_satmae(model_id: str, device: str = "auto"):
    ensure_torch()
    import torch

    try:
        from rshf.satmae import SatMAE
    except Exception as e:
        raise ModelError("SatMAE requires rshf. Install: pip install rshf") from e

    dev = "cuda" if (device == "auto" and torch.cuda.is_available()) else ("cpu" if device == "auto" else device)
    m = SatMAE.from_pretrained(model_id)
    m = m.to(dev).eval()
    return m, dev


def _satmae_forward_tokens(model, rgb_u8: np.ndarray, *, image_size: int, device: str) -> np.ndarray:
    """
    Return tokens [N,D] via forward_encoder(mask_ratio=0.0).
    """
    ensure_torch()
    import torch

    # prefer wrapper transform()
    x = maybe_use_model_transform(model, rgb_u8, image_size)
    if x is None:
        # fallback: generic preprocessing (CLIP norm)
        x = rgb_u8_to_tensor_clipnorm(rgb_u8, image_size)
    x = x.to(device)

    fe = getattr(model, "forward_encoder", None)
    if not callable(fe):
        raise ModelError("SatMAE wrapper does not expose forward_encoder(). Update rshf.")

    with torch.no_grad():
        out = fe(x, mask_ratio=0.0)
        toks = out[0] if isinstance(out, (tuple, list)) else out  # [B,N,D]
        if toks.ndim != 3:
            raise ModelError(f"SatMAE forward_encoder returned {tuple(toks.shape)}; expected [B,N,D].")
        return toks[0].detach().float().cpu().numpy().astype(np.float32)


@register("satmae_rgb")
class SatMAERGBEmbedder(EmbedderBase):
    """
    SatMAE (ViT/MAE) on-the-fly embeddings from Sentinel-2 RGB patch (GEE).

    Strategy aligned via _vit_mae_utils:
      - pooled: pool patch tokens by OutputSpec.pooling (exclude CLS if present)
      - grid: patch token grid (exclude CLS if present)
    """

    DEFAULT_MODEL_ID = "MVRL/satmae-vitlarge-fmow-pretrain-800"
    DEFAULT_IMAGE_SIZE = 224

    def describe(self) -> Dict[str, Any]:
        return {
            "type": "onthefly",
            "backend": ["gee"],
            "model_id_default": self.DEFAULT_MODEL_ID,
            "image_size": self.DEFAULT_IMAGE_SIZE,
            "output": ["pooled", "grid"],
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
            raise ModelError("satmae_rgb expects backend='gee' (or 'auto').")

        if sensor is None:
            sensor = SensorSpec(
                collection="COPERNICUS/S2_SR_HARMONIZED",
                bands=("B4", "B3", "B2"),
                scale_m=10,
                cloudy_pct=30,
                composite="median",
            )

        model_id = os.environ.get("RS_EMBED_SATMAE_ID", self.DEFAULT_MODEL_ID)
        image_size = int(os.environ.get("RS_EMBED_SATMAE_IMG", str(self.DEFAULT_IMAGE_SIZE)))

        t = temporal_to_range(temporal)
        rgb_u8 = fetch_s2_rgb_u8_from_gee(
            spatial=spatial,
            temporal=t,
            sensor=sensor,
            out_size=image_size,
        )

        model, dev = _load_satmae(model_id=model_id, device=device)
        tokens = _satmae_forward_tokens(model, rgb_u8, image_size=image_size, device=dev)  # [N,D]

        meta = base_meta(
            model_name=self.model_name,
            hf_id=model_id,
            backend="gee",
            image_size=image_size,
            sensor=sensor,
            temporal=t,
            source=sensor.collection,
            extra={"tokens_kind": "tokens_forward_encoder", "tokens_shape": tuple(tokens.shape)},
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
