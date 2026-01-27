# src/rs_embed/embedders/onthefly_remoteclip.py
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import xarray as xr

from ..core.registry import register
from ..core.embedding import Embedding
from ..core.errors import ModelError
from ..core.specs import SpatialSpec, TemporalSpec, SensorSpec, OutputSpec
from ..providers.gee import GEEProvider
from .base import EmbedderBase


# -----------------------------
# GEE: Fetch S2 RGB
# -----------------------------
def _s2_rgb_u8_from_chw(s2_chw: np.ndarray) -> np.ndarray:
    """s2_chw: [3,H,W] float in [0,1] -> uint8 [H,W,3]"""
    if s2_chw.ndim != 3 or s2_chw.shape[0] != 3:
        raise ModelError(f"Expected S2 RGB CHW with 3 bands, got shape={s2_chw.shape}")
    x = np.clip(s2_chw, 0.0, 1.0)
    return (x.transpose(1, 2, 0) * 255.0).astype(np.uint8)


def _fetch_s2_rgb_chw(
    provider: GEEProvider,
    spatial: SpatialSpec,
    temporal: TemporalSpec,
    *,
    scale_m: int = 10,
    cloudy_pct: int = 30,
    composite: str = "median",
) -> np.ndarray:
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

    img = img.select(["B4", "B3", "B2"]).reproject(crs="EPSG:3857", scale=scale_m)

    rect = img.sampleRectangle(region=region, defaultValue=0.0).getInfo()
    props = rect["properties"]

    # SR scaling: 0..10000
    r = np.array(props["B4"], dtype=np.float32) / 10000.0
    g = np.array(props["B3"], dtype=np.float32) / 10000.0
    b = np.array(props["B2"], dtype=np.float32) / 10000.0
    return np.stack([r, g, b], axis=0)


# -----------------------------
# HF weight management (strict)
# -----------------------------
def _find_weight_file(path: str) -> Optional[str]:
    for fn in ("model.safetensors", "pytorch_model.bin", "pytorch_model.bin.index.json"):
        p = os.path.join(path, fn)
        if os.path.exists(p):
            return p
    return None


def _ensure_hf_weights(
    repo_id_or_path: str,
    *,
    auto_download: bool = True,
    require_pretrained: bool = True,
    cache_dir: Optional[str] = None,
    min_bytes: int = 50 * 1024 * 1024,  # 50MB: below this is almost surely pointer/metadata
) -> Tuple[str, Optional[str]]:
    """
    Ensure pretrained weights are present locally.
    Returns (local_dir, weight_file_path).
    """
    if os.path.exists(repo_id_or_path):
        wf = _find_weight_file(repo_id_or_path)
        if require_pretrained:
            if wf is None:
                raise ModelError(
                    f"Local ckpt path '{repo_id_or_path}' has no weights file "
                    "(expected model.safetensors or pytorch_model.bin)."
                )
            if wf.endswith(".safetensors") and os.path.getsize(wf) < min_bytes:
                raise ModelError(
                    f"Local '{wf}' is too small to be real weights (size={os.path.getsize(wf)}). "
                    "It looks like a pointer/placeholder."
                )
        return repo_id_or_path, wf

    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        raise ModelError("Install huggingface_hub to download/verify weights: pip install huggingface_hub") from e

    if auto_download:
        local_dir = snapshot_download(
            repo_id=repo_id_or_path,
            cache_dir=cache_dir,
            local_files_only=False,
        )
    else:
        local_dir = snapshot_download(
            repo_id=repo_id_or_path,
            cache_dir=cache_dir,
            local_files_only=True,
        )

    wf = _find_weight_file(local_dir)
    if require_pretrained:
        if wf is None:
            raise ModelError(f"Downloaded snapshot for '{repo_id_or_path}' but no weights file found in {local_dir}.")
        if wf.endswith(".safetensors") and os.path.getsize(wf) < min_bytes:
            raise ModelError(
                f"Found '{wf}' but it's only {os.path.getsize(wf)} bytes — likely a xet/LFS pointer, not real weights.\n"
                "Fix:\n"
                "  pip install -U hf_xet\n"
                "  (optional) pip install -U \"huggingface_hub[hf_transfer]\"\n"
                "Then delete the cached snapshot and re-run.\n"
            )
    return local_dir, wf


def _assert_weights_loaded(model) -> Dict[str, float]:
    """Best-effort sanity check that weights are loaded (do not trust rshf warnings)."""
    import torch

    core = getattr(model, "model", model)
    p = None
    for _, param in core.named_parameters():
        if param is not None and param.numel() > 0:
            p = param.detach()
            break
    if p is None:
        raise ModelError("RemoteCLIP model has no parameters; cannot verify weights.")
    if not torch.isfinite(p).all():
        raise ModelError("RemoteCLIP parameters contain NaN/Inf; load likely failed.")

    p_f = p.float()
    std = float(p_f.std().cpu())
    mx = float(p_f.abs().max().cpu())
    mean = float(p_f.mean().cpu())
    if std < 1e-6 and mx < 1e-5:
        raise ModelError("RemoteCLIP parameters look uninitialized (near-zero stats).")
    return {"param_mean": mean, "param_std": std, "param_absmax": mx}


def _load_rshf_remoteclip(
    ckpt: str,
    *,
    auto_download: bool = True,
    require_pretrained: bool = True,
    cache_dir: Optional[str] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Load rshf RemoteCLIP with explicit weight checks. Returns (model, weight_meta)."""
    try:
        from rshf.remoteclip import RemoteCLIP
    except Exception as e:
        raise ModelError("RemoteCLIP requires rshf. Install: pip install rshf") from e

    local_dir, weight_file = _ensure_hf_weights(
        ckpt,
        auto_download=auto_download,
        require_pretrained=require_pretrained,
        cache_dir=cache_dir,
    )

    model = RemoteCLIP.from_pretrained(local_dir if os.path.exists(local_dir) else ckpt)
    stats = _assert_weights_loaded(model)

    meta = {
        "ckpt_input": ckpt,
        "ckpt_local_dir": local_dir,
        "weight_file": weight_file,
        "weight_file_size": os.path.getsize(weight_file) if (weight_file and os.path.exists(weight_file)) else None,
        "weights_verified": True,
        **stats,
    }
    return model, meta


# -----------------------------
# Token -> grid helpers
# -----------------------------
def _is_perfect_square(n: int) -> bool:
    r = int(np.sqrt(n))
    return r * r == n


def _tokens_to_grid_dhw(tokens_nd: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    tokens_nd: [N, D] or [N+1, D] (maybe with CLS)
    Returns: grid_dhw: [D, Ht, Wt] and meta about CLS handling and grid size.
    """
    if tokens_nd.ndim != 2:
        raise ModelError(f"Expected tokens [N,D], got {tokens_nd.shape}")
    n, d = tokens_nd.shape

    cls_removed = False
    if _is_perfect_square(n):
        ht = wt = int(np.sqrt(n))
        tok = tokens_nd
    elif _is_perfect_square(n - 1):
        # common: first token is CLS
        cls_removed = True
        tok = tokens_nd[1:, :]
        ht = wt = int(np.sqrt(n - 1))
    else:
        raise ModelError(
            f"Token count N={n} is not a square (or N-1). Cannot reshape into HxW grid."
        )

    grid_hwd = tok.reshape(ht, wt, d)               # [Ht, Wt, D]
    grid_dhw = np.transpose(grid_hwd, (2, 0, 1))    # [D, Ht, Wt]
    meta = {"token_count": n, "dim": d, "grid_hw": (ht, wt), "cls_removed": cls_removed}
    return grid_dhw.astype(np.float32), meta


# -----------------------------
# RemoteCLIP inference adapter
# -----------------------------
def _remoteclip_encode_tokens(
    model,
    rgb_u8: np.ndarray,
    *,
    image_size: int = 224,
    device: str = "auto",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Return tokens if possible; else return pooled vector.

    Priority:
      1) forward_encoder -> tokens [N,D] (not available in your current rshf)
      2) open_clip CLIP.forward_intermediates -> tokens (if returns)
      3) hook core.visual.transformer -> capture tokens while running core.encode_image(x)
      4) fallback encode_image -> pooled only

    Returns:
      - tokens: [N,D]  (tokens_kind='tokens' or 'tokens_hook' or 'tokens_intermediates')
      - pooled: [D]    (tokens_kind='pooled')
    """
    import torch
    from torchvision import transforms
    from PIL import Image

    dev = "cuda" if (device == "auto" and torch.cuda.is_available()) else ("cpu" if device == "auto" else device)
    model = model.to(dev).eval()
    core = getattr(model, "model", model)

    # --- preprocess to tensor ---
    if hasattr(model, "transform") and callable(getattr(model, "transform")):
        x = model.transform(rgb_u8.astype(np.float32), image_size).unsqueeze(0)
    else:
        preprocess = transforms.Compose(
            [
                transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
        img = Image.fromarray(rgb_u8, mode="RGB")
        x = preprocess(img).unsqueeze(0)

    x = x.to(dev)

    with torch.no_grad():
        # 1) forward_encoder (not in your current wrapper)
        fe = None
        if hasattr(model, "forward_encoder"):
            fe = model.forward_encoder
        elif hasattr(core, "forward_encoder"):
            fe = core.forward_encoder

        if fe is not None:
            try:
                out = fe(x, mask_ratio=0.0)
            except TypeError:
                out = fe(x)
            toks = out[0] if isinstance(out, (tuple, list)) else out
            if toks.ndim == 3:
                return toks[0].detach().float().cpu().numpy().astype(np.float32), {"tokens_kind": "tokens"}
            if toks.ndim == 2:
                return toks[0].detach().float().cpu().numpy().astype(np.float32), {"tokens_kind": "pooled"}
            raise ModelError(f"Unexpected forward_encoder output shape: {tuple(toks.shape)}")

        # 2) open_clip: forward_intermediates (best if it returns tokens cleanly)
        if hasattr(core, "forward_intermediates"):
            try:
                out = core.forward_intermediates(x)
                # open_clip versions differ: out may be dict-like or tuple-like.
                # We'll search for the first tensor shaped [B,N,D] as tokens.
                tokens_t = None

                if isinstance(out, dict):
                    # common keys (varies by version)
                    candidates = []
                    for k in ("image_intermediates", "intermediates", "image_tokens", "tokens"):
                        if k in out:
                            candidates.append(out[k])
                    # also scan values
                    candidates += list(out.values())
                    for v in candidates:
                        if torch.is_tensor(v) and v.ndim == 3:
                            tokens_t = v
                            break
                        if isinstance(v, (list, tuple)):
                            for vv in v:
                                if torch.is_tensor(vv) and vv.ndim == 3:
                                    tokens_t = vv
                                    break
                            if tokens_t is not None:
                                break

                elif isinstance(out, (tuple, list)):
                    # scan elements
                    for v in out:
                        if torch.is_tensor(v) and v.ndim == 3:
                            tokens_t = v
                            break
                        if isinstance(v, (list, tuple)):
                            for vv in v:
                                if torch.is_tensor(vv) and vv.ndim == 3:
                                    tokens_t = vv
                                    break
                            if tokens_t is not None:
                                break

                if tokens_t is not None:
                    return tokens_t[0].detach().float().cpu().numpy().astype(np.float32), {"tokens_kind": "tokens_intermediates"}
            except Exception:
                # If forward_intermediates exists but signature/return differs, fall back to hook
                pass

        # 3) hook vision transformer transformer output to get tokens
        if hasattr(core, "visual") and hasattr(core.visual, "transformer"):
            captured = {}

            def _hook(_module, _inp, outp):
                # outp is typically [B, N, D]
                captured["tokens"] = outp

            handle = core.visual.transformer.register_forward_hook(_hook)
            try:
                # run a normal encode_image forward; hook captures tokens
                _ = core.encode_image(x) if hasattr(core, "encode_image") else core.forward(x)
            finally:
                handle.remove()

            if "tokens" in captured and torch.is_tensor(captured["tokens"]) and captured["tokens"].ndim == 3:
                toks = captured["tokens"]
                return toks[0].detach().float().cpu().numpy().astype(np.float32), {"tokens_kind": "tokens_hook"}

        # 4) pooled fallback only
        if hasattr(core, "encode_image"):
            v = core.encode_image(x)
            return v[0].detach().float().cpu().numpy().astype(np.float32), {"tokens_kind": "pooled"}
        if hasattr(core, "visual") and callable(getattr(core.visual, "forward", None)):
            v = core.visual(x)
            if v.ndim == 3:
                v = v.mean(dim=1)
            return v[0].detach().float().cpu().numpy().astype(np.float32), {"tokens_kind": "pooled"}

        raise ModelError("RemoteCLIP exposes neither token sequence nor pooled encoding methods.")

@register("remoteclip_s2rgb")
class RemoteCLIPS2RGBEmbedder(EmbedderBase):
    """
    ROI -> (GEE S2 SR Harmonized RGB composite) -> RemoteCLIP -> pooled or token-grid embedding

    - OutputSpec.pooled(): returns vec [D]
    - OutputSpec.grid(): returns token grid [D, Ht, Wt] (ViT patch grid, NOT pixel grid)
    """

    def describe(self) -> Dict[str, Any]:
        return {
            "type": "on_the_fly",
            "backend": ["gee"],
            "inputs": {"collection": "COPERNICUS/S2_SR_HARMONIZED", "bands": ["B4", "B3", "B2"]},
            "temporal": {"mode": "range"},
            "output": ["pooled", "grid"],
            "defaults": {
                "scale_m": 10,
                "cloudy_pct": 30,
                "composite": "median",
                "ckpt": "MVRL/remote-clip-vit-base-patch32",
                "image_size": 224,
            },
            "notes": "grid output is ViT token grid (patch-level), typically 7x7 for ViT-B/32 at 224px.",
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
        if backend.lower() != "gee":
            raise ModelError("remoteclip_s2rgb only supports backend='gee' in v0.1.")
        if temporal is None:
            raise ModelError("remoteclip_s2rgb requires TemporalSpec.range(start,end).")
        temporal.validate()
        if temporal.mode != "range":
            raise ModelError("remoteclip_s2rgb requires TemporalSpec.range in v0.1.")

        provider = GEEProvider(auto_auth=True)
        provider.ensure_ready()

        # overrides via SensorSpec
        scale_m = sensor.scale_m if sensor else 10
        cloudy_pct = sensor.cloudy_pct if sensor else 30
        composite = sensor.composite if sensor else "median"

        ckpt = "MVRL/remote-clip-vit-base-patch32"
        # v0.1 convention: sensor.collection="hf:<repo_id_or_local_path>"
        if sensor and isinstance(sensor.collection, str) and sensor.collection.startswith("hf:"):
            ckpt = sensor.collection.replace("hf:", "", 1).strip()

        image_size = 224

        # fetch image
        s2_rgb_chw = _fetch_s2_rgb_chw(
            provider, spatial, temporal, scale_m=scale_m, cloudy_pct=cloudy_pct, composite=composite
        )
        rgb_u8 = _s2_rgb_u8_from_chw(s2_rgb_chw)

        # HF cache dir
        cache_dir = (
            os.environ.get("HUGGINGFACE_HUB_CACHE")
            or os.environ.get("HF_HOME")
            or os.environ.get("HUGGINGFACE_HOME")
        )

        # load model (strict weights)
        model, wmeta = _load_rshf_remoteclip(
            ckpt,
            auto_download=True,
            require_pretrained=True,
            cache_dir=cache_dir,
        )

        tokens_or_vec, tmeta = _remoteclip_encode_tokens(
            model, rgb_u8, image_size=image_size, device=device
        )

        base_meta = {
            "model": self.model_name,
            "type": "on_the_fly",
            "backend": "gee",
            "source": "COPERNICUS/S2_SR_HARMONIZED",
            "bands": ["B4", "B3", "B2"],
            "scale_m": scale_m,
            "cloudy_pct": cloudy_pct,
            "composite": composite,
            "start": temporal.start,
            "end": temporal.end,
            "ckpt": ckpt,
            "image_size": image_size,
            "device": device,
            "pretrained_required": True,
            "auto_download": True,
            "hf_cache_dir": cache_dir,
            **wmeta,
            **tmeta,
        }

        # ---- pooled output ----
        if output.mode == "pooled":
            if tokens_or_vec.ndim == 1:
                vec = tokens_or_vec.astype(np.float32)
            elif tokens_or_vec.ndim == 2:
                vec = tokens_or_vec.mean(axis=0).astype(np.float32)  # tokens mean
                base_meta["pooling"] = "token_mean"
            else:
                raise ModelError(f"Unexpected tokens/vec shape for pooled: {tokens_or_vec.shape}")
            return Embedding(data=vec, meta=base_meta)

        # ---- grid output ----
        if output.mode == "grid":
            if tokens_or_vec.ndim != 2:
                raise ModelError(
                    "grid output requires token sequence [N,D]. "
                    "Your RemoteCLIP wrapper only provides pooled vectors (no forward_encoder tokens)."
                )

            grid_dhw, gmeta = _tokens_to_grid_dhw(tokens_or_vec)
            meta = {**base_meta, **gmeta, "grid_type": "vit_tokens"}  # patch grid, not pixel grid

            da = xr.DataArray(
                grid_dhw,
                dims=("d", "y", "x"),
                coords={
                    "d": np.arange(grid_dhw.shape[0]),
                    "y": np.arange(grid_dhw.shape[1]),
                    "x": np.arange(grid_dhw.shape[2]),
                },
                name="embedding",
                attrs=meta,
            )
            return Embedding(data=da, meta=meta)

        raise ModelError(f"Unknown output mode: {output.mode}")