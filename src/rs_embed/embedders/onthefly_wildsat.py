from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import xarray as xr

from ..core.embedding import Embedding
from ..core.errors import ModelError
from ..core.registry import register
from ..core.specs import OutputSpec, SensorSpec, SpatialSpec, TemporalSpec
from ._vit_mae_utils import ensure_torch, pool_from_tokens, resize_rgb_u8, tokens_to_grid_dhw
from .base import EmbedderBase
from .meta_utils import build_meta, temporal_midpoint_str, temporal_to_range
from .onthefly_remoteclip import _fetch_s2_rgb_chw


_SUPPORTED_ARCHES = {"vitb16", "vitl16", "resnet50", "swint"}


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _normalize_arch_name(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    x = str(name).strip().lower().replace("-", "").replace("_", "")
    alias = {
        "vitb16": "vitb16",
        "vitbase16": "vitb16",
        "vitbasepatch16": "vitb16",
        "vitl16": "vitl16",
        "vitlarge16": "vitl16",
        "vitlargepatch16": "vitl16",
        "resnet50": "resnet50",
        "swint": "swint",
        "swintiny": "swint",
    }
    return alias.get(x, x)


def _clean_state_key(key: str) -> str:
    k = str(key)
    prefixes = ("module.", "satellite_model.", "model.")
    changed = True
    while changed:
        changed = False
        for p in prefixes:
            if k.startswith(p):
                k = k[len(p) :]
                changed = True
    return k


def _namespace_like_to_dict(x: Any) -> Dict[str, Any]:
    if x is None:
        return {}
    if isinstance(x, dict):
        return dict(x)
    try:
        return dict(vars(x))
    except Exception:
        return {}


def _extract_satellite_state_dict(ckpt_obj: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if not isinstance(ckpt_obj, dict):
        raise ModelError(f"WildSAT checkpoint must be a dict-like object, got {type(ckpt_obj)}")

    opt_meta = _namespace_like_to_dict(ckpt_obj.get("opt"))

    sd: Optional[Dict[str, Any]] = None
    if isinstance(ckpt_obj.get("satellite_state_dict"), dict):
        sd = ckpt_obj["satellite_state_dict"]
    elif isinstance(ckpt_obj.get("state_dict"), dict):
        sd = ckpt_obj["state_dict"]
    else:
        tensor_values = [v for v in ckpt_obj.values() if hasattr(v, "shape")]
        if tensor_values and len(tensor_values) == len(ckpt_obj):
            sd = ckpt_obj  # already plain state_dict

    if not isinstance(sd, dict) or not sd:
        raise ModelError(
            "Failed to locate WildSAT satellite state_dict in checkpoint. "
            "Expected key 'satellite_state_dict' (preferred), or a plain torch state_dict."
        )

    cleaned: Dict[str, Any] = {}
    for k, v in sd.items():
        nk = _clean_state_key(str(k))
        cleaned[nk] = v

    if not any(k.startswith("backbone.") for k in cleaned):
        n_backbone = sum(1 for k in cleaned if "backbone" in k)
        raise ModelError(
            "WildSAT state_dict does not contain backbone.* keys after normalization. "
            f"Found {n_backbone} keys containing 'backbone'."
        )

    return cleaned, opt_meta


def _infer_arch_from_state_dict(state_dict: Dict[str, Any]) -> Optional[str]:
    keys = set(state_dict.keys())
    if any(k.startswith("backbone.conv_proj.") for k in keys):
        return "vitb16"
    if any(k.startswith("backbone.features.") for k in keys):
        return "swint"
    if any(k.startswith("backbone.layer1.") for k in keys):
        return "resnet50"
    return None


def _resolve_wildsat_arch(
    *,
    arch_hint: str,
    opt_meta: Dict[str, Any],
    state_dict: Dict[str, Any],
) -> str:
    ah = _normalize_arch_name(arch_hint)
    if ah and ah != "auto":
        if ah not in _SUPPORTED_ARCHES:
            raise ModelError(f"Unsupported RS_EMBED_WILDSAT_ARCH='{arch_hint}'. Supported: {sorted(_SUPPORTED_ARCHES)}")
        return ah

    opt_arch = _normalize_arch_name(opt_meta.get("satellite_encoder"))
    if opt_arch in _SUPPORTED_ARCHES:
        return str(opt_arch)

    inferred = _infer_arch_from_state_dict(state_dict)
    if inferred in _SUPPORTED_ARCHES:
        return str(inferred)

    raise ModelError(
        "Failed to infer WildSAT backbone architecture from checkpoint. "
        "Set RS_EMBED_WILDSAT_ARCH explicitly (one of: vitb16, vitl16, resnet50, swint)."
    )


def _build_backbone(arch: str):
    ensure_torch()
    import torchvision

    if arch == "vitb16":
        return torchvision.models.vit_b_16(weights=None)
    if arch == "vitl16":
        return torchvision.models.vit_l_16(weights=None)
    if arch == "resnet50":
        return torchvision.models.resnet50(weights=None)
    if arch == "swint":
        return torchvision.models.swin_t(weights=None)
    raise ModelError(f"Unsupported WildSAT arch='{arch}'")


def _build_branch_head_from_state_dict(state_dict: Dict[str, Any], *, prefer_branch: int = 3):
    ensure_torch()
    import torch
    import torch.nn as nn

    branch_pat = re.compile(r"^decoder\.backbone(\d+)\.(\d+)\.(weight|bias)$")
    single_pat = re.compile(r"^decoder\.backbone\.(\d+)\.(weight|bias)$")

    branches: Dict[int, Dict[int, Dict[str, Any]]] = {}

    for k, v in state_dict.items():
        m = branch_pat.match(k)
        if m is not None:
            b = int(m.group(1))
            li = int(m.group(2))
            wb = str(m.group(3))
            branches.setdefault(b, {}).setdefault(li, {})[wb] = v

    head_kind = "branched"
    if not branches:
        simple: Dict[int, Dict[str, Any]] = {}
        for k, v in state_dict.items():
            m = single_pat.match(k)
            if m is None:
                continue
            li = int(m.group(1))
            wb = str(m.group(2))
            simple.setdefault(li, {})[wb] = v
        if simple:
            branches = {1: {li: params for li, params in simple.items()}}
            head_kind = "single"

    if not branches:
        return None, {"image_head_available": False}

    branch = int(prefer_branch) if int(prefer_branch) in branches else int(sorted(branches.keys())[-1])
    params_by_layer = branches[branch]

    linear_ids = []
    for li in sorted(params_by_layer.keys()):
        w = params_by_layer[li].get("weight")
        if hasattr(w, "ndim") and int(w.ndim) == 2:
            linear_ids.append(li)

    if not linear_ids:
        return None, {
            "image_head_available": False,
            "image_head_branch": branch,
            "image_head_kind": head_kind,
        }

    layers: List[Any] = []
    with torch.no_grad():
        for i, li in enumerate(linear_ids):
            w = params_by_layer[li]["weight"].detach().float().cpu()
            b = params_by_layer[li].get("bias")
            b_t = b.detach().float().cpu() if b is not None else None

            lin = nn.Linear(int(w.shape[1]), int(w.shape[0]), bias=(b_t is not None))
            lin.weight.copy_(w)
            if b_t is not None:
                lin.bias.copy_(b_t)
            layers.append(lin)

            if i < len(linear_ids) - 1:
                layers.append(nn.ReLU(inplace=False))

    head = nn.Sequential(*layers)
    meta = {
        "image_head_available": True,
        "image_head_kind": head_kind,
        "image_head_branch": int(branch),
        "image_head_linear_layers": int(len(linear_ids)),
        "image_head_out_dim": int(params_by_layer[linear_ids[-1]]["weight"].shape[0]),
    }
    return head, meta


def _torchvision_vit_tokens(backbone: Any, x_bchw: Any):
    import torch

    n = x_bchw.shape[0]
    x = backbone._process_input(x_bchw)
    cls = backbone.class_token.expand(n, -1, -1)
    x = torch.cat((cls, x), dim=1)
    x = backbone.encoder(x)
    return x


@lru_cache(maxsize=8)
def _load_wildsat_cached(
    *,
    ckpt_path: str,
    arch_hint: str,
    prefer_branch: int,
    dev: str,
) -> Tuple[Any, Optional[Any], Dict[str, Any]]:
    ensure_torch()
    import torch

    p = os.path.expanduser(ckpt_path)
    if not os.path.exists(p):
        raise ModelError(f"WildSAT checkpoint not found: {p}")

    obj = torch.load(p, map_location="cpu", weights_only=False)
    sat_sd, opt_meta = _extract_satellite_state_dict(obj)
    arch = _resolve_wildsat_arch(arch_hint=arch_hint, opt_meta=opt_meta, state_dict=sat_sd)

    backbone = _build_backbone(arch)

    backbone_sd = {
        k[len("backbone.") :]: v
        for k, v in sat_sd.items()
        if k.startswith("backbone.") and hasattr(v, "shape")
    }
    if not backbone_sd:
        raise ModelError("WildSAT checkpoint has no backbone parameters after normalization.")

    msg = backbone.load_state_dict(backbone_sd, strict=False)
    loaded_count = len(backbone_sd)
    if loaded_count < 10:
        raise ModelError(
            "Too few WildSAT backbone parameters were loaded. "
            "Checkpoint may not match the selected architecture."
        )

    image_head, head_meta = _build_branch_head_from_state_dict(sat_sd, prefer_branch=int(prefer_branch))

    try:
        backbone = backbone.to(dev).eval()
    except Exception:
        pass
    if image_head is not None:
        try:
            image_head = image_head.to(dev).eval()
        except Exception:
            pass

    p0 = None
    for _, p0cand in backbone.named_parameters():
        if p0cand is not None and p0cand.numel() > 0:
            p0 = p0cand.detach()
            break
    if p0 is None:
        raise ModelError("WildSAT backbone has no parameters; cannot verify checkpoint load.")
    if not torch.isfinite(p0).all():
        raise ModelError("WildSAT backbone parameters contain NaN/Inf; checkpoint load likely failed.")
    p0f = p0.float()

    meta = {
        "ckpt_path": p,
        "ckpt_size": int(os.path.getsize(p)),
        "arch": arch,
        "device": str(dev),
        "opt_satellite_encoder": opt_meta.get("satellite_encoder"),
        "opt_satellite_head": opt_meta.get("satellite_head"),
        "opt_common_embed_dim": opt_meta.get("common_embed_dim"),
        "backbone_loaded_keys": int(loaded_count),
        "backbone_missing_keys": int(len(getattr(msg, "missing_keys", []))),
        "backbone_unexpected_keys": int(len(getattr(msg, "unexpected_keys", []))),
        "param_mean": float(p0f.mean().cpu()),
        "param_std": float(p0f.std().cpu()),
        "param_absmax": float(p0f.abs().max().cpu()),
        **head_meta,
    }
    return backbone, image_head, meta


def _load_wildsat(
    *,
    ckpt_path: str,
    arch_hint: str,
    prefer_branch: int,
    device: str,
) -> Tuple[Any, Optional[Any], Dict[str, Any], str]:
    dev = _resolve_device(device)
    backbone, image_head, meta = _load_wildsat_cached(
        ckpt_path=os.path.expanduser(ckpt_path),
        arch_hint=str(arch_hint),
        prefer_branch=int(prefer_branch),
        dev=dev,
    )
    return backbone, image_head, meta, dev


def _raw_chw_to_rgb_u8(raw_chw: np.ndarray, *, image_size: int, norm_mode: str) -> np.ndarray:
    if raw_chw.ndim != 3 or int(raw_chw.shape[0]) != 3:
        raise ModelError(f"WildSAT expects input_chw as CHW with C=3, got {getattr(raw_chw, 'shape', None)}")

    x = np.asarray(raw_chw, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.clip(x, 0.0, 10000.0) / 10000.0

    mode = str(norm_mode).lower().strip()
    if mode in {"minmax", "per_tile_minmax", "tile_minmax"}:
        lo = float(np.min(x))
        hi = float(np.max(x))
        if hi - lo > 1e-6:
            x = (x - lo) / (hi - lo)
        else:
            x = np.zeros_like(x, dtype=np.float32)
    elif mode in {"unit", "unit_scale", "reflectance", "none", "raw"}:
        pass
    else:
        raise ModelError(
            f"Unknown WildSAT normalization mode '{norm_mode}'. "
            "Use one of: minmax, unit_scale, none."
        )

    rgb_u8 = np.clip(x.transpose(1, 2, 0) * 255.0, 0.0, 255.0).astype(np.uint8)
    return resize_rgb_u8(rgb_u8, int(image_size))


def _rgb_u8_to_bchw_unit(rgb_u8: np.ndarray) -> np.ndarray:
    if rgb_u8.dtype != np.uint8 or rgb_u8.ndim != 3 or int(rgb_u8.shape[2]) != 3:
        raise ModelError(f"Expected uint8 HWC RGB image, got dtype={rgb_u8.dtype}, shape={rgb_u8.shape}")
    x = rgb_u8.astype(np.float32) / 255.0
    return x.transpose(2, 0, 1)[None, ...].astype(np.float32)


def _wildsat_forward(
    backbone: Any,
    image_head: Optional[Any],
    x_bchw: np.ndarray,
    *,
    arch: str,
    feature_source: str,
    pooled_from_tokens: bool,
    pooling: str,
    make_grid: bool,
    grid_from_tokens: bool,
    device: str,
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    ensure_torch()
    import torch

    dev = _resolve_device(device)
    x = torch.from_numpy(x_bchw.astype(np.float32, copy=False)).to(dev)

    try:
        backbone = backbone.to(dev).eval()
    except Exception:
        pass
    if image_head is not None:
        try:
            image_head = image_head.to(dev).eval()
        except Exception:
            pass

    with torch.no_grad():
        tok_np: Optional[np.ndarray] = None
        if (arch in {"vitb16", "vitl16"}) and (make_grid or pooled_from_tokens) and grid_from_tokens:
            try:
                toks = _torchvision_vit_tokens(backbone, x)
                tok_np = toks[0].detach().float().cpu().numpy().astype(np.float32)
            except Exception:
                tok_np = None

        out = backbone(x)

        if not hasattr(out, "ndim"):
            raise ModelError(f"WildSAT backbone returned unsupported output type: {type(out)}")

        if int(out.ndim) == 2:
            backbone_vec_t = out
        elif int(out.ndim) == 4:
            backbone_vec_t = out.mean(dim=(2, 3))
        else:
            raise ModelError(f"WildSAT backbone output has unsupported shape: {tuple(out.shape)}")

        head_vec_t = None
        if image_head is not None:
            try:
                head_out = image_head(backbone_vec_t)
                if hasattr(head_out, "ndim") and int(head_out.ndim) == 2:
                    head_vec_t = head_out
            except Exception:
                head_vec_t = None

        fs = str(feature_source).lower().strip()
        if fs in {"image_head", "img_head", "image"}:
            if head_vec_t is not None:
                final_vec_t = head_vec_t
                used_source = "image_head"
            else:
                final_vec_t = backbone_vec_t
                used_source = "backbone_fallback"
        elif fs in {"backbone", "encoder"}:
            final_vec_t = backbone_vec_t
            used_source = "backbone"
        elif fs in {"auto", "default"}:
            if head_vec_t is not None:
                final_vec_t = head_vec_t
                used_source = "image_head"
            else:
                final_vec_t = backbone_vec_t
                used_source = "backbone"
        else:
            raise ModelError("RS_EMBED_WILDSAT_FEATURE must be one of: auto, image_head, backbone")

        vec_np = final_vec_t[0].detach().float().cpu().numpy().astype(np.float32)

        pooled_cls_removed = False
        if pooled_from_tokens and tok_np is not None:
            vec_np, pooled_cls_removed = pool_from_tokens(tok_np, pooling)

        grid_np: Optional[np.ndarray] = None
        grid_meta: Dict[str, Any] = {}
        if make_grid:
            if tok_np is not None:
                grid_np, (gh, gw), cls_removed = tokens_to_grid_dhw(tok_np)
                grid_meta = {
                    "grid_kind": "vit_patch_tokens",
                    "grid_hw": (int(gh), int(gw)),
                    "grid_shape": tuple(grid_np.shape),
                    "grid_cls_removed": bool(cls_removed),
                }
            else:
                grid_np = vec_np[:, None, None].astype(np.float32)
                grid_meta = {
                    "grid_kind": "vector_as_1x1",
                    "grid_hw": (1, 1),
                    "grid_shape": tuple(grid_np.shape),
                    "grid_cls_removed": False,
                }

    meta = {
        "feature_source": used_source,
        "feature_dim": int(vec_np.shape[0]),
        "pooled_from_tokens": bool(pooled_from_tokens and (tok_np is not None)),
        "pooled_cls_removed": bool(pooled_cls_removed),
        "tokens_available": bool(tok_np is not None),
        **grid_meta,
    }
    return vec_np, grid_np, meta


@register("wildsat")
class WildSATEmbedder(EmbedderBase):
    DEFAULT_IMAGE_SIZE = 224
    DEFAULT_FETCH_WORKERS = 8

    def __init__(self) -> None:
        self._provider: Optional[Any] = None

    def describe(self) -> Dict[str, Any]:
        return {
            "type": "on_the_fly",
            "backend": ["gee"],
            "inputs": {
                "collection": "COPERNICUS/S2_SR_HARMONIZED",
                "bands": ["B4", "B3", "B2"],
            },
            "temporal": {"mode": "range"},
            "output": ["pooled", "grid"],
            "defaults": {
                "scale_m": 10,
                "cloudy_pct": 30,
                "composite": "median",
                "image_size": self.DEFAULT_IMAGE_SIZE,
                "normalization": "minmax",
                "feature": "image_head",
            },
            "notes": [
                "Requires RS_EMBED_WILDSAT_CKPT pointing to a WildSAT checkpoint (.pth).",
                "If decoder image head weights are present, pooled features default to that branch; otherwise fallback to backbone output.",
            ],
        }

    def _get_provider(self):
        if self._provider is None:
            from ..providers.gee import GEEProvider

            p = GEEProvider(auto_auth=True)
            p.ensure_ready()
            self._provider = p
        return self._provider

    @staticmethod
    def _default_sensor() -> SensorSpec:
        return SensorSpec(
            collection="COPERNICUS/S2_SR_HARMONIZED",
            bands=("B4", "B3", "B2"),
            scale_m=10,
            cloudy_pct=30,
            composite="median",
        )

    @staticmethod
    def _resolve_fetch_workers(n_items: int) -> int:
        v = int(os.environ.get("RS_EMBED_WILDSAT_FETCH_WORKERS", str(WildSATEmbedder.DEFAULT_FETCH_WORKERS)))
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
        input_chw: Optional[np.ndarray] = None,
    ) -> Embedding:
        backend_l = backend.lower().strip()
        if backend_l not in {"gee", "auto"}:
            raise ModelError("wildsat expects backend='gee' (or 'auto').")

        ss = sensor or self._default_sensor()
        t = temporal_to_range(temporal)

        ckpt_path = os.environ.get("RS_EMBED_WILDSAT_CKPT")
        if not ckpt_path:
            raise ModelError(
                "WildSAT checkpoint is required. Set RS_EMBED_WILDSAT_CKPT to a local .pth checkpoint path."
            )

        arch_hint = os.environ.get("RS_EMBED_WILDSAT_ARCH", "auto").strip()
        image_size = int(os.environ.get("RS_EMBED_WILDSAT_IMG", str(self.DEFAULT_IMAGE_SIZE)))
        norm_mode = os.environ.get("RS_EMBED_WILDSAT_NORM", "minmax").strip()
        feature_source = os.environ.get("RS_EMBED_WILDSAT_FEATURE", "image_head").strip()
        prefer_branch = int(os.environ.get("RS_EMBED_WILDSAT_IMAGE_BRANCH", "3"))
        pooled_from_tokens = os.environ.get("RS_EMBED_WILDSAT_POOLED_FROM_TOKENS", "0").strip() in {
            "1",
            "true",
            "True",
        }
        grid_from_tokens = os.environ.get("RS_EMBED_WILDSAT_GRID_FROM_TOKENS", "1").strip() not in {
            "0",
            "false",
            "False",
        }

        if input_chw is None:
            s2_rgb_chw = _fetch_s2_rgb_chw(
                self._get_provider(),
                spatial,
                t,
                scale_m=int(ss.scale_m),
                cloudy_pct=int(ss.cloudy_pct),
                composite=str(ss.composite),
            )
            raw = np.clip(s2_rgb_chw * 10000.0, 0.0, 10000.0).astype(np.float32)
        else:
            raw = np.asarray(input_chw, dtype=np.float32)
            if raw.ndim != 3 or int(raw.shape[0]) != 3:
                raise ModelError(f"input_chw must be CHW with 3 bands for wildsat, got {getattr(raw, 'shape', None)}")
            raw = np.clip(np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 10000.0).astype(np.float32)

        rgb_u8 = _raw_chw_to_rgb_u8(raw, image_size=image_size, norm_mode=norm_mode)
        x_bchw = _rgb_u8_to_bchw_unit(rgb_u8)

        backbone, image_head, lmeta, dev = _load_wildsat(
            ckpt_path=ckpt_path,
            arch_hint=arch_hint,
            prefer_branch=prefer_branch,
            device=device,
        )

        vec, grid, fmeta = _wildsat_forward(
            backbone,
            image_head,
            x_bchw,
            arch=str(lmeta.get("arch") or _normalize_arch_name(arch_hint) or "vitb16"),
            feature_source=feature_source,
            pooled_from_tokens=pooled_from_tokens,
            pooling=output.pooling,
            make_grid=(output.mode == "grid"),
            grid_from_tokens=grid_from_tokens,
            device=dev,
        )

        meta = build_meta(
            model=self.model_name,
            kind="on_the_fly",
            backend="gee",
            source=ss.collection,
            sensor={
                "collection": ss.collection,
                "bands": ("B4", "B3", "B2"),
                "scale_m": int(ss.scale_m),
                "cloudy_pct": int(ss.cloudy_pct),
                "composite": str(ss.composite),
            },
            temporal=t,
            image_size=image_size,
            input_time=temporal_midpoint_str(t),
            extra={
                "start": t.start,
                "end": t.end,
                "normalization": str(norm_mode),
                "device": dev,
                **lmeta,
                **fmeta,
            },
        )

        if output.mode == "pooled":
            ometa = {
                **meta,
                "pooling": output.pooling if fmeta.get("pooled_from_tokens", False) else "identity",
                "pooled_shape": tuple(vec.shape),
            }
            return Embedding(data=vec.astype(np.float32), meta=ometa)

        if output.mode == "grid":
            if grid is None:
                raise ModelError("Internal error: grid output requested but grid tensor is missing.")
            gmeta = {
                **meta,
                "grid_shape": tuple(grid.shape),
                "grid_hw": (int(grid.shape[1]), int(grid.shape[2])),
            }
            da = xr.DataArray(
                grid.astype(np.float32),
                dims=("d", "y", "x"),
                coords={
                    "d": np.arange(grid.shape[0]),
                    "y": np.arange(grid.shape[1]),
                    "x": np.arange(grid.shape[2]),
                },
                name="embedding",
                attrs=gmeta,
            )
            return Embedding(data=da, meta=gmeta)

        raise ModelError(f"Unknown output mode: {output.mode}")

    def get_embeddings_batch(
        self,
        *,
        spatials: list[SpatialSpec],
        temporal: Optional[TemporalSpec] = None,
        sensor: Optional[SensorSpec] = None,
        output: OutputSpec = OutputSpec.pooled(),
        backend: str = "gee",
        device: str = "auto",
    ) -> list[Embedding]:
        if not spatials:
            return []

        backend_l = backend.lower().strip()
        if backend_l not in {"gee", "auto"}:
            raise ModelError("wildsat expects backend='gee' (or 'auto').")

        t = temporal_to_range(temporal)
        ss = sensor or self._default_sensor()
        provider = self._get_provider()
        n = len(spatials)

        scale_m = int(ss.scale_m)
        cloudy_pct = int(ss.cloudy_pct)
        composite = str(ss.composite)

        prefetched_raw: List[Optional[np.ndarray]] = [None] * n

        def _fetch_one(i: int, sp: SpatialSpec) -> Tuple[int, np.ndarray]:
            s2_rgb_chw = _fetch_s2_rgb_chw(
                provider,
                sp,
                t,
                scale_m=scale_m,
                cloudy_pct=cloudy_pct,
                composite=composite,
            )
            raw = np.clip(s2_rgb_chw * 10000.0, 0.0, 10000.0).astype(np.float32)
            return i, raw

        mw = self._resolve_fetch_workers(n)
        if mw == 1:
            for i, sp in enumerate(spatials):
                ii, raw = _fetch_one(i, sp)
                prefetched_raw[ii] = raw
        else:
            with ThreadPoolExecutor(max_workers=mw) as ex:
                futs = [ex.submit(_fetch_one, i, sp) for i, sp in enumerate(spatials)]
                for fut in as_completed(futs):
                    i, raw = fut.result()
                    prefetched_raw[i] = raw

        out: List[Embedding] = []
        for i, sp in enumerate(spatials):
            raw = prefetched_raw[i]
            if raw is None:
                raise ModelError(f"Missing prefetched input at index={i} for wildsat.")
            out.append(
                self.get_embedding(
                    spatial=sp,
                    temporal=t,
                    sensor=ss,
                    output=output,
                    backend=backend,
                    device=device,
                    input_chw=raw,
                )
            )
        return out
