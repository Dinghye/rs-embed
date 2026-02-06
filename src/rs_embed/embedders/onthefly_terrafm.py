# src/rs_embed/embedders/onthefly_terrafm.py
from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import xarray as xr

from ..core.registry import register
from ..core.embedding import Embedding
from ..core.errors import ModelError
from ..core.specs import SpatialSpec, TemporalSpec, SensorSpec, OutputSpec
from ..providers.gee import GEEProvider
from .base import EmbedderBase
from .meta_utils import build_meta, temporal_midpoint_str


HF_REPO_ID = "MBZUAI/TerraFM"
HF_CODE_FILE = "terrafm.py"
HF_WEIGHT_FILE_B = "TerraFM-B.pth"


# -----------------------------
# Small utils
# -----------------------------
def _auto_device(device: str) -> str:
    import torch

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _resize_chw_to_224(x_chw: np.ndarray, *, size: int = 224) -> np.ndarray:
    """Resize CHW float32 -> CHW float32 (bilinear)."""
    import torch
    import torch.nn.functional as F

    if x_chw.ndim != 3:
        raise ModelError(f"Expected CHW array, got {x_chw.shape}")
    x = torch.from_numpy(x_chw).unsqueeze(0)  # [1,C,H,W]
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    return x[0].cpu().numpy().astype(np.float32)


# -----------------------------
# GEE: Fetch S2 (12 bands, SR)
# -----------------------------
_S2_SR_12_BANDS = [
    "B1", "B2", "B3", "B4", "B5", "B6",
    "B7", "B8", "B8A", "B9", "B11", "B12",
]


def _fetch_s2_sr_12_chw(
    provider: GEEProvider,
    spatial: SpatialSpec,
    temporal: TemporalSpec,
    *,
    scale_m: int = 10,
    cloudy_pct: int = 30,
    composite: str = "median",
) -> np.ndarray:
    """
    Returns CHW float32 in [0,1] approx, resized later to 224.
    COPERNICUS/S2_SR_HARMONIZED bands are UINT16 SR scaled by 10000.  [oai_citation:3‡Google for Developers](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED?utm_source=chatgpt.com)
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

    img = img.select(_S2_SR_12_BANDS).reproject(crs="EPSG:3857", scale=scale_m)

    rect = img.sampleRectangle(region=region, defaultValue=0).getInfo()
    props = rect["properties"]

    bands = []
    for b in _S2_SR_12_BANDS:
        arr = np.array(props[b], dtype=np.float32) / 10000.0
        bands.append(arr)

    x = np.stack(bands, axis=0).astype(np.float32)  # [12,H,W]
    x = np.clip(x, 0.0, 1.0)
    return x


# -----------------------------
# GEE: Fetch S1 (VV/VH)
# -----------------------------
def _fetch_s1_vvvh_chw(
    provider: GEEProvider,
    spatial: SpatialSpec,
    temporal: TemporalSpec,
    *,
    scale_m: int = 10,
    orbit: Optional[str] = None,   # "ASCENDING" | "DESCENDING"
    use_float_linear: bool = True,
    composite: str = "median",
) -> np.ndarray:
    """
    Returns CHW float32 [2,H,W].

    Notes:
    - COPERNICUS/S1_GRD catalog mentions "log scaling" in description.  [oai_citation:4‡Google for Developers](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD?utm_source=chatgpt.com)
    - For linear data, community notes COPERNICUS/S1_GRD_FLOAT exists.  [oai_citation:5‡STEP Forum](https://forum.step.esa.int/t/is-google-earth-engine-sentinel-1-grd-product-sigma0/27239?utm_source=chatgpt.com)

    This function defaults to S1_GRD_FLOAT (linear). We then apply a mild log1p compression
    + normalization to keep values numerically sane for a vision backbone.
    """
    import ee  # lazy

    region = provider.get_region_3857(spatial)

    collection_id = "COPERNICUS/S1_GRD_FLOAT" if use_float_linear else "COPERNICUS/S1_GRD"
    col = (
        ee.ImageCollection(collection_id)
        .filterDate(temporal.start, temporal.end)
        .filterBounds(region)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
    )

    if orbit:
        col = col.filter(ee.Filter.eq("orbitProperties_pass", orbit))

    if composite == "median":
        img = col.median()
    elif composite == "mosaic":
        img = col.mosaic()
    else:
        raise ModelError(f"Unknown composite='{composite}'. Use 'median' or 'mosaic'.")

    img = img.select(["VV", "VH"]).reproject(crs="EPSG:3857", scale=scale_m)

    rect = img.sampleRectangle(region=region, defaultValue=0.0).getInfo()
    props = rect["properties"]

    vv = np.array(props["VV"], dtype=np.float32)
    vh = np.array(props["VH"], dtype=np.float32)

    x = np.stack([vv, vh], axis=0).astype(np.float32)  # [2,H,W]

    # simple, robust compression/normalization (works whether linear or dB-ish)
    x = np.log1p(np.maximum(x, 0.0))
    # normalize per-chip (avoid division by 0)
    denom = np.percentile(x, 99) if np.isfinite(x).all() else 1.0
    denom = float(denom) if denom > 0 else 1.0
    x = np.clip(x / denom, 0.0, 1.0)

    return x


# -----------------------------
# HF asset management (strict)
# -----------------------------
@lru_cache(maxsize=8)
def _ensure_hf_terrafm_assets(
    repo_id: str,
    *,
    auto_download: bool = True,
    cache_dir: Optional[str] = None,
    min_bytes: int = 50 * 1024 * 1024,
) -> Tuple[str, str]:
    """
    Returns (local_py_path, local_weight_path).
    TerraFM HF uses .pth weights, not standard transformers files.
    """
    try:
        from huggingface_hub import hf_hub_download
    except Exception as e:
        raise ModelError("Install huggingface_hub: pip install huggingface_hub") from e

    py_path = hf_hub_download(repo_id=repo_id, filename=HF_CODE_FILE, cache_dir=cache_dir)
    wt_path = hf_hub_download(repo_id=repo_id, filename=HF_WEIGHT_FILE_B, cache_dir=cache_dir)

    if not os.path.exists(py_path):
        raise ModelError(f"Failed to download '{HF_CODE_FILE}' from {repo_id}.")
    if not os.path.exists(wt_path):
        raise ModelError(f"Failed to download '{HF_WEIGHT_FILE_B}' from {repo_id}.")

    # size sanity: avoid pointer/placeholder
    sz = os.path.getsize(wt_path)
    if sz < min_bytes:
        raise ModelError(
            f"Found '{wt_path}' but it's only {sz} bytes — likely not real weights.\n"
            "Fix (if using LFS/xet):\n"
            "  pip install -U hf_xet\n"
            "  (optional) pip install -U \"huggingface_hub[hf_transfer]\"\n"
            "Then delete the cached snapshot and re-run.\n"
        )

    return py_path, wt_path


@lru_cache(maxsize=8)
def _load_terrafm_module(local_py_path: str):
    """Dynamic import terrafm.py from downloaded file."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("terrafm_impl", local_py_path)
    if spec is None or spec.loader is None:
        raise ModelError("Failed to create import spec for TerraFM module.")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def _assert_weights_loaded(model) -> Dict[str, float]:
    """Same philosophy as your RemoteCLIP: param stats should not be near-zero."""
    import torch

    p = None
    for _, param in model.named_parameters():
        if param is not None and param.numel() > 0:
            p = param.detach()
            break
    if p is None:
        raise ModelError("TerraFM model has no parameters; cannot verify weights.")
    if not torch.isfinite(p).all():
        raise ModelError("TerraFM parameters contain NaN/Inf; load likely failed.")

    p_f = p.float()
    std = float(p_f.std().cpu())
    mx = float(p_f.abs().max().cpu())
    mean = float(p_f.mean().cpu())
    if std < 1e-6 and mx < 1e-5:
        raise ModelError("TerraFM parameters look uninitialized (near-zero stats).")
    return {"param_mean": mean, "param_std": std, "param_absmax": mx}


@lru_cache(maxsize=4)
def _load_terrafm_b(
    *,
    auto_download: bool = True,
    cache_dir: Optional[str] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Returns (model, weight_meta).
    """
    import torch

    py_path, wt_path = _ensure_hf_terrafm_assets(
        HF_REPO_ID, auto_download=auto_download, cache_dir=cache_dir
    )
    mod = _load_terrafm_module(py_path)

    if not hasattr(mod, "terrafm_base"):
        raise ModelError("Downloaded terrafm.py has no 'terrafm_base()' factory.")

    model = mod.terrafm_base()
    state = torch.load(wt_path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    stats = _assert_weights_loaded(model)

    meta = {
        "hf_repo": HF_REPO_ID,
        "code_file": py_path,
        "weight_file": wt_path,
        "weight_file_size": os.path.getsize(wt_path),
        "weights_verified": True,
        **stats,
    }
    return model, meta


# -----------------------------
# TerraFM forward adapters
# -----------------------------
def _terrafm_pooled_and_grid(
    model,
    x_bchw: "np.ndarray",
    *,
    device: str,
    want_grid: bool,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Returns (pooled_vec[D], grid_dhw[D,Ht,Wt] or None)
    """
    import torch

    dev = _auto_device(device)
    model = model.to(dev).eval()

    x = torch.from_numpy(x_bchw).to(dev)  # [B,C,H,W]
    with torch.no_grad():
        pooled = model(x)  # TerraFM forward returns CLS embedding (B,D)
        pooled_np = pooled[0].detach().float().cpu().numpy().astype(np.float32)

        if not want_grid:
            return pooled_np, None

        # extract_feature returns list of feature maps; we grab last layer by default
        depth = len(getattr(model, "blocks", []))
        if depth <= 0 or not hasattr(model, "extract_feature"):
            raise ModelError("TerraFM model does not expose extract_feature/blocks for grid output.")

        last_idx = depth - 1
        feats = model.extract_feature(x, return_h_w=True, out_indices=[last_idx])
        # feats[-1] is (B, C, H, W) for the requested index
        fmap = feats[-1]
        grid = fmap[0].detach().float().cpu().numpy().astype(np.float32)  # [D,Ht,Wt]
        return pooled_np, grid


# -----------------------------
# Embedder
# -----------------------------
@register("terrafm_b")
class TerraFMBEmbedder(EmbedderBase):
    """
    ROI -> (GEE S2 SR 12-band OR S1 VV/VH) -> TerraFM-B -> pooled or grid embedding

    - OutputSpec.pooled(): vec [D]
    - OutputSpec.grid():  grid [D, Ht, Wt] (model-native feature map grid)
    """

    def describe(self) -> Dict[str, Any]:
        return {
            "type": "on_the_fly",
            "backend": ["gee", "tensor"],
            "inputs": {
                "s2_sr": {"collection": "COPERNICUS/S2_SR_HARMONIZED", "bands": _S2_SR_12_BANDS},
                "s1": {"collection": "COPERNICUS/S1_GRD_FLOAT (default) or COPERNICUS/S1_GRD", "bands": ["VV", "VH"]},
            },
            "temporal": {"mode": "range"},
            "output": ["pooled", "grid"],
            "defaults": {
                "scale_m": 10,
                "cloudy_pct": 30,
                "composite": "median",
                "modality": "s2",  # or "s1"
                "orbit": None,
                "use_float_linear": True,
                "image_size": 224,
            },
            "notes": "grid output is model feature-map grid (not pixel grid).",
        }

    
    def __init__(self) -> None:
        self._provider: Optional[GEEProvider] = None

    def _get_provider(self) -> GEEProvider:
        if self._provider is None:
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
            input_chw: Optional[np.ndarray] = None,
        ) -> Embedding:
            backend_l = backend.lower()

            # defaults / overrides (match your style: sensor carries overrides)
            modality = getattr(sensor, "modality", "s2") if sensor else "s2"
            modality = str(modality).lower()

            scale_m = getattr(sensor, "scale_m", 10) if sensor else 10
            cloudy_pct = getattr(sensor, "cloudy_pct", 30) if sensor else 30
            composite = getattr(sensor, "composite", "median") if sensor else "median"
            orbit = getattr(sensor, "orbit", None) if sensor else None
            use_float_linear = bool(getattr(sensor, "use_float_linear", True)) if sensor else True

            image_size = 224
            cache_dir = (
                os.environ.get("HUGGINGFACE_HUB_CACHE")
                or os.environ.get("HF_HOME")
                or os.environ.get("HUGGINGFACE_HOME")
            )

            # For optional on-the-fly input inspection
            check_meta: Dict[str, Any] = {}

            # -----------------
            # Build input tensor
            # -----------------
            if backend_l == "tensor":
                if sensor is None or not hasattr(sensor, "data"):
                    raise ModelError("backend='tensor' requires sensor.data as CHW or BCHW numpy/torch.")
                x = sensor.data
                # accept np or torch
                try:
                    import torch
                    if torch.is_tensor(x):
                        x = x.detach().cpu().numpy()
                except Exception:
                    pass
                x = np.asarray(x)
                if x.ndim == 3:
                    x_bchw = x[None, ...]
                elif x.ndim == 4:
                    x_bchw = x
                else:
                    raise ModelError(f"Expected CHW or BCHW, got shape={x.shape}")

                # resize to 224 to match TerraFM patch embed
                if x_bchw.shape[-2:] != (image_size, image_size):
                    x_bchw = np.stack([_resize_chw_to_224(xi, size=image_size) for xi in x_bchw], axis=0)

            elif backend_l == "gee":
                if temporal is None:
                    raise ModelError("terrafm_b_gee requires TemporalSpec.range(start,end).")
                temporal.validate()
                if temporal.mode != "range":
                    raise ModelError("terrafm_b_gee requires TemporalSpec.range in v0.1.")

                provider = self._get_provider()

                if input_chw is None:
                    if modality == "s2":
                        x_chw = _fetch_s2_sr_12_chw(
                            provider, spatial, temporal, scale_m=scale_m, cloudy_pct=cloudy_pct, composite=composite
                        )  # [12,H,W]
                    elif modality == "s1":
                        x_chw = _fetch_s1_vvvh_chw(
                            provider,
                            spatial,
                            temporal,
                            scale_m=scale_m,
                            orbit=orbit,
                            use_float_linear=use_float_linear,
                            composite=composite,
                        )  # [2,H,W]
                    else:
                        raise ModelError("modality must be 's2' or 's1'.")
                else:
                    # input_chw is expected to be raw provider values in the order implied by `sensor.bands`
                    if modality == "s2":
                        if input_chw.ndim != 3 or int(input_chw.shape[0]) != 12:
                            raise ModelError(
                                f"input_chw must be CHW with 12 bands for TerraFM S2, got {getattr(input_chw,'shape',None)}"
                            )
                        x_chw = np.clip(input_chw.astype(np.float32) / 10000.0, 0.0, 1.0)
                    elif modality == "s1":
                        if input_chw.ndim != 3 or int(input_chw.shape[0]) != 2:
                            raise ModelError(
                                f"input_chw must be CHW with 2 bands (VV,VH) for TerraFM S1, got {getattr(input_chw,'shape',None)}"
                            )
                        x = input_chw.astype(np.float32)
                        x = np.log1p(np.maximum(x, 0.0))
                        denom = np.percentile(x, 99) if np.isfinite(x).all() else 1.0
                        denom = float(denom) if denom > 0 else 1.0
                        x_chw = np.clip(x / denom, 0.0, 1.0).astype(np.float32)
                    else:
                        raise ModelError("modality must be 's2' or 's1'.")

                # Optional: inspect on-the-fly GEE input
                from ..core.input_checks import maybe_inspect_chw, checks_should_raise
                check_meta.clear()
                exp_c = 12 if modality == "s2" else 2
                report = maybe_inspect_chw(
                    x_chw,
                    sensor=sensor,
                    name=f"gee_{modality}_chw",
                    expected_channels=exp_c,
                    value_range=(0.0, 1.0),
                    fill_value=0.0,
                    meta=check_meta,
                )
                if report is not None and (not report.get("ok", True)) and checks_should_raise(sensor):
                    raise ModelError("GEE input inspection failed: " + "; ".join(report.get("issues", [])))

                # resize to 224
                x_chw = _resize_chw_to_224(x_chw, size=image_size)
                x_bchw = x_chw[None, ...].astype(np.float32)

            else:
                raise ModelError("terrafm_b_gee supports backend='gee' or 'tensor' only.")

            # channel sanity: TerraFM HF terrafm.py routes by C==2 (S1) else (S2). Keep it strict.
            c = int(x_bchw.shape[1])
            if c not in (2, 12):
                raise ModelError(f"TerraFM expects C=2 (S1 VV/VH) or C=12 (S2 SR bands). Got C={c}")

            # -----------------
            # Load model (strict weights)
            # -----------------
            model, wmeta = _load_terrafm_b(auto_download=True, cache_dir=cache_dir)

            pooled, grid = _terrafm_pooled_and_grid(
                model,
                x_bchw.astype(np.float32),
                device=device,
                want_grid=(output.mode == "grid"),
            )

            temporal_used = temporal if backend_l == "gee" else None
            sensor_meta = None
            source = None
            if backend_l == "gee":
                if modality == "s2":
                    sensor_meta = {
                        "collection": "COPERNICUS/S2_SR_HARMONIZED",
                        "bands": tuple(_S2_SR_12_BANDS),
                        "scale_m": scale_m,
                        "cloudy_pct": cloudy_pct,
                        "composite": composite,
                    }
                    source = sensor_meta["collection"]
                elif modality == "s1":
                    sensor_meta = {
                        "collection": "COPERNICUS/S1_GRD_FLOAT" if use_float_linear else "COPERNICUS/S1_GRD",
                        "bands": ("VV", "VH"),
                        "scale_m": scale_m,
                        "cloudy_pct": cloudy_pct,
                        "composite": composite,
                        "orbit": orbit,
                        "use_float_linear": use_float_linear,
                    }
                    source = sensor_meta["collection"]

            base_meta = build_meta(
                model=self.model_name,
                kind="on_the_fly",
                backend=backend_l,
                source=source,
                sensor=sensor_meta,
                temporal=temporal_used,
                image_size=image_size,
                input_time=temporal_midpoint_str(temporal_used),
                extra={
                    "modality": modality,
                    "scale_m": scale_m if backend_l == "gee" else None,
                    "cloudy_pct": cloudy_pct if backend_l == "gee" else None,
                    "composite": composite if backend_l == "gee" else None,
                    "orbit": orbit if (backend_l == "gee" and modality == "s1") else None,
                    "use_float_linear": use_float_linear if (backend_l == "gee" and modality == "s1") else None,
                    "start": getattr(temporal_used, "start", None),
                    "end": getattr(temporal_used, "end", None),
                    "image_size": image_size,
                    "device": device,
                    "hf_cache_dir": cache_dir,
                    **check_meta,
                    **wmeta,
                },
            )

            # ---- pooled output ----
            if output.mode == "pooled":
                return Embedding(data=pooled.astype(np.float32), meta=base_meta)

            # ---- grid output ----
            if output.mode == "grid":
                if grid is None:
                    raise ModelError("Grid output requested but TerraFM grid extraction returned None.")

                meta = {**base_meta, "grid_type": "feature_map", "grid_shape": tuple(grid.shape)}
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

