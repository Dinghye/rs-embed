from __future__ import annotations

import inspect
import os
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, TypeVar

import numpy as np

from ..core.errors import ModelError
from ..core.specs import SensorSpec, SpatialSpec, TemporalSpec
from ..providers import get_provider, has_provider, list_providers
from ..providers.base import ProviderBase

_T = TypeVar("_T")


def normalize_backend_name(backend: str) -> str:
    return str(backend).strip().lower()


def default_provider_backend_name() -> Optional[str]:
    configured = normalize_backend_name(os.environ.get("RS_EMBED_DEFAULT_PROVIDER", ""))
    if configured:
        return configured if has_provider(configured) else None
    providers = list_providers()
    if not providers:
        return None
    if "gee" in providers:
        return "gee"
    return str(providers[0]).strip().lower()


def resolve_provider_backend_name(
    backend: str,
    *,
    allow_auto: bool = True,
    auto_backend: Optional[str] = None,
) -> Optional[str]:
    b = normalize_backend_name(backend)
    if allow_auto and b == "auto":
        resolved_auto = normalize_backend_name(auto_backend) if auto_backend is not None else default_provider_backend_name()
        if not resolved_auto:
            return None
        b = resolved_auto
    if has_provider(b):
        return b
    return None


def is_provider_backend(
    backend: str,
    *,
    allow_auto: bool = True,
    auto_backend: Optional[str] = None,
) -> bool:
    return resolve_provider_backend_name(
        backend,
        allow_auto=allow_auto,
        auto_backend=auto_backend,
    ) is not None


def get_cached_provider(
    provider_cache: Dict[str, ProviderBase],
    *,
    backend: str,
    allow_auto: bool = True,
    auto_backend: Optional[str] = None,
) -> ProviderBase:
    b = resolve_provider_backend_name(
        backend,
        allow_auto=allow_auto,
        auto_backend=auto_backend,
    )
    if b is None:
        raise ModelError(f"Unsupported provider backend={backend!r}.")
    p = provider_cache.get(b)
    if p is None:
        kwargs = provider_init_kwargs(b)
        p = get_provider(b, **kwargs)
        provider_cache[b] = p
    p.ensure_ready()
    return p


def provider_init_kwargs(backend: str) -> Dict[str, Any]:
    """Provider-specific constructor kwargs, centralized outside embedders."""
    b = normalize_backend_name(backend)
    if b == "gee":
        return {"auto_auth": True}
    return {}


def create_provider_for_backend(
    backend: str,
    *,
    allow_auto: bool = True,
    auto_backend: Optional[str] = None,
) -> ProviderBase:
    b = resolve_provider_backend_name(
        backend,
        allow_auto=allow_auto,
        auto_backend=auto_backend,
    )
    if b is None:
        raise ModelError(f"Unsupported provider backend={backend!r}.")
    p = get_provider(b, **provider_init_kwargs(b))
    p.ensure_ready()
    return p


def resolve_device_auto_torch(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def load_cached_with_device(
    cached_loader: Callable[..., _T],
    *,
    device: str,
    **kwargs: Any,
) -> Tuple[_T, str]:
    """Resolve device once and call a cached loader that accepts `dev=...`."""
    dev = resolve_device_auto_torch(device)
    loaded = cached_loader(dev=dev, **kwargs)
    return loaded, dev


def fetch_collection_patch_chw(
    provider: ProviderBase,
    *,
    spatial: SpatialSpec,
    temporal: Optional[TemporalSpec],
    collection: str,
    bands: Tuple[str, ...],
    scale_m: int = 10,
    cloudy_pct: Optional[int] = 30,
    composite: str = "median",
    fill_value: float = 0.0,
) -> np.ndarray:
    """Fetch a provider patch as CHW float32 using shared SensorSpec logic."""
    sensor = SensorSpec(
        collection=str(collection),
        bands=tuple(str(b) for b in bands),
        scale_m=int(scale_m),
        cloudy_pct=(int(cloudy_pct) if cloudy_pct is not None else None),  # type: ignore[arg-type]
        fill_value=float(fill_value),
        composite=str(composite),
    )
    return fetch_sensor_patch_chw(
        provider,
        spatial=spatial,
        temporal=temporal,
        sensor=sensor,
    )


def fetch_gee_patch_chw(
    provider: ProviderBase,
    *,
    spatial: SpatialSpec,
    temporal: Optional[TemporalSpec],
    collection: str,
    bands: Tuple[str, ...],
    scale_m: int = 10,
    cloudy_pct: Optional[int] = 30,
    composite: str = "median",
    fill_value: float = 0.0,
) -> np.ndarray:
    """Backward-compatible alias for historical helper name."""
    return fetch_collection_patch_chw(
        provider,
        spatial=spatial,
        temporal=temporal,
        collection=collection,
        bands=bands,
        scale_m=scale_m,
        cloudy_pct=cloudy_pct,
        composite=composite,
        fill_value=fill_value,
    )


def fetch_sensor_patch_chw(
    provider: ProviderBase,
    *,
    spatial: SpatialSpec,
    temporal: Optional[TemporalSpec],
    sensor: SensorSpec,
    to_float_image: bool = False,
) -> np.ndarray:
    """Fetch a CHW patch from a concrete SensorSpec."""
    x = provider.fetch_sensor_patch_chw(
        spatial=spatial,
        temporal=temporal,
        sensor=sensor,
        to_float_image=to_float_image,
    )
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 3:
        raise ModelError(f"Expected CHW array from provider fetch, got shape={getattr(arr, 'shape', None)}")
    if int(arr.shape[0]) != len(sensor.bands):
        raise ModelError(
            f"Provider fetch channel mismatch: got C={int(arr.shape[0])}, expected C={len(sensor.bands)} "
            f"for collection={sensor.collection}"
        )
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def fetch_s2_rgb_chw(
    provider: ProviderBase,
    *,
    spatial: SpatialSpec,
    temporal: TemporalSpec,
    scale_m: int = 10,
    cloudy_pct: int = 30,
    composite: str = "median",
) -> np.ndarray:
    """Fetch Sentinel-2 RGB as float32 CHW in [0,1]."""
    raw = fetch_collection_patch_chw(
        provider,
        spatial=spatial,
        temporal=temporal,
        collection="COPERNICUS/S2_SR_HARMONIZED",
        bands=("B4", "B3", "B2"),
        scale_m=int(scale_m),
        cloudy_pct=int(cloudy_pct),
        composite=str(composite),
        fill_value=0.0,
    )
    return np.clip(raw / 10000.0, 0.0, 1.0).astype(np.float32)


def fetch_s1_vvvh_raw_chw(
    provider: ProviderBase,
    *,
    spatial: SpatialSpec,
    temporal: TemporalSpec,
    scale_m: int = 10,
    orbit: Optional[str] = None,
    use_float_linear: bool = True,
    composite: str = "median",
    fill_value: float = 0.0,
) -> np.ndarray:
    """Fetch Sentinel-1 VV/VH as raw float32 CHW."""
    arr = provider.fetch_s1_vvvh_raw_chw(
        spatial=spatial,
        temporal=temporal,
        scale_m=int(scale_m),
        orbit=orbit,
        use_float_linear=bool(use_float_linear),
        composite=str(composite),
        fill_value=float(fill_value),
    )
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 3 or int(arr.shape[0]) != 2:
        raise ModelError(f"Expected S1 VV/VH CHW with C=2, got shape={getattr(arr, 'shape', None)}")
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def normalize_s1_vvvh_chw(raw_chw: np.ndarray) -> np.ndarray:
    """Convert raw S1 VV/VH to numerically stable [0,1] CHW."""
    arr = np.asarray(raw_chw, dtype=np.float32)
    if arr.ndim != 3 or int(arr.shape[0]) != 2:
        raise ModelError(f"Expected raw S1 VV/VH CHW with C=2, got shape={getattr(arr, 'shape', None)}")
    x = np.log1p(np.maximum(arr, 0.0))
    denom = np.percentile(x, 99) if np.isfinite(x).all() else 1.0
    denom = float(denom) if float(denom) > 0 else 1.0
    return np.clip(x / denom, 0.0, 1.0).astype(np.float32)


def fetch_s2_multiframe_raw_tchw(
    provider: ProviderBase,
    *,
    spatial: SpatialSpec,
    temporal: TemporalSpec,
    bands: Sequence[str],
    n_frames: int = 8,
    collection: str = "COPERNICUS/S2_SR_HARMONIZED",
    scale_m: int = 10,
    cloudy_pct: Optional[int] = 30,
    composite: str = "median",
    fill_value: float = 0.0,
) -> np.ndarray:
    """Fetch an S2 time series as raw float32 [T,C,H,W] in [0,10000]."""
    arr = provider.fetch_multiframe_collection_raw_tchw(
        spatial=spatial,
        temporal=temporal,
        collection=str(collection),
        bands=tuple(str(b) for b in bands),
        n_frames=int(n_frames),
        scale_m=int(scale_m),
        cloudy_pct=(int(cloudy_pct) if cloudy_pct is not None else None),
        composite=str(composite),
        fill_value=float(fill_value),
    )
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 4:
        raise ModelError(f"Expected TCHW array, got shape={getattr(arr, 'shape', None)}")
    if int(arr.shape[1]) != len(tuple(bands)):
        raise ModelError(
            f"Time series channel mismatch: got C={int(arr.shape[1])}, expected C={len(tuple(bands))}"
        )
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def call_provider_getter(
    getter: Callable[..., ProviderBase],
    backend: str,
) -> ProviderBase:
    """Call _get_provider with backward-compatible signature handling.

    Some tests monkeypatch `_get_provider` as a zero-arg lambda. This helper lets
    new backend-aware call sites remain compatible with those older patches.
    """
    try:
        sig = inspect.signature(getter)
        if len(sig.parameters) == 0:
            return getter()
    except Exception:
        pass

    try:
        return getter(backend)
    except TypeError:
        return getter()
