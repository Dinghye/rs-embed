from __future__ import annotations

import inspect
import time
from functools import lru_cache
from threading import RLock
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar

import numpy as np

from ...core.embedding import Embedding
from ...core.registry import get_embedder_cls
from ...core.specs import OutputSpec, SensorSpec, SpatialSpec, TemporalSpec

_T = TypeVar("_T")


@lru_cache(maxsize=32)
def get_embedder_bundle_cached(model: str, backend: str, device: str, sensor_k: Tuple):
    """Return (embedder instance, instance lock)."""
    cls = get_embedder_cls(model)
    emb = cls()
    return emb, RLock()


def sensor_key(sensor: Optional[SensorSpec]) -> Tuple:
    if sensor is None:
        return ("__none__",)
    return (
        sensor.collection,
        sensor.bands,
        int(sensor.scale_m),
        int(sensor.cloudy_pct),
        float(sensor.fill_value),
        str(sensor.composite),
        bool(getattr(sensor, "check_input", False)),
        bool(getattr(sensor, "check_raise", True)),
        getattr(sensor, "check_save_dir", None),
    )


def supports_batch_api(embedder: Any) -> bool:
    """Return True when embedder overrides EmbedderBase.get_embeddings_batch."""
    fn = getattr(type(embedder), "get_embeddings_batch", None)
    if fn is None:
        return False
    from ...embedders.base import EmbedderBase

    return fn is not EmbedderBase.get_embeddings_batch


def supports_prefetched_batch_api(embedder: Any) -> bool:
    """Return True when embedder overrides batch-from-inputs fast path."""
    fn = getattr(type(embedder), "get_embeddings_batch_from_inputs", None)
    if fn is None:
        return False
    from ...embedders.base import EmbedderBase

    return fn is not EmbedderBase.get_embeddings_batch_from_inputs


@lru_cache(maxsize=128)
def embedder_accepts_input_chw(embedder_cls: type) -> bool:
    fn = getattr(embedder_cls, "get_embedding", None)
    if fn is None:
        return False
    try:
        sig = inspect.signature(fn)
    except Exception:
        return False
    if "input_chw" in sig.parameters:
        return True
    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())


def call_embedder_get_embedding(
    *,
    embedder: Any,
    spatial: SpatialSpec,
    temporal: Optional[TemporalSpec],
    sensor: Optional[SensorSpec],
    output: OutputSpec,
    backend: str,
    device: str,
    input_chw: Optional[np.ndarray] = None,
) -> Embedding:
    kwargs: Dict[str, Any] = {
        "spatial": spatial,
        "temporal": temporal,
        "sensor": sensor,
        "output": output,
        "backend": backend,
        "device": device,
    }
    if input_chw is not None and embedder_accepts_input_chw(type(embedder)):
        kwargs["input_chw"] = input_chw
    return embedder.get_embedding(**kwargs)


def run_with_retry(
    fn: Callable[[], _T],
    *,
    retries: int = 0,
    backoff_s: float = 0.0,
) -> _T:
    """Run a callable with bounded retries and optional exponential backoff."""
    tries = max(0, int(retries))
    backoff = max(0.0, float(backoff_s))
    last_err: Optional[Exception] = None
    for attempt in range(tries + 1):
        try:
            return fn()
        except Exception as e:  # pragma: no cover - exercised by call-sites
            last_err = e
            if attempt >= tries:
                raise
            if backoff > 0:
                time.sleep(backoff * (2 ** attempt))
    if last_err is not None:
        raise last_err
    raise RuntimeError("unreachable retry state")
