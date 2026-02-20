from __future__ import annotations

from typing import Any, Optional

from .errors import ModelError
from .specs import SensorSpec, TemporalSpec
from ..providers import get_provider, has_provider


def build_provider_image(
    *,
    sensor: SensorSpec,
    temporal: Optional[TemporalSpec],
    backend: str = "gee",
    region: Optional[Any] = None,
) -> Any:
    """Build provider image through backend provider abstraction."""
    backend_name = str(backend).strip().lower()
    if not backend_name:
        raise ModelError("backend must be a non-empty provider name.")
    if not has_provider(backend_name):
        raise ModelError(f"Unknown provider backend '{backend_name}'.")
    kwargs = {"auto_auth": True} if backend_name == "gee" else {}
    p = get_provider(backend_name, **kwargs)
    p.ensure_ready()
    return p.build_image(sensor=sensor, temporal=temporal, region=region)


def build_gee_image(
    *,
    sensor: SensorSpec,
    temporal: Optional[TemporalSpec],
    region: Optional[Any] = None,
) -> Any:
    """Backwards-compatible wrapper for GEE backend."""
    return build_provider_image(
        sensor=sensor,
        temporal=temporal,
        backend="gee",
        region=region,
    )
