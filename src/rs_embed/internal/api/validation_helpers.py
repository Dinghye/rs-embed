from __future__ import annotations

from typing import Optional

from ...core.errors import ModelError
from ...core.specs import OutputSpec, SpatialSpec, TemporalSpec
from ...providers import has_provider


def validate_specs(*, spatial: SpatialSpec, temporal: Optional[TemporalSpec], output: OutputSpec) -> None:
    if not hasattr(spatial, "validate"):
        raise ModelError(f"Invalid spatial spec type: {type(spatial)}")
    spatial.validate()  # type: ignore[call-arg]

    if temporal is not None:
        temporal.validate()

    if output.mode not in ("grid", "pooled"):
        raise ModelError(f"Unknown output mode: {output.mode}")
    if output.scale_m <= 0:
        raise ModelError("output.scale_m must be positive.")
    if output.mode == "pooled" and output.pooling not in ("mean", "max"):
        raise ModelError(f"Unknown pooling method: {output.pooling}")


def assert_supported(embedder, *, backend: str, output: OutputSpec, temporal: Optional[TemporalSpec]) -> None:
    try:
        desc = embedder.describe() or {}
    except Exception:
        return

    backends = desc.get("backend")
    if isinstance(backends, list):
        allowed = [str(b).lower() for b in backends]
        auto_provider_compatible = backend == "auto" and ("provider" in allowed or "gee" in allowed)
        provider_compatible = (
            has_provider(backend)
            and (
                "provider" in allowed
                or "gee" in allowed
            )
        )
        if backend not in allowed and not provider_compatible and not auto_provider_compatible:
            raise ModelError(f"Model '{embedder.model_name}' does not support backend='{backend}'. Supported: {backends}")

    outputs = desc.get("output")
    if isinstance(outputs, list) and output.mode not in outputs:
        raise ModelError(f"Model '{embedder.model_name}' does not support output.mode='{output.mode}'. Supported: {outputs}")

    temporal_hint = desc.get("temporal")
    if isinstance(temporal_hint, dict) and "mode" in temporal_hint:
        mode_hint = str(temporal_hint["mode"])
        if "year" in mode_hint and temporal is not None and getattr(temporal, "mode", None) != "year":
            raise ModelError(f"Model '{embedder.model_name}' expects TemporalSpec.mode='year' (or None).")
