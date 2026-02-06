from typing import List, Optional
 
from .core.embedding import Embedding
from .core.errors import ModelError
from .core.registry import get_embedder_cls
from .core.specs import OutputSpec, SensorSpec, SpatialSpec, TemporalSpec




def get_embedding(
    model: str,
    *,
    spatial: SpatialSpec,
    temporal: Optional[TemporalSpec] = None,
    sensor: Optional[SensorSpec] = None,
    output: OutputSpec = OutputSpec.pooled(),
    backend: str = "gee",
    device: str = "auto",
) -> Embedding:
    # Import embedders here so registration happens after registry is initialized
    from . import embedders  # noqa: F401
    backend_n = backend.lower().strip()
    _validate_specs(spatial=spatial, temporal=temporal, output=output)
    cls = get_embedder_cls(model)
    embedder = cls()
    return embedder.get_embedding(
        spatial=spatial,
        temporal=temporal,
        sensor=sensor,
        output=output,
        backend=backend_n,
        device=device,
    )


def get_embeddings_batch(
    model: str,
    *,
    spatials: List[SpatialSpec],  # get embeddings for multiple spatials
    temporal: Optional[TemporalSpec] = None,
    sensor: Optional[SensorSpec] = None,
    output: OutputSpec = OutputSpec.pooled(),
    backend: str = "gee",
    device: str = "auto",
) -> List[Embedding]:
    # 1. initial embedder
    from . import embedders 
    
     
    backend_n = backend.lower().strip()
    if not isinstance(spatials, list) or len(spatials) == 0:
        raise ModelError("spatials must be a non-empty List[SpatialSpec].")
    _validate_specs(spatial=spatials[0], temporal=temporal, output=output)  # validate output + temporal once

    
    
    
    cls = get_embedder_cls(model)
    embedder = cls()
    _assert_supported(embedder, backend=backend_n, output=output, temporal=temporal)
     
    
    # 2. loop over spatials
    results = []
    results: List[Embedding] = []
    for spatial in spatials:
        _validate_specs(spatial=spatial, temporal=temporal, output=output)
        emb = embedder.get_embedding(
            spatial=spatial,
            temporal=temporal,
            sensor=sensor,
            output=output,
            backend=backend_n,
            device=device,
        )
        results.append(emb)
    return results



def _validate_specs(*, spatial: SpatialSpec, temporal: Optional[TemporalSpec], output: OutputSpec) -> None:
    # Spatial
    if not hasattr(spatial, "validate"):
        raise ModelError(f"Invalid spatial spec type: {type(spatial)}")
    spatial.validate()  # type: ignore[call-arg]

    # Temporal (optional)
    if temporal is not None:
        temporal.validate()

    # Output
    if output.mode not in ("grid", "pooled"):
        raise ModelError(f"Unknown output mode: {output.mode}")
    if output.scale_m <= 0:
        raise ModelError("output.scale_m must be positive.")
    if output.mode == "pooled" and output.pooling not in ("mean", "max"):
        raise ModelError(f"Unknown pooling method: {output.pooling}")


def _assert_supported(embedder, *, backend: str, output: OutputSpec, temporal: Optional[TemporalSpec]) -> None:
    """Best-effort capability check using embedder.describe()."""
    try:
        desc = embedder.describe() or {}
    except Exception:
        return

    backends = desc.get("backend")
    if isinstance(backends, list) and backend not in [b.lower() for b in backends]:
        raise ModelError(f"Model '{embedder.model_name}' does not support backend='{backend}'. Supported: {backends}")

    outputs = desc.get("output")
    if isinstance(outputs, list) and output.mode not in outputs:
        raise ModelError(f"Model '{embedder.model_name}' does not support output.mode='{output.mode}'. Supported: {outputs}")

    # Optional temporal mode hint (only enforce if explicitly declared)
    temporal_hint = desc.get("temporal")
    if isinstance(temporal_hint, dict) and "mode" in temporal_hint:
        mode_hint = str(temporal_hint["mode"])
        if "year" in mode_hint and temporal is not None and getattr(temporal, "mode", None) != "year":
            # Only enforce when user provided temporal but it's incompatible.
            raise ModelError(f"Model '{embedder.model_name}' expects TemporalSpec.mode='year' (or None).")
 