
from __future__ import annotations
from typing import Optional

from .core.registry import get_embedder_cls
from .core.specs import SpatialSpec, TemporalSpec, SensorSpec, OutputSpec
from .core.embedding import Embedding

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

    cls = get_embedder_cls(model)
    embedder = cls()
    return embedder.get_embedding(
        spatial=spatial,
        temporal=temporal,
        sensor=sensor,
        output=output,
        backend=backend,
        device=device,
    )