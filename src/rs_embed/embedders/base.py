from __future__ import annotations
from typing import Any, Dict, Optional

from ..core.specs import SpatialSpec, TemporalSpec, SensorSpec, OutputSpec
from ..core.embedding import Embedding

class EmbedderBase:
    model_name: str = "base"

    def describe(self) -> Dict[str, Any]:
        """Return model/product capabilities and requirements."""
        raise NotImplementedError

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
        raise NotImplementedError