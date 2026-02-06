from __future__ import annotations
from typing import Any, Dict, Optional

import numpy as np

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
        input_chw: Optional[np.ndarray] = None,
    ) -> Embedding:
        
        raise NotImplementedError

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
        """Default batch implementation: loop over spatials.

        Embedders that can do true batching (e.g. torch models) should override.
        """
        return [
            self.get_embedding(
                spatial=s,
                temporal=temporal,
                sensor=sensor,
                output=output,
                backend=backend,
                device=device,
            )
            for s in spatials
        ]
