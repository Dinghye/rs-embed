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

    def get_embeddings_batch_from_inputs(
        self,
        *,
        spatials: list[SpatialSpec],
        input_chws: list[np.ndarray],
        temporal: Optional[TemporalSpec] = None,
        sensor: Optional[SensorSpec] = None,
        output: OutputSpec = OutputSpec.pooled(),
        backend: str = "gee",
        device: str = "auto",
    ) -> list[Embedding]:
        """Batch inference with prefetched CHW inputs.

        Default implementation keeps existing behavior by looping through
        get_embedding(..., input_chw=...).
        Embedders that can do true batched model forward should override.
        """
        if len(spatials) != len(input_chws):
            raise ValueError(
                f"spatials/input_chws length mismatch: {len(spatials)} != {len(input_chws)}"
            )
        return [
            self.get_embedding(
                spatial=s,
                temporal=temporal,
                sensor=sensor,
                output=output,
                backend=backend,
                device=device,
                input_chw=x,
            )
            for s, x in zip(spatials, input_chws)
        ]
