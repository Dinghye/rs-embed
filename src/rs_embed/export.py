from __future__ import annotations

"""High-level export entrypoints.

`export_npz` is a convenience wrapper around `rs_embed.api.export_batch`.
Legacy helper names are re-exported from `core.export_helpers` for compatibility.
"""

import os
from typing import Any, Dict, List, Optional

from .core.export_helpers import (
    embedding_to_numpy as _embedding_to_numpy,
    jsonable as _jsonable,
    sanitize_key as _sanitize_key,
    sha1 as _sha1,
    utc_ts as _utc_ts,
)
from .core.specs import OutputSpec, SensorSpec, SpatialSpec, TemporalSpec
from .internal.api.model_defaults_helpers import (
    default_sensor_for_model as _default_sensor_for_model,
)


def export_npz(
    *,
    spatial: SpatialSpec,
    temporal: Optional[TemporalSpec],
    models: List[str],
    out_path: str,
    backend: str = "gee",
    device: str = "auto",
    output: OutputSpec = OutputSpec.pooled(),
    sensor: Optional[SensorSpec] = None,
    per_model_sensors: Optional[Dict[str, SensorSpec]] = None,
    save_inputs: bool = True,
    save_embeddings: bool = True,
    save_manifest: bool = True,
    fail_on_bad_input: bool = False,
    continue_on_error: bool = False,
    max_retries: int = 0,
    retry_backoff_s: float = 0.0,
) -> Dict[str, Any]:
    """Export inputs + embeddings for one spatial query to a single `.npz`."""
    from .api import export_batch

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if not out_path.endswith(".npz"):
        out_path = out_path + ".npz"

    return export_batch(
        spatials=[spatial],
        temporal=temporal,
        models=models,
        out_path=out_path,
        backend=backend,
        device=device,
        output=output,
        sensor=sensor,
        per_model_sensors=per_model_sensors,
        format="npz",
        save_inputs=save_inputs,
        save_embeddings=save_embeddings,
        save_manifest=save_manifest,
        fail_on_bad_input=fail_on_bad_input,
        continue_on_error=continue_on_error,
        max_retries=max_retries,
        retry_backoff_s=retry_backoff_s,
    )
