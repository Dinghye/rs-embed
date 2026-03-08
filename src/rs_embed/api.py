"""Public Facade and Entry Point.

This module is the single boundary between callers and the internal pipeline
stack.  It should contain **no** heavy execution logic — only configuration
and delegation.

Responsibilities
----------------
1. **Validation** — ensure all input Specs (Spatial, Temporal, Output) are
   valid before any processing begins.
2. **Normalisation** — convert user-friendly strings (e.g. ``"sentinel-2"``,
   ``"cuda"``) into strict internal objects.
3. **Context Resolution** — route each request to the correct backend
   (Provider vs. Precomputed) and device.
4. **Delegation** — instantiate the appropriate Pipeline or Embedder and
   hand off execution.

Flow summary
------------
1. Validate and normalise user inputs / specs.
2. Resolve request context (model / backend / device / sensor / input prep).
3. Execute single / batch embedding, or delegate batch export to
   :class:`BatchExporter`.

Backward compatibility
----------------------
Some tests and downstream integrations monkeypatch symbols on ``rs_embed.api``.
For batch export, those hook symbols are mirrored into pipeline module globals
via ``_sync_export_pipeline_hooks`` so monkeypatch behaviour remains stable
after the pipeline refactor.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional

from .providers.gee_utils import (
    fetch_gee_patch_raw as _fetch_gee_patch_raw,
    inspect_input_raw as _inspect_input_raw,
)
from .tools.normalization import (
    _resolve_embedding_api_backend,
    normalize_backend_name as _normalize_backend_name,
    normalize_device_name as _normalize_device_name,
    normalize_model_name as _normalize_model_name,
)
from .tools.checkpoint_utils import (
    is_incomplete_combined_manifest as _is_incomplete_combined_manifest,
)
from .tools.runtime import (
    _EmbeddingRequestContext,
    _prepare_embedding_request_context as _prepare_embedding_request_context_shared,
    provider_factory_for_backend as _provider_factory_for_backend_shared,
    run_embedding_request as _run_embedding_request_shared,
)
from .core.validation import (
    assert_supported as _assert_supported,
    validate_specs as _validate_specs,
)
from .tools.manifest import (
    combined_resume_manifest as _combined_resume_manifest,
    load_json_dict as _load_json_dict,
)
from .tools.model_defaults import (
    default_sensor_for_model as _default_sensor_for_model,
)
from .tools.progress import create_progress as _create_progress
from .core.embedding import Embedding
from .core.errors import ModelError
from .core.registry import get_embedder_cls
from .core.specs import (
    InputPrepSpec,
    OutputSpec,
    SensorSpec,
    SpatialSpec,
    TemporalSpec,
)
from .core.types import ExportConfig, ExportLayout, ExportTarget, ModelConfig
from .embedders.catalog import MODEL_ALIASES, MODEL_SPECS
from .providers import ProviderBase

# Backward-compatibility hook: tests/downstream may monkeypatch api.GEEProvider.
GEEProvider: Optional[Callable[..., ProviderBase]] = None


def _provider_factory_for_backend(backend: str) -> Optional[Callable[[], ProviderBase]]:
    return _provider_factory_for_backend_shared(
        backend,
        gee_provider_cls=GEEProvider,
    )


# ---------------------------------------------------------------------------
# Internal: export target resolution
# ---------------------------------------------------------------------------


def _normalize_export_layout(layout: str) -> ExportLayout:
    layout_n = str(layout).strip().lower().replace("-", "_")
    if layout_n in {"combined", "single_file", "file"}:
        return ExportLayout.COMBINED
    if layout_n in {"per_item", "dir", "directory"}:
        return ExportLayout.PER_ITEM
    raise ModelError(
        f"Unsupported export layout: {layout!r}. Supported: 'combined', 'per_item'."
    )


def _resolve_export_batch_target(
    *,
    n_spatials: int,
    ext: str,
    out: Optional[str],
    layout: Optional[str],
    out_dir: Optional[str],
    out_path: Optional[str],
    names: Optional[List[str]],
) -> ExportTarget:
    if (out is not None) or (layout is not None):
        if out is None or layout is None:
            raise ModelError(
                "Provide both out and layout when using the decoupled output API."
            )
        if out_dir is not None or out_path is not None:
            raise ModelError("Use either out+layout or out_dir/out_path, not both.")
        layout_enum = _normalize_export_layout(layout)
        if layout_enum == ExportLayout.COMBINED:
            out_path = out
        else:
            out_dir = out

    if out_dir is None and out_path is None:
        raise ModelError("export_batch requires out_dir or out_path.")
    if out_dir is not None and out_path is not None:
        raise ModelError("Provide only one of out_dir or out_path.")

    if out_path is not None:
        out_file = out_path if out_path.endswith(ext) else (out_path + ext)
        return ExportTarget(layout=ExportLayout.COMBINED, out_file=out_file)

    assert out_dir is not None
    point_names = (
        names if names is not None else [f"p{i:05d}" for i in range(n_spatials)]
    )
    if len(point_names) != n_spatials:
        raise ModelError("names must have the same length as spatials.")
    return ExportTarget(
        layout=ExportLayout.PER_ITEM, out_dir=out_dir, names=point_names
    )


def _sync_export_pipeline_hooks() -> None:
    """Propagate api-level hook variables into refactored pipeline modules.

    Tests and downstream users monkeypatch symbols on ``rs_embed.api``
    (for example ``_inspect_input_raw`` or ``_create_progress``).
    The pipeline refactor moved implementations into separate modules, so
    we mirror those hooks back into pipeline module globals before running
    batch export to preserve backward-compatible monkeypatch behavior.
    """

    from .pipelines import combined_flow as _combined_flow_mod
    from .pipelines import exporter as _exporter_mod
    from .pipelines import point_payload as _point_payload_mod
    from .pipelines import prefetch as _prefetch_mod

    _prefetch_mod.fetch_gee_patch_raw = _fetch_gee_patch_raw
    _prefetch_mod.inspect_input_raw = _inspect_input_raw
    _point_payload_mod.fetch_gee_patch_raw = _fetch_gee_patch_raw
    _point_payload_mod.inspect_input_raw = _inspect_input_raw
    _exporter_mod.create_progress = _create_progress
    _combined_flow_mod.create_progress = _create_progress


# ---------------------------------------------------------------------------
# Internal: embedding request helpers (for get_embedding / get_embeddings_batch)
# ---------------------------------------------------------------------------


def _validate_spatials(
    *,
    spatials: List[SpatialSpec],
    temporal: Optional[TemporalSpec],
    output: OutputSpec,
) -> None:
    if not isinstance(spatials, list) or len(spatials) == 0:
        raise ModelError("spatials must be a non-empty List[SpatialSpec].")
    for spatial in spatials:
        _validate_specs(spatial=spatial, temporal=temporal, output=output)


def _prepare_embedding_request_context(
    *,
    model: str,
    temporal: Optional[TemporalSpec],
    sensor: Optional[SensorSpec],
    output: OutputSpec,
    backend: str,
    device: str,
    input_prep: Optional[InputPrepSpec | str],
) -> _EmbeddingRequestContext:
    return _prepare_embedding_request_context_shared(
        model=model,
        temporal=temporal,
        sensor=sensor,
        output=output,
        backend=backend,
        device=device,
        input_prep=input_prep,
    )


def _run_embedding_request(
    *,
    spatials: List[SpatialSpec],
    temporal: Optional[TemporalSpec],
    sensor: Optional[SensorSpec],
    output: OutputSpec,
    ctx: _EmbeddingRequestContext,
) -> List[Embedding]:
    return _run_embedding_request_shared(
        spatials=spatials,
        temporal=temporal,
        sensor=sensor,
        output=output,
        ctx=ctx,
        gee_provider_cls=GEEProvider,
    )


# -----------------------------------------------------------------------------
# Public: embeddings
# -----------------------------------------------------------------------------


def list_models(*, include_aliases: bool = False) -> List[str]:
    """Return the stable model catalog, independent of runtime lazy-load state.

    Parameters
    ----------
    include_aliases : bool
        If ``True``, include alias names in addition to canonical ids.

    Returns
    -------
    list[str]
        Sorted model names available in the catalog.
    """
    model_ids = set(MODEL_SPECS.keys())
    if include_aliases:
        model_ids.update(MODEL_ALIASES.keys())
    return sorted(model_ids)


def get_embedding(
    model: str,
    *,
    spatial: SpatialSpec,
    temporal: Optional[TemporalSpec] = None,
    sensor: Optional[SensorSpec] = None,
    output: OutputSpec = OutputSpec.pooled(),
    backend: str = "auto",
    device: str = "auto",
    input_prep: Optional[InputPrepSpec | str] = "resize",
) -> Embedding:
    """Compute a single embedding.

    Parameters
    ----------
    model : str
        Model identifier or alias.
    spatial : SpatialSpec
        Spatial location/extent to embed.
    temporal : TemporalSpec or None
        Optional temporal filter.
    sensor : SensorSpec or None
        Optional sensor override.
    output : OutputSpec
        Output representation policy.
    backend : str
        Backend/provider selector (for example ``"auto"`` or ``"gee"``).
    device : str
        Target inference device.
    input_prep : InputPrepSpec or str or None
        Optional API-side input preprocessing policy.

    Returns
    -------
    Embedding
        Normalized embedding output for the requested location.

    Raises
    ------
    ModelError
        If inputs/specs are invalid or requested model/backend configuration is
        unsupported.
    SpecError
        If spatial or temporal specifications fail validation.

    Notes
    -----
    This function reuses a cached embedder instance when possible to avoid
    repeatedly loading model weights / initializing providers.
    """
    _validate_spatials(spatials=[spatial], temporal=temporal, output=output)
    ctx = _prepare_embedding_request_context(
        model=model,
        temporal=temporal,
        sensor=sensor,
        output=output,
        backend=backend,
        device=device,
        input_prep=input_prep,
    )
    return _run_embedding_request(
        spatials=[spatial],
        temporal=temporal,
        sensor=sensor,
        output=output,
        ctx=ctx,
    )[0]


def get_embeddings_batch(
    model: str,
    *,
    spatials: List[SpatialSpec],
    temporal: Optional[TemporalSpec] = None,
    sensor: Optional[SensorSpec] = None,
    output: OutputSpec = OutputSpec.pooled(),
    backend: str = "auto",
    device: str = "auto",
    input_prep: Optional[InputPrepSpec | str] = "resize",
) -> List[Embedding]:
    """Compute embeddings for multiple spatials using a shared embedder instance.

    Parameters
    ----------
    model : str
        Model identifier or alias.
    spatials : list[SpatialSpec]
        Spatial requests to embed.
    temporal : TemporalSpec or None
        Optional temporal filter.
    sensor : SensorSpec or None
        Optional sensor override.
    output : OutputSpec
        Output representation policy.
    backend : str
        Backend/provider selector.
    device : str
        Target inference device.
    input_prep : InputPrepSpec or str or None
        Optional API-side input preprocessing policy.

    Returns
    -------
    list[Embedding]
        Embeddings in the same order as ``spatials``.

    Raises
    ------
    ModelError
        If inputs/specs are invalid or requested model/backend configuration is
        unsupported.
    SpecError
        If spatial or temporal specifications fail validation.
    """
    _validate_spatials(spatials=spatials, temporal=temporal, output=output)
    ctx = _prepare_embedding_request_context(
        model=model,
        temporal=temporal,
        sensor=sensor,
        output=output,
        backend=backend,
        device=device,
        input_prep=input_prep,
    )
    return _run_embedding_request(
        spatials=spatials,
        temporal=temporal,
        sensor=sensor,
        output=output,
        ctx=ctx,
    )


# -----------------------------------------------------------------------------
# Public: batch export (core)
# -----------------------------------------------------------------------------


def export_batch(
    *,
    spatials: List[SpatialSpec],
    temporal: Optional[TemporalSpec],
    models: List[str],
    out: Optional[str] = None,
    layout: Optional[str] = None,
    out_dir: Optional[str] = None,
    out_path: Optional[str] = None,
    names: Optional[List[str]] = None,
    backend: str = "auto",
    device: str = "auto",
    output: OutputSpec = OutputSpec.pooled(),
    sensor: Optional[SensorSpec] = None,
    per_model_sensors: Optional[Dict[str, SensorSpec]] = None,
    format: str = "npz",
    save_inputs: bool = True,
    save_embeddings: bool = True,
    save_manifest: bool = True,
    fail_on_bad_input: bool = False,
    chunk_size: int = 16,
    infer_batch_size: Optional[int] = None,
    num_workers: int = 8,
    continue_on_error: bool = False,
    max_retries: int = 0,
    retry_backoff_s: float = 0.0,
    async_write: bool = True,
    writer_workers: int = 2,
    resume: bool = False,
    show_progress: bool = True,
    input_prep: Optional[InputPrepSpec | str] = "resize",
) -> Any:
    """Export inputs + embeddings for many spatials and many models.

    This is the recommended high-level entrypoint for batch export.
    Delegates to :class:`~rs_embed.pipelines.exporter.BatchExporter`.

    Parameters
    ----------
    spatials : list[SpatialSpec]
        Spatial requests to export.
    temporal : TemporalSpec or None
        Optional temporal filter applied to all spatial requests.
    models : list[str]
        Model identifiers to run.
    out : str or None
        Convenience output path hint. Combined layout when file-like, per-item
        layout when directory-like.
    layout : str or None
        Explicit layout override (``"combined"`` or ``"per_item"``).
    out_dir : str or None
        Directory path for per-item exports.
    out_path : str or None
        File path for combined exports.
    names : list[str] or None
        Optional names aligned with ``spatials`` for per-item outputs.
    backend : str
        Backend/provider selector.
    device : str
        Target inference device.
    output : OutputSpec
        Embedding output representation policy.
    sensor : SensorSpec or None
        Default sensor for all models unless overridden.
    per_model_sensors : dict[str, SensorSpec] or None
        Per-model sensor overrides keyed by model name.
    format : str
        Output serialization format.
    save_inputs : bool
        Whether to persist fetched/model input arrays.
    save_embeddings : bool
        Whether to persist embeddings.
    save_manifest : bool
        Whether to write export manifest metadata.
    fail_on_bad_input : bool
        If ``True``, treat invalid inputs as hard failures.
    chunk_size : int
        Spatial chunk size for scheduling work.
    infer_batch_size : int or None
        Optional explicit model inference batch size.
    num_workers : int
        Number of preprocessing/inference workers.
    continue_on_error : bool
        If ``True``, continue processing after per-item failures.
    max_retries : int
        Retry count for retryable operations.
    retry_backoff_s : float
        Backoff delay in seconds between retries.
    async_write : bool
        If ``True``, enable asynchronous output writing.
    writer_workers : int
        Number of writer workers when ``async_write`` is enabled.
    resume : bool
        If ``True``, attempt to resume from prior export state.
    show_progress : bool
        Whether to display progress bars.
    input_prep : InputPrepSpec or str or None
        Optional API-side input preprocessing policy.

    Returns
    -------
    Any
        Export result object returned by :class:`BatchExporter`.

    Raises
    ------
    ModelError
        If arguments are invalid or unsupported (for example empty inputs,
        unsupported format, or incompatible model/backend settings).
    SpecError
        If spatial or temporal specifications fail validation.
    """
    from .pipelines.exporter import BatchExporter

    if not isinstance(spatials, list) or len(spatials) == 0:
        raise ModelError("spatials must be a non-empty List[SpatialSpec].")
    if not isinstance(models, list) or len(models) == 0:
        raise ModelError("models must be a non-empty List[str].")

    backend_n = _normalize_backend_name(backend)
    device_n = _normalize_device_name(device)
    fmt = format.lower().strip()

    from .writers import SUPPORTED_FORMATS, get_extension

    if fmt not in SUPPORTED_FORMATS:
        raise ModelError(
            f"Unsupported export format: {format!r}. Supported: {SUPPORTED_FORMATS}."
        )
    ext = get_extension(fmt)

    target = _resolve_export_batch_target(
        n_spatials=len(spatials),
        ext=ext,
        out=out,
        layout=layout,
        out_dir=out_dir,
        out_path=out_path,
        names=names,
    )

    _validate_spatials(spatials=spatials, temporal=temporal, output=output)

    # Early resume check for combined layout
    if target.layout == ExportLayout.COMBINED and bool(resume) and target.out_file:
        if os.path.exists(target.out_file):
            json_path = os.path.splitext(target.out_file)[0] + ".json"
            resume_manifest = _load_json_dict(json_path)
            if not _is_incomplete_combined_manifest(resume_manifest):
                return _combined_resume_manifest(
                    spatials=spatials,
                    temporal=temporal,
                    output=output,
                    backend=backend_n,
                    device=device_n,
                    out_file=target.out_file,
                )

    per_model_sensors = per_model_sensors or {}

    # Resolve per-model config
    model_configs: List[ModelConfig] = []
    resolved_backend: Dict[str, str] = {}
    for m in models:
        m_n = _normalize_model_name(m)
        eff_backend = _resolve_embedding_api_backend(m_n, backend_n)
        resolved_backend[m] = eff_backend
        cls = get_embedder_cls(m_n)
        try:
            emb_check = cls()
            _assert_supported(
                emb_check, backend=eff_backend, output=output, temporal=temporal
            )
            desc = emb_check.describe() or {}
        except ModelError:
            raise
        except Exception:
            desc = {}
        m_type = str(desc.get("type", "")).lower()
        if m in per_model_sensors:
            s = per_model_sensors[m]
        elif sensor is not None:
            s = sensor
        else:
            s = _default_sensor_for_model(m_n)
        model_configs.append(
            ModelConfig(name=m, backend=eff_backend, sensor=s, model_type=m_type)
        )

    config = ExportConfig(
        format=fmt,
        save_inputs=save_inputs,
        save_embeddings=save_embeddings,
        save_manifest=save_manifest,
        fail_on_bad_input=fail_on_bad_input,
        chunk_size=chunk_size,
        infer_batch_size=infer_batch_size,
        num_workers=num_workers,
        continue_on_error=continue_on_error,
        max_retries=max_retries,
        retry_backoff_s=retry_backoff_s,
        async_write=async_write,
        writer_workers=writer_workers,
        resume=resume,
        show_progress=show_progress,
        input_prep=input_prep,
    )

    _sync_export_pipeline_hooks()

    exporter = BatchExporter(
        spatials=spatials,
        temporal=temporal,
        models=model_configs,
        target=target,
        output=output,
        config=config,
        backend=backend_n,
        resolved_backend=resolved_backend,
        device=device_n,
        provider_factory=_provider_factory_for_backend(backend_n),
    )
    return exporter.run()
