from __future__ import annotations

import json
import os
import sys
import time
from functools import lru_cache
from threading import RLock
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

import numpy as np

from .core.export_helpers import (
    default_sensor_for_model as _default_sensor_for_model,
    embedding_to_numpy as _embedding_to_numpy,
    jsonable as _jsonable,
    sanitize_key as _sanitize_key,
    sensor_cache_key as _sensor_cache_key,
    sha1 as _sha1,
    utc_ts as _utc_ts,
)
from .internal.api.api_helpers import (
    fetch_gee_patch_raw as _fetch_gee_patch_raw_impl,
    inspect_input_raw as _inspect_input_raw_impl,
    normalize_backend_name as _normalize_backend_name,
    normalize_device_name as _normalize_device_name,
    normalize_input_chw as _normalize_input_chw,
    normalize_model_name as _normalize_model_name,
)
from .internal.api.checkpoint_helpers import (
    drop_model_arrays as _drop_model_arrays_impl,
    drop_prefetch_checkpoint_arrays as _drop_prefetch_checkpoint_arrays_impl,
    is_incomplete_combined_manifest as _is_incomplete_combined_manifest_impl,
    load_saved_arrays as _load_saved_arrays_impl,
    restore_prefetch_checkpoint_cache as _restore_prefetch_checkpoint_cache_impl,
    store_prefetch_checkpoint_arrays as _store_prefetch_checkpoint_arrays_impl,
)
from .internal.api.combined_helpers import (
    collect_input_refs_by_sensor as _collect_input_refs_by_sensor_impl,
    init_combined_export_state as _init_combined_export_state_impl,
    summarize_combined_models as _summarize_combined_models_impl,
)
from .internal.api.combined_orchestration_helpers import (
    build_combined_prefetch_tasks as _build_combined_prefetch_tasks_impl,
    init_combined_provider as _init_combined_provider_impl,
    restore_prefetch_cache_from_manifest as _restore_prefetch_cache_from_manifest_impl,
    write_combined_checkpoint as _write_combined_checkpoint_impl,
)
from .internal.api.combined_flow_helpers import (
    CombinedModelDeps as _CombinedModelDeps,
    CombinedPrefetchDeps as _CombinedPrefetchDeps,
    get_or_fetch_input as _get_or_fetch_input_impl,
    run_combined_prefetch_tasks as _run_combined_prefetch_tasks_impl,
    run_pending_models as _run_pending_models_impl,
)
from .internal.api.prefetch_helpers import (
    build_gee_prefetch_plan as _build_gee_prefetch_plan_impl,
    select_prefetched_channels as _select_prefetched_channels_impl,
    sensor_fetch_group_key as _sensor_fetch_group_key_impl,
)
from .internal.api.point_payload_helpers import (
    PointPayloadDeps as _PointPayloadDeps,
    build_one_point_payload as _build_one_point_payload_impl,
)
from .internal.api.runtime_helpers import (
    call_embedder_get_embedding as _call_embedder_get_embedding_impl,
    embedder_accepts_input_chw as _embedder_accepts_input_chw_impl,
    sensor_key as _sensor_key_impl,
    supports_batch_api as _supports_batch_api_impl,
    supports_prefetched_batch_api as _supports_prefetched_batch_api_impl,
)
from .internal.api.validation_helpers import (
    assert_supported as _assert_supported_impl,
    validate_specs as _validate_specs_impl,
)
from .core.embedding import Embedding
from .core.errors import ModelError
from .core.registry import get_embedder_cls
from .core.specs import OutputSpec, SensorSpec, SpatialSpec, TemporalSpec
from .providers.gee import GEEProvider

_T = TypeVar("_T")


# -----------------------------------------------------------------------------
# Internal: progress + resume helpers
# -----------------------------------------------------------------------------

class _NoOpProgress:
    def update(self, n: int = 1) -> None:
        _ = n

    def close(self) -> None:
        return None


class _SimpleProgress:
    """Minimal fallback progress indicator when tqdm is unavailable."""

    def __init__(self, *, total: int, desc: str):
        self.total = max(0, int(total))
        self.desc = desc
        self.done = 0
        self._last_pct = -1

    def update(self, n: int = 1) -> None:
        if self.total <= 0:
            return
        self.done = min(self.total, self.done + max(0, int(n)))
        pct = int((100 * self.done) / self.total)
        if pct == self._last_pct and self.done < self.total:
            return
        self._last_pct = pct

        width = 24
        fill = int((width * self.done) / self.total)
        bar = ("#" * fill) + ("." * (width - fill))
        sys.stderr.write(f"\r{self.desc} [{bar}] {self.done}/{self.total} ({pct:3d}%)")
        if self.done >= self.total:
            sys.stderr.write("\n")
        sys.stderr.flush()

    def close(self) -> None:
        if self.total > 0 and self.done < self.total:
            sys.stderr.write("\n")
            sys.stderr.flush()


def _create_progress(*, enabled: bool, total: int, desc: str, unit: str = "item") -> Any:
    if (not enabled) or int(total) <= 0:
        return _NoOpProgress()

    try:
        from tqdm.auto import tqdm

        return tqdm(total=int(total), desc=desc, unit=unit, leave=False)
    except Exception:
        return _SimpleProgress(total=int(total), desc=desc)


def _load_json_dict(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            return payload
    except Exception:
        return None
    return None


def _point_resume_manifest(
    *,
    point_index: int,
    spatial: SpatialSpec,
    temporal: Optional[TemporalSpec],
    output: OutputSpec,
    backend: str,
    device: str,
    out_file: str,
) -> Dict[str, Any]:
    json_path = os.path.splitext(out_file)[0] + ".json"
    manifest = _load_json_dict(json_path)
    if manifest is None:
        manifest = {
            "created_at": _utc_ts(),
            "point_index": int(point_index),
            "status": "skipped",
            "stage": "resume",
            "reason": "output_exists",
            "backend": backend,
            "device": device,
            "models": [],
            "spatial": _jsonable(spatial),
            "temporal": _jsonable(temporal),
            "output": _jsonable(output),
        }
    manifest["resume_skipped"] = True
    manifest["resume_output_path"] = out_file
    manifest.setdefault("point_index", int(point_index))
    manifest.setdefault("status", "ok")
    return manifest


def _combined_resume_manifest(
    *,
    spatials: List[SpatialSpec],
    temporal: Optional[TemporalSpec],
    output: OutputSpec,
    backend: str,
    device: str,
    out_file: str,
) -> Dict[str, Any]:
    json_path = os.path.splitext(out_file)[0] + ".json"
    manifest = _load_json_dict(json_path)
    if manifest is None:
        manifest = {
            "created_at": _utc_ts(),
            "status": "skipped",
            "stage": "resume",
            "reason": "output_exists",
            "backend": backend,
            "device": device,
            "n_items": len(spatials),
            "temporal": _jsonable(temporal),
            "output": _jsonable(output),
            "spatials": [_jsonable(s) for s in spatials],
            "models": [],
        }
    manifest["resume_skipped"] = True
    manifest["resume_output_path"] = out_file
    manifest.setdefault("status", "ok")
    return manifest


def _is_incomplete_combined_manifest(manifest: Optional[Dict[str, Any]]) -> bool:
    return _is_incomplete_combined_manifest_impl(manifest)


def _load_saved_arrays(*, fmt: str, out_path: str) -> Dict[str, np.ndarray]:
    return _load_saved_arrays_impl(fmt=fmt, out_path=out_path)


def _drop_prefetch_checkpoint_arrays(arrays: Dict[str, np.ndarray]) -> None:
    _drop_prefetch_checkpoint_arrays_impl(arrays)


def _store_prefetch_checkpoint_arrays(
    *,
    arrays: Dict[str, np.ndarray],
    manifest: Dict[str, Any],
    sensor_by_key: Dict[str, SensorSpec],
    inputs_cache: Dict[Tuple[int, str], np.ndarray],
    n_items: int,
) -> None:
    _store_prefetch_checkpoint_arrays_impl(
        arrays=arrays,
        manifest=manifest,
        sensor_by_key=sensor_by_key,
        inputs_cache=inputs_cache,
        n_items=n_items,
    )


def _restore_prefetch_checkpoint_cache(
    *,
    arrays: Dict[str, np.ndarray],
    prefetch_meta: Dict[str, Any],
) -> Dict[Tuple[int, str], np.ndarray]:
    return _restore_prefetch_checkpoint_cache_impl(
        arrays=arrays,
        prefetch_meta=prefetch_meta,
    )


def _drop_model_arrays(arrays: Dict[str, np.ndarray], model_name: str) -> None:
    _drop_model_arrays_impl(
        arrays=arrays,
        model_name=model_name,
        sanitize_key=_sanitize_key,
    )


# -----------------------------------------------------------------------------
# Public: embeddings
# -----------------------------------------------------------------------------

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
    """Compute a single embedding.

    Notes
    -----
    This function reuses a cached embedder instance when possible to avoid
    repeatedly loading model weights / initializing providers.
    """
    # Import embedders so registration happens before resolving model IDs
    from . import embedders  # noqa: F401

    backend_n = _normalize_backend_name(backend)
    model_n = _normalize_model_name(model)
    device = _normalize_device_name(device)

    _validate_specs(spatial=spatial, temporal=temporal, output=output)

    sensor_k = _sensor_key(sensor)
    embedder, lock = _get_embedder_bundle_cached(model_n, backend_n, device, sensor_k)

    _assert_supported(embedder, backend=backend_n, output=output, temporal=temporal)

    with lock:
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
    spatials: List[SpatialSpec],
    temporal: Optional[TemporalSpec] = None,
    sensor: Optional[SensorSpec] = None,
    output: OutputSpec = OutputSpec.pooled(),
    backend: str = "gee",
    device: str = "auto",
) -> List[Embedding]:
    """Compute embeddings for multiple SpatialSpecs using a shared embedder instance."""
    from . import embedders  # noqa: F401

    backend_n = _normalize_backend_name(backend)
    model_n = _normalize_model_name(model)
    device = _normalize_device_name(device)

    if not isinstance(spatials, list) or len(spatials) == 0:
        raise ModelError("spatials must be a non-empty List[SpatialSpec].")

    # validate once + per item
    _validate_specs(spatial=spatials[0], temporal=temporal, output=output)
    for s in spatials:
        _validate_specs(spatial=s, temporal=temporal, output=output)

    sensor_k = _sensor_key(sensor)
    embedder, lock = _get_embedder_bundle_cached(model_n, backend_n, device, sensor_k)

    _assert_supported(embedder, backend=backend_n, output=output, temporal=temporal)

    with lock:
        return embedder.get_embeddings_batch(
            spatials=spatials,
            temporal=temporal,
            sensor=sensor,
            output=output,
            backend=backend_n,
            device=device,
        )


# -----------------------------------------------------------------------------
# Public: batch export (core)
# -----------------------------------------------------------------------------


def export_batch(
    *,
    spatials: List[SpatialSpec],
    temporal: Optional[TemporalSpec],
    models: List[str],
    out_dir: Optional[str] = None,
    out_path: Optional[str] = None,
    names: Optional[List[str]] = None,
    backend: str = "gee",
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
    num_workers: int = 8,
    continue_on_error: bool = False,
    max_retries: int = 0,
    retry_backoff_s: float = 0.0,
    async_write: bool = True,
    writer_workers: int = 2,
    resume: bool = False,
    show_progress: bool = True,
) -> Any:
    """Export inputs + embeddings for many spatials and many models.

    This is the recommended high-level entrypoint for batch export.

    - Accept any SpatialSpec list (like get_embeddings_batch).
    - Reuse cached embedder instances to avoid re-loading models/providers.
    - For GEE backends, prefetch raw inputs once per (point, sensor) and reuse
      them for input export and embedding inference.
    """
    from . import embedders  # noqa: F401 (ensure registration)

    if not isinstance(spatials, list) or len(spatials) == 0:
        raise ModelError("spatials must be a non-empty List[SpatialSpec].")
    if not isinstance(models, list) or len(models) == 0:
        raise ModelError("models must be a non-empty List[str].")

    if out_dir is None and out_path is None:
        raise ModelError("export_batch requires out_dir or out_path.")
    if out_dir is not None and out_path is not None:
        raise ModelError("Provide only one of out_dir or out_path.")

    backend_n = _normalize_backend_name(backend)
    device = _normalize_device_name(device)
    fmt = format.lower().strip()
    from .writers import SUPPORTED_FORMATS, get_extension
    if fmt not in SUPPORTED_FORMATS:
        raise ModelError(f"Unsupported export format: {format!r}. Supported: {SUPPORTED_FORMATS}.")
    ext = get_extension(fmt)

    # validate specs early
    _validate_specs(spatial=spatials[0], temporal=temporal, output=output)
    for s in spatials:
        _validate_specs(spatial=s, temporal=temporal, output=output)

    per_model_sensors = per_model_sensors or {}

    # resolve sensors + type per model (best effort)
    resolved_sensor: Dict[str, Optional[SensorSpec]] = {}
    model_type: Dict[str, str] = {}
    for m in models:
        m_n = _normalize_model_name(m)
        cls = get_embedder_cls(m_n)
        try:
            desc = cls().describe() or {}
        except Exception:
            desc = {}
        model_type[m] = str(desc.get("type", "")).lower()
        if m in per_model_sensors:
            resolved_sensor[m] = per_model_sensors[m]
        elif sensor is not None:
            resolved_sensor[m] = sensor
        else:
            resolved_sensor[m] = _default_sensor_for_model(m_n)

    # combined mode
    if out_path is not None:
        out_file = out_path if out_path.endswith(ext) else (out_path + ext)
        if bool(resume) and os.path.exists(out_file):
            json_path = os.path.splitext(out_file)[0] + ".json"
            resume_manifest = _load_json_dict(json_path)
            if not _is_incomplete_combined_manifest(resume_manifest):
                return _combined_resume_manifest(
                    spatials=spatials,
                    temporal=temporal,
                    output=output,
                    backend=backend_n,
                    device=device,
                    out_file=out_file,
                )
        return _export_combined_npz(
            spatials=spatials,
            temporal=temporal,
            models=models,
            out_path=out_file,
            backend=backend_n,
            device=device,
            output=output,
            resolved_sensor=resolved_sensor,
            model_type=model_type,
            save_inputs=save_inputs,
            save_embeddings=save_embeddings,
            save_manifest=save_manifest,
            fail_on_bad_input=fail_on_bad_input,
            chunk_size=chunk_size,
            num_workers=num_workers,
            fmt=fmt,
            continue_on_error=continue_on_error,
            max_retries=max_retries,
            retry_backoff_s=retry_backoff_s,
            resume=resume,
            show_progress=show_progress,
        )

    # per-item mode (out_dir)
    assert out_dir is not None
    os.makedirs(out_dir, exist_ok=True)
    if names is None:
        names = [f"p{i:05d}" for i in range(len(spatials))]
    if len(names) != len(spatials):
        raise ModelError("names must have the same length as spatials.")

    n = len(spatials)
    progress = _create_progress(
        enabled=bool(show_progress),
        total=n,
        desc="export_batch",
        unit="point",
    )
    model_progress: Dict[str, Any] = {}

    manifests: List[Dict[str, Any]] = []
    pending_idxs: List[int] = []
    try:
        for i in range(n):
            out_file = os.path.join(out_dir, f"{names[i]}{ext}")
            if bool(resume) and os.path.exists(out_file):
                manifests.append(
                    _point_resume_manifest(
                        point_index=i,
                        spatial=spatials[i],
                        temporal=temporal,
                        output=output,
                        backend=backend_n,
                        device=device,
                        out_file=out_file,
                    )
                )
                progress.update(1)
            else:
                pending_idxs.append(i)

        if not pending_idxs:
            manifests.sort(key=lambda x: int(x.get("point_index", -1)))
            return manifests

        if save_embeddings:
            model_progress = {
                m: _create_progress(
                    enabled=bool(show_progress),
                    total=len(pending_idxs),
                    desc=f"infer[{m}]",
                    unit="point",
                )
                for m in models
            }

        def _on_model_done(model_name: str) -> None:
            bar = model_progress.get(model_name)
            if bar is not None:
                bar.update(1)

        # For GEE, prefetch once and reuse for export and/or embedding inference.
        need_prefetch = backend_n == "gee" and bool(save_inputs or save_embeddings) and bool(pending_idxs)
        pass_input_into_embedder = backend_n == "gee" and bool(save_embeddings)
        (
            sensor_by_key,
            fetch_sensor_by_key,
            sensor_to_fetch,
            sensor_models,
            fetch_members,
        ) = _build_gee_prefetch_plan(
            models=models,
            resolved_sensor=resolved_sensor,
            model_type=model_type,
        )

        provider: Optional[GEEProvider] = None
        if need_prefetch:
            provider = GEEProvider(auto_auth=True)
            _run_with_retry(
                lambda: provider.ensure_ready(),
                retries=max_retries,
                backoff_s=retry_backoff_s,
            )

        csize = max(1, int(chunk_size))
        for chunk_start in range(0, len(pending_idxs), csize):
            idxs = pending_idxs[chunk_start: chunk_start + csize]

            # (i, sensor_cache_key) -> input_chw
            inputs_cache: Dict[Tuple[int, str], np.ndarray] = {}
            input_reports: Dict[Tuple[int, str], Dict[str, Any]] = {}
            prefetch_errors: Dict[Tuple[int, str], str] = {}

            if need_prefetch and provider is not None:
                tasks = [
                    (i, fetch_key, fetch_sensor)
                    for i in idxs
                    for fetch_key, fetch_sensor in fetch_sensor_by_key.items()
                ]

                if tasks:
                    from concurrent.futures import ThreadPoolExecutor, as_completed

                    def _fetch_one(ii: int, sk: str, ss: SensorSpec):
                        assert provider is not None
                        x = _run_with_retry(
                            lambda: _fetch_gee_patch_raw(provider, spatial=spatials[ii], temporal=temporal, sensor=ss),
                            retries=max_retries,
                            backoff_s=retry_backoff_s,
                        )
                        return ii, sk, x

                    mw = max(1, int(num_workers))
                    with ThreadPoolExecutor(max_workers=mw) as ex:
                        fut_map = {ex.submit(_fetch_one, ii, sk, ss): (ii, sk) for (ii, sk, ss) in tasks}
                        for fut in as_completed(fut_map):
                            ii, sk = fut_map[fut]
                            try:
                                ii, sk, x = fut.result()
                            except Exception as e:
                                if not continue_on_error:
                                    raise
                                err_s = repr(e)
                                for member_skey in fetch_members.get(sk, []):
                                    prefetch_errors[(ii, member_skey)] = err_s
                                continue
                            for member_skey in fetch_members.get(sk, []):
                                member_idx = sensor_to_fetch[member_skey][1]
                                x_member = _normalize_input_chw(
                                    _select_prefetched_channels(x, member_idx),
                                    expected_channels=len(member_idx),
                                    name=f"gee_input_{member_skey}",
                                )
                                if fail_on_bad_input:
                                    sspec_member = sensor_by_key[member_skey]
                                    rep = _inspect_input_raw(x_member, sensor=sspec_member, name=f"gee_input_{member_skey}")
                                    if not bool(rep.get("ok", True)):
                                        issues = (rep.get("report", {}) or {}).get("issues", [])
                                        mlist = sorted(set(sensor_models.get(member_skey, [])))
                                        err = RuntimeError(
                                            f"Input inspection failed for index={ii}, sensor={member_skey}, models={mlist}: {issues}"
                                        )
                                        if not continue_on_error:
                                            raise err
                                        prefetch_errors[(ii, member_skey)] = repr(err)
                                        continue
                                    input_reports[(ii, member_skey)] = rep
                                inputs_cache[(ii, member_skey)] = x_member

            # export each point in chunk
            writer_async = bool(async_write)
            writer_mw = max(1, int(writer_workers))
            write_futs = []
            writer_ex = None
            if writer_async:
                from concurrent.futures import ThreadPoolExecutor
                writer_ex = ThreadPoolExecutor(max_workers=writer_mw)
            for i in idxs:
                out_file = os.path.join(out_dir, f"{names[i]}{ext}")
                try:
                    arrays, manifest = _build_one_point_payload(
                        point_index=i,
                        spatial=spatials[i],
                        temporal=temporal,
                        models=models,
                        backend=backend_n,
                        device=device,
                        output=output,
                        resolved_sensor=resolved_sensor,
                        model_type=model_type,
                        inputs_cache=inputs_cache,
                        input_reports=input_reports,
                        prefetch_errors=prefetch_errors,
                        pass_input_into_embedder=pass_input_into_embedder,
                        save_inputs=save_inputs,
                        save_embeddings=save_embeddings,
                        fail_on_bad_input=fail_on_bad_input,
                        continue_on_error=continue_on_error,
                        max_retries=max_retries,
                        retry_backoff_s=retry_backoff_s,
                        model_progress_cb=(_on_model_done if save_embeddings else None),
                    )
                except Exception as e:
                    if not continue_on_error:
                        if writer_ex is not None:
                            writer_ex.shutdown(wait=False)
                        raise
                    manifests.append(
                        _point_failure_manifest(
                            point_index=i,
                            spatial=spatials[i],
                            temporal=temporal,
                            output=output,
                            backend=backend_n,
                            device=device,
                            stage="build",
                            error=e,
                        )
                    )
                    progress.update(1)
                    continue

                if writer_ex is not None:
                    fut = writer_ex.submit(
                        _write_one_payload,
                        out_path=out_file,
                        arrays=arrays,
                        manifest=manifest,
                        save_manifest=save_manifest,
                        fmt=fmt,
                        max_retries=max_retries,
                        retry_backoff_s=retry_backoff_s,
                    )
                    write_futs.append((i, fut))
                else:
                    try:
                        mani = _write_one_payload(
                            out_path=out_file,
                            arrays=arrays,
                            manifest=manifest,
                            save_manifest=save_manifest,
                            fmt=fmt,
                            max_retries=max_retries,
                            retry_backoff_s=retry_backoff_s,
                        )
                    except Exception as e:
                        if not continue_on_error:
                            raise
                        mani = _point_failure_manifest(
                            point_index=i,
                            spatial=spatials[i],
                            temporal=temporal,
                            output=output,
                            backend=backend_n,
                            device=device,
                            stage="write",
                            error=e,
                        )
                    manifests.append(mani)
                    progress.update(1)

            if writer_ex is not None:
                from concurrent.futures import as_completed
                try:
                    fut_map = {fut: i for (i, fut) in write_futs}
                    for fut in as_completed(fut_map):
                        i = fut_map[fut]
                        try:
                            manifests.append(fut.result())
                        except Exception as e:
                            if not continue_on_error:
                                raise
                            manifests.append(
                                _point_failure_manifest(
                                    point_index=i,
                                    spatial=spatials[i],
                                    temporal=temporal,
                                    output=output,
                                    backend=backend_n,
                                    device=device,
                                    stage="write",
                                    error=e,
                                )
                            )
                        finally:
                            progress.update(1)
                finally:
                    writer_ex.shutdown(wait=True)

        manifests.sort(key=lambda x: int(x.get("point_index", -1)))
        return manifests
    finally:
        for bar in model_progress.values():
            bar.close()
        progress.close()

# -----------------------------------------------------------------------------
# Internal: embedder caching
# -----------------------------------------------------------------------------

def _sensor_key(sensor: Optional[SensorSpec]) -> Tuple:
    return _sensor_key_impl(sensor)


@lru_cache(maxsize=32)
def _get_embedder_bundle_cached(model: str, backend: str, device: str, sensor_k: Tuple):
    """Return (embedder instance, instance lock)."""
    cls = get_embedder_cls(model)
    emb = cls()
    return emb, RLock()


def _run_with_retry(
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


def _supports_batch_api(embedder: Any) -> bool:
    return _supports_batch_api_impl(embedder)


def _supports_prefetched_batch_api(embedder: Any) -> bool:
    return _supports_prefetched_batch_api_impl(embedder)


def _embedder_accepts_input_chw(embedder_cls: type) -> bool:
    return _embedder_accepts_input_chw_impl(embedder_cls)


def _call_embedder_get_embedding(
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
    return _call_embedder_get_embedding_impl(
        embedder=embedder,
        spatial=spatial,
        temporal=temporal,
        sensor=sensor,
        output=output,
        backend=backend,
        device=device,
        input_chw=input_chw,
    )


def _sensor_fetch_group_key(sensor: SensorSpec) -> Tuple[str, int, int, float, str]:
    return _sensor_fetch_group_key_impl(sensor)


def _select_prefetched_channels(x_chw: np.ndarray, idx: Tuple[int, ...]) -> np.ndarray:
    return _select_prefetched_channels_impl(x_chw, idx)


def _build_gee_prefetch_plan(
    *,
    models: List[str],
    resolved_sensor: Dict[str, Optional[SensorSpec]],
    model_type: Dict[str, str],
) -> Tuple[
    Dict[str, SensorSpec],  # sensor_by_key
    Dict[str, SensorSpec],  # fetch_sensor_by_key
    Dict[str, Tuple[str, Tuple[int, ...]]],  # sensor_key -> (fetch_key, channel_idx)
    Dict[str, List[str]],  # sensor_models
    Dict[str, List[str]],  # fetch_members
]:
    return _build_gee_prefetch_plan_impl(
        models=models,
        resolved_sensor=resolved_sensor,
        model_type=model_type,
    )


# -----------------------------------------------------------------------------
# Internal: validation and capability checks
# -----------------------------------------------------------------------------

def _validate_specs(*, spatial: SpatialSpec, temporal: Optional[TemporalSpec], output: OutputSpec) -> None:
    _validate_specs_impl(spatial=spatial, temporal=temporal, output=output)


def _assert_supported(embedder, *, backend: str, output: OutputSpec, temporal: Optional[TemporalSpec]) -> None:
    _assert_supported_impl(embedder, backend=backend, output=output, temporal=temporal)


def _fetch_gee_patch_raw(provider: GEEProvider, *, spatial: SpatialSpec, temporal: Optional[TemporalSpec], sensor: SensorSpec) -> np.ndarray:
    return _fetch_gee_patch_raw_impl(
        provider,
        spatial=spatial,
        temporal=temporal,
        sensor=sensor,
    )


def _inspect_input_raw(x_chw: np.ndarray, *, sensor: SensorSpec, name: str) -> Dict[str, Any]:
    return _inspect_input_raw_impl(
        x_chw,
        sensor=sensor,
        name=name,
    )


def _point_failure_manifest(
    *,
    point_index: int,
    spatial: SpatialSpec,
    temporal: Optional[TemporalSpec],
    output: OutputSpec,
    backend: str,
    device: str,
    stage: str,
    error: Exception,
) -> Dict[str, Any]:
    return {
        "created_at": _utc_ts(),
        "point_index": int(point_index),
        "status": "failed",
        "stage": stage,
        "error": repr(error),
        "backend": backend,
        "device": device,
        "models": [],
        "spatial": _jsonable(spatial),
        "temporal": _jsonable(temporal),
        "output": _jsonable(output),
    }


def _build_one_point_payload(
    *,
    point_index: int,
    spatial: SpatialSpec,
    temporal: Optional[TemporalSpec],
    models: List[str],
    backend: str,
    device: str,
    output: OutputSpec,
    resolved_sensor: Dict[str, Optional[SensorSpec]],
    model_type: Dict[str, str],
    inputs_cache: Dict[Tuple[int, str], np.ndarray],
    input_reports: Dict[Tuple[int, str], Dict[str, Any]],
    prefetch_errors: Dict[Tuple[int, str], str],
    pass_input_into_embedder: bool,
    save_inputs: bool,
    save_embeddings: bool,
    fail_on_bad_input: bool,
    continue_on_error: bool,
    max_retries: int,
    retry_backoff_s: float,
    model_progress_cb: Optional[Callable[[str], None]] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    deps = _PointPayloadDeps(
        utc_ts=_utc_ts,
        jsonable=_jsonable,
        sanitize_key=_sanitize_key,
        sha1=_sha1,
        embedding_to_numpy=_embedding_to_numpy,
        sensor_cache_key=_sensor_cache_key,
        sensor_key=_sensor_key,
        normalize_model_name=_normalize_model_name,
        get_embedder_bundle_cached=_get_embedder_bundle_cached,
        run_with_retry=_run_with_retry,
        fetch_gee_patch_raw=_fetch_gee_patch_raw,
        inspect_input_raw=_inspect_input_raw,
        call_embedder_get_embedding=_call_embedder_get_embedding,
        provider_factory=lambda: GEEProvider(auto_auth=True),
    )
    return _build_one_point_payload_impl(
        point_index=point_index,
        spatial=spatial,
        temporal=temporal,
        models=models,
        backend=backend,
        device=device,
        output=output,
        resolved_sensor=resolved_sensor,
        model_type=model_type,
        inputs_cache=inputs_cache,
        input_reports=input_reports,
        prefetch_errors=prefetch_errors,
        pass_input_into_embedder=pass_input_into_embedder,
        save_inputs=save_inputs,
        save_embeddings=save_embeddings,
        fail_on_bad_input=fail_on_bad_input,
        continue_on_error=continue_on_error,
        max_retries=max_retries,
        retry_backoff_s=retry_backoff_s,
        deps=deps,
        model_progress_cb=model_progress_cb,
    )


def _write_one_payload(
    *,
    out_path: str,
    arrays: Dict[str, np.ndarray],
    manifest: Dict[str, Any],
    save_manifest: bool,
    fmt: str,
    max_retries: int,
    retry_backoff_s: float,
) -> Dict[str, Any]:
    from .writers import write_arrays
    return _run_with_retry(
        lambda: write_arrays(
            fmt=fmt,
            out_path=out_path,
            arrays=arrays,
            manifest=_jsonable(manifest),
            save_manifest=save_manifest,
        ),
        retries=max_retries,
        backoff_s=retry_backoff_s,
    )


def _export_one_point_npz(
    *,
    point_index: int,
    spatial: SpatialSpec,
    temporal: Optional[TemporalSpec],
    models: List[str],
    out_path: str,
    backend: str,
    device: str,
    output: OutputSpec,
    resolved_sensor: Dict[str, Optional[SensorSpec]],
    model_type: Dict[str, str],
    inputs_cache: Dict[Tuple[int, str], np.ndarray],
    input_reports: Dict[Tuple[int, str], Dict[str, Any]],
    pass_input_into_embedder: bool,
    save_inputs: bool,
    save_embeddings: bool,
    save_manifest: bool,
    fail_on_bad_input: bool,
    fmt: str = "npz",
    continue_on_error: bool = False,
    max_retries: int = 0,
    retry_backoff_s: float = 0.0,
) -> Dict[str, Any]:
    arrays, manifest = _build_one_point_payload(
        point_index=point_index,
        spatial=spatial,
        temporal=temporal,
        models=models,
        backend=backend,
        device=device,
        output=output,
        resolved_sensor=resolved_sensor,
        model_type=model_type,
        inputs_cache=inputs_cache,
        input_reports=input_reports,
        prefetch_errors={},
        pass_input_into_embedder=pass_input_into_embedder,
        save_inputs=save_inputs,
        save_embeddings=save_embeddings,
        fail_on_bad_input=fail_on_bad_input,
        continue_on_error=continue_on_error,
        max_retries=max_retries,
        retry_backoff_s=retry_backoff_s,
    )
    return _write_one_payload(
        out_path=out_path,
        arrays=arrays,
        manifest=manifest,
        save_manifest=save_manifest,
        fmt=fmt,
        max_retries=max_retries,
        retry_backoff_s=retry_backoff_s,
    )


def _export_combined_npz(
    *,
    spatials: List[SpatialSpec],
    temporal: Optional[TemporalSpec],
    models: List[str],
    out_path: str,
    backend: str,
    device: str,
    output: OutputSpec,
    resolved_sensor: Dict[str, Optional[SensorSpec]],
    model_type: Dict[str, str],
    save_inputs: bool,
    save_embeddings: bool,
    save_manifest: bool,
    fail_on_bad_input: bool,
    chunk_size: int,
    num_workers: int,
    fmt: str = "npz",
    continue_on_error: bool = False,
    max_retries: int = 0,
    retry_backoff_s: float = 0.0,
    resume: bool = False,
    show_progress: bool = False,
) -> Dict[str, Any]:
    # Simple combined implementation (no chunking): for large N prefer out_dir.
    arrays, manifest, pending_models, json_path = _init_combined_export_state_impl(
        spatials=spatials,
        temporal=temporal,
        output=output,
        backend=backend,
        device=device,
        models=models,
        out_path=out_path,
        fmt=fmt,
        resume=resume,
        load_json_dict=_load_json_dict,
        is_incomplete_combined_manifest=_is_incomplete_combined_manifest,
        load_saved_arrays=_load_saved_arrays,
        jsonable=_jsonable,
        utc_ts=_utc_ts,
    )

    provider = _init_combined_provider_impl(
        backend=backend,
        save_inputs=save_inputs,
        save_embeddings=save_embeddings,
        provider_factory=lambda: GEEProvider(auto_auth=True),
        run_with_retry=_run_with_retry,
        max_retries=max_retries,
        retry_backoff_s=retry_backoff_s,
    )

    # prefetch raw inputs in parallel per sensor
    input_reports: Dict[Tuple[int, str], Dict[str, Any]] = {}
    prefetch_errors: Dict[Tuple[int, str], str] = {}
    (
        sensor_by_key,
        fetch_sensor_by_key,
        sensor_to_fetch,
        sensor_models,
        fetch_members,
    ) = _build_gee_prefetch_plan(
        models=models,
        resolved_sensor=resolved_sensor,
        model_type=model_type,
    )

    inputs_cache = _restore_prefetch_cache_from_manifest_impl(
        manifest=manifest,
        arrays=arrays,
        restore_prefetch_checkpoint_cache=_restore_prefetch_checkpoint_cache,
    )
    tasks = _build_combined_prefetch_tasks_impl(
        provider=provider,
        spatials=spatials,
        fetch_sensor_by_key=fetch_sensor_by_key,
        fetch_members=fetch_members,
        inputs_cache=inputs_cache,
    )

    progress = _create_progress(
        enabled=bool(show_progress),
        total=(len(tasks) + len(pending_models)),
        desc="export_batch[combined]",
        unit="step",
    )

    def _write_checkpoint(*, stage: str, final: bool = False) -> Dict[str, Any]:
        return _write_combined_checkpoint_impl(
            manifest=manifest,
            arrays=arrays,
            stage=stage,
            final=final,
            out_path=out_path,
            fmt=fmt,
            save_manifest=save_manifest,
            json_path=json_path,
            max_retries=max_retries,
            retry_backoff_s=retry_backoff_s,
            write_one_payload=_write_one_payload,
        )

    prefetch_deps = _CombinedPrefetchDeps(
        run_with_retry=_run_with_retry,
        fetch_gee_patch_raw=_fetch_gee_patch_raw,
        normalize_input_chw=_normalize_input_chw,
        select_prefetched_channels=_select_prefetched_channels,
        inspect_input_raw=_inspect_input_raw,
    )

    try:
        if provider is not None and tasks:
            _run_combined_prefetch_tasks_impl(
                provider=provider,
                tasks=tasks,
                spatials=spatials,
                temporal=temporal,
                max_retries=max_retries,
                retry_backoff_s=retry_backoff_s,
                num_workers=num_workers,
                continue_on_error=continue_on_error,
                fail_on_bad_input=fail_on_bad_input,
                fetch_members=fetch_members,
                sensor_to_fetch=sensor_to_fetch,
                sensor_by_key=sensor_by_key,
                sensor_models=sensor_models,
                inputs_cache=inputs_cache,
                input_reports=input_reports,
                prefetch_errors=prefetch_errors,
                progress=progress,
                deps=prefetch_deps,
            )

        if provider is not None:
            _store_prefetch_checkpoint_arrays(
                arrays=arrays,
                manifest=manifest,
                sensor_by_key=sensor_by_key,
                inputs_cache=inputs_cache,
                n_items=len(spatials),
            )
            manifest = _write_checkpoint(stage="prefetched", final=False)

        def _get_or_fetch_input(i: int, skey: str, sspec: SensorSpec) -> np.ndarray:
            return _get_or_fetch_input_impl(
                i=i,
                skey=skey,
                sspec=sspec,
                provider=provider,
                spatials=spatials,
                temporal=temporal,
                max_retries=max_retries,
                retry_backoff_s=retry_backoff_s,
                fail_on_bad_input=fail_on_bad_input,
                inputs_cache=inputs_cache,
                input_reports=input_reports,
                prefetch_errors=prefetch_errors,
                deps=prefetch_deps,
            )

        input_refs_by_sensor = _collect_input_refs_by_sensor_impl(
            manifest=manifest,
            resolved_sensor=resolved_sensor,
            sensor_cache_key=_sensor_cache_key,
        )

        model_deps = _CombinedModelDeps(
            create_progress=_create_progress,
            drop_model_arrays=_drop_model_arrays,
            jsonable=_jsonable,
            sensor_key=_sensor_key,
            normalize_model_name=_normalize_model_name,
            get_embedder_bundle_cached=_get_embedder_bundle_cached,
            sensor_cache_key=_sensor_cache_key,
            sanitize_key=_sanitize_key,
            run_with_retry=_run_with_retry,
            call_embedder_get_embedding=_call_embedder_get_embedding,
            supports_prefetched_batch_api=_supports_prefetched_batch_api,
            supports_batch_api=_supports_batch_api,
            embedding_to_numpy=_embedding_to_numpy,
        )
        manifest = _run_pending_models_impl(
            pending_models=pending_models,
            arrays=arrays,
            manifest=manifest,
            spatials=spatials,
            temporal=temporal,
            output=output,
            resolved_sensor=resolved_sensor,
            model_type=model_type,
            backend=backend,
            device=device,
            save_inputs=save_inputs,
            save_embeddings=save_embeddings,
            continue_on_error=continue_on_error,
            chunk_size=chunk_size,
            max_retries=max_retries,
            retry_backoff_s=retry_backoff_s,
            show_progress=show_progress,
            input_refs_by_sensor=input_refs_by_sensor,
            get_or_fetch_input_fn=_get_or_fetch_input,
            write_checkpoint_fn=_write_checkpoint,
            progress=progress,
            deps=model_deps,
        )

        manifest["status"], manifest["summary"] = _summarize_combined_models_impl(manifest["models"])

        _drop_prefetch_checkpoint_arrays(arrays)
        manifest.pop("prefetch", None)
        manifest = _write_checkpoint(stage="done", final=True)
        return manifest
    finally:
        progress.close()
