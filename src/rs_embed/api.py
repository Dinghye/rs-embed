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
from .core.gee_image import build_gee_image as _build_gee_image
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

    backend_n = backend.lower().strip()
    model_n = model.lower().strip()

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

    backend_n = backend.lower().strip()
    model_n = model.lower().strip()

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

    backend_n = backend.lower().strip()
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
        m_n = m.lower().strip()
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
            out_path=out_path,
            backend=backend_n,
            device=device,
            output=output,
            resolved_sensor=resolved_sensor,
            model_type=model_type,
            save_inputs=save_inputs,
            save_embeddings=save_embeddings,
            save_manifest=save_manifest,
            fail_on_bad_input=fail_on_bad_input,
            num_workers=num_workers,
            fmt=fmt,
            continue_on_error=continue_on_error,
            max_retries=max_retries,
            retry_backoff_s=retry_backoff_s,
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

        # For GEE, prefetch once and reuse for export and/or embedding inference.
        need_prefetch = backend_n == "gee" and bool(save_inputs or save_embeddings) and bool(pending_idxs)
        pass_input_into_embedder = backend_n == "gee" and bool(save_embeddings)

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
                tasks: List[Tuple[int, str, SensorSpec]] = []
                seen: set[Tuple[int, str]] = set()
                for i in idxs:
                    for m in models:
                        sspec = resolved_sensor.get(m)
                        if sspec is None:
                            continue
                        if "precomputed" in (model_type.get(m) or ""):
                            continue
                        skey = _sensor_cache_key(sspec)
                        key = (i, skey)
                        if key in seen:
                            continue
                        seen.add(key)
                        tasks.append((i, skey, sspec))

                if tasks:
                    from concurrent.futures import ThreadPoolExecutor, as_completed

                    def _fetch_one(ii: int, sk: str, ss: SensorSpec):
                        assert provider is not None
                        x = _run_with_retry(
                            lambda: _fetch_gee_patch_raw(provider, spatial=spatials[ii], temporal=temporal, sensor=ss),
                            retries=max_retries,
                            backoff_s=retry_backoff_s,
                        )
                        rep = _inspect_input_raw(x, sensor=ss, name=f"gee_input_{sk}")
                        return ii, sk, x, rep

                    mw = max(1, int(num_workers))
                    with ThreadPoolExecutor(max_workers=mw) as ex:
                        fut_map = {ex.submit(_fetch_one, ii, sk, ss): (ii, sk) for (ii, sk, ss) in tasks}
                        for fut in as_completed(fut_map):
                            ii, sk = fut_map[fut]
                            try:
                                ii, sk, x, rep = fut.result()
                            except Exception as e:
                                if not continue_on_error:
                                    raise
                                prefetch_errors[(ii, sk)] = repr(e)
                                continue
                            if fail_on_bad_input and (not bool(rep.get("ok", True))):
                                issues = (rep.get("report", {}) or {}).get("issues", [])
                                err = RuntimeError(f"Input inspection failed for index={ii}, sensor={sk}: {issues}")
                                if not continue_on_error:
                                    raise err
                                prefetch_errors[(ii, sk)] = repr(err)
                                continue
                            inputs_cache[(ii, sk)] = x
                            input_reports[(ii, sk)] = rep

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
        progress.close()

# -----------------------------------------------------------------------------
# Internal: embedder caching
# -----------------------------------------------------------------------------

def _sensor_key(sensor: Optional[SensorSpec]) -> Tuple:
    if sensor is None:
        return ("__none__",)
    return (
        sensor.collection,
        sensor.bands,
        int(sensor.scale_m),
        int(sensor.cloudy_pct),
        float(sensor.fill_value),
        str(sensor.composite),
        bool(getattr(sensor, "check_input", False)),
        bool(getattr(sensor, "check_raise", True)),
        getattr(sensor, "check_save_dir", None),
    )


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
    """Return True when embedder overrides EmbedderBase.get_embeddings_batch."""
    fn = getattr(type(embedder), "get_embeddings_batch", None)
    if fn is None:
        return False
    from .embedders.base import EmbedderBase
    return fn is not EmbedderBase.get_embeddings_batch


# -----------------------------------------------------------------------------
# Internal: validation and capability checks
# -----------------------------------------------------------------------------

def _validate_specs(*, spatial: SpatialSpec, temporal: Optional[TemporalSpec], output: OutputSpec) -> None:
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


def _assert_supported(embedder, *, backend: str, output: OutputSpec, temporal: Optional[TemporalSpec]) -> None:
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

    temporal_hint = desc.get("temporal")
    if isinstance(temporal_hint, dict) and "mode" in temporal_hint:
        mode_hint = str(temporal_hint["mode"])
        if "year" in mode_hint and temporal is not None and getattr(temporal, "mode", None) != "year":
            raise ModelError(f"Model '{embedder.model_name}' expects TemporalSpec.mode='year' (or None).")


def _fetch_gee_patch_raw(provider: GEEProvider, *, spatial: SpatialSpec, temporal: Optional[TemporalSpec], sensor: SensorSpec) -> np.ndarray:
    region = provider.get_region_3857(spatial)
    img = _build_gee_image(sensor=sensor, temporal=temporal, region=region)
    return provider.fetch_array_chw(
        image=img,
        bands=sensor.bands,
        region=region,
        scale_m=int(sensor.scale_m),
        fill_value=float(sensor.fill_value),
        collection=sensor.collection,
    )


def _inspect_input_raw(x_chw: np.ndarray, *, sensor: SensorSpec, name: str) -> Dict[str, Any]:
    from .core.input_checks import inspect_chw

    rep = inspect_chw(
        x_chw,
        name=name,
        expected_channels=len(sensor.bands),
        value_range=None,
        fill_value=float(sensor.fill_value),
    )
    return {"ok": bool(rep.get("ok", False)), "report": rep, "sensor": _jsonable(sensor)}


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
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    arrays: Dict[str, np.ndarray] = {}
    manifest: Dict[str, Any] = {
        "created_at": _utc_ts(),
        "point_index": int(point_index),
        "status": "ok",
        "backend": backend,
        "device": device,
        "models": [],
        "spatial": _jsonable(spatial),
        "temporal": _jsonable(temporal),
        "output": _jsonable(output),
    }

    try:
        from importlib.metadata import version
        manifest["package_version"] = version("rs-embed")
    except Exception:
        manifest["package_version"] = None

    # local input cache: sensor_key -> x_chw
    local_inp: Dict[str, np.ndarray] = {}
    local_input_meta: Dict[str, Dict[str, Any]] = {}

    for m in models:
        m_entry: Dict[str, Any] = {"model": m, "status": "ok"}
        sspec = resolved_sensor.get(m)
        m_entry["sensor"] = _jsonable(sspec)

        try:
            # load embedder (cached)
            sensor_k = _sensor_key(sspec)
            embedder, lock = _get_embedder_bundle_cached(m.lower().strip(), backend, device, sensor_k)

            # describe once per point per model (cheap)
            try:
                m_entry["describe"] = _jsonable(embedder.describe())
            except Exception as e:
                m_entry["describe"] = {"error": repr(e)}

            # input: for on-the-fly models with a sensor
            input_chw: Optional[np.ndarray] = None
            report: Optional[Dict[str, Any]] = None
            needs_provider_input = backend == "gee" and sspec is not None and "precomputed" not in (model_type.get(m) or "")
            needs_input_for_embed = bool(pass_input_into_embedder and save_embeddings and needs_provider_input)
            needs_input_for_export = bool(save_inputs and needs_provider_input)
            if needs_input_for_embed or needs_input_for_export:
                skey = _sensor_cache_key(sspec)
                if skey in local_inp:
                    input_chw = local_inp[skey]
                else:
                    cached = inputs_cache.get((point_index, skey))
                    if cached is not None:
                        input_chw = cached
                        local_inp[skey] = input_chw
                    else:
                        # Fall back to direct fetch if chunk prefetch missed this item.
                        pref_err = prefetch_errors.get((point_index, skey))
                        if pref_err and continue_on_error:
                            raise RuntimeError(
                                f"Prefetch previously failed for model={m}, index={point_index}, sensor={skey}: {pref_err}"
                            )
                        prov = GEEProvider(auto_auth=True)
                        _run_with_retry(
                            lambda: prov.ensure_ready(),
                            retries=max_retries,
                            backoff_s=retry_backoff_s,
                        )
                        input_chw = _run_with_retry(
                            lambda: _fetch_gee_patch_raw(prov, spatial=spatial, temporal=temporal, sensor=sspec),
                            retries=max_retries,
                            backoff_s=retry_backoff_s,
                        )
                        local_inp[skey] = input_chw

                report = input_reports.get((point_index, skey))
                if report is None and input_chw is not None:
                    report = _inspect_input_raw(input_chw, sensor=sspec, name=f"gee_input_{skey}")

                if fail_on_bad_input and report is not None and (not bool(report.get("ok", True))):
                    issues = (report.get("report", {}) or {}).get("issues", [])
                    raise RuntimeError(f"Input inspection failed for model={m}: {issues}")

                if save_inputs and input_chw is not None:
                    if skey in local_input_meta:
                        input_meta = dict(local_input_meta[skey])
                        input_meta["dedup_reused"] = True
                    else:
                        input_key = f"input_chw__{_sanitize_key(m)}"
                        arrays[input_key] = np.asarray(input_chw, dtype=np.float32)
                        input_meta = {
                            "npz_key": input_key,
                            "dtype": str(arrays[input_key].dtype),
                            "shape": list(arrays[input_key].shape),
                            "sha1": _sha1(arrays[input_key]),
                            "inspection": _jsonable(report),
                        }
                        local_input_meta[skey] = dict(input_meta)
                    m_entry["input"] = input_meta
                else:
                    m_entry["input"] = None
            else:
                m_entry["input"] = None

            if save_embeddings:
                def _infer_once():
                    with lock:
                        return embedder.get_embedding(
                            spatial=spatial,
                            temporal=temporal,
                            sensor=sspec,
                            output=output,
                            backend=backend,
                            device=device,
                            input_chw=(input_chw if pass_input_into_embedder else None),
                        )
                emb = _run_with_retry(
                    _infer_once,
                    retries=max_retries,
                    backoff_s=retry_backoff_s,
                )
                e_np = _embedding_to_numpy(emb)
                emb_key = f"embedding__{_sanitize_key(m)}"
                arrays[emb_key] = e_np
                m_entry["embedding"] = {"npz_key": emb_key, "dtype": str(e_np.dtype), "shape": list(e_np.shape), "sha1": _sha1(e_np)}
                m_entry["meta"] = _jsonable(emb.meta)
            else:
                m_entry["embedding"] = None
                m_entry["meta"] = None
        except Exception as e:
            if not continue_on_error:
                raise
            m_entry["status"] = "failed"
            m_entry["error"] = repr(e)
            m_entry["input"] = m_entry.get("input")
            m_entry["embedding"] = None
            m_entry["meta"] = None

        manifest["models"].append(m_entry)

    n_failed = sum(1 for x in manifest["models"] if x.get("status") == "failed")
    if n_failed == 0:
        manifest["status"] = "ok"
    elif n_failed < len(manifest["models"]):
        manifest["status"] = "partial"
    else:
        manifest["status"] = "failed"
    manifest["summary"] = {
        "total_models": len(manifest["models"]),
        "failed_models": n_failed,
        "ok_models": len(manifest["models"]) - n_failed,
    }
    return arrays, manifest


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
    num_workers: int,
    fmt: str = "npz",
    continue_on_error: bool = False,
    max_retries: int = 0,
    retry_backoff_s: float = 0.0,
    show_progress: bool = False,
) -> Dict[str, Any]:
    # Simple combined implementation (no chunking): for large N prefer out_dir.
    arrays: Dict[str, np.ndarray] = {}
    manifest: Dict[str, Any] = {
        "created_at": _utc_ts(),
        "status": "ok",
        "backend": backend,
        "device": device,
        "models": [],
        "n_items": len(spatials),
        "temporal": _jsonable(temporal),
        "output": _jsonable(output),
        "spatials": [_jsonable(s) for s in spatials],
    }

    provider: Optional[GEEProvider] = None
    if backend == "gee" and (save_inputs or save_embeddings):
        provider = GEEProvider(auto_auth=True)
        _run_with_retry(
            lambda: provider.ensure_ready(),
            retries=max_retries,
            backoff_s=retry_backoff_s,
        )

    # prefetch raw inputs in parallel per sensor
    inputs_cache: Dict[Tuple[int, str], np.ndarray] = {}
    input_reports: Dict[Tuple[int, str], Dict[str, Any]] = {}
    prefetch_errors: Dict[Tuple[int, str], str] = {}
    tasks: List[Tuple[int, str, SensorSpec]] = []
    sensor_models: Dict[str, List[str]] = {}
    if provider is not None:
        seen: set[Tuple[int, str]] = set()
        for i, _sp in enumerate(spatials):
            for m in models:
                sspec = resolved_sensor.get(m)
                if sspec is None or "precomputed" in (model_type.get(m) or ""):
                    continue
                skey = _sensor_cache_key(sspec)
                sensor_models.setdefault(skey, []).append(m)
                key = (i, skey)
                if key in seen:
                    continue
                seen.add(key)
                tasks.append((i, skey, sspec))

    progress = _create_progress(
        enabled=bool(show_progress),
        total=(len(tasks) + len(models)),
        desc="export_batch[combined]",
        unit="step",
    )
    try:
        if provider is not None and tasks:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            def _fetch_one(i: int, skey: str, sspec: SensorSpec):
                assert provider is not None
                x = _run_with_retry(
                    lambda: _fetch_gee_patch_raw(provider, spatial=spatials[i], temporal=temporal, sensor=sspec),
                    retries=max_retries,
                    backoff_s=retry_backoff_s,
                )
                rep = _inspect_input_raw(x, sensor=sspec, name=f"gee_input_{skey}")
                return i, skey, x, rep

            mw = max(1, int(num_workers))
            with ThreadPoolExecutor(max_workers=mw) as ex:
                fut_map = {ex.submit(_fetch_one, i, sk, ss): (i, sk) for (i, sk, ss) in tasks}
                for fut in as_completed(fut_map):
                    i, skey = fut_map[fut]
                    try:
                        i, skey, x, rep = fut.result()
                    except Exception as e:
                        if not continue_on_error:
                            raise
                        prefetch_errors[(i, skey)] = repr(e)
                    else:
                        if fail_on_bad_input and (not bool(rep.get("ok", True))):
                            issues = (rep.get("report", {}) or {}).get("issues", [])
                            mlist = sorted(set(sensor_models.get(skey, [])))
                            err = RuntimeError(f"Input inspection failed for index={i}, models={mlist}: {issues}")
                            if not continue_on_error:
                                raise err
                            prefetch_errors[(i, skey)] = repr(err)
                            continue
                        inputs_cache[(i, skey)] = x
                        input_reports[(i, skey)] = rep
                    finally:
                        progress.update(1)

        def _get_or_fetch_input(i: int, skey: str, sspec: SensorSpec) -> np.ndarray:
            hit = inputs_cache.get((i, skey))
            if hit is not None:
                return hit
            pref_err = prefetch_errors.get((i, skey))
            if pref_err:
                raise RuntimeError(f"Prefetch previously failed for index={i}, sensor={skey}: {pref_err}")
            if provider is None:
                raise RuntimeError(f"Missing provider for input fetch: index={i}, sensor={skey}")
            x = _run_with_retry(
                lambda: _fetch_gee_patch_raw(provider, spatial=spatials[i], temporal=temporal, sensor=sspec),
                retries=max_retries,
                backoff_s=retry_backoff_s,
            )
            rep = _inspect_input_raw(x, sensor=sspec, name=f"gee_input_{skey}")
            if fail_on_bad_input and (not bool(rep.get("ok", True))):
                issues = (rep.get("report", {}) or {}).get("issues", [])
                raise RuntimeError(f"Input inspection failed for index={i}, sensor={skey}: {issues}")
            inputs_cache[(i, skey)] = x
            input_reports[(i, skey)] = rep
            return x

        input_refs_by_sensor: Dict[str, Dict[str, Any]] = {}

        # compute embeddings and fill arrays
        for m in models:
            m_entry: Dict[str, Any] = {
                "model": m,
                "sensor": _jsonable(resolved_sensor.get(m)),
                "status": "ok",
            }
            sspec = resolved_sensor.get(m)
            try:
                sensor_k = _sensor_key(sspec)
                embedder, lock = _get_embedder_bundle_cached(m.lower().strip(), backend, device, sensor_k)
                try:
                    m_entry["describe"] = _jsonable(embedder.describe())
                except Exception as e:
                    m_entry["describe"] = {"error": repr(e)}

                needs_provider_input = backend == "gee" and sspec is not None and "precomputed" not in (model_type.get(m) or "")
                skey = _sensor_cache_key(sspec) if needs_provider_input and sspec is not None else None

                # inputs: store once per sensor and let other models reference.
                if save_inputs and needs_provider_input and skey is not None:
                    if skey in input_refs_by_sensor:
                        m_entry["inputs"] = {**input_refs_by_sensor[skey], "dedup_reused": True}
                    else:
                        xs = []
                        missing = []
                        for i in range(len(spatials)):
                            try:
                                x = _get_or_fetch_input(i, skey, sspec)
                            except Exception as e:
                                missing.append((i, repr(e)))
                                continue
                            xs.append(np.asarray(x, dtype=np.float32))
                        if missing and not continue_on_error:
                            raise RuntimeError(f"Missing prefetched inputs for model={m}: {missing}")
                        if not xs:
                            m_entry["inputs"] = None
                        else:
                            try:
                                arr = np.stack(xs, axis=0)
                                in_key = f"inputs_bchw__{_sanitize_key(m)}"
                                arrays[in_key] = arr
                                ref = {"npz_key": in_key, "shape": list(arr.shape), "dtype": str(arr.dtype)}
                                input_refs_by_sensor[skey] = dict(ref)
                                m_entry["inputs"] = ref
                            except Exception:
                                keys = []
                                for i in range(len(spatials)):
                                    try:
                                        x = _get_or_fetch_input(i, skey, sspec)
                                    except Exception:
                                        continue
                                    k = f"input_chw__{_sanitize_key(m)}__{i:05d}"
                                    arrays[k] = np.asarray(x, dtype=np.float32)
                                    keys.append(k)
                                ref = {"npz_keys": keys}
                                input_refs_by_sensor[skey] = dict(ref)
                                m_entry["inputs"] = ref
                else:
                    m_entry["inputs"] = None

                if save_embeddings:
                    n = len(spatials)
                    embs_by_idx: List[Optional[np.ndarray]] = [None] * n
                    metas_by_idx: List[Optional[Dict[str, Any]]] = [None] * n
                    errors_by_idx: Dict[int, str] = {}

                    def _infer_one(i: int) -> Embedding:
                        inp = None
                        if needs_provider_input and skey is not None and sspec is not None:
                            inp = _get_or_fetch_input(i, skey, sspec)
                        with lock:
                            return embedder.get_embedding(
                                spatial=spatials[i],
                                temporal=temporal,
                                sensor=sspec,
                                output=output,
                                backend=backend,
                                device=device,
                                input_chw=inp,
                            )

                    can_batch = _supports_batch_api(embedder) and not needs_provider_input
                    batch_error: Optional[Exception] = None
                    if can_batch:
                        try:
                            def _infer_batch():
                                with lock:
                                    return embedder.get_embeddings_batch(
                                        spatials=spatials,
                                        temporal=temporal,
                                        sensor=sspec,
                                        output=output,
                                        backend=backend,
                                        device=device,
                                    )
                            batch_out = _run_with_retry(
                                _infer_batch,
                                retries=max_retries,
                                backoff_s=retry_backoff_s,
                            )
                            if len(batch_out) != n:
                                raise RuntimeError(
                                    f"Model {m} returned {len(batch_out)} embeddings for {n} inputs."
                                )
                            for i, emb in enumerate(batch_out):
                                embs_by_idx[i] = _embedding_to_numpy(emb)
                                metas_by_idx[i] = _jsonable(emb.meta)
                        except Exception as e:
                            batch_error = e
                            if not continue_on_error:
                                raise

                    if (not can_batch) or (batch_error is not None):
                        for i in range(n):
                            try:
                                emb = _run_with_retry(
                                    lambda i=i: _infer_one(i),
                                    retries=max_retries,
                                    backoff_s=retry_backoff_s,
                                )
                                embs_by_idx[i] = _embedding_to_numpy(emb)
                                metas_by_idx[i] = _jsonable(emb.meta)
                            except Exception as e:
                                if not continue_on_error:
                                    raise
                                errors_by_idx[i] = repr(e)

                    ok_indices = [i for i, e in enumerate(embs_by_idx) if e is not None]
                    if ok_indices:
                        try:
                            e_arr = np.stack([embs_by_idx[i] for i in ok_indices], axis=0)  # type: ignore[list-item]
                            if len(ok_indices) == n:
                                e_key = f"embeddings__{_sanitize_key(m)}"
                                arrays[e_key] = e_arr
                                m_entry["embeddings"] = {"npz_key": e_key, "shape": list(e_arr.shape), "dtype": str(e_arr.dtype)}
                            else:
                                keys = []
                                index_map = []
                                for j, i in enumerate(ok_indices):
                                    k = f"embedding__{_sanitize_key(m)}__{i:05d}"
                                    arrays[k] = e_arr[j]
                                    keys.append(k)
                                    index_map.append(i)
                                m_entry["embeddings"] = {"npz_keys": keys, "indices": index_map}
                        except Exception:
                            keys = []
                            index_map = []
                            for i in ok_indices:
                                k = f"embedding__{_sanitize_key(m)}__{i:05d}"
                                arrays[k] = embs_by_idx[i]  # type: ignore[index]
                                keys.append(k)
                                index_map.append(i)
                            m_entry["embeddings"] = {"npz_keys": keys, "indices": index_map}
                    else:
                        m_entry["embeddings"] = None

                    m_entry["metas"] = metas_by_idx
                    if errors_by_idx:
                        m_entry["failed_indices"] = sorted(errors_by_idx.keys())
                        m_entry["errors_by_index"] = errors_by_idx
                        if ok_indices:
                            m_entry["status"] = "partial"
                        else:
                            m_entry["status"] = "failed"
                else:
                    m_entry["embeddings"] = None
                    m_entry["metas"] = None
            except Exception as e:
                if not continue_on_error:
                    raise
                m_entry["status"] = "failed"
                m_entry["error"] = repr(e)
                m_entry["embeddings"] = None
                m_entry["metas"] = None
            finally:
                progress.update(1)

            manifest["models"].append(m_entry)

        n_failed = sum(1 for x in manifest["models"] if x.get("status") == "failed")
        n_partial = sum(1 for x in manifest["models"] if x.get("status") == "partial")
        if n_failed == 0 and n_partial == 0:
            manifest["status"] = "ok"
        elif n_failed < len(manifest["models"]):
            manifest["status"] = "partial"
        else:
            manifest["status"] = "failed"
        manifest["summary"] = {
            "total_models": len(manifest["models"]),
            "failed_models": n_failed,
            "partial_models": n_partial,
            "ok_models": len(manifest["models"]) - n_failed - n_partial,
        }

        manifest = _write_one_payload(
            out_path=out_path,
            arrays=arrays,
            manifest=manifest,
            save_manifest=save_manifest,
            fmt=fmt,
            max_retries=max_retries,
            retry_backoff_s=retry_backoff_s,
        )
        return manifest
    finally:
        progress.close()
