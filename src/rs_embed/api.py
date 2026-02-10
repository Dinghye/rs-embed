from __future__ import annotations

import os
from functools import lru_cache
from threading import RLock
from typing import Any, Dict, List, Optional, Tuple

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
) -> Any:
    """Export inputs + embeddings for many spatials and many models.

    This is the recommended high-level entrypoint for batch export.

    - Accept any SpatialSpec list (like get_embeddings_batch).
    - Reuse cached embedder instances to avoid re-loading models/providers.
    - When `save_inputs=True` and `save_embeddings=True` (and backend is GEE),
      prefetch the raw GEE patch once and pass it to embedders via `input_chw`
      to avoid downloading the same patch twice.
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
        )

    # per-item mode (out_dir)
    assert out_dir is not None
    os.makedirs(out_dir, exist_ok=True)
    if names is None:
        names = [f"p{i:05d}" for i in range(len(spatials))]
    if len(names) != len(spatials):
        raise ModelError("names must have the same length as spatials.")

    # When saving inputs + embeddings for on-the-fly models on GEE, prefetch so we don't download twice.
    need_prefetch = bool(save_inputs) and backend_n == "gee"
    pass_input_into_embedder = bool(save_inputs and save_embeddings and backend_n == "gee")

    provider: Optional[GEEProvider] = None
    if need_prefetch:
        provider = GEEProvider(auto_auth=True)
        provider.ensure_ready()

    manifests: List[Dict[str, Any]] = []

    n = len(spatials)
    csize = max(1, int(chunk_size))
    for chunk_start in range(0, n, csize):
        chunk_end = min(n, chunk_start + csize)
        idxs = list(range(chunk_start, chunk_end))

        # (i, sensor_cache_key) -> input_chw
        inputs_cache: Dict[Tuple[int, str], np.ndarray] = {}
        input_reports: Dict[Tuple[int, str], Dict[str, Any]] = {}

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
                    x = _fetch_gee_patch_raw(provider, spatial=spatials[ii], temporal=temporal, sensor=ss)
                    rep = _inspect_input_raw(x, sensor=ss, name=f"gee_input_{sk}")
                    return ii, sk, x, rep

                mw = max(1, int(num_workers))
                with ThreadPoolExecutor(max_workers=mw) as ex:
                    futs = [ex.submit(_fetch_one, ii, sk, ss) for (ii, sk, ss) in tasks]
                    for fut in as_completed(futs):
                        ii, sk, x, rep = fut.result()
                        inputs_cache[(ii, sk)] = x
                        input_reports[(ii, sk)] = rep

        # export each point in chunk
        ext = get_extension(fmt)
        for i in idxs:
            out_file = os.path.join(out_dir, f"{names[i]}{ext}")
            mani = _export_one_point_npz(
                point_index=i,
                spatial=spatials[i],
                temporal=temporal,
                models=models,
                out_path=out_file,
                backend=backend_n,
                device=device,
                output=output,
                resolved_sensor=resolved_sensor,
                model_type=model_type,
                inputs_cache=inputs_cache,
                input_reports=input_reports,
                pass_input_into_embedder=pass_input_into_embedder,
                save_inputs=save_inputs,
                save_embeddings=save_embeddings,
                save_manifest=save_manifest,
                fail_on_bad_input=fail_on_bad_input,
                fmt=fmt,
            )
            manifests.append(mani)

    return manifests

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
) -> Dict[str, Any]:
    arrays: Dict[str, np.ndarray] = {}
    manifest: Dict[str, Any] = {
        "created_at": _utc_ts(),
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

    for m in models:
        m_entry: Dict[str, Any] = {"model": m}
        sspec = resolved_sensor.get(m)
        m_entry["sensor"] = _jsonable(sspec)

        # load embedder (cached)
        sensor_k = _sensor_key(sspec)
        embedder, lock = _get_embedder_bundle_cached(m.lower().strip(), backend, device, sensor_k)

        # describe once per point per model (cheap)
        try:
            m_entry["describe"] = _jsonable(embedder.describe())
        except Exception as e:
            m_entry["describe"] = {"error": repr(e)}

        # input: only for on-the-fly models with a sensor
        input_chw: Optional[np.ndarray] = None
        report: Optional[Dict[str, Any]] = None
        if save_inputs and backend == "gee" and sspec is not None and "precomputed" not in (model_type.get(m) or ""):
            skey = _sensor_cache_key(sspec)
            # try chunk cache first (we don't know index here; it might not exist)
            # for one-point export we cache only in local_inp; export_batch fills local_inp before calling us
            if skey in local_inp:
                input_chw = local_inp[skey]
            else:
                # try batch-level cache
                cached = inputs_cache.get((point_index, skey))
                if cached is not None:
                    input_chw = cached
                    local_inp[skey] = input_chw
                else:
                    # fall back: just fetch once (should be rare when called from export_batch)
                    prov = GEEProvider(auto_auth=True)
                    prov.ensure_ready()
                    input_chw = _fetch_gee_patch_raw(prov, spatial=spatial, temporal=temporal, sensor=sspec)
                    local_inp[skey] = input_chw

            # inspection report (from cache if available)
            report = input_reports.get((point_index, skey))
            if report is None:
                report = _inspect_input_raw(input_chw, sensor=sspec, name=f"gee_input_{skey}")

            input_key = f"input_chw__{_sanitize_key(m)}"
            arrays[input_key] = np.asarray(input_chw, dtype=np.float32)

            m_entry["input"] = {
                "npz_key": input_key,
                "dtype": str(arrays[input_key].dtype),
                "shape": list(arrays[input_key].shape),
                "sha1": _sha1(arrays[input_key]),
                "inspection": _jsonable(report),
            }

            if fail_on_bad_input and report is not None and (not bool(report.get("ok", True))):
                issues = (report.get("report", {}) or {}).get("issues", [])
                raise RuntimeError(f"Input inspection failed for model={m}: {issues}")
        else:
            m_entry["input"] = None

        # embeddings
        if save_embeddings:
            with lock:
                emb = embedder.get_embedding(
                    spatial=spatial,
                    temporal=temporal,
                    sensor=sspec,
                    output=output,
                    backend=backend,
                    device=device,
                    input_chw=(input_chw if pass_input_into_embedder else None),
                )
            e_np = _embedding_to_numpy(emb)
            emb_key = f"embedding__{_sanitize_key(m)}"
            arrays[emb_key] = e_np
            m_entry["embedding"] = {"npz_key": emb_key, "dtype": str(e_np.dtype), "shape": list(e_np.shape), "sha1": _sha1(e_np)}
            m_entry["meta"] = _jsonable(emb.meta)
        else:
            m_entry["embedding"] = None
            m_entry["meta"] = None

        manifest["models"].append(m_entry)

    from .writers import write_arrays
    manifest = write_arrays(
        fmt=fmt, out_path=out_path, arrays=arrays,
        manifest=_jsonable(manifest), save_manifest=save_manifest,
    )
    return manifest


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
) -> Dict[str, Any]:
    # Simple combined implementation (no chunking): for large N prefer out_dir.
    arrays: Dict[str, np.ndarray] = {}
    manifest: Dict[str, Any] = {
        "created_at": _utc_ts(),
        "backend": backend,
        "device": device,
        "models": [],
        "n_items": len(spatials),
        "temporal": _jsonable(temporal),
        "output": _jsonable(output),
        "spatials": [_jsonable(s) for s in spatials],
    }

    provider: Optional[GEEProvider] = None
    if save_inputs and backend == "gee":
        provider = GEEProvider(auto_auth=True)
        provider.ensure_ready()

    # prefetch raw inputs in parallel per sensor
    inputs_cache: Dict[Tuple[int, str], np.ndarray] = {}
    input_reports: Dict[Tuple[int, str], Dict[str, Any]] = {}
    if provider is not None:
        tasks: List[Tuple[int, str, SensorSpec]] = []
        sensor_models: Dict[str, List[str]] = {}
        seen: set[Tuple[int, str]] = set()
        for i, sp in enumerate(spatials):
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
        from concurrent.futures import ThreadPoolExecutor, as_completed
        def _fetch_one(i: int, skey: str, sspec: SensorSpec):
            assert provider is not None
            x = _fetch_gee_patch_raw(provider, spatial=spatials[i], temporal=temporal, sensor=sspec)
            rep = _inspect_input_raw(x, sensor=sspec, name=f"gee_input_{skey}")
            return i, skey, x, rep
        mw = max(1, int(num_workers))
        with ThreadPoolExecutor(max_workers=mw) as ex:
            futs = [ex.submit(_fetch_one, i, sk, ss) for (i, sk, ss) in tasks]
            for fut in as_completed(futs):
                i, skey, x, rep = fut.result()
                if fail_on_bad_input and (not bool(rep.get("ok", True))):
                    issues = (rep.get("report", {}) or {}).get("issues", [])
                    mlist = sorted(set(sensor_models.get(skey, [])))
                    raise RuntimeError(
                        f"Input inspection failed for index={i}, models={mlist}: {issues}"
                    )
                inputs_cache[(i, skey)] = x
                input_reports[(i, skey)] = rep

    # compute embeddings and fill arrays
    for m in models:
        m_entry: Dict[str, Any] = {"model": m, "sensor": _jsonable(resolved_sensor.get(m))}
        sspec = resolved_sensor.get(m)
        sensor_k = _sensor_key(sspec)
        embedder, lock = _get_embedder_bundle_cached(m.lower().strip(), backend, device, sensor_k)
        try:
            m_entry["describe"] = _jsonable(embedder.describe())
        except Exception as e:
            m_entry["describe"] = {"error": repr(e)}

        # inputs: store as [N,C,H,W] if possible (same shape)
        if save_inputs and backend == "gee" and sspec is not None and "precomputed" not in (model_type.get(m) or ""):
            skey = _sensor_cache_key(sspec)
            xs = []
            ok = True
            for i in range(len(spatials)):
                x = inputs_cache.get((i, skey))
                if x is None:
                    ok = False
                    break
                xs.append(np.asarray(x, dtype=np.float32))
            if ok:
                try:
                    arr = np.stack(xs, axis=0)
                    in_key = f"inputs_bchw__{_sanitize_key(m)}"
                    arrays[in_key] = arr
                    m_entry["inputs"] = {"npz_key": in_key, "shape": list(arr.shape), "dtype": str(arr.dtype)}
                except Exception:
                    # fallback: per-item keys
                    keys = []
                    for i, x in enumerate(xs):
                        k = f"input_chw__{_sanitize_key(m)}__{i:05d}"
                        arrays[k] = x
                        keys.append(k)
                    m_entry["inputs"] = {"npz_keys": keys}
            else:
                m_entry["inputs"] = None
        else:
            m_entry["inputs"] = None

        if save_embeddings:
            embs = []
            metas = []
            with lock:
                for i, sp in enumerate(spatials):
                    inp = None
                    if save_inputs and backend == "gee" and sspec is not None and "precomputed" not in (model_type.get(m) or ""):
                        skey = _sensor_cache_key(sspec)
                        inp = inputs_cache.get((i, skey))
                    emb = embedder.get_embedding(
                        spatial=sp,
                        temporal=temporal,
                        sensor=sspec,
                        output=output,
                        backend=backend,
                        device=device,
                        input_chw=inp,
                    )
                    embs.append(_embedding_to_numpy(emb))
                    metas.append(_jsonable(emb.meta))
            # try stack
            try:
                e_arr = np.stack(embs, axis=0)
                e_key = f"embeddings__{_sanitize_key(m)}"
                arrays[e_key] = e_arr
                m_entry["embeddings"] = {"npz_key": e_key, "shape": list(e_arr.shape), "dtype": str(e_arr.dtype)}
            except Exception:
                keys = []
                for i, e in enumerate(embs):
                    k = f"embedding__{_sanitize_key(m)}__{i:05d}"
                    arrays[k] = e
                    keys.append(k)
                m_entry["embeddings"] = {"npz_keys": keys}
            m_entry["metas"] = metas
        else:
            m_entry["embeddings"] = None
            m_entry["metas"] = None

        manifest["models"].append(m_entry)

    from .writers import write_arrays
    manifest = write_arrays(
        fmt=fmt, out_path=out_path, arrays=arrays,
        manifest=_jsonable(manifest), save_manifest=save_manifest,
    )
    return manifest
