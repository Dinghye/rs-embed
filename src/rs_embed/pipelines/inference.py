"""Inference engine for export pipelines.

This module resolves embedders and executes model inference in either
single-point or batch mode, with consistent retry/error shaping.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..core.embedding import Embedding
from ..tools.serialization import embedding_to_numpy, jsonable, sensor_cache_key
from ..core.specs import OutputSpec, SensorSpec, SpatialSpec, TemporalSpec
from ..core.types import ExportConfig, ModelConfig, Status, TaskResult
from ..tools.output import normalize_embedding_output
from ..tools.runtime import (
    call_embedder_get_embedding,
    get_embedder_bundle_cached,
    sensor_key,
    supports_batch_api,
    supports_prefetched_batch_api,
)
from ..tools.tiling import (
    _call_embedder_get_embedding_with_input_prep,
    _resolve_input_prep_spec,
)
from .runner import run_with_retry


class InferenceEngine:
    """Manages embedder lifecycle and dispatches single/batch inference.

    It centralizes inference policy (batch vs. single fallback), embedder
    bundle reuse, and result normalization into :class:`TaskResult` records.

    Parameters
    ----------
    device : str
        Target device (``"auto"``, ``"cpu"``, ``"cuda"``, …).
    output : OutputSpec
        Embedding output spec (pooled/grid).
    config : ExportConfig
        Behavioral flags.
    """

    def __init__(
        self,
        *,
        device: str,
        output: OutputSpec,
        config: ExportConfig,
    ) -> None:
        self.device = device
        self.output = output
        self.config = config
        self.input_prep_resolved = _resolve_input_prep_spec(config.input_prep)
        self.prefer_batch = _device_has_gpu(device)
        self._explicit_nonresize = (config.input_prep is not None) and (
            self.input_prep_resolved.mode in {"tile", "auto"}
        )

    # ── single-point inference ─────────────────────────────────────

    def infer_single(
        self,
        *,
        embedder: Any,
        lock: Any,
        spatial: SpatialSpec,
        temporal: Optional[TemporalSpec],
        sensor: Optional[SensorSpec],
        backend: str,
        input_chw: Optional[np.ndarray] = None,
    ) -> Embedding:
        """Run a single embedding with retry + optional tiling."""
        cfg = self.config
        return run_with_retry(
            lambda: _call_embedder_get_embedding_with_input_prep(
                embedder=embedder,
                spatial=spatial,
                temporal=temporal,
                sensor=sensor,
                output=self.output,
                backend=backend,
                device=self.device,
                input_chw=input_chw,
                input_prep=cfg.input_prep,
            ),
            retries=cfg.max_retries,
            backoff_s=cfg.retry_backoff_s,
        )

    # ── chunk inference (multi-point × multi-model) ────────────────

    def infer_chunk(
        self,
        *,
        idxs: List[int],
        spatials: List[SpatialSpec],
        temporal: Optional[TemporalSpec],
        models: List[ModelConfig],
        prefetch_cache: Dict[Tuple[int, str], np.ndarray],
        prefetch_errors: Dict[Tuple[int, str], str],
        model_progress_cb: Optional[Any] = None,
    ) -> Dict[Tuple[int, str], TaskResult]:
        """Infer embeddings for a chunk of spatial indices across all models.

        Returns ``{(point_idx, model_name): TaskResult}``.
        """
        out: Dict[Tuple[int, str], TaskResult] = {}
        cfg = self.config
        infer_bs = cfg.effective_infer_batch_size

        for mc in models:
            sensor_k = sensor_key(mc.sensor)
            skey = (
                sensor_cache_key(mc.sensor)
                if (not mc.is_precomputed) and mc.sensor is not None
                else None
            )
            needs_provider_input = skey is not None

            from ..tools.normalization import normalize_model_name

            embedder, lock = get_embedder_bundle_cached(
                normalize_model_name(mc.name), mc.backend, self.device, sensor_k
            )

            def _get_input(i: int) -> Optional[np.ndarray]:
                if not needs_provider_input or skey is None:
                    return None
                hit = prefetch_cache.get((i, skey))
                if hit is not None:
                    return hit
                err = prefetch_errors.get((i, skey))
                if err:
                    raise RuntimeError(
                        f"Prefetch failed for model={mc.name}, index={i}: {err}"
                    )
                raise RuntimeError(
                    f"Missing prefetched input for model={mc.name}, index={i}"
                )

            def _single(i: int) -> Embedding:
                inp = _get_input(i)
                return self.infer_single(
                    embedder=embedder,
                    lock=lock,
                    spatial=spatials[i],
                    temporal=temporal,
                    sensor=mc.sensor,
                    backend=mc.backend,
                    input_chw=inp,
                )

            def _mark_done(model_name: str) -> None:
                if model_progress_cb is not None:
                    try:
                        model_progress_cb(model_name)
                    except Exception:
                        pass

            def _record_ok(i: int) -> None:
                pass  # placeholder for progress; actual recording below

            # -- try batch paths first --
            batch_succeeded = False

            can_batch_prefetched = (
                self.prefer_batch
                and not self._explicit_nonresize
                and supports_prefetched_batch_api(embedder)
                and needs_provider_input
                and mc.sensor is not None
                and skey is not None
            )
            can_batch_no_input = (
                self.prefer_batch
                and not self._explicit_nonresize
                and supports_batch_api(embedder)
                and not needs_provider_input
            )

            if can_batch_prefetched:
                try:
                    ready: List[Tuple[int, SpatialSpec, np.ndarray]] = []
                    for i in idxs:
                        try:
                            inp = _get_input(i)
                            assert inp is not None
                            ready.append(
                                (i, spatials[i], np.asarray(inp, dtype=np.float32))
                            )
                        except Exception as e:
                            if not cfg.continue_on_error:
                                raise
                            out[(i, mc.name)] = TaskResult.failed(e)
                            _mark_done(mc.name)

                    for start in range(0, len(ready), infer_bs):
                        sub = ready[start : start + infer_bs]
                        if not sub:
                            continue
                        sub_idx = [t[0] for t in sub]
                        sub_sp = [t[1] for t in sub]
                        sub_inp = [t[2] for t in sub]

                        batch_out = run_with_retry(
                            lambda: embedder.get_embeddings_batch_from_inputs(
                                spatials=sub_sp,
                                input_chws=sub_inp,
                                temporal=temporal,
                                sensor=mc.sensor,
                                output=self.output,
                                backend=mc.backend,
                                device=self.device,
                            ),
                            retries=cfg.max_retries,
                            backoff_s=cfg.retry_backoff_s,
                        )
                        if len(batch_out) != len(sub_idx):
                            raise RuntimeError(
                                f"Model {mc.name} returned {len(batch_out)} embeddings "
                                f"for {len(sub_idx)} prefetched inputs."
                            )
                        for j, emb in enumerate(batch_out):
                            emb_n = normalize_embedding_output(
                                emb=emb, output=self.output
                            )
                            out[(sub_idx[j], mc.name)] = TaskResult.ok(
                                embedding_to_numpy(emb_n),
                                jsonable(getattr(emb_n, "meta", None)),
                            )
                            _mark_done(mc.name)
                    batch_succeeded = True
                except Exception:
                    batch_succeeded = False

            if not batch_succeeded and can_batch_no_input:
                try:
                    for start in range(0, len(idxs), infer_bs):
                        sub_idx = idxs[start : start + infer_bs]
                        sub_sp = [spatials[i] for i in sub_idx]

                        batch_out = run_with_retry(
                            lambda: embedder.get_embeddings_batch(
                                spatials=sub_sp,
                                temporal=temporal,
                                sensor=mc.sensor,
                                output=self.output,
                                backend=mc.backend,
                                device=self.device,
                            ),
                            retries=cfg.max_retries,
                            backoff_s=cfg.retry_backoff_s,
                        )
                        if len(batch_out) != len(sub_idx):
                            raise RuntimeError(
                                f"Model {mc.name} returned {len(batch_out)} embeddings "
                                f"for {len(sub_idx)} inputs."
                            )
                        for j, emb in enumerate(batch_out):
                            emb_n = normalize_embedding_output(
                                emb=emb, output=self.output
                            )
                            out[(sub_idx[j], mc.name)] = TaskResult.ok(
                                embedding_to_numpy(emb_n),
                                jsonable(getattr(emb_n, "meta", None)),
                            )
                            _mark_done(mc.name)
                    batch_succeeded = True
                except Exception:
                    batch_succeeded = False

            # -- fallback: single-item --
            if not batch_succeeded:
                for i in idxs:
                    if (i, mc.name) in out:
                        continue
                    try:
                        emb = _single(i)
                        out[(i, mc.name)] = TaskResult.ok(
                            embedding_to_numpy(emb),
                            jsonable(getattr(emb, "meta", None)),
                        )
                    except Exception as e:
                        if not cfg.continue_on_error:
                            raise
                        out[(i, mc.name)] = TaskResult.failed(e)
                    _mark_done(mc.name)

        return out

    # ── embedder helpers ───────────────────────────────────────────

    @staticmethod
    def resolve_embedder(model_config: ModelConfig, device: str) -> Tuple[Any, Any]:
        """Return ``(embedder, lock)`` for the given model config."""
        from ..tools.normalization import normalize_model_name

        sensor_k = sensor_key(model_config.sensor)
        return get_embedder_bundle_cached(
            normalize_model_name(model_config.name),
            model_config.backend,
            device,
            sensor_k,
        )

    # ── combined-export: infer all spatials for one model ──────────

    def infer_model(
        self,
        *,
        model_name: str,
        model_backend: str,
        sensor: Optional[SensorSpec],
        is_precomputed: bool,
        provider_enabled: bool,
        spatials: List[SpatialSpec],
        temporal: Optional[TemporalSpec],
        inference_strategy: str,
        get_input_fn: Callable[[int, str, SensorSpec], np.ndarray],
        progress_cb: Optional[Callable[[int], None]] = None,
    ) -> Dict[int, TaskResult]:
        """Infer embeddings for ALL spatial indices for a single model.

        Used by combined export.  Returns ``{spatial_idx: TaskResult}``.

        Parameters
        ----------
        get_input_fn
            ``fn(idx, sensor_cache_key, sensor_spec) -> np.ndarray`` that
            retrieves the (possibly cached) provider input for a point.
        progress_cb
            Called with spatial index after each point finishes inference.
        """
        from ..tools.normalization import normalize_model_name

        cfg = self.config
        n = len(spatials)
        infer_bs = cfg.effective_infer_batch_size
        out: Dict[int, TaskResult] = {}
        done: set[int] = set()

        sensor_k = sensor_key(sensor)
        skey = (
            sensor_cache_key(sensor)
            if provider_enabled and sensor is not None and not is_precomputed
            else None
        )
        needs_provider_input = skey is not None

        embedder, lock = get_embedder_bundle_cached(
            normalize_model_name(model_name), model_backend, self.device, sensor_k
        )

        strategy = str(inference_strategy).strip().lower()
        prefer_batch = (strategy == "batch") or (strategy == "auto")
        allow_batch = strategy != "single"

        def _mark_done(i: int) -> None:
            if i in done:
                return
            done.add(i)
            if progress_cb is not None:
                progress_cb(i)

        def _infer_one(i: int) -> Embedding:
            inp = None
            if needs_provider_input and skey is not None and sensor is not None:
                inp = get_input_fn(i, skey, sensor)
            with lock:
                return call_embedder_get_embedding(
                    embedder=embedder,
                    spatial=spatials[i],
                    temporal=temporal,
                    sensor=sensor,
                    output=self.output,
                    backend=model_backend,
                    device=self.device,
                    input_chw=inp,
                )

        can_batch_prefetched = (
            allow_batch
            and prefer_batch
            and supports_prefetched_batch_api(embedder)
            and needs_provider_input
            and skey is not None
            and sensor is not None
        )
        can_batch = (
            allow_batch
            and prefer_batch
            and supports_batch_api(embedder)
            and not needs_provider_input
        )
        batch_attempted = False
        batch_succeeded = False

        # Tier 1: batch with prefetched inputs
        if can_batch_prefetched:
            batch_attempted = True
            try:
                batch_indices: List[int] = []
                batch_spatials: List[SpatialSpec] = []
                batch_inputs: List[np.ndarray] = []
                for i in range(n):
                    try:
                        assert skey is not None and sensor is not None
                        inp = get_input_fn(i, skey, sensor)
                    except Exception as e:
                        if not cfg.continue_on_error:
                            raise
                        out[i] = TaskResult.failed(e)
                        continue
                    batch_indices.append(i)
                    batch_spatials.append(spatials[i])
                    batch_inputs.append(np.asarray(inp, dtype=np.float32))

                if batch_spatials:
                    for start in range(0, len(batch_spatials), infer_bs):
                        sub_spatials = batch_spatials[start : start + infer_bs]
                        sub_inputs = batch_inputs[start : start + infer_bs]
                        sub_indices = batch_indices[start : start + infer_bs]

                        def _infer_prefetched(_sp=sub_spatials, _inp=sub_inputs):
                            with lock:
                                return embedder.get_embeddings_batch_from_inputs(
                                    spatials=_sp,
                                    input_chws=_inp,
                                    temporal=temporal,
                                    sensor=sensor,
                                    output=self.output,
                                    backend=model_backend,
                                    device=self.device,
                                )

                        batch_out = run_with_retry(
                            _infer_prefetched,
                            retries=cfg.max_retries,
                            backoff_s=cfg.retry_backoff_s,
                        )
                        if len(batch_out) != len(sub_spatials):
                            raise RuntimeError(
                                f"Model {model_name} returned {len(batch_out)} "
                                f"embeddings for {len(sub_spatials)} prefetched inputs."
                            )
                        for j, emb in enumerate(batch_out):
                            emb = normalize_embedding_output(
                                emb=emb, output=self.output
                            )
                            out[sub_indices[j]] = TaskResult.ok(
                                embedding_to_numpy(emb),
                                jsonable(emb.meta),
                            )
                            _mark_done(sub_indices[j])

                batch_succeeded = True
                # Mark non-batch items done (they had errors)
                for i in range(n):
                    if i not in done:
                        _mark_done(i)
            except Exception:
                batch_succeeded = False

        # Tier 2: batch without inputs
        if not batch_attempted and can_batch:
            batch_attempted = True
            try:
                for start in range(0, n, infer_bs):
                    sub_spatials = spatials[start : start + infer_bs]

                    def _infer_batch(_sp=sub_spatials):
                        with lock:
                            return embedder.get_embeddings_batch(
                                spatials=_sp,
                                temporal=temporal,
                                sensor=sensor,
                                output=self.output,
                                backend=model_backend,
                                device=self.device,
                            )

                    batch_out = run_with_retry(
                        _infer_batch,
                        retries=cfg.max_retries,
                        backoff_s=cfg.retry_backoff_s,
                    )
                    if len(batch_out) != len(sub_spatials):
                        raise RuntimeError(
                            f"Model {model_name} returned {len(batch_out)} "
                            f"embeddings for {len(sub_spatials)} inputs."
                        )
                    for j, emb in enumerate(batch_out):
                        emb = normalize_embedding_output(emb=emb, output=self.output)
                        i = start + j
                        out[i] = TaskResult.ok(
                            embedding_to_numpy(emb),
                            jsonable(emb.meta),
                        )
                        _mark_done(i)
                batch_succeeded = True
            except Exception:
                batch_succeeded = False

        # Tier 3: single-item fallback
        if not batch_succeeded:
            for i in range(n):
                if i in done:
                    continue
                try:
                    emb = run_with_retry(
                        lambda i=i: _infer_one(i),
                        retries=cfg.max_retries,
                        backoff_s=cfg.retry_backoff_s,
                    )
                    out[i] = TaskResult.ok(
                        embedding_to_numpy(emb),
                        jsonable(getattr(emb, "meta", None)),
                    )
                except Exception as e:
                    if not cfg.continue_on_error:
                        raise
                    out[i] = TaskResult.failed(e)
                finally:
                    _mark_done(i)

        return out


# ── module-level helpers ───────────────────────────────────────────


def _device_has_gpu(device: str) -> bool:
    dev = str(device or "").strip().lower()
    if dev and dev not in {"auto", "cpu"}:
        return True
    if dev == "cpu":
        return False
    try:
        import torch

        if bool(getattr(torch.cuda, "is_available", lambda: False)()):
            return True
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and bool(getattr(mps, "is_available", lambda: False)()):
            return True
    except Exception:
        return False
    return False
