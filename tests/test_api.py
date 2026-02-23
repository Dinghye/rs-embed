"""Tests for the public API (get_embedding, get_embeddings_batch, export_batch).

These use a mock embedder registered in the test so they don't require
GEE, torch, or any real model weights.
"""
import numpy as np
import pytest

from rs_embed.core import registry
from rs_embed.core.embedding import Embedding
from rs_embed.core.errors import ModelError
from rs_embed.core.specs import PointBuffer, TemporalSpec, OutputSpec, SensorSpec
from rs_embed.embedders.base import EmbedderBase


# ── mock embedder ──────────────────────────────────────────────────

class _MockEmbedder(EmbedderBase):
    """Returns a deterministic embedding without any I/O."""

    def describe(self):
        return {"type": "mock", "dim": 8}

    def get_embedding(self, *, spatial, temporal, sensor, output, backend, device="auto", input_chw=None):
        vec = np.arange(8, dtype=np.float32)
        return Embedding(data=vec, meta={"model": self.model_name, "output": output.mode})


@pytest.fixture(autouse=True)
def register_mock():
    registry._REGISTRY.clear()
    registry.register("mock_model")(_MockEmbedder)
    yield
    registry._REGISTRY.clear()


# ── helpers ────────────────────────────────────────────────────────

_SPATIAL = PointBuffer(lon=0.0, lat=0.0, buffer_m=512)
_TEMPORAL = TemporalSpec.year(2024)


# ══════════════════════════════════════════════════════════════════════
# get_embedding
# ══════════════════════════════════════════════════════════════════════

def test_get_embedding_returns_embedding():
    from rs_embed.api import get_embedding

    emb = get_embedding("mock_model", spatial=_SPATIAL, temporal=_TEMPORAL)
    assert isinstance(emb, Embedding)
    assert emb.data.shape == (8,)
    assert emb.meta["model"] == "mock_model"


def test_get_embedding_output_modes():
    from rs_embed.api import get_embedding

    emb_pooled = get_embedding("mock_model", spatial=_SPATIAL, output=OutputSpec.pooled())
    assert emb_pooled.meta["output"] == "pooled"

    emb_grid = get_embedding("mock_model", spatial=_SPATIAL, output=OutputSpec.grid())
    assert emb_grid.meta["output"] == "grid"


def test_get_embedding_unknown_model():
    from rs_embed.api import get_embedding

    with pytest.raises(ModelError, match="Unknown model"):
        get_embedding("nonexistent", spatial=_SPATIAL)


# ══════════════════════════════════════════════════════════════════════
# get_embeddings_batch
# ══════════════════════════════════════════════════════════════════════

def test_get_embeddings_batch():
    from rs_embed.api import get_embeddings_batch

    spatials = [
        PointBuffer(lon=0.0, lat=0.0, buffer_m=256),
        PointBuffer(lon=1.0, lat=1.0, buffer_m=256),
        PointBuffer(lon=2.0, lat=2.0, buffer_m=256),
    ]
    results = get_embeddings_batch("mock_model", spatials=spatials, temporal=_TEMPORAL)
    assert len(results) == 3
    for emb in results:
        assert isinstance(emb, Embedding)


def test_get_embeddings_batch_empty():
    from rs_embed.api import get_embeddings_batch

    with pytest.raises(ModelError, match="non-empty"):
        get_embeddings_batch("mock_model", spatials=[], temporal=_TEMPORAL)


def test_get_embeddings_batch_with_sensor():
    """Ensures sensor param flows through _sensor_key without errors."""
    from rs_embed.api import get_embeddings_batch

    sensor = SensorSpec(collection="COLL", bands=("B1",))
    spatials = [PointBuffer(lon=0.0, lat=0.0, buffer_m=256)]
    results = get_embeddings_batch(
        "mock_model", spatials=spatials, temporal=_TEMPORAL, sensor=sensor,
    )
    assert len(results) == 1


# ══════════════════════════════════════════════════════════════════════
# _validate_specs
# ══════════════════════════════════════════════════════════════════════

def test_validate_specs_invalid_spatial_type():
    from rs_embed.api import _validate_specs

    with pytest.raises(ModelError, match="Invalid spatial spec type"):
        _validate_specs(spatial="not-spatial", temporal=None, output=OutputSpec.pooled())


def test_validate_specs_bad_output_mode():
    from rs_embed.api import _validate_specs

    bad_output = OutputSpec.__new__(OutputSpec)
    object.__setattr__(bad_output, "mode", "unknown")
    object.__setattr__(bad_output, "scale_m", 10)
    object.__setattr__(bad_output, "pooling", "mean")
    with pytest.raises(ModelError, match="Unknown output mode"):
        _validate_specs(spatial=_SPATIAL, temporal=None, output=bad_output)


def test_validate_specs_non_positive_scale():
    from rs_embed.api import _validate_specs

    bad_output = OutputSpec.__new__(OutputSpec)
    object.__setattr__(bad_output, "mode", "pooled")
    object.__setattr__(bad_output, "scale_m", 0)
    object.__setattr__(bad_output, "pooling", "mean")
    with pytest.raises(ModelError, match="scale_m must be positive"):
        _validate_specs(spatial=_SPATIAL, temporal=None, output=bad_output)


def test_validate_specs_bad_pooling():
    from rs_embed.api import _validate_specs

    bad_output = OutputSpec.__new__(OutputSpec)
    object.__setattr__(bad_output, "mode", "pooled")
    object.__setattr__(bad_output, "scale_m", 10)
    object.__setattr__(bad_output, "pooling", "median")
    with pytest.raises(ModelError, match="Unknown pooling"):
        _validate_specs(spatial=_SPATIAL, temporal=None, output=bad_output)


def test_validate_specs_ok():
    from rs_embed.api import _validate_specs

    _validate_specs(spatial=_SPATIAL, temporal=_TEMPORAL, output=OutputSpec.pooled())
    _validate_specs(spatial=_SPATIAL, temporal=None, output=OutputSpec.grid())


# ══════════════════════════════════════════════════════════════════════
# _assert_supported
# ══════════════════════════════════════════════════════════════════════

class _BackendLimitedEmbedder(EmbedderBase):
    """Embedder that only supports a specific backend."""
    def describe(self):
        return {
            "type": "mock", "dim": 8,
            "backend": ["gee"],
            "output": ["pooled"],
            "temporal": {"mode": "year"},
        }

    def get_embedding(self, *, spatial, temporal, sensor, output, backend, device="auto", input_chw=None):
        return Embedding(data=np.arange(8, dtype=np.float32), meta={})


class _BrokenDescribeEmbedder(EmbedderBase):
    """Embedder whose describe() raises — _assert_supported should not crash."""
    def describe(self):
        raise RuntimeError("broken")

    def get_embedding(self, **kw):
        return Embedding(data=np.zeros(4, dtype=np.float32), meta={})


def test_assert_supported_wrong_backend():
    from rs_embed.api import _assert_supported

    emb = _BackendLimitedEmbedder()
    emb.model_name = "limited"
    with pytest.raises(ModelError, match="does not support backend"):
        _assert_supported(emb, backend="local", output=OutputSpec.pooled(), temporal=None)


def test_assert_supported_wrong_output():
    from rs_embed.api import _assert_supported

    emb = _BackendLimitedEmbedder()
    emb.model_name = "limited"
    with pytest.raises(ModelError, match="does not support output.mode"):
        _assert_supported(emb, backend="gee", output=OutputSpec.grid(), temporal=None)


def test_assert_supported_wrong_temporal():
    from rs_embed.api import _assert_supported

    emb = _BackendLimitedEmbedder()
    emb.model_name = "limited"
    with pytest.raises(ModelError, match="expects TemporalSpec.mode='year'"):
        _assert_supported(
            emb, backend="gee", output=OutputSpec.pooled(),
            temporal=TemporalSpec.range("2022-01-01", "2022-06-01"),
        )


def test_assert_supported_ok():
    from rs_embed.api import _assert_supported

    emb = _BackendLimitedEmbedder()
    emb.model_name = "limited"
    _assert_supported(emb, backend="gee", output=OutputSpec.pooled(), temporal=TemporalSpec.year(2024))


def test_assert_supported_broken_describe_graceful():
    """_assert_supported should silently return when describe() throws."""
    from rs_embed.api import _assert_supported

    emb = _BrokenDescribeEmbedder()
    emb.model_name = "broken"
    _assert_supported(emb, backend="gee", output=OutputSpec.pooled(), temporal=None)


# ══════════════════════════════════════════════════════════════════════
# _sensor_key / _sensor_cache_key
# ══════════════════════════════════════════════════════════════════════

def test_sensor_key_none():
    from rs_embed.api import _sensor_key
    assert _sensor_key(None) == ("__none__",)


def test_sensor_key_deterministic_and_differs():
    from rs_embed.api import _sensor_key

    s1 = SensorSpec(collection="A", bands=("B1",))
    s2 = SensorSpec(collection="B", bands=("B1",))
    assert _sensor_key(s1) == _sensor_key(s1)
    assert _sensor_key(s1) != _sensor_key(s2)


def test_sensor_cache_key_deterministic_and_differs():
    from rs_embed.api import _sensor_cache_key

    s1 = SensorSpec(collection="A", bands=("B1",))
    s2 = SensorSpec(collection="B", bands=("B1",))
    assert isinstance(_sensor_cache_key(s1), str)
    assert _sensor_cache_key(s1) == _sensor_cache_key(s1)
    assert _sensor_cache_key(s1) != _sensor_cache_key(s2)


# ══════════════════════════════════════════════════════════════════════
# export_batch — argument validation (no GEE needed)
# ══════════════════════════════════════════════════════════════════════

def test_export_batch_empty_spatials():
    from rs_embed.api import export_batch

    with pytest.raises(ModelError, match="non-empty"):
        export_batch(spatials=[], temporal=_TEMPORAL, models=["mock_model"], out_dir="/tmp")


def test_export_batch_empty_models():
    from rs_embed.api import export_batch

    with pytest.raises(ModelError, match="non-empty"):
        export_batch(spatials=[_SPATIAL], temporal=_TEMPORAL, models=[], out_dir="/tmp")


def test_export_batch_no_output_arg():
    from rs_embed.api import export_batch

    with pytest.raises(ModelError, match="out_dir or out_path"):
        export_batch(spatials=[_SPATIAL], temporal=_TEMPORAL, models=["mock_model"])


def test_export_batch_both_output_args():
    from rs_embed.api import export_batch

    with pytest.raises(ModelError, match="only one"):
        export_batch(
            spatials=[_SPATIAL], temporal=_TEMPORAL, models=["mock_model"],
            out_dir="/tmp/a", out_path="/tmp/b.npz",
        )


def test_export_batch_decoupled_output_api_requires_out_and_layout():
    from rs_embed.api import export_batch

    with pytest.raises(ModelError, match="both out and layout"):
        export_batch(spatials=[_SPATIAL], temporal=_TEMPORAL, models=["mock_model"], out="/tmp/x")


def test_export_batch_decoupled_output_api_disallows_mixing_with_legacy_args():
    from rs_embed.api import export_batch

    with pytest.raises(ModelError, match="either out\\+layout or out_dir/out_path"):
        export_batch(
            spatials=[_SPATIAL],
            temporal=_TEMPORAL,
            models=["mock_model"],
            out="/tmp/x",
            layout="combined",
            out_path="/tmp/y.npz",
        )


def test_export_batch_unsupported_format():
    from rs_embed.api import export_batch

    with pytest.raises(ModelError, match="Unsupported export format"):
        export_batch(
            spatials=[_SPATIAL], temporal=_TEMPORAL, models=["mock_model"],
            out_dir="/tmp", format="parquet",
        )


def test_export_batch_accepts_netcdf_format(tmp_path):
    """format='netcdf' should pass validation (no GEE needed — it fails later, not at format check)."""
    from rs_embed.api import export_batch

    # This will proceed past format validation and attempt to actually export.
    # With our mock embedder it should succeed.
    results = export_batch(
        spatials=[_SPATIAL], temporal=_TEMPORAL, models=["mock_model"],
        out_dir=str(tmp_path), format="netcdf",
        save_inputs=False, save_embeddings=True, save_manifest=False,
    )
    assert len(results) == 1
    nc_file = tmp_path / "p00000.nc"
    assert nc_file.exists()


def test_export_batch_names_length_mismatch(tmp_path):
    from rs_embed.api import export_batch

    with pytest.raises(ModelError, match="same length"):
        export_batch(
            spatials=[_SPATIAL, _SPATIAL], temporal=_TEMPORAL, models=["mock_model"],
            out_dir=str(tmp_path), names=["only_one"],
        )


def test_export_batch_decoupled_layout_per_item(tmp_path):
    from rs_embed.api import export_batch

    export_batch(
        spatials=[_SPATIAL],
        temporal=_TEMPORAL,
        models=["mock_model"],
        out=str(tmp_path / "dir_out"),
        layout="per_item",
        save_inputs=False,
        save_embeddings=True,
        save_manifest=False,
        show_progress=False,
    )
    assert (tmp_path / "dir_out" / "p00000.npz").exists()


def test_export_batch_decoupled_layout_combined(tmp_path):
    from rs_embed.api import export_batch

    export_batch(
        spatials=[_SPATIAL],
        temporal=_TEMPORAL,
        models=["mock_model"],
        out=str(tmp_path / "combined_out"),
        layout="combined",
        save_inputs=False,
        save_embeddings=True,
        save_manifest=False,
        show_progress=False,
    )
    assert (tmp_path / "combined_out.npz").exists()
