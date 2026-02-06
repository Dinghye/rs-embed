import numpy as np
import pytest

from rs_embed.core import registry
from rs_embed.core.embedding import Embedding
from rs_embed.core.specs import PointBuffer, TemporalSpec, SensorSpec, OutputSpec


@pytest.fixture(autouse=True)
def clean_registry():
    registry._REGISTRY.clear()
    yield
    registry._REGISTRY.clear()


def test_export_batch_prefetch_dedup_across_models(tmp_path, monkeypatch):
    # Register two on-the-fly models that require input_chw to be provided.
    class DummyA:
        calls = 0

        def describe(self):
            return {
                "type": "onthefly",
                "inputs": {"collection": "C", "bands": ["B1", "B2", "B3"]},
                "defaults": {"scale_m": 10, "cloudy_pct": 30, "composite": "median", "fill_value": 0.0},
            }

        def get_embedding(self, *, spatial, temporal, sensor, output, backend, device="auto", input_chw=None):
            DummyA.calls += 1
            assert backend == "gee"
            assert input_chw is not None
            return Embedding(data=np.array([float(np.sum(input_chw))], dtype=np.float32), meta={})

    class DummyB:
        calls = 0

        def describe(self):
            return {
                "type": "onthefly",
                "inputs": {"collection": "C", "bands": ["B1", "B2", "B3"]},
                "defaults": {"scale_m": 10, "cloudy_pct": 30, "composite": "median", "fill_value": 0.0},
            }

        def get_embedding(self, *, spatial, temporal, sensor, output, backend, device="auto", input_chw=None):
            DummyB.calls += 1
            assert backend == "gee"
            assert input_chw is not None
            return Embedding(data=np.array([1.0], dtype=np.float32), meta={})

    registry.register("dummy_a")(DummyA)
    registry.register("dummy_b")(DummyB)

    import rs_embed.api as api

    class DummyProvider:
        def __init__(self, *args, **kwargs):
            pass

        def ensure_ready(self):
            return None

    fetch_calls = {"n": 0}

    def fake_fetch(provider, *, spatial, temporal, sensor):
        fetch_calls["n"] += 1
        return np.ones((3, 4, 4), dtype=np.float32)

    monkeypatch.setattr(api, "GEEProvider", DummyProvider)
    monkeypatch.setattr(api, "_fetch_gee_patch_raw", fake_fetch)
    monkeypatch.setattr(api, "_inspect_input_raw", lambda x_chw, *, sensor, name: {"ok": True})
    api._get_embedder_bundle_cached.cache_clear()

    spatials = [
        PointBuffer(lon=-122.4, lat=37.8, buffer_m=50),
        PointBuffer(lon=-122.3, lat=37.7, buffer_m=50),
    ]
    temporal = TemporalSpec.range("2020-01-01", "2020-02-01")
    sensor = SensorSpec(collection="C", bands=("B1", "B2", "B3"), scale_m=10, cloudy_pct=30, composite="median")

    out_dir = tmp_path / "out"
    res = api.export_batch(
        spatials=spatials,
        temporal=temporal,
        models=["dummy_a", "dummy_b"],
        out_dir=str(out_dir),
        backend="gee",
        device="cpu",
        output=OutputSpec.pooled(),
        sensor=sensor,
        save_inputs=True,
        save_embeddings=True,
        chunk_size=10,
        num_workers=4,
    )

    # One fetch per spatial (dedup across models sharing identical sensor)
    assert fetch_calls["n"] == len(spatials)

    # One embedding per spatial per model
    assert DummyA.calls == len(spatials)
    assert DummyB.calls == len(spatials)

    assert len(res) == len(spatials)
    for i in range(len(spatials)):
        assert (out_dir / f"p{i:05d}.npz").exists()
        assert (out_dir / f"p{i:05d}.json").exists()


def test_export_batch_combined_npz_dedup(tmp_path, monkeypatch):
    class DummyC:
        def describe(self):
            return {
                "type": "onthefly",
                "inputs": {"collection": "C", "bands": ["B1", "B2", "B3"]},
                "defaults": {"scale_m": 10, "cloudy_pct": 30, "composite": "median", "fill_value": 0.0},
            }

        def get_embedding(self, *, spatial, temporal, sensor, output, backend, device="auto", input_chw=None):
            assert input_chw is not None
            return Embedding(data=np.array([1.0], dtype=np.float32), meta={})

    registry.register("dummy_c")(DummyC)

    import rs_embed.api as api

    class DummyProvider:
        def __init__(self, *args, **kwargs):
            pass

        def ensure_ready(self):
            return None

    fetch_calls = {"n": 0}

    def fake_fetch(provider, *, spatial, temporal, sensor):
        fetch_calls["n"] += 1
        return np.zeros((3, 2, 2), dtype=np.float32)

    monkeypatch.setattr(api, "GEEProvider", DummyProvider)
    monkeypatch.setattr(api, "_fetch_gee_patch_raw", fake_fetch)
    monkeypatch.setattr(api, "_inspect_input_raw", lambda x_chw, *, sensor, name: {"ok": True})
    api._get_embedder_bundle_cached.cache_clear()

    spatials = [PointBuffer(lon=0, lat=0, buffer_m=10), PointBuffer(lon=0.1, lat=0.1, buffer_m=10)]
    temporal = TemporalSpec.range("2020-01-01", "2020-02-01")
    sensor = SensorSpec(collection="C", bands=("B1", "B2", "B3"), scale_m=10, cloudy_pct=30, composite="median")

    out_path = tmp_path / "combined.npz"
    api.export_batch(
        spatials=spatials,
        temporal=temporal,
        models=["dummy_c"],
        out_path=str(out_path),
        backend="gee",
        device="cpu",
        output=OutputSpec.pooled(),
        sensor=sensor,
        save_inputs=True,
        save_embeddings=True,
        num_workers=4,
    )

    assert out_path.exists()
    assert fetch_calls["n"] == len(spatials)


def test_export_batch_netcdf_per_item(tmp_path, monkeypatch):
    """export_batch with format='netcdf' writes .nc files with correct variables."""
    class DummyNC:
        def describe(self):
            return {
                "type": "onthefly",
                "inputs": {"collection": "C", "bands": ["B1", "B2"]},
                "defaults": {"scale_m": 10, "cloudy_pct": 30, "composite": "median", "fill_value": 0.0},
            }

        def get_embedding(self, *, spatial, temporal, sensor, output, backend, device="auto", input_chw=None):
            return Embedding(data=np.arange(4, dtype=np.float32), meta={})

    registry.register("dummy_nc")(DummyNC)

    import rs_embed.api as api

    class DummyProvider:
        def __init__(self, *a, **kw): pass
        def ensure_ready(self): return None

    monkeypatch.setattr(api, "GEEProvider", DummyProvider)
    monkeypatch.setattr(api, "_fetch_gee_patch_raw",
                        lambda prov, *, spatial, temporal, sensor: np.ones((2, 4, 4), dtype=np.float32))
    monkeypatch.setattr(api, "_inspect_input_raw", lambda x, *, sensor, name: {"ok": True})
    api._get_embedder_bundle_cached.cache_clear()

    spatials = [PointBuffer(lon=0, lat=0, buffer_m=10), PointBuffer(lon=1, lat=1, buffer_m=10)]
    temporal = TemporalSpec.range("2021-01-01", "2021-06-01")
    sensor = SensorSpec(collection="C", bands=("B1", "B2"))

    out_dir = tmp_path / "nc_out"
    res = api.export_batch(
        spatials=spatials,
        temporal=temporal,
        models=["dummy_nc"],
        out_dir=str(out_dir),
        backend="gee",
        device="cpu",
        output=OutputSpec.pooled(),
        sensor=sensor,
        format="netcdf",
        save_inputs=True,
        save_embeddings=True,
        save_manifest=True,
    )

    assert len(res) == len(spatials)
    for i in range(len(spatials)):
        nc = out_dir / f"p{i:05d}.nc"
        assert nc.exists(), f"Missing {nc}"
        json_f = out_dir / f"p{i:05d}.json"
        assert json_f.exists(), f"Missing {json_f}"

    # Verify NetCDF contents
    import xarray as xr
    ds = xr.open_dataset(str(out_dir / "p00000.nc"))
    assert "embedding__dummy_nc" in ds.data_vars
    assert "input_chw__dummy_nc" in ds.data_vars
    assert tuple(ds["embedding__dummy_nc"].dims) == ("dim",)
    assert tuple(ds["input_chw__dummy_nc"].dims) == ("band", "y", "x")
    ds.close()


def test_export_batch_netcdf_combined(tmp_path, monkeypatch):
    """export_batch with format='netcdf' and out_path produces a combined .nc."""
    class DummyComb:
        def describe(self):
            return {
                "type": "onthefly",
                "inputs": {"collection": "C", "bands": ["B1"]},
                "defaults": {"scale_m": 10, "cloudy_pct": 30, "composite": "median", "fill_value": 0.0},
            }

        def get_embedding(self, *, spatial, temporal, sensor, output, backend, device="auto", input_chw=None):
            return Embedding(data=np.array([1.0, 2.0], dtype=np.float32), meta={})

    registry.register("dummy_comb")(DummyComb)

    import rs_embed.api as api

    class DummyProvider:
        def __init__(self, *a, **kw): pass
        def ensure_ready(self): return None

    monkeypatch.setattr(api, "GEEProvider", DummyProvider)
    monkeypatch.setattr(api, "_fetch_gee_patch_raw",
                        lambda prov, *, spatial, temporal, sensor: np.ones((1, 2, 2), dtype=np.float32))
    monkeypatch.setattr(api, "_inspect_input_raw", lambda x, *, sensor, name: {"ok": True})
    api._get_embedder_bundle_cached.cache_clear()

    spatials = [PointBuffer(lon=0, lat=0, buffer_m=10), PointBuffer(lon=1, lat=1, buffer_m=10)]
    temporal = TemporalSpec.year(2022)
    sensor = SensorSpec(collection="C", bands=("B1",))

    out_path = tmp_path / "combined.nc"
    result = api.export_batch(
        spatials=spatials,
        temporal=temporal,
        models=["dummy_comb"],
        out_path=str(out_path),
        backend="gee",
        device="cpu",
        output=OutputSpec.pooled(),
        sensor=sensor,
        format="netcdf",
        save_inputs=True,
        save_embeddings=True,
    )

    assert out_path.exists()
    assert "nc_path" in result

    import xarray as xr
    ds = xr.open_dataset(str(out_path))
    assert "embeddings__dummy_comb" in ds.data_vars
    assert ds["embeddings__dummy_comb"].shape == (2, 2)  # (point, dim)
    ds.close()
