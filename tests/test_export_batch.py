import json
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


def test_export_batch_combined_fail_on_bad_input(tmp_path, monkeypatch):
    class DummyBad:
        def describe(self):
            return {
                "type": "onthefly",
                "inputs": {"collection": "C", "bands": ["B1"]},
                "defaults": {"scale_m": 10, "cloudy_pct": 30, "composite": "median", "fill_value": 0.0},
            }

        def get_embedding(self, *, spatial, temporal, sensor, output, backend, device="auto", input_chw=None):
            return Embedding(data=np.array([1.0], dtype=np.float32), meta={})

    registry.register("dummy_bad")(DummyBad)

    import rs_embed.api as api

    class DummyProvider:
        def __init__(self, *a, **kw):
            pass

        def ensure_ready(self):
            return None

    monkeypatch.setattr(api, "GEEProvider", DummyProvider)
    monkeypatch.setattr(
        api,
        "_fetch_gee_patch_raw",
        lambda prov, *, spatial, temporal, sensor: np.zeros((1, 2, 2), dtype=np.float32),
    )
    monkeypatch.setattr(
        api,
        "_inspect_input_raw",
        lambda x, *, sensor, name: {"ok": False, "report": {"issues": ["all fill"]}},
    )
    api._get_embedder_bundle_cached.cache_clear()

    with pytest.raises(RuntimeError, match="Input inspection failed"):
        api.export_batch(
            spatials=[PointBuffer(lon=0, lat=0, buffer_m=10)],
            temporal=TemporalSpec.year(2022),
            models=["dummy_bad"],
            out_path=str(tmp_path / "bad.npz"),
            backend="gee",
            output=OutputSpec.pooled(),
            sensor=SensorSpec(collection="C", bands=("B1",)),
            save_inputs=True,
            save_embeddings=False,
            fail_on_bad_input=True,
        )


def test_export_batch_prefetch_used_even_without_saving_inputs(tmp_path, monkeypatch):
    class DummyNeedInput:
        def describe(self):
            return {
                "type": "onthefly",
                "inputs": {"collection": "C", "bands": ["B1", "B2"]},
                "defaults": {"scale_m": 10, "cloudy_pct": 30, "composite": "median", "fill_value": 0.0},
            }

        def get_embedding(self, *, spatial, temporal, sensor, output, backend, device="auto", input_chw=None):
            assert input_chw is not None
            return Embedding(data=np.array([float(np.sum(input_chw))], dtype=np.float32), meta={})

    registry.register("dummy_need_input")(DummyNeedInput)

    import rs_embed.api as api

    class DummyProvider:
        def __init__(self, *a, **kw):
            pass

        def ensure_ready(self):
            return None

    calls = {"fetch": 0}

    def fake_fetch(provider, *, spatial, temporal, sensor):
        calls["fetch"] += 1
        return np.ones((2, 3, 3), dtype=np.float32)

    monkeypatch.setattr(api, "GEEProvider", DummyProvider)
    monkeypatch.setattr(api, "_fetch_gee_patch_raw", fake_fetch)
    monkeypatch.setattr(api, "_inspect_input_raw", lambda x, *, sensor, name: {"ok": True})
    api._get_embedder_bundle_cached.cache_clear()

    spatials = [PointBuffer(lon=0, lat=0, buffer_m=10), PointBuffer(lon=1, lat=1, buffer_m=10)]
    sensor = SensorSpec(collection="C", bands=("B1", "B2"))
    out_dir = tmp_path / "out_no_inputs"
    api.export_batch(
        spatials=spatials,
        temporal=TemporalSpec.year(2022),
        models=["dummy_need_input"],
        out_dir=str(out_dir),
        backend="gee",
        output=OutputSpec.pooled(),
        sensor=sensor,
        save_inputs=False,
        save_embeddings=True,
    )

    assert calls["fetch"] == len(spatials)
    assert (out_dir / "p00000.npz").exists()


def test_export_batch_continue_on_error_partial_manifest(tmp_path, monkeypatch):
    class DummyGood:
        def describe(self):
            return {
                "type": "onthefly",
                "inputs": {"collection": "C", "bands": ["B1"]},
                "defaults": {"scale_m": 10, "cloudy_pct": 30, "composite": "median", "fill_value": 0.0},
            }

        def get_embedding(self, *, spatial, temporal, sensor, output, backend, device="auto", input_chw=None):
            return Embedding(data=np.array([1.0], dtype=np.float32), meta={"ok": True})

    class DummyBad:
        def describe(self):
            return {
                "type": "onthefly",
                "inputs": {"collection": "C", "bands": ["B1"]},
                "defaults": {"scale_m": 10, "cloudy_pct": 30, "composite": "median", "fill_value": 0.0},
            }

        def get_embedding(self, *, spatial, temporal, sensor, output, backend, device="auto", input_chw=None):
            raise RuntimeError("boom")

    registry.register("dummy_good")(DummyGood)
    registry.register("dummy_bad2")(DummyBad)

    import rs_embed.api as api

    class DummyProvider:
        def __init__(self, *a, **kw):
            pass

        def ensure_ready(self):
            return None

    monkeypatch.setattr(api, "GEEProvider", DummyProvider)
    monkeypatch.setattr(api, "_fetch_gee_patch_raw", lambda prov, *, spatial, temporal, sensor: np.ones((1, 2, 2), dtype=np.float32))
    monkeypatch.setattr(api, "_inspect_input_raw", lambda x, *, sensor, name: {"ok": True})
    api._get_embedder_bundle_cached.cache_clear()

    out_dir = tmp_path / "partial"
    res = api.export_batch(
        spatials=[PointBuffer(lon=0, lat=0, buffer_m=10)],
        temporal=TemporalSpec.year(2022),
        models=["dummy_good", "dummy_bad2"],
        out_dir=str(out_dir),
        backend="gee",
        output=OutputSpec.pooled(),
        sensor=SensorSpec(collection="C", bands=("B1",)),
        save_inputs=True,
        save_embeddings=True,
        continue_on_error=True,
    )

    assert len(res) == 1
    assert res[0]["status"] == "partial"
    assert any(m["model"] == "dummy_bad2" and m["status"] == "failed" for m in res[0]["models"])
    assert (out_dir / "p00000.npz").exists()


def test_export_batch_combined_prefers_model_batch_api(tmp_path):
    class DummyBatch:
        batch_calls = 0
        single_calls = 0

        def describe(self):
            return {"type": "precomputed", "backend": ["local"], "output": ["pooled"]}

        def get_embedding(self, *, spatial, temporal, sensor, output, backend, device="auto", input_chw=None):
            DummyBatch.single_calls += 1
            raise RuntimeError("single path should not be used")

        def get_embeddings_batch(self, *, spatials, temporal=None, sensor=None, output=OutputSpec.pooled(), backend="local", device="auto"):
            DummyBatch.batch_calls += 1
            return [Embedding(data=np.array([float(i)], dtype=np.float32), meta={}) for i in range(len(spatials))]

    registry.register("dummy_batch")(DummyBatch)

    import rs_embed.api as api
    api._get_embedder_bundle_cached.cache_clear()

    out_path = tmp_path / "combined_batch.npz"
    mani = api.export_batch(
        spatials=[PointBuffer(lon=0, lat=0, buffer_m=10), PointBuffer(lon=1, lat=1, buffer_m=10)],
        temporal=TemporalSpec.year(2022),
        models=["dummy_batch"],
        out_path=str(out_path),
        backend="local",
        output=OutputSpec.pooled(),
        save_inputs=False,
        save_embeddings=True,
    )

    assert out_path.exists()
    assert mani["status"] == "ok"
    assert DummyBatch.batch_calls == 1
    assert DummyBatch.single_calls == 0


def test_export_batch_dedup_inputs_across_models_in_file(tmp_path, monkeypatch):
    class DummyA:
        def describe(self):
            return {
                "type": "onthefly",
                "inputs": {"collection": "C", "bands": ["B1"]},
                "defaults": {"scale_m": 10, "cloudy_pct": 30, "composite": "median", "fill_value": 0.0},
            }

        def get_embedding(self, *, spatial, temporal, sensor, output, backend, device="auto", input_chw=None):
            return Embedding(data=np.array([1.0], dtype=np.float32), meta={})

    class DummyB:
        def describe(self):
            return {
                "type": "onthefly",
                "inputs": {"collection": "C", "bands": ["B1"]},
                "defaults": {"scale_m": 10, "cloudy_pct": 30, "composite": "median", "fill_value": 0.0},
            }

        def get_embedding(self, *, spatial, temporal, sensor, output, backend, device="auto", input_chw=None):
            return Embedding(data=np.array([2.0], dtype=np.float32), meta={})

    registry.register("dummy_dedup_a")(DummyA)
    registry.register("dummy_dedup_b")(DummyB)

    import rs_embed.api as api

    class DummyProvider:
        def __init__(self, *a, **kw):
            pass

        def ensure_ready(self):
            return None

    monkeypatch.setattr(api, "GEEProvider", DummyProvider)
    monkeypatch.setattr(api, "_fetch_gee_patch_raw", lambda prov, *, spatial, temporal, sensor: np.ones((1, 2, 2), dtype=np.float32))
    monkeypatch.setattr(api, "_inspect_input_raw", lambda x, *, sensor, name: {"ok": True})
    api._get_embedder_bundle_cached.cache_clear()

    out_dir = tmp_path / "dedup_inputs"
    api.export_batch(
        spatials=[PointBuffer(lon=0, lat=0, buffer_m=10)],
        temporal=TemporalSpec.year(2022),
        models=["dummy_dedup_a", "dummy_dedup_b"],
        out_dir=str(out_dir),
        backend="gee",
        output=OutputSpec.pooled(),
        sensor=SensorSpec(collection="C", bands=("B1",)),
        save_inputs=True,
        save_embeddings=True,
    )

    npz = np.load(out_dir / "p00000.npz")
    input_keys = [k for k in npz.keys() if k.startswith("input_chw__")]
    assert len(input_keys) == 1

    with open(out_dir / "p00000.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)
    model_entries = {m["model"]: m for m in manifest["models"]}
    assert model_entries["dummy_dedup_b"]["input"].get("dedup_reused") is True


def test_export_batch_resume_out_dir_skips_existing(tmp_path):
    class DummyResumeDir:
        calls = 0

        def describe(self):
            return {"type": "precomputed", "backend": ["local"], "output": ["pooled"]}

        def get_embedding(self, *, spatial, temporal, sensor, output, backend, device="auto", input_chw=None):
            DummyResumeDir.calls += 1
            return Embedding(data=np.array([1.0], dtype=np.float32), meta={})

    registry.register("dummy_resume_dir")(DummyResumeDir)

    import rs_embed.api as api
    api._get_embedder_bundle_cached.cache_clear()

    spatials = [PointBuffer(lon=0, lat=0, buffer_m=10), PointBuffer(lon=1, lat=1, buffer_m=10)]
    out_dir = tmp_path / "resume_dir"

    first = api.export_batch(
        spatials=spatials,
        temporal=TemporalSpec.year(2022),
        models=["dummy_resume_dir"],
        out_dir=str(out_dir),
        backend="local",
        output=OutputSpec.pooled(),
        save_inputs=False,
        save_embeddings=True,
        show_progress=False,
    )
    assert len(first) == len(spatials)
    assert DummyResumeDir.calls == len(spatials)

    second = api.export_batch(
        spatials=spatials,
        temporal=TemporalSpec.year(2022),
        models=["dummy_resume_dir"],
        out_dir=str(out_dir),
        backend="local",
        output=OutputSpec.pooled(),
        save_inputs=False,
        save_embeddings=True,
        resume=True,
        show_progress=False,
    )
    assert len(second) == len(spatials)
    assert DummyResumeDir.calls == len(spatials)
    assert all(bool(m.get("resume_skipped")) for m in second)


def test_export_batch_resume_out_path_skips_existing(tmp_path):
    class DummyResumeCombined:
        calls = 0

        def describe(self):
            return {"type": "precomputed", "backend": ["local"], "output": ["pooled"]}

        def get_embedding(self, *, spatial, temporal, sensor, output, backend, device="auto", input_chw=None):
            DummyResumeCombined.calls += 1
            return Embedding(data=np.array([2.0], dtype=np.float32), meta={})

    registry.register("dummy_resume_combined")(DummyResumeCombined)

    import rs_embed.api as api
    api._get_embedder_bundle_cached.cache_clear()

    out_path = tmp_path / "combined_resume.npz"
    spatials = [PointBuffer(lon=0, lat=0, buffer_m=10), PointBuffer(lon=1, lat=1, buffer_m=10)]

    api.export_batch(
        spatials=spatials,
        temporal=TemporalSpec.year(2022),
        models=["dummy_resume_combined"],
        out_path=str(out_path),
        backend="local",
        output=OutputSpec.pooled(),
        save_inputs=False,
        save_embeddings=True,
        show_progress=False,
    )
    assert DummyResumeCombined.calls == len(spatials)

    skipped = api.export_batch(
        spatials=spatials,
        temporal=TemporalSpec.year(2022),
        models=["dummy_resume_combined"],
        out_path=str(out_path),
        backend="local",
        output=OutputSpec.pooled(),
        save_inputs=False,
        save_embeddings=True,
        resume=True,
        show_progress=False,
    )
    assert bool(skipped.get("resume_skipped"))
    assert DummyResumeCombined.calls == len(spatials)


def test_export_batch_progress_updates_once_per_point(tmp_path, monkeypatch):
    class DummyProgressModel:
        def describe(self):
            return {"type": "precomputed", "backend": ["local"], "output": ["pooled"]}

        def get_embedding(self, *, spatial, temporal, sensor, output, backend, device="auto", input_chw=None):
            return Embedding(data=np.array([3.0], dtype=np.float32), meta={})

    registry.register("dummy_progress")(DummyProgressModel)

    import rs_embed.api as api
    api._get_embedder_bundle_cached.cache_clear()

    state = {"total": None, "updates": 0, "closed": False}

    class _FakeProgress:
        def __init__(self, *, total: int):
            state["total"] = total

        def update(self, n: int = 1):
            state["updates"] += int(n)

        def close(self):
            state["closed"] = True

    monkeypatch.setattr(
        api,
        "_create_progress",
        lambda *, enabled, total, desc, unit="item": _FakeProgress(total=total),
    )

    spatials = [PointBuffer(lon=0, lat=0, buffer_m=10), PointBuffer(lon=1, lat=1, buffer_m=10)]
    api.export_batch(
        spatials=spatials,
        temporal=TemporalSpec.year(2022),
        models=["dummy_progress"],
        out_dir=str(tmp_path / "progress"),
        backend="local",
        output=OutputSpec.pooled(),
        save_inputs=False,
        save_embeddings=True,
        show_progress=True,
    )

    assert state["total"] == len(spatials)
    assert state["updates"] == len(spatials)
    assert state["closed"] is True
