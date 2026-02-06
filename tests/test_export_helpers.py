import numpy as np
import xarray as xr

from rs_embed.core import registry
from rs_embed.core.specs import BBox, SensorSpec
from rs_embed.export import _sanitize_key, _sha1, _jsonable, _embedding_to_numpy, _default_sensor_for_model


def test_sanitize_key():
    assert _sanitize_key("foo/bar baz") == "foo_bar_baz"
    assert _sanitize_key("___") == "item"


def test_sha1_deterministic():
    arr = np.arange(10, dtype=np.int32)
    h1 = _sha1(arr)
    h2 = _sha1(arr.copy())
    assert h1 == h2
    assert len(h1) == 40


def test_jsonable_numpy_and_dataclass():
    arr = np.array([1.0, 2.0], dtype=np.float32)
    out = _jsonable(arr)
    assert out.get("__ndarray__") is True
    assert out.get("shape") == [2]
    bbox = BBox(minlon=0.0, minlat=0.0, maxlon=1.0, maxlat=1.0)
    out_bbox = _jsonable(bbox)
    assert out_bbox["minlon"] == 0.0
    assert out_bbox["maxlat"] == 1.0


def test_embedding_to_numpy_xarray():
    da = xr.DataArray(np.array([[1.0, 2.0]], dtype=np.float64))
    out = _embedding_to_numpy(da)
    assert out.dtype == np.float32
    assert out.shape == (1, 2)


def test_default_sensor_for_model_precomputed():
    registry._REGISTRY.clear()

    @registry.register("precomputed_test")
    class DummyPrecomputed:
        def describe(self):
            return {"type": "precomputed"}

    assert _default_sensor_for_model("precomputed_test") is None
    registry._REGISTRY.clear()


def test_default_sensor_for_model_inputs_dict():
    registry._REGISTRY.clear()

    @registry.register("onthefly_test")
    class DummyOnTheFly:
        def describe(self):
            return {
                "type": "onthefly",
                "inputs": {"collection": "COLL", "bands": ["B1", "B2"]},
                "defaults": {
                    "scale_m": 20,
                    "cloudy_pct": 5,
                    "composite": "mosaic",
                    "fill_value": 1.0,
                },
            }

    sensor = _default_sensor_for_model("onthefly_test")
    assert isinstance(sensor, SensorSpec)
    assert sensor.collection == "COLL"
    assert sensor.bands == ("B1", "B2")
    assert sensor.scale_m == 20
    assert sensor.cloudy_pct == 5
    assert sensor.composite == "mosaic"
    assert sensor.fill_value == 1.0
    registry._REGISTRY.clear()
