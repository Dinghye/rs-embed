# API: Inspect

This page documents raw input inspection utilities (patch checks) used before model inference.

Related pages:

- [API: Specs and Data Structures](api_specs.md)
- [API: Embedding](api_embedding.md)
- [API: Export](api_export.md)

---

## inspect_provider_patch (recommended)

```python
inspect_provider_patch(
    *,
    spatial: SpatialSpec,
    temporal: Optional[TemporalSpec] = None,
    sensor: SensorSpec,
    backend: str = "gee",
    name: str = "gee_patch",
    value_range: Optional[Tuple[float, float]] = None,
    return_array: bool = False,
) -> Dict[str, Any]
```

Provider-agnostic patch inspection utility (recommended entry point).
Use this when you want the same inspection flow but with a non-GEE provider backend.

---

## inspect_gee_patch

```python
inspect_gee_patch(
    *,
    spatial: SpatialSpec,
    temporal: Optional[TemporalSpec] = None,
    sensor: SensorSpec,
    backend: str = "gee",
    name: str = "gee_patch",
    value_range: Optional[Tuple[float, float]] = None,
    return_array: bool = False,
) -> Dict[str, Any]
```

Backwards-compatible GEE-focused wrapper around `inspect_provider_patch(...)` (compatibility wrapper).
New code should prefer `inspect_provider_patch(...)` unless you specifically want the older GEE-focused name.
It performs the same input quality checks (**without running the model**).

**Returns**

- A JSON-serializable dict:
  - `ok`: bool
  - `report`: stats/check report
  - `sensor`, `temporal`, `backend`
  - `artifacts`: optional quicklook save paths
  - If `return_array=True`, includes `array_chw` (numpy array, not JSON-serializable)

**Example**

```python
from rs_embed import inspect_gee_patch, PointBuffer, TemporalSpec, SensorSpec

rep = inspect_gee_patch(
    spatial=PointBuffer(121.5, 31.2, 2048),
    temporal=TemporalSpec.range("2022-06-01", "2022-09-01"),
    sensor=SensorSpec(
        collection="COPERNICUS/S2_SR_HARMONIZED",
        bands=("B4", "B3", "B2"),
        scale_m=10,
        cloudy_pct=30,
        composite="median",
        check_input=True,
        check_save_dir="artifacts",
    ),
    return_array=False,
)
```

---
