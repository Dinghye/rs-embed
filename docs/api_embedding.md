# API: Embedding

This page covers single-ROI and batch embedding APIs.

Model IDs in examples use canonical short names (for example `remoteclip`). Legacy IDs (for example `remoteclip_s2rgb`) remain supported as aliases.

Related pages:

- [API: Specs and Data Structures](api_specs.md)
- [API: Export](api_export.md)
- [API: Inspect](api_inspect.md)

---

## Embedding Data Structure

`get_embedding` / `get_embeddings_batch` return an `Embedding`:

```python
from rs_embed.core.embedding import Embedding

Embedding(
    data: np.ndarray | xarray.DataArray,
    meta: Dict[str, Any],
)
```

- `data`: the embedding data (float32, vector or grid)
- `meta`: includes model info, input info (optional), and export/check reports, etc.

---

## Embedding Functions

### get_embedding

```python
get_embedding(
    model: str,
    *,
    spatial: SpatialSpec,
    temporal: Optional[TemporalSpec] = None,
    sensor: Optional[SensorSpec] = None,
    modality: Optional[str] = None,
    output: OutputSpec = OutputSpec.pooled(),
    backend: str = "auto",
    device: str = "auto",
    input_prep: InputPrepSpec | str | None = "resize",
) -> Embedding
```

Computes the embedding for a single ROI.

**Parameters**

- `model`: model ID (see the *Supported Models* page, or use `rs_embed.list_models()`)
- `spatial`: `BBox` or `PointBuffer`
- `temporal`: `TemporalSpec` or `None`
- `sensor`: input descriptor for on-the-fly models; for most precomputed models this can be `None`
- `modality`: optional model-facing modality selector (for example `s1`, `s2`, `s2_l2a`) for models that expose multiple input branches
- `output`: `OutputSpec.pooled()` or `OutputSpec.grid(...)`
- `backend`: access backend. `backend="auto"` is the public default and the recommended choice. For provider-backed on-the-fly models it resolves to a compatible provider backend; for precomputed models it lets rs-embed choose the model-compatible access path.
- `device`: `"auto" / "cpu" / "cuda"` (if the model depends on torch)
- `input_prep`: `"resize"` (default), `"tile"`, `"auto"`, or `InputPrepSpec(...)`

Modality contract:

- All public embedding APIs accept `modality`.
- Only models that explicitly expose a given modality can use it.
- Unsupported modality selections raise a `ModelError`.

**Returns**

- `Embedding`

**Example**

```python
from rs_embed import PointBuffer, TemporalSpec, OutputSpec, get_embedding

emb = get_embedding(
    "remoteclip",
    spatial=PointBuffer(lon=121.5, lat=31.2, buffer_m=2048),
    temporal=TemporalSpec.range("2022-06-01", "2022-09-01"),
    output=OutputSpec.pooled(pooling="mean"),
    backend="gee",
    device="auto",
    input_prep="resize",  # default
)
vec = emb.data  # (D,)
```

---

### get_embeddings_batch

```python
get_embeddings_batch(
    model: str,
    *,
    spatials: List[SpatialSpec],
    temporal: Optional[TemporalSpec] = None,
    sensor: Optional[SensorSpec] = None,
    modality: Optional[str] = None,
    output: OutputSpec = OutputSpec.pooled(),
    backend: str = "auto",
    device: str = "auto",
    input_prep: InputPrepSpec | str | None = "resize",
) -> List[Embedding]
```

Batch-computes embeddings for multiple ROIs using the same embedder instance (often more efficient than looping over `get_embedding`).

**Parameters**

- `spatials`: a non-empty `List[SpatialSpec]`
- Others are the same as `get_embedding`

**Returns**

- `List[Embedding]` (same length as `spatials`)

**Example**

```python
from rs_embed import PointBuffer, TemporalSpec, get_embeddings_batch

spatials = [
    PointBuffer(121.5, 31.2, 2048),
    PointBuffer(120.5, 30.2, 2048),
]
embs = get_embeddings_batch(
    "remoteclip",
    spatials=spatials,
    temporal=TemporalSpec.range("2022-06-01", "2022-09-01"),
)
```

---

## Model (class-based API) { #model }

`Model` is the stateful, class-based alternative to the function API.
Set it up once — model loading and backend resolution happen in `__init__` — then call
`get_embedding` / `get_embeddings_batch` as many times as you need without repeating
configuration on every call.

```python
Model(
    name: str,
    *,
    backend: str = "auto",
    device: str = "auto",
    sensor: Optional[SensorSpec] = None,
    modality: Optional[str] = None,
    output: OutputSpec = OutputSpec.pooled(),
    input_prep: InputPrepSpec | str | None = "resize",
)
```

**Parameters** — same semantics as `get_embedding(...)`:

- `name`: model ID (e.g. `"remoteclip"`, `"prithvi"`)
- `backend`: `"auto"` / `"gee"` / provider name
- `device`: `"auto"` / `"cpu"` / `"cuda"`
- `sensor`: `SensorSpec` override for on-the-fly models; `None` for precomputed
- `modality`: optional modality selector for models with multiple input branches
- `output`: `OutputSpec.pooled()` (default) or `OutputSpec.grid(...)`
- `input_prep`: `"resize"` (default), `"tile"`, `"auto"`, `InputPrepSpec(...)`, or `None`

**Example**

```python
from rs_embed import Model, PointBuffer, TemporalSpec

model = Model("remoteclip", backend="gee", device="auto")

spatials = [
    PointBuffer(121.5, 31.2, 2048),
    PointBuffer(120.5, 30.2, 2048),
]
temporal = TemporalSpec.range("2022-06-01", "2022-09-01")

# embed once
emb = model.get_embedding(spatials[0], temporal=temporal)

# embed many — embedder instance is reused
embs = model.get_embeddings_batch(spatials, temporal=temporal)
```

---

### `Model.get_embedding`

```python
model.get_embedding(
    spatial: SpatialSpec,
    *,
    temporal: Optional[TemporalSpec] = None,
) -> Embedding
```

Compute one embedding using this model instance.

---

### `Model.get_embeddings_batch`

```python
model.get_embeddings_batch(
    spatials: list[SpatialSpec],
    *,
    temporal: Optional[TemporalSpec] = None,
) -> list[Embedding]
```

Compute embeddings for multiple spatial locations. Same order as `spatials`.

Raises `ModelError` if `spatials` is empty or not a list.

---

### `Model.describe`

```python
model.describe() -> dict[str, Any]
```

Return capability metadata from the underlying embedder (e.g. supported backends,
default image size, output dimension). Returns `{}` if unavailable.

---

### `Model.list_models` (static)

```python
Model.list_models(*, include_aliases: bool = False) -> list[str]
```

Return the stable public model catalog. Equivalent to `rs_embed.list_models()`.

---
