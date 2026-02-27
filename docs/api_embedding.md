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
    output: OutputSpec = OutputSpec.pooled(),
    backend: str = "gee",
    device: str = "auto",
    input_prep: InputPrepSpec | str = "resize",
) -> Embedding
```

Computes the embedding for a single ROI.

**Parameters**

- `model`: model ID (see the *Supported Models* page, or use `rs_embed.core.registry.list_models()`)
- `spatial`: `BBox` or `PointBuffer`
- `temporal`: `TemporalSpec` or `None`
- `sensor`: input descriptor for on-the-fly models; for most precomputed models this can be `None`
- `output`: `OutputSpec.pooled()` or `OutputSpec.grid(...)`
- `backend`: access backend (mainly for on-the-fly models, typically `"gee"`). For precomputed models, the data `source` is fixed by the model; use `backend="auto"` and let rs-embed resolve the concrete access path.
- `device`: `"auto" / "cpu" / "cuda"` (if the model depends on torch)
- `input_prep`: `"resize"` (default), `"tile"`, `"auto"`, or `InputPrepSpec(...)`



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

!!! tip "Performance tip"
    `get_embedding` tries to reuse a **cached embedder instance** internally to avoid repeatedly initializing the provider / loading model weights (especially for torch models).

---

### get_embeddings_batch

```python
get_embeddings_batch(
    model: str,
    *,
    spatials: List[SpatialSpec],
    temporal: Optional[TemporalSpec] = None,
    sensor: Optional[SensorSpec] = None,
    output: OutputSpec = OutputSpec.pooled(),
    backend: str = "gee",
    device: str = "auto",
    input_prep: InputPrepSpec | str = "resize",
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
