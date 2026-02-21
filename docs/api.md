
## Imports

```python
from rs_embed import (
    # Specs
    BBox, PointBuffer, TemporalSpec, SensorSpec, OutputSpec,
    # Core APIs
    get_embedding, get_embeddings_batch, export_batch, export_npz,
    # Utilities
    inspect_gee_patch,
)
```

---

## Data Structures

### SpatialSpec

`SpatialSpec` describes the spatial region for which you want to extract an embedding.

#### `BBox`

```python
BBox(minlon: float, minlat: float, maxlon: float, maxlat: float, crs: str = "EPSG:4326")
```

- An **EPSG:4326** lat/lon bounding box (the current version supports only EPSG:4326).
- `validate()` checks that bounds are valid.

#### `PointBuffer`

```python
PointBuffer(lon: float, lat: float, buffer_m: float, crs: str = "EPSG:4326")
```

- A buffer centered at a point, measured in meters (a square ROI; internally projected into the coordinate system required by the provider).
- Requires `buffer_m > 0`.

---

### TemporalSpec

`TemporalSpec` describes the time range (by year or by date range).

```python
TemporalSpec(mode: Literal["year", "range"], year: int | None, start: str | None, end: str | None)
```

Recommended constructors:

```python
TemporalSpec.year(2022)
TemporalSpec.range("2022-06-01", "2022-09-01")
```

Temporal semantics in provider/on-the-fly paths:

- `TemporalSpec.range(start, end)` is interpreted as a half-open window `[start, end)`, where `end` is excluded.
- In GEE-backed on-the-fly fetch, `range` is used to filter an image collection over the full window, then apply a compositing reducer (default `median`, optional `mosaic`).
- So the fetched input is usually a composite over the whole time window, not an automatically selected single-day scene.
- To approximate a single-day query, pass a one-day window such as `TemporalSpec.range("2022-06-01", "2022-06-02")`.

About `input_time` in metadata:

- Many embedders store `meta["input_time"]` as the midpoint date of the temporal window.
- This midpoint is metadata (and for some models, an auxiliary time signal), not evidence that imagery was fetched from exactly that single date.

---

### SensorSpec

`SensorSpec` is mainly for **on-the-fly** models (fetch a patch from GEE online and feed it into the model). It specifies which collection to pull from, which bands, and what resolution/compositing strategy to use.

```python
SensorSpec(
    collection: str,
    bands: Tuple[str, ...],
    scale_m: int = 10,
    cloudy_pct: int = 30,
    fill_value: float = 0.0,
    composite: Literal["median", "mosaic"] = "median",
    check_input: bool = False,
    check_raise: bool = True,
    check_save_dir: Optional[str] = None,
)
```

- `collection`: GEE collection or image ID
- `bands`: band names (tuple)
- `scale_m`: sampling resolution (meters)
- `cloudy_pct`: cloud filter (best-effort; depends on collection properties)
- `fill_value`: no-data fill value
- `composite`: image compositing method over the temporal window (median/mosaic)
- `check_*`: optional input checks and quicklook saving (see `inspect_gee_patch`)

!!! note
    For **precomputed** models (e.g., directly reading offline embedding products), `sensor` is usually ignored or set to `None`.

---

### OutputSpec

`OutputSpec` controls the embedding output shape: a **pooled vector** or a **dense grid**.

```python
OutputSpec(
    mode: Literal["grid", "pooled"],
    scale_m: int = 10,
    pooling: Literal["mean", "max"] = "mean",
    grid_orientation: Literal["north_up", "native"] = "north_up",
)
```

Recommended constructors:

```python
OutputSpec.pooled(pooling="mean")   # shape: (D,)
OutputSpec.grid(scale_m=10)         # shape: (D, H, W), normalized to north-up when possible
OutputSpec.grid(scale_m=10, grid_orientation="native")  # keep model/provider native orientation
```


#### `OutputSpec.pooled()`: ROI-level Vector Embedding

**Semantic meaning**

`pooled` represents one whole ROI (Region of Interest) using a single vector `(D,)`.

Best suited for:

- Classification / regression
- Retrieval / similarity search
- Clustering
- Cross-model comparison (recommended)

Unified output format:

```python
Embedding.data.shape == (D,)
```

How it is produced:

ViT / MAE-style models (e.g., RemoteCLIP / Prithvi / SatMAE / ScaleMAE):

- Native output is patch tokens `(N, D)` (with optional CLS token)
- Remove CLS token if present, then pool tokens across the token axis (`mean` by default, optional `max`)

Mean-pooling formula:

$$
v_d = \frac{1}{N'} \sum_{i=1}^{N'} t_{i,d}
$$

Precomputed embeddings (e.g., Tessera / GSE / Copernicus):

- Native output is an embedding grid `(D, H, W)`
- Pool over spatial dimensions `(H, W)`

$$
v_d = \frac{1}{HW} \sum_{y,x} g_{d,y,x}
$$

Why prefer `pooled` for benchmarks:

- Model-agnostic and stable
- Less sensitive to spatial/token layout differences
- Easiest output to compare across models

#### `OutputSpec.grid()`: ROI-level Spatial Embedding Field

**Semantic meaning**

`grid` returns a spatial embedding field `(D, H, W)`, where each spatial location maps to a vector.

Best suited for:

- Spatial visualization (PCA / norm / similarity maps)
- Pixel-wise / patch-wise tasks
- Intra-ROI structure analysis

Unified output format:

```python
Embedding.data.shape == (D, H, W)
```

Notes:

- `data` can be returned as `xarray.DataArray` with metadata in `meta`/`attrs`
- For precomputed geospatial products, metadata may include CRS/crop context
- For ViT token grids, this is usually patch-grid metadata (not georeferenced pixel coordinates)

How it is produced:

ViT / MAE-style models:

- Native output: tokens `(N, D)`
- Remove CLS token if present, reshape remaining tokens:
- `(N', D) -> (H, W, D) -> (D, H, W)`
- `(H, W)` comes from patch layout (for example, `8x8`, `14x14`)

Precomputed embeddings:

- Native output is already `(D, H, W)`

---

### Embedding

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

## Core Functions

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
) -> Embedding
```

Computes the embedding for a single ROI.

**Parameters**

- `model`: model ID (see the *Supported Models* page, or use `rs_embed.core.registry.list_models()`)
- `spatial`: `BBox` or `PointBuffer`
- `temporal`: `TemporalSpec` or `None`
- `sensor`: input descriptor for on-the-fly models; for most precomputed models this can be `None`
- `output`: `OutputSpec.pooled()` or `OutputSpec.grid(...)`
- `backend`: currently mainly `"gee"` (Google Earth Engine)
- `device`: `"auto" / "cpu" / "cuda"` (if the model depends on torch)

**Returns**

- `Embedding`

**Example**

```python
from rs_embed import PointBuffer, TemporalSpec, OutputSpec, get_embedding

emb = get_embedding(
    "remoteclip_s2rgb",
    spatial=PointBuffer(lon=121.5, lat=31.2, buffer_m=2048),
    temporal=TemporalSpec.range("2022-06-01", "2022-09-01"),
    output=OutputSpec.pooled(pooling="mean"),
    backend="gee",
    device="auto",
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
    "remoteclip_s2rgb",
    spatials=spatials,
    temporal=TemporalSpec.range("2022-06-01", "2022-09-01"),
)
```

---

### export_batch (core)

```python
export_batch(
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
) -> Any
```

**Recommended batch export entry point**: export `inputs + embeddings + manifest` for **multiple ROIs × multiple models** in one go.

- `out_dir` mode: one file per point (recommended for massive numbers of points)
- `out_path` mode: merge into a single output file (good for fewer points and portability)

**Parameters**

- `spatials`: non-empty list
- `temporal`: can be `None` (some models don’t require time)
- `models`: non-empty list of model IDs
- `out_dir` / `out_path`: choose one
- `names`: used only in `out_dir` mode, for output filenames (length must equal `spatials`)
- `sensor`: a shared `SensorSpec` for all models (if models are on-the-fly)
- `per_model_sensors`: override `SensorSpec` per model; keys are model strings
- `format`: currently only `"npz"` (may be extended in the future)
- `save_inputs`: whether to save model input patches (CHW numpy)
- `save_embeddings`: whether to save embedding arrays
- `save_manifest`: whether to save JSON manifests (each export artifact will have an accompanying `.json`)
- `fail_on_bad_input`: whether to raise immediately if input checks fail
- `chunk_size`: process points in chunks (controls memory/throughput)
- `num_workers`: concurrency for GEE patch prefetching (ThreadPool)
- `continue_on_error`: keep exporting remaining points/models even if one item fails
- `max_retries`: retry count for provider fetch/write operations
- `retry_backoff_s`: sleep seconds between retries
- `async_write`: write output files asynchronously in `out_dir` mode
- `writer_workers`: writer thread count when `async_write=True`
- `resume`: skip already-exported outputs and continue from remaining items
- `show_progress`: show progress during batch export (overall progress + per-model inference progress)

**Returns**

- `out_dir` mode: `List[dict]` (manifest for each point)
- `out_path` mode: `dict` (combined manifest)

**Example: out_dir (recommended)**

```python
from rs_embed import export_batch, PointBuffer, TemporalSpec

spatials = [
    PointBuffer(121.5, 31.2, 2048),
    PointBuffer(120.5, 30.2, 2048),
]
export_batch(
    spatials=spatials,
    temporal=TemporalSpec.range("2022-06-01", "2022-09-01"),
    models=["remoteclip_s2rgb", "prithvi_eo_v2_s2_6b"],
    out_dir="exports",
    names=["p1", "p2"],
    save_inputs=True,
    save_embeddings=True,
    chunk_size=32,
    num_workers=8,
)
```

**Example: out_path (single merged file)**

```python
from rs_embed import export_batch, PointBuffer, TemporalSpec

export_batch(
    spatials=[PointBuffer(121.5, 31.2, 2048)],
    temporal=TemporalSpec.range("2022-06-01", "2022-09-01"),
    models=["remoteclip_s2rgb"],
    out_path="combined.npz",
)
```

!!! tip "Key performance feature: avoid duplicate downloads"
    When `backend="gee"` and `save_inputs=True` and `save_embeddings=True`, `export_batch` **prefetches the raw patch once**,
    and passes that same patch into the embedder via `input_chw` to compute embeddings—avoiding the pattern of “download once to save inputs + download again for embeddings”.

!!! warning "About parallelism"
    By default, only **GEE prefetching** is parallelized (network-IO friendly). Inference is run serially by default to avoid GPU/model thread-safety issues (but model instances are reused and not repeatedly loaded).
    If you need faster inference later, you can implement true batched forward for specific torch models (override `get_embeddings_batch`).

---

### inspect_gee_patch

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

Downloads a GEE patch and performs input quality checks (**without running the model**).

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

## Model Registry (Advanced)

If you need a stable model list in code, use the model catalog:

```python
from rs_embed.embedders.catalog import MODEL_SPECS
print(sorted(MODEL_SPECS.keys()))
```

`list_models()` from `rs_embed.core.registry` only reports models currently loaded into the runtime registry.

---

## Errors

rs-embed raises several explicit exception types (all in `rs_embed.core.errors`):

- `SpecError`: spec validation failure (invalid bbox, missing temporal fields, etc.)
- `ProviderError`: provider/backend errors (e.g., GEE initialization or fetch failure)
- `ModelError`: unknown model ID, unsupported parameters, unsupported export format, etc.

---

## Optional Dependencies

Different features require different optional dependencies:

- `pip install "rs-embed[gee]"`: use the Earth Engine backend
- `pip install "rs-embed[torch]"`: torch model inference
- `pip install "rs-embed[models]"`: dependencies for some model wrappers (e.g., rshf)
- `pip install "rs-embed[dev]"`: dev dependencies such as pytest

---

## Versioning Notes

The current version is still early stage (`0.1.x`):

- `BBox/PointBuffer` currently require `crs="EPSG:4326"`
- Precomputed models mainly use `backend="local"`; on-the-fly models mainly use provider backends (typically `"gee"`)
- `export_batch(format=...)` currently implements only `"npz"`; it may be extended to parquet/zarr/hdf5, etc.
