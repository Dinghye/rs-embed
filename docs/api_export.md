# API: Export

This page documents dataset export APIs and export-specific runtime behavior.

Related pages:

- [API: Specs and Data Structures](api_specs.md)
- [API: Embedding](api_embedding.md)
- [API: Inspect](api_inspect.md)

---

## export_batch (core)

```python
export_batch(
    *,
    spatials: List[SpatialSpec],
    temporal: Optional[TemporalSpec],
    models: List[str],
    out: Optional[str] = None,
    layout: Optional[str] = None,
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
    input_prep: InputPrepSpec | str = "resize",
) -> Any
```

**Recommended batch export entry point**: export `inputs + embeddings + manifest` for **multiple ROIs × multiple models** in one go.

- `out_dir` mode: one file per point (recommended for massive numbers of points)
- `out_path` mode: merge into a single output file (good for fewer points and portability)
- Decoupled output target API: `out + layout` (maps to the same two modes)

**Parameters**

- `spatials`: non-empty list
- `temporal`: can be `None` (some models don’t require time)
- `models`: non-empty list of model IDs
- Output target:
  - legacy API: choose one of `out_dir` / `out_path`
  - decoupled API: provide both `out` and `layout` (`"per_item"` or `"combined"`)
  - do not mix `out+layout` with `out_dir/out_path`
- `names`: used only in `out_dir` mode, for output filenames (length must equal `spatials`)
- `sensor`: a shared `SensorSpec` for all models (if models are on-the-fly)
- `per_model_sensors`: override `SensorSpec` per model; keys are model strings
- `format`: `"npz"` or `"netcdf"`
- `save_inputs`: whether to save model input patches (CHW numpy)
- `save_embeddings`: whether to save embedding arrays
- `save_manifest`: whether to save JSON manifests (each export artifact will have an accompanying `.json`)
- `fail_on_bad_input`: whether to raise immediately if input checks fail
- `chunk_size`: process points in chunks (controls memory/throughput)
  - also used as the default inference batch size when batched inference is enabled
  - in `per_item` mode with GEE prefetch enabled, rs-embed uses a one-slot prefetch pipeline (double buffering), so input-cache peak memory can be roughly up to 2 chunks (to overlap `prefetch(chunk k+1)` with `infer/write(chunk k)`)
- `num_workers`: concurrency for GEE patch prefetching (ThreadPool)
- `continue_on_error`: keep exporting remaining points/models even if one item fails
- `max_retries`: retry count for provider fetch/write operations
- `retry_backoff_s`: sleep seconds between retries
- `async_write`: write output files asynchronously in `out_dir` mode
- `writer_workers`: writer thread count when `async_write=True`
- `resume`: skip already-exported outputs and continue from remaining items
- `show_progress`: show progress during batch export (overall progress + per-model inference progress)
- `input_prep`: large-ROI input policy (`"resize"` default, `"tile"`, `"auto"`, or `InputPrepSpec(...)`)

**Automatic inference behavior**

- In **per-item output mode** (`out_dir` / `layout="per_item"`):
  - `device="cpu"` (or auto-resolved CPU): defaults to per-item inference
  - `device="cuda"` / `mps` / other accelerators (or auto-resolved GPU): prefers batched inference when the embedder implements batch APIs
- In **combined output mode** (`out_path` / `layout="combined"`), rs-embed keeps the historical behavior of attempting batched model APIs (with fallback to single-item inference if batched execution fails)
- Model-level scheduling remains serial (one model at a time)

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
    input_prep="tile",  # optional: API-side tiled inference for large ROIs
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

**Example: decoupled output target (`out + layout`)**

```python
from rs_embed import export_batch, PointBuffer, TemporalSpec

export_batch(
    spatials=[PointBuffer(121.5, 31.2, 2048)],
    temporal=TemporalSpec.range("2022-06-01", "2022-09-01"),
    models=["remoteclip_s2rgb"],
    out="exports/combined_run",
    layout="combined",  # writes exports/combined_run.npz
)
```

!!! tip "Key performance feature: avoid duplicate downloads"
    When `backend="gee"` and `save_inputs=True` and `save_embeddings=True`, `export_batch` **prefetches the raw patch once**,
    and passes that same patch into the embedder via `input_chw` to compute embeddings—avoiding the pattern of “download once to save inputs + download again for embeddings”.

!!! warning "About parallelism"
    `export_batch` currently has two levels of execution behavior:
    - **IO level**: GEE prefetching is parallelized (ThreadPool, controlled by `num_workers`).
      - In **per-item mode**, rs-embed uses a **one-slot double buffer**: while chunk `k` is running inference / writing outputs, chunk `k+1` can be prefetched in the background (after the first chunk).
      - This improves throughput when fetch and inference times are comparable, at the cost of a higher input-cache peak (roughly up to 2 chunks).
      - In **combined mode**, prefetch still runs as a distinct stage before model execution (to keep checkpoint/resume semantics simpler and memory behavior predictable).
    - **Inference level**:
      - **model level**: models are executed serially (one model at a time), to keep runtime/GPU behavior stable.
      - **batch level**: for a single model, rs-embed can run batched inference when the embedder implements batch APIs (for example `get_embeddings_batch` / `get_embeddings_batch_from_inputs`). This is used in combined mode and, on GPU/accelerators, also in per-item mode.
    In short: rs-embed supports batch-level inference acceleration, while keeping model-level scheduling serial by design.

---

## export_npz (single-ROI convenience wrapper)

```python
export_npz(
    *,
    spatial: SpatialSpec,
    temporal: Optional[TemporalSpec],
    models: List[str],
    out_path: str,
    backend: str = "gee",
    device: str = "auto",
    output: OutputSpec = OutputSpec.pooled(),
    sensor: Optional[SensorSpec] = None,
    per_model_sensors: Optional[Dict[str, SensorSpec]] = None,
    save_inputs: bool = True,
    save_embeddings: bool = True,
    save_manifest: bool = True,
    fail_on_bad_input: bool = False,
    continue_on_error: bool = False,
    max_retries: int = 0,
    retry_backoff_s: float = 0.0,
    input_prep: InputPrepSpec | str = "resize",
) -> Dict[str, Any]
```

Convenience wrapper around `export_batch(...)` for a single `spatial` query that always writes a single `.npz` file.

- Creates parent directory if needed
- Appends `.npz` suffix if missing
- Delegates to `export_batch(..., out_path=..., format="npz")`

Use `export_batch(...)` directly when you need:

- multiple spatials
- non-`npz` formats (for example `netcdf`)
- output layout control (`out + layout` / `out_dir` / `out_path`)
