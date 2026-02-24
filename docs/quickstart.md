## Install (temporary)

```bash
git clone https://github.com/Dinghye/rs-embed.git
# or: git clone git@github.com:Dinghye/rs-embed.git
cd rs-embed
conda env create -f environment.yml
conda activate rs-embed
pip install -e .
```

For on-the-fly model demos (GEE + torch wrappers), install optional dependencies if needed:

```bash
pip install -e ".[gee,torch,models]"
```

Examples notebook: examples/playground.ipynb


## Authenticate Google Earth Engine

If you are using GEE for the first time, you need to complete the authentication process by using the following command.

```bash
earthengine authenticate
```

## Run `examples/quickstart.py`

You can run the packaged quickstart script directly:

```bash
# show CLI options
python examples/quickstart.py --help
```

### Local mode (default, precomputed)

Runs `tessera` examples for:
- single embedding (`pooled` + `grid`)
- batch embeddings (`get_embeddings_batch`)
- optional export (`export_batch`)

```bash
python examples/quickstart.py --mode local
python examples/quickstart.py --mode local --run-export
```

### GEE mode (on-the-fly)

Runs `remoteclip_s2rgb` examples for:
- `inspect_gee_patch`
- single embedding
- batch embeddings
- optional export

```bash
python examples/quickstart.py --mode gee --device auto
python examples/quickstart.py --mode gee --run-export --out-dir examples/_outputs/quickstart
```

### Run all demos

```bash
python examples/quickstart.py --mode all
python examples/quickstart.py --mode all --run-export
```

!!! tip
    If you see `ModuleNotFoundError: No module named 'rs_embed'`, run from repository root after installation:
    `pip install -e .`


### 1. Compute a single embedding

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

vec = emb.data  # shape: (D,)
meta = emb.meta
```

!!! note
    `TemporalSpec.range(start, end)` is treated as a temporal window (half-open: `[start, end)`).
    On GEE-backed on-the-fly paths, inputs are typically composites over that window (`median` by default), not an auto-selected single-day scene.

### 2. Batch compute embeddings for many points

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
    backend="gee",
    device="auto",
)
```

### 3. Export at scale (recommended workflow)

`export_batch` is the **core** export API. It supports:

- arbitrary point / ROI lists
- multiple models per ROI
- saving inputs and embeddings
- manifests for downstream bookkeeping

```python
from rs_embed import export_batch, PointBuffer, TemporalSpec

spatials = [
    PointBuffer(121.5, 31.2, 2048),
    PointBuffer(120.5, 30.2, 2048),
]

export_batch(
    spatials=spatials,
    names=["p1", "p2"],
    temporal=TemporalSpec.range("2022-06-01", "2022-09-01"),
    models=["remoteclip_s2rgb", "prithvi_eo_v2_s2_6b"],
    out_dir="exports",
    backend="gee",
    device="auto",
    save_inputs=True,
    save_embeddings=True,
    chunk_size=32,
    num_workers=8,
    resume=True,
    show_progress=True,
)
```

## Working with Providers / Backends

rs-embed supports pluggable backends. In most setups:

- `backend="gee"` uses **Google Earth Engine** for patch retrieval and preprocessing (best for on-the-fly models).

!!! tip
    For large exports, tune:
    - `chunk_size`: how many ROIs per chunk (controls memory peak)
    - `num_workers`: how many concurrent fetch workers (controls IO parallelism)
    - `resume=True`: skip files already exported in previous runs


## Export Formats

`export_batch(format=...)` is designed to be extensible.

- Current formats: `npz`, `netcdf`
- Planned: parquet / zarr / hdf5 (depending on your roadmap)

`export_npz(...)` is provided as a convenience wrapper for single-ROI exports and shares the same performance optimizations.


## Performance Notes

### 1. Avoid repeated model initialization
rs-embed caches embedder instances internally (per `model + backend + device + sensor`), so repeated calls do not re-initialize providers or reload weights.

### 2. Avoid repeated input downloads
When you use:

- `backend="gee"`
- `save_inputs=True`
- `save_embeddings=True`

`export_batch` will **prefetch each input patch once** and reuse it for both:
- saving the input patch
- computing embeddings (via `input_chw`)

### 3. IO parallelism vs inference safety
`export_batch` currently uses two-level scheduling:
- **IO level**: remote patch prefetch is parallelized (`num_workers`).
- **Inference level**:
  - model-to-model execution is serial (stability-first default),
  - but each model can use batched inference over many points when batch APIs are available (such as `get_embeddings_batch` / `get_embeddings_batch_from_inputs`): in combined mode by default, and in per-item mode when running on GPU/accelerators.

So rs-embed supports batch-level inference acceleration, while model-level scheduling remains serial by design.
