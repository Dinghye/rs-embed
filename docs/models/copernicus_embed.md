# Copernicus Embed (`copernicus_embed`)

> Local precomputed embedding adapter using `torchgeo.datasets.CopernicusEmbed`, with bbox slicing and optional bbox expansion to improve tile overlap on small ROIs.

## Quick Facts

| Field | Value |
|---|---|
| Model ID | `copernicus_embed` |
| Family / Source | TorchGeo `CopernicusEmbed` dataset |
| Adapter type | `precomputed` |
| Typical backend | `local` (or `auto`) |
| Primary input | `BBox` / `PointBuffer` in EPSG:4326, sliced via TorchGeo dataset bbox indexing |
| Temporal mode | **strict** `TemporalSpec.year(2021)` in v0.1 |
| Output modes | `pooled`, `grid` |
| Extra side inputs | none |
| Training alignment (adapter path) | N/A (precomputed product) |

---

## When To Use This Model

### Good fit for

- local/offline precomputed embedding workflows via TorchGeo
- quick spatial baseline features without provider requests
- experiments where coarse precomputed coverage is acceptable

### Be careful when

- requesting years other than `2021` (unsupported in current adapter)
- assuming exact ROI slicing without expansion (adapter expands bbox by default)
- using non-local backends (`copernicus_embed` is local/auto only)

---

## Input Contract (Current Adapter Path)

### Spatial

Accepted `SpatialSpec`:

- `BBox`
- `PointBuffer` (converted to EPSG:4326 bbox)

The adapter internally slices `CopernicusEmbed` with bbox indexing:

- `ds[minlon:maxlon, minlat:maxlat]`

### Temporal

- requires `TemporalSpec.year(...)`
- current adapter supports only `2021`
- adapter validates the year before dataset access

### Backend / data directory

- backend must be `local` or `auto`
- data directory resolution:
  - `RS_EMBED_COP_DIR` (default `data/copernicus_embed`)
  - optional per-call override via `sensor.collection="dir:/path/to/copernicus_embed"`

---

## Retrieval Pipeline (Current rs-embed Path)

1. Validate `TemporalSpec.year(...)` and supported year (`2021`)
2. Resolve local `data_dir` (env or `sensor.collection` override)
3. Load/cache TorchGeo `CopernicusEmbed` dataset (`download=True` in current adapter)
4. Convert `SpatialSpec` to EPSG:4326 bbox
5. Expand bbox by fixed `expand_deg=1.0` (centered) to increase tile overlap chance for small ROIs
6. Slice dataset with bbox indexing and get `sample["image"]` (`CHW`)
7. Return pooled vector or grid

Notes:

- `temporal` is validated but metadata in current adapter is built with `temporal=None`; record the requested year externally if strict provenance matters.

---

## Environment Variables / Tuning Knobs

| Env var | Default | Effect |
|---|---|---|
| `RS_EMBED_COP_DIR` | `data/copernicus_embed` | Local TorchGeo CopernicusEmbed data directory |
| `RS_EMBED_COPERNICUS_BATCH_WORKERS` | `4` | Batch worker count for `get_embeddings_batch(...)` |

Non-env override:

- `sensor.collection="dir:/path/to/copernicus_embed"` overrides data directory per call

Current fixed adapter behavior (not env-configurable in v0.1):

- `download=True`
- `expand_deg=1.0`

---

## Output Semantics

### `OutputSpec.pooled()`

- Pools `CHW` embedding grid over spatial dims:
  - `mean` -> `mean_hw`
  - `max` -> `max_hw`

### `OutputSpec.grid()`

- Returns TorchGeo sample embedding tensor as `xarray.DataArray` `(D,H,W)`
- Grid is precomputed product space (dataset slice), not raw imagery pixels

---

## Examples

### Minimal local example

```python
from rs_embed import get_embedding, PointBuffer, TemporalSpec, OutputSpec

emb = get_embedding(
    "copernicus_embed",
    spatial=PointBuffer(lon=121.5, lat=31.2, buffer_m=5000),
    temporal=TemporalSpec.year(2021),
    output=OutputSpec.pooled(),
    backend="local",
)
```

### Local dataset directory override

```python
# Example (shell):
# export RS_EMBED_COP_DIR=/data/copernicus_embed
```

---

## Common Failure Modes / Debugging

- year not supported (`2021` only in current adapter)
- backend is not `local`/`auto`
- missing `torchgeo` dependency
- dataset files missing/corrupt under `RS_EMBED_COP_DIR`
- small ROI misses coverage even after expansion (returns dataset slicing issues)

Recommended first checks:

- confirm `TemporalSpec.year(2021)`
- inspect metadata `data_dir`, `expand_deg`, `chw_shape`, `bbox_4326`
- test a larger ROI if coverage seems empty

---

## Reproducibility Notes

Keep fixed and record:

- local dataset path/version snapshot
- requested year (must be `2021`)
- ROI geometry
- output mode / pooling choice

---

## Source of Truth (Code Pointers)

- Registration/catalog: `src/rs_embed/embedders/catalog.py`
- Adapter implementation: `src/rs_embed/embedders/precomputed_copernicus_embed.py`

