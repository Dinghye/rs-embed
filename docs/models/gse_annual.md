# Google Satellite Embedding Annual (`gse_annual`)

> Provider-backed precomputed annual embedding adapter for `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`, supporting pooled vectors or provider-sampled embedding grids.

## Quick Facts

| Field | Value |
|---|---|
| Model ID | `gse_annual` |
| Family / Source | Google Satellite Embedding annual product (`GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`) |
| Adapter type | `precomputed` |
| Typical backend | provider backend (`gee`) |
| Primary input | Provider-sampled annual embedding image collection |
| Temporal mode | **strict** `TemporalSpec.year(...)` |
| Output modes | `pooled`, `grid` |
| Extra side inputs | none |
| Training alignment (adapter path) | N/A (precomputed product) |

---

## When To Use This Model

### Good fit for

- quick annual baselines from a maintained provider-hosted embedding product
- low-friction comparisons using `OutputSpec.pooled()`
- workflows that want provider-based sampling (no local tile cache management)

### Be careful when

- using `TemporalSpec.range(...)` (not supported in v0.1)
- assuming native imagery semantics (this is an embedding product)
- forgetting that `output.scale_m` affects provider sampling resolution

---

## Input Contract (Current Adapter Path)

### Backend / temporal

- provider backend only (`gee` / provider-compatible; not `auto`)
- requires `TemporalSpec.year(year=...)`
- adapter validates `temporal.mode == "year"`

### Spatial / sampling

- accepts normal `SpatialSpec` provider sampling path
- fetches all embedding bands from annual collection using provider helper
- `OutputSpec.scale_m` controls sampling scale passed to provider

Fixed provider fetch settings in current adapter:

- collection: `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`
- `fill_value=-9999.0`
- `composite="mosaic"`

---

## Retrieval Pipeline (Current rs-embed Path)

1. Validate provider backend + `TemporalSpec.year(...)`
2. Fetch all embedding bands as `CHW` from annual collection
3. Replace fill value `-9999` with `NaN`
4. Build metadata (year, scale, band names)
5. Return:
   - pooled vector via `pool_chw_to_vec(...)`
   - grid as `xarray.DataArray` `(D,H,W)` with `d` coords set to band names

---

## Environment Variables / Tuning Knobs

| Env var | Default | Effect |
|---|---|---|
| `RS_EMBED_GSE_BATCH_WORKERS` | `4` | Batch worker count for `get_embeddings_batch(...)` |

Primary non-env sampling knob:

- `OutputSpec.scale_m` (passed to provider sampling)

---

## Output Semantics

### `OutputSpec.pooled()`

- Pools precomputed embedding grid over spatial dims using `OutputSpec.pooling`
- Metadata records pooling mode (`mean` / `max`)

### `OutputSpec.grid()`

- Returns provider-sampled embedding grid as `xarray.DataArray` `(D,H,W)`
- `d` coordinate uses product band names (not integer indices only)
- Grid is provider-sampled embedding image in product space, not raw imagery pixels

---

## Examples

### Minimal annual example

```python
from rs_embed import get_embedding, PointBuffer, TemporalSpec, OutputSpec

emb = get_embedding(
    "gse_annual",
    spatial=PointBuffer(lon=121.5, lat=31.2, buffer_m=2048),
    temporal=TemporalSpec.year(2021),
    output=OutputSpec.pooled(),
    backend="gee",
)
```

### Sampling resolution example

```python
from rs_embed import OutputSpec

out = OutputSpec.pooled(scale_m=30)
```

---

## Common Failure Modes / Debugging

- backend is not provider-compatible
- missing `TemporalSpec.year(...)`
- `TemporalSpec.range(...)` used instead of `year`
- provider sampling issues / permissions (GEE auth or provider config)
- unexpected NaNs due to fill regions (`-9999` -> `NaN`)

Recommended first checks:

- inspect metadata `year`, `scale_m`, `bands`
- try `OutputSpec.pooled()` first to validate access
- adjust `OutputSpec.scale_m` if sampling is too coarse/fine

---

## Reproducibility Notes

Keep fixed and record:

- `TemporalSpec.year(...)`
- provider backend config / auth context
- `OutputSpec.scale_m`
- output mode and pooling choice

---

## Source of Truth (Code Pointers)

- Registration/catalog: `src/rs_embed/embedders/catalog.py`
- Adapter implementation: `src/rs_embed/embedders/precomputed_gse_annual.py`

