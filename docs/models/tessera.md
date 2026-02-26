# Tessera (`tessera`)

> Precomputed embedding adapter backed by GeoTessera local tiles, with strict tile mosaic + ROI crop behavior and local-only (`backend="local"`/`"auto"`) access.

## Quick Facts

| Field | Value |
|---|---|
| Model ID | `tessera` |
| Family / Source | GeoTessera local precomputed embeddings |
| Adapter type | `precomputed` |
| Typical backend | `local` (or `auto`) |
| Primary input | `BBox` / `PointBuffer` in EPSG:4326 (converted to bbox) |
| Temporal mode | year-like selection (`year`; `range` uses start year fallback) |
| Output modes | `pooled`, `grid` |
| Extra side inputs | none |
| Training alignment (adapter path) | N/A (precomputed product) |

---

## When To Use This Model

### Good fit for

- fast offline/local baselines using existing GeoTessera cache tiles
- large-area ROI embedding extraction without model inference runtime
- workflows where local tile mosaic/crop behavior is preferable to provider fetches

### Be careful when

- expecting arbitrary backends (`tessera` is local/auto only)
- using `TemporalSpec.range(...)` and assuming exact temporal semantics (adapter picks the start year)
- ROI crosses tiles with inconsistent CRS/resolution (adapter requires strict mosaic compatibility)

---

## Input Contract (Current Adapter Path)

### Spatial

Accepted `SpatialSpec`:

- `BBox` (validated)
- `PointBuffer` (converted to `BBox` in EPSG:4326 using approximate meter-to-degree conversion)

Unsupported spatial types raise `ModelError`.

### Temporal

- `temporal=None` -> defaults to year `2021`
- `TemporalSpec.year(...)` -> uses `temporal.year`
- `TemporalSpec.range(start, end)` -> uses the year parsed from `start`

This is a year selector for tile product lookup, not scene-level temporal filtering.

### Backend / cache

- backend must be `local` or `auto`
- adapter reads GeoTessera cache from:
  - `RS_EMBED_TESSERA_CACHE`, or
  - `sensor.collection="cache:/path/to/cache"` override

---

## Preprocessing / Retrieval Pipeline (Current rs-embed Path)

1. Convert `SpatialSpec` to `BBox` in EPSG:4326
2. Resolve year from `TemporalSpec` (with fallback behavior)
3. Open/cache `geotessera.GeoTessera` instance by `cache_dir`
4. Query tile blocks covering ROI/year
5. Fetch tile embeddings and normalize array layout (`HWC` or `CHW` -> internal `HWC`)
6. Strict mosaic + crop:
   - requires north-up transforms (no rotation/shear)
   - requires consistent tile CRS and resolution across fetched tiles
   - reprojects ROI bbox into tile CRS if needed (requires `pyproj`)
7. Convert cropped result to `CHW`
8. Return pooled vector or grid

---

## Environment Variables / Tuning Knobs

| Env var | Default | Effect |
|---|---|---|
| `RS_EMBED_TESSERA_CACHE` | unset (GeoTessera default) | Local GeoTessera cache directory |
| `RS_EMBED_TESSERA_BATCH_WORKERS` | `4` | Batch worker count for `get_embeddings_batch(...)` |

Non-env override:

- `sensor.collection="cache:/path/to/cache"` overrides cache directory for the call

---

## Output Semantics

### `OutputSpec.pooled()`

- Pools cropped precomputed `CHW` grid over spatial dims:
  - `mean` -> `mean_hw`
  - `max` -> `max_hw`

### `OutputSpec.grid()`

- Returns cropped precomputed embedding grid as `xarray.DataArray` `(D,H,W)`
- Grid is product pixel/grid space from the precomputed tiles (after local mosaic+crop)
- Metadata includes crop/mosaic info (CRS, crop window, transform)

---

## Examples

### Minimal local example

```python
from rs_embed import get_embedding, PointBuffer, TemporalSpec, OutputSpec

emb = get_embedding(
    "tessera",
    spatial=PointBuffer(lon=121.5, lat=31.2, buffer_m=5000),
    temporal=TemporalSpec.year(2021),
    output=OutputSpec.pooled(),
    backend="local",
)
```

### Example cache override

```python
# Example (shell):
# export RS_EMBED_TESSERA_CACHE=/data/geotessera
```

---

## Common Failure Modes / Debugging

- backend is not `local`/`auto`
- no tiles found for ROI/year
- tile CRS/resolution mismatch during mosaic
- tile transform has rotation/shear (not north-up)
- missing optional deps for CRS reprojection (`pyproj`) on non-4326 tiles

Recommended first checks:

- inspect metadata `preferred_year`, `bbox_4326`, `chw_shape`
- inspect crop metadata (`tile_crs`, `mosaic_hw`, `crop_px_window`)
- try a larger ROI if no tiles are found

---

## Reproducibility Notes

Keep fixed and record:

- GeoTessera cache snapshot/path
- year selection logic (`year` vs `range(start, ...)`)
- ROI geometry and CRS
- output mode / pooling choice

---

## Source of Truth (Code Pointers)

- Registration/catalog: `src/rs_embed/embedders/catalog.py`
- Adapter implementation: `src/rs_embed/embedders/precomputed_tessera.py`

