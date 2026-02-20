## Current Limitations (v0.1.x)

This page summarizes user-facing limitations in the current implementation.

### 1) Spatial input CRS

- `BBox` and `PointBuffer` currently require `crs="EPSG:4326"`.
- Other CRS inputs are not accepted at the API boundary.

### 2) Temporal behavior for on-the-fly models

- Most on-the-fly adapters treat `TemporalSpec.range(start, end)` as a window filter and build one composite image (`median` by default).
- This is not an automatic single-scene selection by acquisition date.

### 3) Temporal constraints are model-specific

- `gse_annual` currently requires `TemporalSpec.year(...)`.
- `copernicus_embed` currently supports only year `2021`.
- Some precomputed models may ignore finer temporal granularity.

### 4) Grid output is not always georeferenced raster space

- For ViT-like models, `OutputSpec.grid()` is often a token/patch grid `(D, H, W)`.
- Treat it as model-internal spatial structure, not as guaranteed geo-aligned pixel coordinates.

### 5) Export format support

- `export_batch(format=...)` currently supports only `npz`.

### 6) Optional dependency surface is large

- Different models/providers require extra packages (`earthengine-api`, `torch`, `rshf`, `torchgeo`, etc.).
- Missing optional dependencies will fail at runtime for the corresponding model path.

### 7) Known edge case: tessera tile boundary mosaic

- Near some UTM-zone boundaries, fetched tiles may have different CRS/resolution and strict mosaic can fail.
- If this occurs, try shifting ROI slightly or using a smaller ROI/window.

## Practical Workarounds

- Use `TemporalSpec.range("YYYY-MM-DD", "YYYY-MM-DD+1")` when you want a near single-day query.
- Prefer `OutputSpec.pooled()` for cross-model comparisons.
- Use `inspect_gee_patch(...)` before large exports to catch invalid inputs early.
- For robust long exports, use `resume=True` and tune `chunk_size` / `num_workers`.
