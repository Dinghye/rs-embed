# Core Concepts

This page explains the core mental model of `rs-embed` before you dive into the full API reference.

If you are new to the project, read this page first, then go to [Quick Start](quickstart.md).

---

## The Core Abstraction

`rs-embed` exposes a unified interface:

`(model, spatial, temporal, sensor, output) -> Embedding`

In practice, most users work with:

- `spatial`: where (ROI)
- `temporal`: when (year or time window)
- `output`: what shape you want (`pooled` or `grid`)
- `backend`: where data comes from (usually `gee` for on-the-fly models)

---

## Spatial Specs: What Area You Want

Two common ways to define a region:

- `PointBuffer(lon, lat, buffer_m)`: square ROI centered at a point
- `BBox(minlon, minlat, maxlon, maxlat)`: explicit lat/lon bounds

Current limitation:

- API currently accepts only `crs="EPSG:4326"` for `BBox` / `PointBuffer`

See: [API Reference](api.md) and [Limitations](limitations.md).

---

## Temporal Specs: Window, Not Necessarily a Single Scene

This is the most important concept for readability and correct usage.

### `TemporalSpec.year(...)`

Used mainly for annual precomputed products (for example `gse_annual`).

### `TemporalSpec.range(start, end)`

For most on-the-fly GEE-backed models, this means:

1. Filter imagery within the half-open window `[start, end)`
2. Composite the images (default `median`, optional `mosaic`)
3. Feed the composite patch into the model

It usually does **not** mean "pick a single image acquired exactly on this date."

!!! tip
    If you want a near single-day query, use a one-day window such as
    `TemporalSpec.range("2022-06-01", "2022-06-02")`.

See detailed model-specific temporal behavior in [Supported Models](models.md).

---

## Output Specs: `pooled` vs `grid`

### `OutputSpec.pooled()`

Returns one vector `(D,)` for the whole ROI.

Use this for:

- classification / regression
- similarity search / retrieval
- clustering
- cross-model benchmarking (recommended default)

### `OutputSpec.grid()`

Returns a spatial feature grid `(D, H, W)`.

Use this for:

- visualization (PCA / norm maps)
- patch-wise analysis
- spatial structure inspection

!!! note
    For ViT-like models, `grid` is often a token/patch grid, not guaranteed georeferenced raster pixels.

---

## Backends and Providers

Think of backend as the input retrieval/runtime path.

- `backend="gee"`: fetch imagery from Google Earth Engine (common for on-the-fly models)
- `backend="local"`: local/offline access for precomputed products (common for precomputed models)

You usually do not need to customize providers directly unless you are debugging inputs or extending the library.

---

## `sensor`: Only Needed for Some Paths

For on-the-fly models, `SensorSpec(...)` describes:

- collection
- bands
- scale (meters)
- cloud filtering
- composite mode (`median` / `mosaic`)

For most precomputed models, `sensor` is often `None` or ignored.

---

## Input Prep (`resize` / `tile` / `auto`)

`input_prep` is an API-level policy for large on-the-fly inputs:

- `"resize"`: fast default
- `"tile"`: API-side tiled inference for large ROIs
- `"auto"`: conservative automatic choice (mainly useful for some `grid` outputs)

Use tiling when:

- you care about preserving more spatial detail for large ROIs
- model default resize would be too destructive

See the tiled behavior details in [API Reference](api.md).

---

## Precomputed vs On-the-fly Models

### Precomputed

- Reads embeddings from existing embedding products
- Faster and simpler runtime
- Temporal coverage and resolution are fixed by the product

Examples:

- `tessera`
- `gse_annual`
- `copernicus_embed`

### On-the-fly

- Fetches imagery patch, preprocesses, then runs model inference
- More flexible but heavier dependencies/runtime
- Requires careful attention to bands, temporal windows, and normalization

Examples:

- `remoteclip_s2rgb`
- `prithvi_eo_v2_s2_6b`
- `anysat`
- `terramind`

---

## Reproducibility Checklist (Recommended)

When comparing models, keep these fixed first:

1. Same `spatial` ROI definition
2. Same `temporal` window
3. Same `SensorSpec.composite` policy
4. Same `OutputSpec` mode (usually `pooled`)
5. Default model preprocessing unless you can exactly reproduce training pipelines

The model-specific preprocessing knobs are summarized in [Supported Models](models.md).

---

## Where To Go Next

- New user: [Quick Start](quickstart.md)
- Task-oriented usage: [Common Workflows](workflows.md)
- Model capabilities and preprocessing: [Supported Models](models.md)
- Full signatures and parameters: [API Reference](api.md)
