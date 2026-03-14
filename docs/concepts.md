# Core Concepts

This page explains the semantic meaning of the main rs-embed abstractions.
It is a supplement to [Quickstart](quickstart.md) and [API: Specs and Data Structures](api_specs.md), not a separate getting-started path.

---

## The Core Abstraction

`rs-embed` exposes a unified interface:

`(model, spatial, temporal, sensor, output) -> Embedding`

In practice, most users work with:

- `spatial`: where (ROI)
- `temporal`: when (year or time window)
- `output`: what shape you want (`pooled` or `grid`)
- `backend`: data access route (`auto` recommended; `gee` is a common explicit provider override)

This page explains what those words mean in practice.

---

## Spatial Specs: What Area You Want

Two common ways to define a region:

- `PointBuffer(lon, lat, buffer_m)`: square ROI centered at a point
- `BBox(minlon, minlat, maxlon, maxlat)`: explicit lat/lon bounds

Current limitation:

- API currently accepts only `crs="EPSG:4326"` for `BBox` / `PointBuffer`

For exact constructors and validation rules, see [API: Specs and Data Structures](api_specs.md).

---

## Temporal Specs: Window, Not Necessarily a Single Scene

This is usually the most important semantic distinction to get right.

### `TemporalSpec.year(...)`

Used mainly for annual precomputed products (for example `gse`).

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

Interpretation:

- one ROI in, one embedding vector out
- easiest output to compare across different model families

### `OutputSpec.grid()`

Returns a spatial feature grid `(D, H, W)`.

Use this for:

- visualization (PCA / norm maps)
- patch-wise analysis
- spatial structure inspection

!!! note
    For ViT-like models, `grid` is often a token/patch grid, not guaranteed georeferenced raster pixels.

Interpretation:

- one ROI in, one spatial embedding field out
- useful when spatial layout matters more than a single pooled descriptor

---

## Backends and Providers

Think of backend as the input retrieval/runtime path.

- `backend="auto"`: recommended default; lets rs-embed choose the model-compatible access path
- `backend="gee"`: fetch imagery from Google Earth Engine (common for on-the-fly models)

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

This is a runtime policy choice, not a model identity choice.
Use it when the same model needs different large-ROI handling in different workflows.

---

## Precomputed vs On-the-fly Models

### Precomputed

- Reads embeddings from existing embedding products
- Faster and simpler runtime
- Temporal coverage and resolution are fixed by the product

Examples:

- `tessera`
- `gse`
- `copernicus`

### On-the-fly

- Fetches imagery patch, preprocesses, then runs model inference
- More flexible but heavier dependencies/runtime
- Requires careful attention to bands, temporal windows, and normalization

Examples:

- `remoteclip`
- `prithvi`
- `anysat`
- `terramind`

---

## Where To Go Next

- Need runnable examples: [Quickstart](quickstart.md)
- Need task recipes: [Workflows](workflows.md)
- Need model-specific assumptions: [Models](models.md)
- Need exact type definitions: [API: Specs and Data Structures](api_specs.md)
