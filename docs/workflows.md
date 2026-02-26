# Workflows

This page is task-first: start from what you want to do, then use the smallest API surface that gets you there.

For full signatures and edge cases, see [API Reference](api.md).

=== ":material-map-marker-path: One ROI"

    Use [`get_embedding(...)`](api_embedding.md#get_embedding) for the fastest path to a single embedding.

=== ":material-dots-grid: Many ROIs (one model)"

    Use [`get_embeddings_batch(...)`](api_embedding.md#get_embeddings_batch) when the model is fixed and you have many ROIs.

=== ":material-database-export-outline: Dataset export"

    Use [`export_batch(...)`](api_export.md#export_batch) for reproducible, resumable exports across many ROIs/models.

=== ":material-image-search-outline: Debug inputs"

    Use [`inspect_provider_patch(...)`](api_inspect.md#inspect_provider_patch) before blaming the model.

!!! info "How to read this page"
    Start from the task tab above, then scroll to the matching section for a runnable example and "Choose this when" guidance.

---

## Single Embedding (Fastest Path)

Use `get_embedding(...)` when you want one ROI embedding now.

```python
from rs_embed import PointBuffer, TemporalSpec, OutputSpec, get_embedding

emb = get_embedding(
    "remoteclip",  # (1)!
    spatial=PointBuffer(lon=121.5, lat=31.2, buffer_m=2048),  # (2)!
    temporal=TemporalSpec.range("2022-06-01", "2022-09-01"),  # (3)!
    output=OutputSpec.pooled(),  # (4)!
    backend="gee",
    device="auto",
)
```
1. Model ID (see [Model Overview](models.md) / [Model Reference](models_reference.md)).
2. ROI centered at a point with a square buffer (meters).
3. Date range is a window, not a guaranteed single scene.
4. `pooled()` is the best default for comparison/classification workflows.

Choose this when:

- you are prototyping
- you want to inspect metadata
- you are debugging model behavior on one location

---

## Batch Embeddings for One Model

Use `get_embeddings_batch(...)` when the model is fixed and you have multiple ROIs.

```python
from rs_embed import PointBuffer, TemporalSpec, OutputSpec, get_embeddings_batch

spatials = [
    PointBuffer(121.5, 31.2, 2048),
    PointBuffer(120.5, 30.2, 2048),
]

embs = get_embeddings_batch(
    "remoteclip",
    spatials=spatials,
    temporal=TemporalSpec.range("2022-06-01", "2022-09-01"),
    output=OutputSpec.pooled(),
    backend="gee",
)
```

Choose this when:

- same model, many points
- you want simpler code than manual loops
- you may benefit from embedder-level batch inference

---

## Dataset Export (Recommended)

Use `export_batch(...)` for reproducible data pipelines and downstream experiments.
For new code, prefer the `out + layout` target style so the same API pattern works for single-ROI and multi-ROI exports.

```python
from rs_embed import export_batch, PointBuffer, TemporalSpec

spatials = [
    PointBuffer(121.5, 31.2, 2048),
    PointBuffer(120.5, 30.2, 2048),
]

export_batch(
    spatials=spatials,
    names=["p1", "p2"],  # (1)!
    temporal=TemporalSpec.range("2022-06-01", "2022-09-01"),  # (2)!
    models=["remoteclip", "prithvi"],  # (3)!
    out="exports",  # (4)!
    layout="per_item",  # (5)!
    backend="gee",
    save_inputs=True,
    save_embeddings=True,
    resume=True,
)
```
1. Stable ROI names make exports/manifests easier to track.
2. Apply one temporal policy consistently across all items for fair comparisons.
3. Mix multiple models in one export job when building benchmark datasets.
4. Root output directory for manifests, inputs, and embeddings.
5. `per_item` keeps each ROI grouped together; useful for inspection and resume.

Choose this when:

- multiple models and/or many points
- you need manifests for bookkeeping
- you want resumable exports
- you want to avoid duplicate input downloads

---

## Inspect Inputs Before Modeling

Use patch inspection when outputs look suspicious (clouds, wrong band order, bad dynamic range, etc.).

### Preferred: provider-agnostic

```python
from rs_embed import inspect_provider_patch, PointBuffer, TemporalSpec, SensorSpec

report = inspect_provider_patch(
    spatial=PointBuffer(121.5, 31.2, 2048),
    temporal=TemporalSpec.range("2022-06-01", "2022-09-01"),
    sensor=SensorSpec(
        collection="COPERNICUS/S2_SR_HARMONIZED",
        bands=("B4", "B3", "B2"),
        scale_m=10,
    ),
    backend="gee",
)
```

### Backward-compatible alias

- `inspect_gee_patch(...)` calls the same underlying inspection flow for GEE paths.

### Export convenience wrapper (optional)

- `export_npz(...)` is a single-ROI `.npz` convenience wrapper around `export_batch(...)`.
- Prefer `export_batch(...)` in tutorials and pipelines so one API scales from one ROI to many ROIs.

---

## Large ROI with Tiling

If you request large ROIs for on-the-fly models, try API-side tiling:

```python
from rs_embed import get_embedding, PointBuffer, TemporalSpec, OutputSpec

emb = get_embedding(
    "remoteclip",
    spatial=PointBuffer(121.5, 31.2, 8000),
    temporal=TemporalSpec.range("2022-06-01", "2022-09-01"),
    output=OutputSpec.grid(),
    backend="gee",
    input_prep="tile",
)
```

Use `input_prep="tile"` when:

- `OutputSpec.grid()` matters
- large ROI resize would lose too much detail
- you accept extra runtime cost for better spatial structure preservation

---

## Fair Cross-Model Comparison

When benchmarking models, prefer:

- same ROI list
- same temporal window
- same compositing policy (`SensorSpec.composite`)
- `OutputSpec.pooled()` first
- default model normalization unless replicating original training setup

Then use [Supported Models](models.md) to review model-specific preprocessing and required side inputs.

---

## Choosing the Right Page

- Need runnable setup steps: [Quickstart](quickstart.md)
- Need mental model and semantics: [Concepts](concepts.md)
- Need model capability matrix: [Model Overview](models.md)
- Need exact function signatures/options: [API Reference](api.md)
