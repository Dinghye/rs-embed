# Common Workflows

This page is task-first: start from what you want to do, then use the smallest API surface that gets you there.

For full signatures and edge cases, see [API Reference](api.md).

---

## Workflow 1: Get a Single Embedding (Fastest Path)

Use `get_embedding(...)` when you want one ROI embedding now.

```python
from rs_embed import PointBuffer, TemporalSpec, OutputSpec, get_embedding

emb = get_embedding(
    "remoteclip_s2rgb",
    spatial=PointBuffer(lon=121.5, lat=31.2, buffer_m=2048),
    temporal=TemporalSpec.range("2022-06-01", "2022-09-01"),
    output=OutputSpec.pooled(),
    backend="gee",
    device="auto",
)
```

Choose this when:

- you are prototyping
- you want to inspect metadata
- you are debugging model behavior on one location

---

## Workflow 2: Compare Many Points for One Model

Use `get_embeddings_batch(...)` when the model is fixed and you have multiple ROIs.

```python
from rs_embed import PointBuffer, TemporalSpec, OutputSpec, get_embeddings_batch

spatials = [
    PointBuffer(121.5, 31.2, 2048),
    PointBuffer(120.5, 30.2, 2048),
]

embs = get_embeddings_batch(
    "remoteclip_s2rgb",
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

## Workflow 3: Export a Dataset (Recommended for Real Projects)

Use `export_batch(...)` for reproducible data pipelines and downstream experiments.

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
    save_inputs=True,
    save_embeddings=True,
    resume=True,
)
```

Choose this when:

- multiple models and/or many points
- you need manifests for bookkeeping
- you want resumable exports
- you want to avoid duplicate input downloads

---

## Workflow 4: Inspect Inputs Before Running a Model

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

---

## Workflow 5: Large ROI with Better Spatial Fidelity

If you request large ROIs for on-the-fly models, try API-side tiling:

```python
from rs_embed import get_embedding, PointBuffer, TemporalSpec, OutputSpec

emb = get_embedding(
    "remoteclip_s2rgb",
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

## Workflow 6: Fair Cross-Model Comparisons

When benchmarking models, prefer:

- same ROI list
- same temporal window
- same compositing policy (`SensorSpec.composite`)
- `OutputSpec.pooled()` first
- default model normalization unless replicating original training setup

Then use [Supported Models](models.md) to review model-specific preprocessing and required side inputs.

---

## Choosing the Right Page

- Need runnable setup steps: [Quick Start](quickstart.md)
- Need mental model and semantics: [Core Concepts](concepts.md)
- Need model capability matrix: [Supported Models](models.md)
- Need exact function signatures/options: [API Reference](api.md)
