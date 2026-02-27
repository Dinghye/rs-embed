# API Reference

This is the API reference entry page.

`rs-embed` API docs are split by topic for readability:

- [API: Specs and Data Structures](api_specs.md)
- [API: Embedding](api_embedding.md)
- [API: Export](api_export.md)
- [API: Inspect](api_inspect.md)

If you are looking for task-oriented usage first:

- [Quick Start](quickstart.md): fastest first run
- [Common Workflows](workflows.md): task-first recipes
- [Core Concepts](concepts.md): semantics for `TemporalSpec`, `OutputSpec`, and backends

---

## Imports

```python
from rs_embed import (
    # Specs
    BBox, PointBuffer, TemporalSpec, SensorSpec, OutputSpec, InputPrepSpec,
    # Core APIs
    get_embedding, get_embeddings_batch, export_batch, export_npz,
    # Utilities
    inspect_provider_patch,
    inspect_gee_patch,
)
```

---

## Recommended Starting Points

For new code, most users only need these entry points:

- `get_embedding(...)`
- `get_embeddings_batch(...)`
- `export_batch(...)`
- `inspect_provider_patch(...)`

Compatibility / convenience wrappers (still supported):

- `export_npz(...)` -> wrapper around `export_batch(...)` for single-ROI `.npz`
- `inspect_gee_patch(...)` -> wrapper around `inspect_provider_patch(...)`

---

## Choose by Task

| I want to... | Read this page |
|---|---|
| understand spatial/temporal/output specs | [API: Specs and Data Structures](api_specs.md) |
| get one embedding or batch embeddings | [API: Embedding](api_embedding.md) |
| build export pipelines and datasets | [API: Export](api_export.md) |
| inspect raw provider patches before inference | [API: Inspect](api_inspect.md) |

---
## Model Registry (Advanced)

If you need a stable model list in code, use the model catalog:

```python
from rs_embed.embedders.catalog import MODEL_SPECS
print(sorted(MODEL_SPECS.keys()))
```

`list_models()` from `rs_embed.core.registry` only reports models currently loaded into the runtime registry.

---

## Errors

rs-embed raises several explicit exception types (all in `rs_embed.core.errors`):

- `SpecError`: spec validation failure (invalid bbox, missing temporal fields, etc.)
- `ProviderError`: provider/backend errors (e.g., GEE initialization or fetch failure)
- `ModelError`: unknown model ID, unsupported parameters, unsupported export format, etc.

---

## Optional Dependencies

Different features require different optional dependencies:

- `pip install "rs-embed[gee]"`: use the Earth Engine backend
- `pip install "rs-embed[torch]"`: torch model inference
- `pip install "rs-embed[models]"`: dependencies for some model wrappers (e.g., rshf)
- `pip install "rs-embed[dev]"`: dev dependencies such as pytest

---

## Versioning Notes

The current version is still early stage (`0.1.x`):

- `BBox/PointBuffer` currently require `crs="EPSG:4326"`
- Precomputed models should use `backend="auto"`; on-the-fly models mainly use provider backends (typically `"gee"` or explicit provider names)
- `export_batch(format=...)` currently supports `"npz"` and `"netcdf"`; it may be extended to parquet/zarr/hdf5, etc.
