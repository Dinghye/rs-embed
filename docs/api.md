# API Reference

This section is the exact reference for the public API.
If you want installation and first-run examples, start with [Quickstart](quickstart.md) instead.

---

## Core Entry Points

Most users only need these:

- `get_embedding(...)` — single-ROI embedding (function)
- `get_embeddings_batch(...)` — batch embeddings (function)
- `Model(name, ...)` — class-based API (set up once, embed many)
- `export_batch(...)` — export embeddings to disk
- `inspect_provider_patch(...)` — inspect raw provider patches

---

## Choose by Task

| I want to... | Read this page |
|---|---|
| understand spatial/temporal/output specs | [API: Specs and Data Structures](api_specs.md) |
| get one embedding or batch embeddings | [API: Embedding](api_embedding.md) |
| use a stateful model instance across many calls | [API: Embedding](api_embedding.md) |
| build export pipelines and datasets | [API: Export](api_export.md) |
| inspect raw provider patches before inference | [API: Inspect](api_inspect.md) |

---
## Useful Extras

- `export_npz(...)`: compatibility wrapper around `export_batch(...)` for single-ROI `.npz`
- `inspect_gee_patch(...)`: compatibility wrapper around `inspect_provider_patch(...)`
- `list_models()`: stable public model catalog helper

If you need a stable model list in code:

```python
from rs_embed import list_models

print(list_models())
```

`rs_embed.core.registry.list_models()` only reports models currently loaded into the runtime registry.

---

## Errors

rs-embed raises several explicit exception types (all in `rs_embed.core.errors`):

- `SpecError`: spec validation failure (invalid bbox, missing temporal fields, etc.)
- `ProviderError`: provider/backend errors (e.g., GEE initialization or fetch failure)
- `ModelError`: unknown model ID, unsupported parameters, unsupported export format, etc.

---

## Versioning Notes

The current version is still early stage (`0.1.x`):

- `BBox/PointBuffer` currently require `crs="EPSG:4326"`
- Precomputed models should use `backend="auto"`; on-the-fly models mainly use provider backends (typically `"gee"` or explicit provider names)
- `ExportConfig(format=...)` is the recommended way to choose export format; supported values are currently `"npz"` and `"netcdf"` and may be extended to parquet/zarr/hdf5, etc.
