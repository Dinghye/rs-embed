![rs-embed banner](assets/banner.png)

> One line of code to get embeddings from **any Remote Sensing Foundation Model (RSFM)** for **any location** and **any time**

---

## Start Here

### If you are new

1. Read [Quick Start](quickstart.md) to install and run a first example
2. Read [Core Concepts](concepts.md) to understand temporal/output semantics
3. Use [Common Workflows](workflows.md) to pick the right API for your task

### If you want to choose a model

- Go to [Supported Models](models.md) for the comparison matrix, preprocessing notes, and temporal behavior

### If you want exact signatures and parameters

- Go to [API Reference](api.md)

---

## Common Tasks

| Goal | Best Entry Point | Main API |
|---|---|---|
| Get one embedding for one ROI | [Quick Start](quickstart.md) | `get_embedding(...)` |
| Compute embeddings for many ROIs (same model) | [Common Workflows](workflows.md) | `get_embeddings_batch(...)` |
| Build an export dataset for experiments | [Common Workflows](workflows.md) | `export_batch(...)` |
| Debug bad inputs/clouds/band issues | [Common Workflows](workflows.md) | `inspect_provider_patch(...)` (recommended) |
| Compare model preprocessing and I/O assumptions | [Supported Models](models.md) | model matrix + notes |

---

## Motivation

![rs-embed background](assets/background.png)


The remote sensing community has seen an explosion of foundation models in recent years.
Yet, using them in practice remains surprisingly painful:
* Inconsistent model interfaces (imagery vs. tile embeddings)
* Ambiguous input semantics (patch / tile / grid / pooled)
* Large differences in temporal, spectral, and spatial requirements
* No easy way to fairly compare multiple models in a single experiment


RS-Embed aims to fix this.

!!! success "Goal"
    Provide a **minimal**, **unified**, and **stable API** that turns diverse RS foundation models into a simple `ROI → embedding service` — so researchers can focus on **downstream tasks**, **benchmarking**, and **analysis**, not glue code.

## Why rs-embed?

- **Unified interface** for diverse embedding models (on-the-fly models and precomputed products).
- **Spatial + temporal specs** to describe what you want, not how to fetch it.
- **Batch export as a first-class workflow** via `export_batch`.
- **Compatibility wrappers preserved** (for example `export_npz`, `inspect_gee_patch`) without changing the main learning path.

---

## Documentation Map

### Learn

- [Quick Start](quickstart.md): installation + first successful runs
- [Core Concepts](concepts.md): mental model (`TemporalSpec`, `OutputSpec`, backends)

### Guides

- [Common Workflows](workflows.md): task-oriented usage patterns
- [Supported Models](models.md): model capabilities, preprocessing, env knobs

### Reference

- [API Reference](api.md): exact signatures and parameter details
- [Limitations](limitations.md): current constraints and known edge cases

### Development

- [Extending](extending.md): add new model adapters and integrate with registry/export
