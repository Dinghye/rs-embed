![rs-embed banner](assets/banner.png)

> One line of code to get embeddings from **any Remote Sensing Foundation Model (RSFM)** for **any location** and **any time**

---

## Start Here

=== ":material-rocket-launch: First Run"

    1. Read [Quickstart](quickstart.md) to install and run a first example
    2. Read [Concepts](concepts.md) to understand temporal/output semantics
    3. Use [Workflows](workflows.md) to pick the right API for your task

=== ":material-view-grid-outline: Choose a Model"

    - Go to [Model Overview](models.md) for the comparison matrix, preprocessing notes, and temporal behavior
    - Open [Model Reference](models_reference.md) when you need advanced details and exact assumptions

=== ":material-api: Exact Signatures"

    - Start with [API Reference](api.md)
    - Then use [Specs & Data Structures](api_specs.md) for `TemporalSpec`, `OutputSpec`, and related types

!!! tip "Suggested reading path"
    If you're unsure where to start, use **Quickstart → Concepts → Workflows**.

---

## :material-format-list-checks: Common Tasks

| Goal | Best Entry Point | Main API |
|---|---|---|
| Get one embedding for one ROI | [Quick Start](quickstart.md) | `get_embedding(...)` |
| Compute embeddings for many ROIs (same model) | [Common Workflows](workflows.md) | `get_embeddings_batch(...)` |
| Build an export dataset for experiments | [Common Workflows](workflows.md) | `export_batch(...)` |
| Debug bad inputs/clouds/band issues | [Common Workflows](workflows.md) | `inspect_provider_patch(...)` (recommended) |
| Compare model preprocessing and I/O assumptions | [Supported Models](models.md) | model matrix + notes |

---

## :material-lightbulb-on-outline: Motivation

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

## :material-map-search-outline: Documentation Map

### Learn

- [Quickstart](quickstart.md): installation + first successful runs
- [Concepts](concepts.md): mental model (`TemporalSpec`, `OutputSpec`, backends)

### Guides

- [Workflows](workflows.md): task-oriented usage patterns
- [Model Overview](models.md): model capabilities, preprocessing, env knobs

### Reference

- [API Reference](api.md): exact signatures and parameter details
- [Limitations](limitations.md): current constraints and known edge cases

### Development

- [Extending](extending.md): add new model adapters and integrate with registry/export
