![rs-embed banner](assets/banner.png)

> **One line of code** to get embeddings from **any Remote Sensing (RS) foundation model** for a given ROI.  

---

## Motivation

The remote sensing community has seen an explosion of foundation models in recent years.
Yet, using them in practice remains surprisingly painful:
* Inconsistent model interfaces (imagery vs. tile embeddings)
* Ambiguous input semantics (patch / tile / grid / pooled)
* Large differences in temporal, spectral, and spatial requirements
* No easy way to fairly compare multiple models in a single experiment

RS-Embed aims to fix this.

!!! success “Goal”
    Provide a **minimal**, **unified**, and **stable API** that turns diverse RS foundation models into a simple `ROI → embedding service` — so researchers can focus on **downstream tasks**, **benchmarking**, and **analysis**, not glue code.

## Why rs-embed?

- **Unified interface** for diverse embedding models (on-the-fly models and precomputed products).
- **Spatial + temporal specs** to describe what you want, not how to fetch it.
- **Batch export as a first-class workflow** via `export_batch`.



## Where to go next
- 🚀 Start here: Quick Start
- 📦 Browse models: Supported Models
- 🧠 Understand outputs: Output Semantics
- 🧩 Code examples: API