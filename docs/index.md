# rs-embed

![rs-embed banner](assets/banner.png)

> **One line of code** to get embeddings from **any Remote Sensing (RS) foundation model** for a given ROI.  
> Supports both **precomputed embeddings** (e.g., Tessera, GSE, Copernicus Embed) and  
> **on-the-fly models** (e.g., Prithvi-EO v2, RemoteCLIP, SatMAE, ScaleMAE).

---

## TL;DR

```python
emb = get_embedding("tessera", spatial=..., temporal=..., output=...)
```

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



## **🚀 Key Features**
- ✅ **Unified input interface**
    - Spatial: BBox, PointBuffer
    - Temporal: TemporalSpec.year(), TemporalSpec.range()
- ✅ **Unified output interface**
    - Pooled embeddings → (D,)
    - Grid embeddings → (D, H, W) (as xarray.DataArray)
- ✅ **Model-agnostic API**
```python
emb = get_embedding("tessera", spatial=..., temporal=..., output=...)
```

- ✅ **Two classes of supported models**
    - 🧊 **Precomputed embeddings** (no deep learning environment required)
    - 🔥 **On-the-fly models** (imagery fetched automatically from Google Earth Engine)

## Where to go next
- 🚀 Start here: Quick Start
- 📦 Browse models: Supported Models
- 🧠 Understand outputs: Output Semantics
- 🧩 Code examples: API