# **rs-embed**

![alt text](img/image.png)


**A unified Python toolkit for Remote Sensing Foundation Model embeddings**


> One line of code to get embeddings from **any Remote Sensing (RS) foundation model** for a given ROI.
> Supports both **precomputed embeddings** (Tessera, GSE, Copernicus Embed) and
> **on-the-fly models** (Prithvi-EO v2, RemoteCLIP, SatMAE, ScaleMAE).

---

## **✨ Motivation**


The remote sensing community has seen an explosion of **foundation models** in recent years.
Yet, using them in practice remains surprisingly painful:
- Inconsistent model interfaces (imagery vs. tile embeddings)
- Ambiguous input semantics (patch / tile / grid / pooled)
- Large differences in temporal, spectral, and spatial requirements
- No easy way to **fairly compare multiple models** in a single experiment
- 
**RS-Foundation-Kit** aims to fix this.

  

> 🎯 **Goal**

> Provide a **minimal, unified, and stable API** that turns diverse RS foundation models into a simple   
> **ROI → embedding** service
> so researchers can focus on **downstream tasks, benchmarking, and analysis** — not glue code.

---
## **🚀 Key Features**
- ✅ **Unified input interface**
    - Spatial: BBox, PointBuffer
    - Temporal: TemporalSpec.year(), TemporalSpec.range()
- ✅ **Unified output interface**
    - Pooled embeddings → (D,)
    - Grid embeddings → (D, H, W) (as xarray.DataArray)
- ✅ **Model-agnostic API**
```
emb = get_embedding("tessera", spatial=..., temporal=..., output=...)
```

- ✅ **Two classes of supported models**
    - 🧊 **Precomputed embeddings** (no deep learning environment required)
    - 🔥 **On-the-fly models** (imagery fetched automatically from Google Earth Engine)
- ✅ **Research-friendly design**
    - Explicit metadata (CRS, time range, crop window, tokens, etc.)
    - Built-in embedding visualization (norm, PCA pseudo-color, similarity maps)
    

---

## **Supported Models**
### **Precomputed Embeddings**

|**Model**|**ID**|**Output**|**Resolution**|**Dim**|**Time Coverage**|**Notes**|
|---|---|---|---|---|---|---|
|**Tessera**|tessera|pooled / grid|–|128|2017–2025|GeoTessera global tile embeddings|
|**Google Satellite Embedding (Alpha Earth)**|gse_annual|pooled / grid|10 m|64|2017–2024|Annual embeddings via GEE|
|**Copernicus Embed**|copernicus_embed|pooled / grid|0.25°|768|2021|Official Copernicus embeddings|

---
### **On-the-fly Foundation Models**

|**Model**|**ID**|**Input Imagery**|**Output**|
|---|---|---|---|
|**Prithvi-EO v2**|prithvi_eo_v2_s2_6b|Sentinel-2 (6 bands)|pooled / grid|
|**RemoteCLIP**|remoteclip_s2rgb|Sentinel-2 RGB|pooled / grid|
|**SatMAE**|satmae_s2rgb|Sentinel-2 RGB|pooled / grid|
|**ScaleMAE**|scalemae_s2rgb|Sentinel-2 RGB|pooled / grid|

**Notes**
- Imagery is fetched automatically from **Google Earth Engine**
- Token-level (patch/grid) outputs are supported
- Suitable for fine-grained semantic modeling and transfer learning

---

## **🧠 Unified Interface**

### **Spatial Specification**

```
BBox(minlon, minlat, maxlon, maxlat)
PointBuffer(lon, lat, buffer_m)
```
### **Temporal Specification**

```
TemporalSpec.year(2022)
TemporalSpec.range("2022-06-01", "2022-09-01")
```
### **Output Specification**

```
OutputSpec.pooled(pooling="mean")  # (D,)
OutputSpec.grid(scale_m=10)        # (D, H, W)
```

---
## **🧩 Core API**

```
from rs_embed import (
    BBox,
    PointBuffer,
    TemporalSpec,
    OutputSpec,
    get_embedding,
)
```

### **Example 1 — Tessera (Precomputed)**

```
spatial = PointBuffer(lon=121.5, lat=31.2, buffer_m=2048)

emb = get_embedding(
    "tessera",
    spatial=spatial,
    temporal=TemporalSpec.range("2021-06-01", "2021-08-31"),
    output=OutputSpec.pooled(),
    backend="local",
)

print(emb.data.shape)  # (D,)
```

### **Example 2 — RemoteCLIP (On-the-fly)**

```
emb = get_embedding(
    "remoteclip_s2rgb",
    spatial=spatial,
    temporal=TemporalSpec.range("2022-06-01", "2022-09-01"),
    output=OutputSpec.grid(),
    backend="gee",
)

print(emb.data.shape)  # (D, H, W)
```

---

## **🏗️ Project Structure**

```
rs_embed/
├── api.py                # Unified entry point
├── core/
│   ├── specs.py          # Spatial / Temporal / Output specs
│   ├── embedding.py      # Embedding(data, meta)
│   ├── registry.py       # Model registry
│   └── errors.py
├── embedders/
│   ├── base.py
│   ├── precomputed_*.py  # Tessera / GSE
│   ├── onthefly_*.py     # Prithvi / RemoteCLIP / MAE
│   └── _vit_mae_utils.py # Shared ViT / MAE utilities
├── providers/
│   └── gee.py            # Google Earth Engine
└── examples/
    ├── quickstart.py
    └── plot_utils.py
```


---
## **📊 Embedding Visualization**
Built-in utilities include:
- Embedding norm heatmaps
- Single-channel visualization
- PCA pseudo-color (RGB)
- Pixel-wise cosine similarity maps

```
plot_embedding_grid(emb, agg="norm")
plot_embedding_pseudocolor(emb)
plot_cosine_similarity_map(emb1, emb2)
```

---

## **🔌 Extending the Toolkit**

1. Create a new embedder:
```
@register("my_model")
class MyEmbedder(EmbedderBase):
    def get_embedding(...):
        ...
```    
2. Return a unified Embedding(data, meta)
3. Automatically available via get_embedding(...)

> You only need to implement:
> **ROI → embedding**
> Everything else (specs, metadata, output formatting) is handled by the framework.

---
## **⚠️ Known Limitations**
- This is a very preliminary task and there might be potential issues. Please use it with caution.
- Tessera / GSE cropping depends on CRS projection (UTM requires pyproj)
- Embedding dimensions differ across models — take care in cross-model comparisons
- Some models impose patch-size constraints (handled via automatic padding)

---
## **🗺️ Roadmap**
- Full Copernicus Embed support
- Multi-temporal embeddings (T × D × H × W)
- Strict CRS-aware grid alignment
- Deeper integration with TorchGeo / xarray / rasterio
- pip install rs-embed

