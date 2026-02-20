

This project introduces OutputSpec to provide a unified abstraction over the outputs of different Remote Sensing Foundation Models (RS FMs).


## OutputSpec.pooled(): ROI-level Vector Embedding

**Semantic Meaning**

> pooled represents an entire ROI (Region of Interest) with a single vector (D,).

This is the most stable and comparable representation, suitable for:

* Classification / regression
* Retrieval / similarity search
* Clustering
* Cross-model comparison (recommended)

Unified Output Format
```
Embedding.data.shape == (D,)
```

**(a) ViT / MAE-style models**
(RemoteCLIP / Prithvi / SatMAE / ScaleMAE)

* Native output: patch tokens
* tokens: (N, D)   # N = patch tokens (+ optional CLS)

Processing steps:
	1.	Remove CLS token if present
	2.	Aggregate tokens along the token dimension (default: mean, optional: max)

Mean pooling:

$$
v_d = \frac{1}{N'} \sum_{i=1}^{N'} t_{i,d}
$$


**(b) Precomputed embeddings** (Tessera / GSE / Copernicus)

* Native output: embedding grid
grid: (D, H, W)

* Processing: Pool over spatial dimensions (H, W):

$$
v_d = \frac{1}{HW} \sum_{y,x} g_{d,y,x}
$$

Why pooled?

* Model-agnostic, stable, and comparable
* Avoids differences in spatial resolution or token structure
* Strongly recommended for cross-model benchmarks

## **OutputSpec.grid(): ROI-level Spatial Embedding Field**

**Semantic Meaning**

> grid outputs a spatially structured embedding field:
> An embedding tensor (D, H, W),
where each spatial location corresponds to a vector.

Suitable for:

* Spatial visualization (PCA / norm / similarity maps)
* Pixel-wise / patch-wise tasks
* Intra-ROI structure analysis

Unified Output Format
```
Embedding.data.shape == (D, H, W)
```

Usually returned as `xarray.DataArray` with metadata in `attrs`.
For precomputed geospatial products, metadata may include CRS/crop context.
For ViT token grids, it is typically patch-grid metadata (not georeferenced pixel coordinates).

**(a) ViT / MAE-style models**
* Native output: tokens (N, D)

Processing steps:

	1.	Remove CLS token (if present)
	2.	Reshape remaining tokens into a patch grid

`(N', D) → (H, W, D) → (D, H, W)`

(H, W) is determined by the model’s patch layout (e.g. 8×8, 14×14)


**(b) Precomputed embeddings**: (Tessera / GSE ...): Native output is already (D, H, W)
