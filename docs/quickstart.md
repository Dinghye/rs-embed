## Install (temporary)

```bash
git clone git@github.com:Dinghye/rs-embed.git
cd rs-embed
conda env create -f environment.yml
conda activate rs-embed
pip install -e .
```

Examples notebook: examples/playground.ipynb

### **Example A — Tessera (Precomputed)**

```python
spatial = PointBuffer(lon=121.5, lat=31.2, buffer_m=2048)
temporal = TemporalSpec.year(2024)

emb_tes_g = get_embedding(
    "tessera",
    spatial=spatial,
    temporal=temporal,
    output=OutputSpec.grid(),
    backend="local",
)
pca_tes = plot_embedding_pseudocolor(emb_tes_g, title="Tessera PCA pseudocolor")
print("tessera grid meta:", emb_tes_g.meta)
```

### **Example B — RemoteCLIP (On-the-fly)**

```python
emb = get_embedding(
    "remoteclip_s2rgb",
    spatial=spatial,
    temporal=TemporalSpec.range("2022-06-01", "2022-09-01"),
    output=OutputSpec.grid(),
    backend="gee",
)

print(emb.data.shape)  # (D, H, W)
```

## Notes / prerequisites
- `Google Earth Engine `is required for backend="gee" workflows.
- Imagery fetching + patch checks are handled automatically by the framework.

!!! tip
If you only want to use precomputed embeddings, you can start with backend="local" and skip GEE.