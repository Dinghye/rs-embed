<div align="center">

# <img src="./docs/assets/icon.png" width="35" alt="icon" />  rs-embed
**A single line of code to get embeddings from Any Remote Sensing Foundation Model(RSFM) for Any location and Any time**


[![Docs](https://img.shields.io/badge/docs-online-brightgreen)](https://dinghye.github.io/rs-embed/)
![Visitors](https://visitor-badge.laobi.icu/badge?page_id=Dinghye.rs-embed)
![License](https://img.shields.io/github/license/Dinghye/rs-embed)
![Last Commit](https://img.shields.io/github/last-commit/Dinghye/rs-embed)

[Docs](https://dinghye.github.io/rs-embed/) · [Examples](./examples) · [Playground](./examples/playground.ipynb)

</div>


<img src="./docs/assets/background.png" /> 


## TL;DR

```python
emb = get_embedding("tessera", spatial=..., temporal=..., output=...)
```


## Install(tempory)
```bash
# temporay
git clone git@github.com:Dinghye/rs-embed.git
cd rs-embed
conda env create -f environment.yml
conda activate rs-embed
pip install -e .

# If you are using GEE for the first time, run:
earthengine authenticate
```

## Quick Example
```python
from rs_embed import PointBuffer, TemporalSpec, OutputSpec, get_embedding

spatial = PointBuffer(lon=121.5, lat=31.2, buffer_m=2048)
temporal = TemporalSpec.year(2024)

emb = get_embedding(
    "tessera",
    spatial=spatial,
    temporal=temporal,
    output=OutputSpec.grid(),
    backend="local",
)

```
You can also visualize the embedding by :

```python
from rs_embed plot_embedding_pseudocolor

plot_embedding_pseudocolor(
    emb_tessera_grid,
    title="Tessera grid PCA pseudocolor",
)
```


<img src="./docs/assets/vis.png" width=600 /> 

## Supported Models (Quick Reference)

This is a convenience index with basic model info only (for quick scanning / future links).
For detailed I/O behavior and preprocessing notes, see `docs/models.md`.

### Precomputed Embeddings

| Model | ID | Type | Resolution | Time Coverage | 
|---|---|---|---|---|
| Tessera | `tessera` | Precomputed | 0.1° | 2017-2025 |
| Google Satellite Embedding (Alpha Earth) | `gse_annual` | Precomputed | 10 m | 2017-2024 |
| Copernicus Embed | `copernicus_embed` | Precomputed | 0.25° | 2021 |

### On-the-fly Foundation Models

| Model ID |  Primary Input  | Publication | Link |
|---|---|---|---|
| `remoteclip_s2rgb` |  S2 RGB | IEEE TGRS2024 |[link](https://github.com/ChenDelong1999/RemoteCLIP) |
| `satmae_rgb` |  S2 RGB | NeurIPS 2022 |[link](https://github.com/sustainlab-group/SatMAE)|
| `scalemae_rgb` | S2 RGB (+ scale) | ICCV 2023 | [link](https://github.com/bair-climate-initiative/scale-mae) |
| `anysat` |  S2 time series (10-band) | CVPR 2025 | [link](https://github.com/gastruc/AnySat) |
| `galileo` | S2 time series (10-band) | ICML 2025 | [link](https://github.com/nasaharvest/galileo) |
| `wildsat` | S2 RGB | ICCV 2025 | [link](https://github.com/mdchuc/HRSFM) |
| `prithvi_eo_v2_s2_6b` | S2 6-band | Arvix 2023 | [link](https://huggingface.co/ibm-nasa-geospatial) |
| `terrafm_b` | S2 12-band / S1 VV-VH | ICLR 2026 | [link](https://github.com/mbzuai-oryx/TerraFM) |
| `terramind` | S2 12-band | ICCV 2025 | [link](https://github.com/IBM/terramind) |
| `dofa` |  Multi-band + wavelengths | Arvix 2024 | [link](https://github.com/zhu-xlab/DOFA) |
| `fomo` | S2 12-band | AAAI 2025 |[link](https://github.com/RolnickLab/FoMo-Bench)|
| `thor_1_0_base` | S2 10-band | Arvix 2026 | [link](https://github.com/FM4CS/THOR) |
| `agrifm` | S2 time series (10-band) | RSE 2026 | [link](https://github.com/flyakon/AgriFM) |
| `satvision_toa` |  TOA 14-channel | Arvix 2024 | [link](https://github.com/nasa-nccs-hpda/pytorch-caney)|

See [Supported Models](https://dinghye.github.io/rs-embed/models/) for exact behavior.

## Learn More
📚 [Full documentation](https://dinghye.github.io/rs-embed/)



## License
Apache-2.0
