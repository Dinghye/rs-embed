<div align="center">

# <img src="./docs/assets/icon.png" width="30" alt="icon" />  rs-embed
**One line of code to get embeddings from any Remote Sensing foundation model for a given ROI.**


<!-- Badges: replace YOUR_USER/YOUR_REPO + workflow file name -->
[![Docs](https://img.shields.io/badge/docs-online-brightgreen)](https://dinghye.github.io/rs-embed/)
![Visitors](https://visitor-badge.laobi.icu/badge?page_id=Dinghye.rs-embed)
![License](https://img.shields.io/github/license/Dinghye/rs-embed)
![Last Commit](https://img.shields.io/github/last-commit/Dinghye/rs-embed)

[Docs](https://dinghye.github.io/rs-embed/) · [Examples](./examples) · [Playground](./examples/playground.ipynb)

</div>



## TL;DR

```python
emb = get_embedding("tessera", spatial=..., temporal=..., output=...)
```

* 🧊 Precomputed embeddings: Tessera / GSE / Copernicus Embed
* 🔥 On-the-fly models: Prithvi-EO v2 / RemoteCLIP / SatMAE / ScaleMAE (imagery via GEE)

## Install(tempory)
```bash
git git@github.com:Dinghye/rs-embed.git
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

print(emb.data.shape)  # (D,)
```

## Learn More
📚 Full documentation: https://dinghye.github.io/rs-embed/




## License
Apache-2.0