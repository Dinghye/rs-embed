# API

RS-Embed exposes a unified API centered around:

- `SpatialSpec` (`BBox`, `PointBuffer`)
- `TemporalSpec` (`year`, `range`)
- `OutputSpec` (`pooled`, `grid`)
- `get_embedding(model_id, spatial, temporal, output, backend=...)`

![rs-embed banner](assets/api.png)


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

## Core imports

```python
from rs_embed import (
    BBox,
    PointBuffer,
    TemporalSpec,
    OutputSpec,
    get_embedding,
)