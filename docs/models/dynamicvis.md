# DynamicVis (`dynamicvis`)

> Sentinel-2 RGB on-the-fly adapter for DynamicVis, loading the official DynamicVis backbone from upstream code + Hugging Face checkpoint and returning last-stage feature maps as pooled/grid embeddings.

## Quick Facts

| Field | Value |
|---|---|
| Model ID | `dynamicvis` |
| Family / Backbone | DynamicVis `DynamicVisBackbone` (official repo code) |
| Adapter type | `on-the-fly` |
| Typical backend | provider backend (`gee`) |
| Primary input | S2 RGB (`B4,B3,B2`) |
| Temporal mode | `range` in practice (normalized via shared helper) |
| Output modes | `pooled`, `grid` |
| Extra side inputs | none (but backbone config knobs are exposed via env) |
| Training alignment (adapter path) | Medium-High when checkpoint + DynamicVis repo/runtime deps match expected upstream environment |

---

## When To Use This Model

### Good fit for

- RGB experiments where you want a non-ViT feature-map backbone path
- larger input-size runs (`512` default) with dense feature maps
- comparisons against other RGB models using feature-map pooling instead of token pooling

### Be careful when

- OpenMMLab runtime deps (`mmengine`, `mmcv`) are not installed
- assuming `grid` is token grid (here it is last-stage feature map grid)
- changing `arch/path_type/sampling_scale` without recording them

---

## Input Contract (Current Adapter Path)

### Spatial / temporal

- Provider backend only (`backend="gee"` / provider-compatible backend)
- `TemporalSpec` normalized via shared helper; use `TemporalSpec.range(...)` for reproducibility
- Temporal window is used for compositing/filtering

### Sensor / channels

Default `SensorSpec` if omitted:

- Collection: `COPERNICUS/S2_SR_HARMONIZED`
- Bands: `("B4", "B3", "B2")`
- `scale_m=10`, `cloudy_pct=30`, `composite="median"`

`input_chw` contract:

- must be `CHW` with 3 bands in `(B4,B3,B2)` order
- expected raw S2 SR values in `0..10000`
- adapter converts to `[0,1]` then `uint8` RGB before ImageNet normalization

---

## Preprocessing Pipeline (Current rs-embed Path)

1. Fetch S2 RGB patch as `uint8` (provider path) or convert `input_chw` raw SR -> `[0,1]` -> `uint8`
2. Resize to `RS_EMBED_DYNAMICVIS_IMG` (default `512`)
3. Convert to tensor with ImageNet normalization (mean/std)
4. Load DynamicVis backbone from:
   - DynamicVis repo code (installed / local path / auto-clone)
   - Hugging Face checkpoint (`hf_repo` + `ckpt_file`)
5. Forward backbone and extract last returned feature tensor
6. Adapter normalizes output:
   - if feature map `[B,C,H,W]`: use last-stage feature map
   - if vector-like `[B,D]`: convert to `1x1` feature map
7. Return pooled vector or grid

---

## Environment Variables / Tuning Knobs

### Backbone / checkpoint / repo loading

| Env var | Default | Effect |
|---|---|---|
| `RS_EMBED_DYNAMICVIS_HF_REPO` | `KyanChen/DynamicVis` | HF repo for checkpoint download |
| `RS_EMBED_DYNAMICVIS_CKPT_FILE` | upstream default ckpt filename | Checkpoint file in HF repo |
| `RS_EMBED_DYNAMICVIS_REPO_PATH` | unset | Local DynamicVis repo path |
| `RS_EMBED_DYNAMICVIS_REPO_URL` | upstream GitHub URL | Repo clone source |
| `RS_EMBED_DYNAMICVIS_REPO_CACHE` | `~/.cache/rs_embed/dynamicvis` | Repo cache root |
| `RS_EMBED_DYNAMICVIS_AUTO_DOWNLOAD_REPO` | `1` | Auto-clone DynamicVis repo when missing |

### Model/runtime settings

| Env var | Default | Effect |
|---|---|---|
| `RS_EMBED_DYNAMICVIS_IMG` | `512` | Resize target image size |
| `RS_EMBED_DYNAMICVIS_ARCH` | `auto` | Backbone arch hint (`b/base` or `l/large`; auto infers from ckpt filename) |
| `RS_EMBED_DYNAMICVIS_PATH_TYPE` | `forward_reverse_mean` | DynamicVis backbone path type |
| `RS_EMBED_DYNAMICVIS_SAMPLING_SCALE` | `0.1` | Sampling scale parameter |
| `RS_EMBED_DYNAMICVIS_MAMBA2` | `0` | Enable Mamba2 variant flag |
| `RS_EMBED_DYNAMICVIS_FETCH_WORKERS` | `8` | Provider prefetch workers for batch APIs |

Related cache envs (used by HF download path):

- `HUGGINGFACE_HUB_CACHE`, `HF_HOME`, `HUGGINGFACE_HOME`

---

## Output Semantics

### `OutputSpec.pooled()`

- Pools the last-stage feature map spatially:
  - `mean` -> `featmap_mean`
  - `max` -> `featmap_max`
- Not token pooling

### `OutputSpec.grid()`

- Returns last-stage backbone feature map as `xarray.DataArray` `(D,H,W)`
- Metadata marks `grid_kind="last_stage_featmap"`
- Grid is model feature-map layout, not georeferenced raster pixels

---

## Examples

### Minimal provider-backed example

```python
from rs_embed import get_embedding, PointBuffer, TemporalSpec, OutputSpec

emb = get_embedding(
    "dynamicvis",
    spatial=PointBuffer(lon=121.5, lat=31.2, buffer_m=2048),
    temporal=TemporalSpec.range("2022-06-01", "2022-09-01"),
    output=OutputSpec.pooled(),
    backend="gee",
)
```

### Example backbone tuning (env-controlled)

```python
# Example (shell):
# export RS_EMBED_DYNAMICVIS_IMG=512
# export RS_EMBED_DYNAMICVIS_ARCH=b
# export RS_EMBED_DYNAMICVIS_PATH_TYPE=forward_reverse_mean
# export RS_EMBED_DYNAMICVIS_SAMPLING_SCALE=0.1
```

---

## Common Failure Modes / Debugging

- missing OpenMMLab runtime deps (`mmengine`, `mmcv`)
- DynamicVis repo import failures (repo present but dependencies missing)
- HF checkpoint download issues / invalid checkpoint file
- wrong `input_chw` shape (`CHW`, `C=3`)
- confusion between feature-map grid and token grid semantics

Recommended first checks:

- inspect metadata: `arch`, `path_type`, `sampling_scale`, `mamba2`, `feature_kind`
- verify repo path and checkpoint source
- start with `OutputSpec.pooled()` before debugging grid shape differences

---

## Reproducibility Notes

Keep fixed and record:

- HF checkpoint repo/file
- DynamicVis repo source/path (or commit if using local clone)
- `IMG`, `ARCH`, `PATH_TYPE`, `SAMPLING_SCALE`, `MAMBA2`
- temporal window + compositing settings
- output mode and pooling choice

---

## Source of Truth (Code Pointers)

- Registration/catalog: `src/rs_embed/embedders/catalog.py`
- Adapter implementation: `src/rs_embed/embedders/onthefly_dynamicvis.py`

