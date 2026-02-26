# SatMAE RGB (`satmae_rgb`)

> Sentinel-2 RGB on-the-fly adapter for SatMAE (`rshf.satmae.SatMAE`), returning pooled vectors or ViT patch-token grids from `forward_encoder(mask_ratio=0.0)`.

## Quick Facts

| Field | Value |
|---|---|
| Model ID | `satmae_rgb` |
| Family / Backbone | SatMAE via `rshf.satmae.SatMAE` |
| Adapter type | `on-the-fly` |
| Typical backend | provider backend (`gee`) |
| Primary input | S2 RGB (`B4,B3,B2`) |
| Temporal mode | range window in practice (normalized via shared helper) |
| Output modes | `pooled`, `grid` |
| Extra side inputs | none |
| Training alignment (adapter path) | Medium-High (higher when wrapper `model.transform(...)` is available and used) |

---

## When To Use This Model

### Good fit for

- strong RGB-only SatMAE baseline on Sentinel-2
- MAE-style token-grid analysis (`OutputSpec.grid()`)
- comparisons with other RGB ViT adapters (`remoteclip_s2rgb`, `scalemae_rgb`, `wildsat`)

### Be careful when

- you need multispectral semantics (this is RGB-only)
- assuming `grid` is georeferenced pixels (it is a patch-token grid)
- comparing runs across environments where `rshf` wrapper preprocessing behavior may differ

---

## Input Contract (Current Adapter Path)

### Spatial / temporal

- Provider backend only (`backend="gee"` / provider-compatible backend)
- `TemporalSpec` is normalized via shared helper; use `TemporalSpec.range(...)` for reproducibility
- Temporal window is used for compositing/filtering, not single-scene identity selection

### Sensor / channels

Default `SensorSpec` if omitted:

- Collection: `COPERNICUS/S2_SR_HARMONIZED`
- Bands: `("B4", "B3", "B2")`
- `scale_m=10`, `cloudy_pct=30`, `composite="median"`

`input_chw` contract:

- must be `CHW` with exactly 3 bands in `(B4,B3,B2)` order
- expected raw S2 SR values in `0..10000`
- adapter converts to `[0,1]`, then `uint8` RGB before model preprocessing

---

## Preprocessing Pipeline (Current rs-embed Path)

1. Fetch S2 RGB patch as `uint8` RGB (provider path) or convert `input_chw` raw SR -> `[0,1]` -> `uint8`
2. Resize to `RS_EMBED_SATMAE_IMG` (default `224`)
3. Model preprocessing inside adapter:
   - preferred: `model.transform(rgb_u8, image_size)` if wrapper exposes it
   - fallback: generic CLIP-style tensor preprocessing (`rgb_u8_to_tensor_clipnorm`)
4. Run `forward_encoder(mask_ratio=0.0)` to get token sequence `[N,D]`
5. Return:
   - `pooled`: patch-token pooling (`mean` / `max`)
   - `grid`: patch-token reshape to `xarray.DataArray`

Notes:

- Current adapter path always targets token output (not pooled wrapper outputs).
- CLS token is removed automatically when pooling / grid reshape helpers detect it.

---

## Environment Variables / Tuning Knobs

| Env var | Default | Effect |
|---|---|---|
| `RS_EMBED_SATMAE_ID` | `MVRL/satmae-vitlarge-fmow-pretrain-800` | HF model ID used by `SatMAE.from_pretrained(...)` |
| `RS_EMBED_SATMAE_IMG` | `224` | Resize / preprocess image size |
| `RS_EMBED_SATMAE_FETCH_WORKERS` | `8` | Provider prefetch workers for batch APIs |
| `RS_EMBED_SATMAE_BATCH_SIZE` | CPU:`8`, CUDA:`32` | Inference batch size for batch APIs |

---

## Output Semantics

### `OutputSpec.pooled()`

- Pools SatMAE patch tokens using `OutputSpec.pooling`
- Metadata records `pooling="patch_mean"` or `patch_max`, plus `cls_removed`

### `OutputSpec.grid()`

- Reshapes SatMAE token sequence `[N,D]` to `xarray.DataArray` `(D,H,W)`
- Grid is ViT patch-token layout, not georeferenced raster pixels

---

## Examples

### Minimal provider-backed example

```python
from rs_embed import get_embedding, PointBuffer, TemporalSpec, OutputSpec

emb = get_embedding(
    "satmae_rgb",
    spatial=PointBuffer(lon=121.5, lat=31.2, buffer_m=2048),
    temporal=TemporalSpec.range("2022-06-01", "2022-09-01"),
    output=OutputSpec.pooled(),
    backend="gee",
)
```

### Example model/image-size tuning (env-controlled)

```python
# Example (shell):
# export RS_EMBED_SATMAE_ID=MVRL/satmae-vitlarge-fmow-pretrain-800
# export RS_EMBED_SATMAE_IMG=224
```

---

## Common Failure Modes / Debugging

- backend mismatch (`satmae_rgb` is provider-only)
- wrong `input_chw` shape or band order (must be `CHW`, `C=3`, `(B4,B3,B2)`)
- missing `rshf` / incompatible `rshf` version (no `SatMAE` wrapper or `forward_encoder`)
- `grid` requests failing if token output shape is unexpected

Recommended first checks:

- inspect metadata `tokens_shape` and `grid_hw`
- confirm `RS_EMBED_SATMAE_ID` and `RS_EMBED_SATMAE_IMG`
- verify your custom `input_chw` is raw SR (not already 0..1 unless you intentionally converted it)

---

## Reproducibility Notes

Keep fixed and record:

- `RS_EMBED_SATMAE_ID`
- image size (`RS_EMBED_SATMAE_IMG`)
- temporal window and compositing settings
- output mode (`pooled` / `grid`) and pooling choice
- `rshf` version (wrapper preprocessing behavior can matter)

---

## Source of Truth (Code Pointers)

- Registration/catalog: `src/rs_embed/embedders/catalog.py`
- Adapter implementation: `src/rs_embed/embedders/onthefly_satmae.py`
- Shared RGB/token helpers: `src/rs_embed/embedders/_vit_mae_utils.py`

