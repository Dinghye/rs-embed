# SatMAE++ Family (`satmaepp`, `satmaepp_s2_10b`)

> `rs-embed` provides two SatMAE++ adapter paths: RGB (`satmaepp`) and Sentinel-2 10-band (`satmaepp_s2_10b`). This page documents both variants in one place.

## Quick Facts

| Field | `satmaepp` (RGB) | `satmaepp_s2_10b` (S2-10B) |
|---|---|---|
| Canonical ID | `satmaepp` | `satmaepp_s2_10b` |
| Aliases | `satmaepp_rgb`, `satmae++` | `satmaepp_sentinel10`, `satmaepp_s2` |
| Backend | provider (`gee`) | provider (`gee`) |
| Primary input | S2 RGB (`B4,B3,B2`) | S2 SR 10-band (`B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12`) |
| Temporal mode | range window + single composite | range window + single composite |
| Outputs | `pooled`, `grid` | `pooled`, `grid` |
| Core extraction | `forward_encoder(mask_ratio=0.0)` | `forward_encoder(mask_ratio=0.0)` |

---

## Variant A: `satmaepp` (RGB)

### Input contract

- Default `SensorSpec`:
  - `collection="COPERNICUS/S2_SR_HARMONIZED"`
  - `bands=("B4","B3","B2")`
  - `scale_m=10`, `cloudy_pct=30`, `composite="median"`
- `input_chw` must be 3-channel `CHW` in `(B4,B3,B2)` order, using raw S2 SR values in `0..10000`

### Preprocess

1. Fetch RGB `uint8` from provider (or convert `input_chw` with `[0,1] -> uint8`)
2. Apply SatMAE++ fMoW RGB eval preprocessing:
   - optional channel order: `rgb` or `bgr` (default for `fmow_rgb` checkpoints is `bgr`)
   - `ToTensor -> Normalize(mean/std) -> Resize(short side) -> CenterCrop(image_size)`
3. Run `forward_encoder(mask_ratio=0.0)` to extract tokens

### Key env vars

- `RS_EMBED_SATMAEPP_ID`
- `RS_EMBED_SATMAEPP_IMG`
- `RS_EMBED_SATMAEPP_CHANNEL_ORDER`
- `RS_EMBED_SATMAEPP_BGR` (legacy)
- `RS_EMBED_SATMAEPP_FETCH_WORKERS`
- `RS_EMBED_SATMAEPP_BATCH_SIZE`

---

## Variant B: `satmaepp_s2_10b` (Sentinel-2 10-band)

### Input contract

- Default `SensorSpec`:
  - `collection="COPERNICUS/S2_SR_HARMONIZED"`
  - `bands=("B2","B3","B4","B5","B6","B7","B8","B8A","B11","B12")`
  - `scale_m=10`, `cloudy_pct=30`, `composite="median"`, `fill_value=0.0`
- Strict requirement: `sensor.bands` must exactly match the 10-band order above
- `input_chw` must be 10-channel `CHW`, using raw S2 SR values in `0..10000`

### Preprocess + runtime loading

1. Fetch 10-band `CHW` (or use `input_chw`)
2. Apply source-style Sentinel statistics mapping (`mean ± 2*std`) to `uint8`
3. `ToTensor -> Resize(short side) -> CenterCrop(image_size)`
4. Download runtime weights + source code and construct grouped-channel model:
   - channel groups: `((0,1,2,6),(3,4,5,7),(8,9))`
5. Run `forward_encoder(mask_ratio=0.0)` to extract tokens

### Key env vars

- `RS_EMBED_SATMAEPP_S2_CKPT_REPO`
- `RS_EMBED_SATMAEPP_S2_CKPT_FILE`
- `RS_EMBED_SATMAEPP_S2_CODE_REPO`
- `RS_EMBED_SATMAEPP_S2_CODE_REF`
- `RS_EMBED_SATMAEPP_S2_MODEL_FN`
- `RS_EMBED_SATMAEPP_S2_IMG`
- `RS_EMBED_SATMAEPP_S2_PATCH`
- `RS_EMBED_SATMAEPP_S2_GRID_REDUCE`
- `RS_EMBED_SATMAEPP_S2_WEIGHTS_ONLY`
- `RS_EMBED_SATMAEPP_S2_FETCH_WORKERS`
- `RS_EMBED_SATMAEPP_S2_BATCH_SIZE`

---

## Output Semantics

### `OutputSpec.pooled()`

- Both variants pool patch tokens with `mean/max`
- `satmaepp`: `pooling` metadata values like `patch_mean` / `patch_max`
- `satmaepp_s2_10b`: `pooling` metadata values like `group_tokens_mean` / `group_tokens_max`

### `OutputSpec.grid()`

- `satmaepp`: standard ViT patch-token grid `(D,H,W)`
- `satmaepp_s2_10b`: grouped tokens are reduced across groups (`mean/max`) then reshaped to `(D,H,W)`
- Both are model token grids, not georeferenced raster grids

---

## Minimal Examples

```python
from rs_embed import get_embedding, PointBuffer, TemporalSpec, OutputSpec

spatial = PointBuffer(lon=121.5, lat=31.2, buffer_m=2048)
temporal = TemporalSpec.range("2022-06-01", "2022-09-01")

emb_rgb = get_embedding("satmaepp", spatial=spatial, temporal=temporal, output=OutputSpec.pooled(), backend="gee")
emb_s2  = get_embedding("satmaepp_s2_10b", spatial=spatial, temporal=temporal, output=OutputSpec.pooled(), backend="gee")
```

---

## Source of Truth

- `src/rs_embed/embedders/catalog.py`
- `src/rs_embed/embedders/onthefly_satmaepp.py`
- `src/rs_embed/embedders/onthefly_satmaepp_s2.py`
