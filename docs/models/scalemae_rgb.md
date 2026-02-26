# ScaleMAE RGB (`scalemae_rgb`)

> Sentinel-2 RGB on-the-fly adapter for ScaleMAE (`rshf.scalemae.ScaleMAE`), with explicit scale conditioning via `sensor.scale_m -> input_res_m`.

## Quick Facts

| Field | Value |
|---|---|
| Model ID | `scalemae_rgb` |
| Family / Backbone | ScaleMAE via `rshf.scalemae.ScaleMAE` |
| Adapter type | `on-the-fly` |
| Typical backend | provider backend (`gee`) |
| Primary input | S2 RGB (`B4,B3,B2`) + `input_res_m` |
| Temporal mode | range window in practice (normalized via shared helper) |
| Output modes | `pooled`, `grid` |
| Extra side inputs | **required semantic scale** (`sensor.scale_m` passed as `input_res_m`) |
| Training alignment (adapter path) | Medium-High when `sensor.scale_m` matches the actual input resolution semantics |

---

## When To Use This Model

### Good fit for

- RGB experiments where spatial scale conditioning matters
- comparisons against SatMAE/RemoteCLIP with a scale-aware backbone
- benchmarking robustness to resolution changes (while explicitly logging `scale_m`)

### Be careful when

- `sensor.scale_m` is missing/incorrect for your input patch
- assuming `grid` is always available (some wrapper outputs may be pooled vectors only)
- comparing results without recording model output type (`tokens_kind`)

---

## Input Contract (Current Adapter Path)

### Spatial / temporal

- Provider backend only (`backend="gee"` / provider-compatible backend)
- `TemporalSpec` normalized via shared helper; use `TemporalSpec.range(...)` for reproducibility

### Sensor / channels + scale

Default `SensorSpec` if omitted:

- Collection: `COPERNICUS/S2_SR_HARMONIZED`
- Bands: `("B4", "B3", "B2")`
- `scale_m=10`, `cloudy_pct=30`, `composite="median"`

`input_chw` contract:

- must be `CHW` with 3 channels in `(B4,B3,B2)` order
- raw S2 SR values expected in `0..10000`

Scale requirement:

- adapter passes `float(sensor.scale_m)` to ScaleMAE as `input_res_m`
- if `sensor.scale_m` does not reflect actual patch resolution semantics, embeddings are not comparable

---

## Preprocessing Pipeline (Current rs-embed Path)

1. Fetch S2 RGB patch (provider path) or convert `input_chw` raw SR -> `[0,1]` -> `uint8`
2. Resize to `RS_EMBED_SCALEMAE_IMG` (default `224`)
3. Convert to tensor with CLIP-style preprocessing (`rgb_u8_to_tensor_clipnorm`)
4. Build `input_res` tensor from `sensor.scale_m`
5. Call ScaleMAE `forward_features(...)` (preferred) or `forward(...)`
   - adapter handles signature differences across `rshf` versions
   - passes both `patch_size` and `input_res`
6. Normalize output format:
   - `[N,D]` tokens
   - `[D]` pooled vector
   - `[C,H,W]` feature map reshaped to tokens
7. Return pooled vector or token grid

Important:

- `grid` output requires token sequence (`[N,D]` after adapter normalization).
- If the model/wrapper returns pooled vectors only, `OutputSpec.grid()` raises a clear error.

---

## Environment Variables / Tuning Knobs

| Env var | Default | Effect |
|---|---|---|
| `RS_EMBED_SCALEMAE_ID` | `MVRL/scalemae-vitlarge-800` | HF model ID for `ScaleMAE.from_pretrained(...)` |
| `RS_EMBED_SCALEMAE_IMG` | `224` | Resize / preprocess image size |
| `RS_EMBED_SCALEMAE_FETCH_WORKERS` | `8` | Provider prefetch workers for batch APIs |
| `RS_EMBED_SCALEMAE_BATCH_SIZE` | CPU:`8`, CUDA:`32` | Inference batch size for batch APIs |

Non-env but critical:

- `sensor.scale_m` (used as `input_res_m`)

---

## Output Semantics

### `OutputSpec.pooled()`

- If adapter gets token sequence `[N,D]`, it pools patch tokens (`mean` / `max`)
- If adapter gets pooled vector `[D]`, it returns it directly (`pooling="model_pooled"`)
- Metadata includes `tokens_kind`, `used_patch_size`, and `used_scale_m`

### `OutputSpec.grid()`

- Requires token sequence output after adapter normalization
- Returns patch-token grid as `xarray.DataArray` `(D,H,W)`
- Grid is model token layout, not georeferenced raster pixels

---

## Examples

### Minimal provider-backed example

```python
from rs_embed import get_embedding, PointBuffer, TemporalSpec, OutputSpec

emb = get_embedding(
    "scalemae_rgb",
    spatial=PointBuffer(lon=121.5, lat=31.2, buffer_m=2048),
    temporal=TemporalSpec.range("2022-06-01", "2022-09-01"),
    output=OutputSpec.pooled(),
    backend="gee",
)
```

### Example tuning (env + scale semantics)

```python
# Example (shell):
# export RS_EMBED_SCALEMAE_ID=MVRL/scalemae-vitlarge-800
# export RS_EMBED_SCALEMAE_IMG=224
#
# In code, keep sensor.scale_m correct (this is passed to ScaleMAE as input_res_m).
```

---

## Common Failure Modes / Debugging

- backend mismatch (`scalemae_rgb` is provider-only)
- wrong `input_chw` shape / band order (`CHW`, 3 channels, `(B4,B3,B2)`)
- missing `rshf.scalemae.ScaleMAE`
- wrapper signature mismatch in older/newer `rshf` versions (adapter has fallbacks, but still possible)
- `grid` requested when model output is pooled vector only
- incorrect `sensor.scale_m` causing silent comparison drift

Recommended first checks:

- inspect metadata `tokens_kind`, `used_patch_size`, `input_res_m` / `used_scale_m`
- verify `sensor.scale_m` and `RS_EMBED_SCALEMAE_IMG`
- start with `OutputSpec.pooled()` before debugging `grid`

---

## Reproducibility Notes

Keep fixed and record:

- `RS_EMBED_SCALEMAE_ID`
- `RS_EMBED_SCALEMAE_IMG`
- `sensor.scale_m` (critical)
- temporal window + compositing settings
- output mode and pooling choice
- `rshf` version

---

## Source of Truth (Code Pointers)

- Registration/catalog: `src/rs_embed/embedders/catalog.py`
- Adapter implementation: `src/rs_embed/embedders/onthefly_scalemae.py`
- Shared RGB/token helpers: `src/rs_embed/embedders/_vit_mae_utils.py`

