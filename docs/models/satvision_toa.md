# SatVision-TOA (`satvision_toa`)

> Provider-backed SatVision-TOA adapter for 14-channel TOA inputs (default MODIS band order), with channel-aware reflectance/emissive normalization and token/grid outputs.

## Quick Facts

| Field | Value |
|---|---|
| Model ID | `satvision_toa` |
| Family / Backbone | SatVision-TOA checkpoint (HF/local checkpoint loader) |
| Adapter type | `on-the-fly` |
| Typical backend | provider backend (`gee`) |
| Primary input | 14-channel TOA `CHW` (default MODIS/061/MOD021KM band order) |
| Temporal mode | `range` in practice (normalized via shared helper) |
| Output modes | `pooled`, `grid` |
| Extra side inputs | none (but channel calibration settings matter) |
| Training alignment (adapter path) | High only when channel order + calibration match checkpoint expectations |

---

## When To Use This Model

### Good fit for

- TOA-based experiments where you want SatVision checkpoints instead of S2 SR-specific encoders
- MODIS-style provider workflows with explicit control over channel normalization/calibration
- testing pooled vs patch-token grid outputs from the same model path

### Be careful when

- changing `SensorSpec.bands` order without updating `RS_EMBED_SATVISION_TOA_*` channel settings
- mixing raw TOA and unit-scaled inputs without logging the effective normalization mode
- assuming arbitrary 14-channel inputs will work if they do not match checkpoint training semantics

---

## Input Contract (Current Adapter Path)

### Spatial / temporal

- `SpatialSpec`: `BBox` or `PointBuffer`
- `TemporalSpec`: normalized to a range via shared helper (`TemporalSpec.range(...)` recommended)
- Provider backend only (`backend="gee"` / other provider-compatible backend)

### Sensor / channels

Default `SensorSpec` if omitted:

- Collection: `MODIS/061/MOD021KM`
- Bands (strict default order): `1,2,3,26,6,20,7,27,28,29,31,32,33,34`
- `scale_m=1000`, `cloudy_pct=100`, `composite="mosaic"`, `fill_value=0.0`

`input_chw` contract:

- must be `CHW`
- `C` must equal `RS_EMBED_SATVISION_TOA_IN_CHANS` (default `14`)
- adapter checks `len(sensor.bands) == in_chans`
- values may be raw TOA-like or already unit-scaled depending on normalization mode

Important:

- The adapter does not infer semantic channel order from values.
- `sensor.bands` order must match what the checkpoint expects.

---

## Preprocessing Pipeline (Current rs-embed Path)

1. Resolve runtime settings (checkpoint/model ID, `in_chans`, normalization, calibration arrays)
2. Fetch provider patch (`CHW`) or use `input_chw`
3. If provider metadata says input is already unit-scaled, effective norm mode is forced to `unit`
4. Normalize with `RS_EMBED_SATVISION_TOA_NORM`:
   - `auto`: use `[0,1]` directly if values look unit-scaled, else fall back to raw scaling
   - `raw`: channel-wise scaling
     - reflectance channels: divide by `RS_EMBED_SATVISION_TOA_REF_DIV` (default `100`)
     - emissive channels: min-max normalize using `RS_EMBED_SATVISION_TOA_EMISSIVE_MINS/MAXS`
   - `unit`: clip to `[0,1]`
5. Resize to `RS_EMBED_SATVISION_TOA_IMG` (default `128`)
6. Forward model and decode output tensor
7. Return:
   - pooled vector (from token pooling or model-pooled output)
   - grid via token reshape if output is token sequence `[N,D]`

---

## Environment Variables / Tuning Knobs

### Model / weights

| Env var | Default | Effect |
|---|---|---|
| `RS_EMBED_SATVISION_TOA_ID` | SatVision TOA HF model ID | HF model identifier |
| `RS_EMBED_SATVISION_TOA_CKPT` | unset | Local checkpoint path override |
| `RS_EMBED_SATVISION_TOA_AUTO_DOWNLOAD` | `1` | Allow HF download when local checkpoint not set |
| `RS_EMBED_SATVISION_TOA_IMG` | `128` | Resize target image size |
| `RS_EMBED_SATVISION_TOA_IN_CHANS` | `14` | Expected channel count |
| `RS_EMBED_SATVISION_TOA_BATCH_SIZE` | CPU:`2`, CUDA:`8` | Inference batch size (batch APIs) |
| `RS_EMBED_SATVISION_TOA_FETCH_WORKERS` | `8` | Provider prefetch workers (batch APIs) |

### Normalization / calibration

| Env var | Default | Effect |
|---|---|---|
| `RS_EMBED_SATVISION_TOA_NORM` | `auto` | `auto`, `raw`, or `unit` |
| `RS_EMBED_SATVISION_TOA_REFLECTANCE_IDXS` | adapter defaults | Reflectance channel indices |
| `RS_EMBED_SATVISION_TOA_EMISSIVE_IDXS` | adapter defaults | Emissive channel indices |
| `RS_EMBED_SATVISION_TOA_REF_DIV` | `100` | Reflectance divisor |
| `RS_EMBED_SATVISION_TOA_EMISSIVE_MINS` | adapter defaults | Emissive min calibration values |
| `RS_EMBED_SATVISION_TOA_EMISSIVE_MAXS` | adapter defaults | Emissive max calibration values |

### Default sensor overrides (if `sensor` omitted)

| Env var | Default | Effect |
|---|---|---|
| `RS_EMBED_SATVISION_TOA_COLLECTION` | `MODIS/061/MOD021KM` | Default provider collection |
| `RS_EMBED_SATVISION_TOA_BANDS` | default 14-band MODIS order | Override default band list |
| `RS_EMBED_SATVISION_TOA_SCALE_M` | `1000` | Default fetch scale |
| `RS_EMBED_SATVISION_TOA_CLOUDY_PCT` | `100` | Default cloud filter |
| `RS_EMBED_SATVISION_TOA_FILL` | `0` | Default fill value |
| `RS_EMBED_SATVISION_TOA_COMPOSITE` | `mosaic` | Default composite method |

---

## Output Semantics

### `OutputSpec.pooled()`

- If model output is token sequence `[N,D]`, adapter pools patch tokens (`mean` / `max`)
- If model output is already `(D,)`, adapter returns it directly as `model_pooled`
- Metadata records pooling behavior and whether CLS token was removed

### `OutputSpec.grid()`

- Requires token sequence output `[N,D]`
- Reshapes patch tokens to `xarray.DataArray` `(D,H,W)`
- Grid is model patch-token layout, not georeferenced raster pixels

---

## Examples

### Minimal provider-backed example

```python
from rs_embed import get_embedding, PointBuffer, TemporalSpec, OutputSpec

emb = get_embedding(
    "satvision_toa",
    spatial=PointBuffer(lon=121.5, lat=31.2, buffer_m=5000),
    temporal=TemporalSpec.range("2022-07-01", "2022-07-31"),
    output=OutputSpec.pooled(),
    backend="gee",
)
```

### Example normalization tuning (env-controlled)

```python
# Example (shell):
# export RS_EMBED_SATVISION_TOA_NORM=raw
# export RS_EMBED_SATVISION_TOA_IMG=128
# export RS_EMBED_SATVISION_TOA_REF_DIV=100
# export RS_EMBED_SATVISION_TOA_IN_CHANS=14
```

---

## Common Failure Modes / Debugging

- backend is not provider-compatible (`satvision_toa` is provider-only)
- `sensor.bands` count does not match `RS_EMBED_SATVISION_TOA_IN_CHANS`
- calibration list lengths mismatch (`EMISSIVE_IDXS` vs `EMISSIVE_MINS/MAXS`)
- wrong channel order for the chosen checkpoint (results look unstable even if shapes pass)
- grid requested but model output is not a token sequence
- HF download/auth issues when using gated/private checkpoints

Recommended first checks:

- inspect metadata for `norm_mode` and `norm_mode_effective`
- log `sensor.collection`, `sensor.bands`, and calibration env values used
- test `OutputSpec.pooled()` first before `grid`

---

## Reproducibility Notes

Keep fixed and record:

- checkpoint source (`RS_EMBED_SATVISION_TOA_ID` or local `CKPT`)
- exact band order and `in_chans`
- normalization mode plus calibration arrays/divisor
- temporal window + compositing settings
- output mode (`pooled` / `grid`) and pooling choice

---

## Source of Truth (Code Pointers)

- Registration/catalog: `src/rs_embed/embedders/catalog.py`
- Adapter implementation: `src/rs_embed/embedders/onthefly_satvision_toa.py`
- Token/grid helpers: `src/rs_embed/embedders/_vit_mae_utils.py`

