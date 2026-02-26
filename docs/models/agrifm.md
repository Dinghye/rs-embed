# AgriFM (`agrifm`)

> Multi-frame Sentinel-2 adapter for AgriFM (Video Swin-style backbone), with fixed-frame temporal packaging and provider-backed S2 sequence fetching.

## Quick Facts

| Field | Value |
|---|---|
| Model ID | `agrifm` |
| Family / Backbone | AgriFM (repo + checkpoint loader, lightweight shim path supported) |
| Adapter type | `on-the-fly` |
| Typical backend | provider backend (`gee`) |
| Primary input | S2 SR 10-band time series (`T,10,H,W`) |
| Temporal mode | `range` in practice (window split into `T` frames) |
| Output modes | `pooled`, `grid` |
| Extra side inputs | none required (adapter builds fixed frame stack) |
| Training alignment (adapter path) | High when `n_frames`, normalization, and checkpoint/repo versions are fixed |

---

## When To Use This Model

### Good fit for

- crop/agriculture-oriented temporal S2 experiments
- comparisons against other multi-frame adapters (`anysat`, `galileo`)
- workflows where a consistent fixed-length frame stack is useful

### Be careful when

- changing `RS_EMBED_AGRIFM_FRAMES` across experiments (changes temporal packaging)
- feeding `CHW` inputs and forgetting they get repeated to `T` frames
- comparing with single-window models without documenting temporal/frame construction

---

## Input Contract (Current Adapter Path)

### Spatial / temporal

- Provider backend only (`backend="gee"` / provider-compatible backend)
- `TemporalSpec` normalized to range via shared helper (`TemporalSpec.range(...)` recommended)
- Provider path fetches a multi-frame S2 sequence over the range and composes it into fixed `T` frames

### Sensor / channels

Default `SensorSpec` if omitted:

- Collection: `COPERNICUS/S2_SR_HARMONIZED`
- Bands: `B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12`
- `scale_m=10`, `cloudy_pct=30`, `composite="median"`, `fill_value=0.0`

`input_chw` contract:

- accepts `CHW` with `C=10`
  - adapter repeats the same frame to `T = RS_EMBED_AGRIFM_FRAMES`
- accepts `TCHW` with `C=10`
  - adapter pads (repeat last frame) or truncates to exact `T`
- values are clipped to raw S2 SR range `0..10000`

---

## Preprocessing Pipeline (Current rs-embed Path)

1. Resolve runtime settings:
   - `n_frames`, `image_size`, normalization mode
   - checkpoint path (local or auto-download)
   - AgriFM repo path (local or auto-clone)
2. Fetch provider multi-frame raw `TCHW` or coerce `input_chw` (`CHW`/`TCHW`) to exact `T`
3. Optional input inspection on frame 0 (temporarily scaled to `[0,1]`)
4. Normalize with `RS_EMBED_AGRIFM_NORM`:
   - `agrifm_stats` (default): z-score with AgriFM S2 statistics
   - `unit_scale`: divide by `10000` and clip `[0,1]`
   - `none` / `raw`: keep raw `0..10000` (clipped)
5. Resize all frames to `RS_EMBED_AGRIFM_IMG` (default `224`)
6. Load AgriFM model (checkpoint + repo code)
7. Forward to produce embedding grid `(D,H,W)`
8. Return:
   - pooled vector = spatial mean/max over grid
   - grid = `xarray.DataArray`

---

## Environment Variables / Tuning Knobs

### Temporal / preprocessing

| Env var | Default | Effect |
|---|---|---|
| `RS_EMBED_AGRIFM_FRAMES` | `8` | Fixed frame count `T` |
| `RS_EMBED_AGRIFM_IMG` | `224` | Resize target image size |
| `RS_EMBED_AGRIFM_NORM` | `agrifm_stats` | `agrifm_stats`, `unit_scale`, or `none` |
| `RS_EMBED_AGRIFM_FETCH_WORKERS` | `8` | Provider prefetch workers for batch APIs |

### Checkpoint loading

| Env var | Default | Effect |
|---|---|---|
| `RS_EMBED_AGRIFM_CKPT` | unset | Local checkpoint path |
| `RS_EMBED_AGRIFM_AUTO_DOWNLOAD` | `1` | Allow checkpoint auto-download |
| `RS_EMBED_AGRIFM_CACHE_DIR` | `~/.cache/rs_embed/agrifm` | Checkpoint cache dir |
| `RS_EMBED_AGRIFM_CKPT_FILE` | `AgriFM.pth` | Cached checkpoint filename |
| `RS_EMBED_AGRIFM_CKPT_URL` | project default URL | Checkpoint download URL |
| `RS_EMBED_AGRIFM_CKPT_MIN_BYTES` | large-size threshold | Download validation threshold |

### Repo loading (AgriFM source code)

| Env var | Default | Effect |
|---|---|---|
| `RS_EMBED_AGRIFM_REPO_PATH` | unset | Local AgriFM repo path |
| `RS_EMBED_AGRIFM_REPO_URL` | `https://github.com/flyakon/AgriFM` | Repo clone source |
| `RS_EMBED_AGRIFM_REPO_CACHE` | `~/.cache/rs_embed` | Repo cache root |
| `RS_EMBED_AGRIFM_AUTO_DOWNLOAD_REPO` | `1` | Auto-clone repo when missing |

---

## Output Semantics

### `OutputSpec.pooled()`

- Adapter pools the AgriFM grid spatially:
  - `mean` -> spatial mean over `(H,W)`
  - `max` -> spatial max over `(H,W)`
- Metadata records `pooling="spatial_mean"` or `pooling="spatial_max"`

### `OutputSpec.grid()`

- Returns AgriFM feature grid as `xarray.DataArray` `(D,H,W)`
- Grid is model feature map space (not georeferenced raster pixels)
- Metadata includes `n_frames`, `grid_hw`, normalization mode, and input sizes

---

## Examples

### Minimal provider-backed example

```python
from rs_embed import get_embedding, PointBuffer, TemporalSpec, OutputSpec

emb = get_embedding(
    "agrifm",
    spatial=PointBuffer(lon=121.5, lat=31.2, buffer_m=2048),
    temporal=TemporalSpec.range("2022-01-01", "2022-12-31"),
    output=OutputSpec.pooled(),
    backend="gee",
)
```

### Example temporal packaging and normalization tuning

```python
# Example (shell):
# export RS_EMBED_AGRIFM_FRAMES=8
# export RS_EMBED_AGRIFM_IMG=224
# export RS_EMBED_AGRIFM_NORM=agrifm_stats
```

---

## Common Failure Modes / Debugging

- backend mismatch (`agrifm` is provider-only)
- wrong `input_chw` shape (must be `CHW`/`TCHW` with `C=10`)
- silent temporal mismatch from `CHW` repeat-to-`T` behavior
- checkpoint download failure or invalid local checkpoint path
- AgriFM repo missing when auto-download is disabled
- missing lightweight import deps (`torch`, `timm`, `einops`) for repo backbone import

Recommended first checks:

- inspect metadata for `n_frames`, `norm_mode`, `input_frames`, `grid_hw`
- verify whether input came from provider fetch vs repeated `CHW`
- pin checkpoint and repo path before benchmarking

---

## Reproducibility Notes

Keep fixed and record:

- `RS_EMBED_AGRIFM_FRAMES`, `RS_EMBED_AGRIFM_IMG`, normalization mode
- checkpoint source/path and repo source/path
- temporal window and compositing settings
- output mode / pooling choice

Temporal note:

- Multi-frame models are very sensitive to frame count and frame construction; changing `FRAMES` is a different experiment.

---

## Source of Truth (Code Pointers)

- Registration/catalog: `src/rs_embed/embedders/catalog.py`
- Adapter implementation: `src/rs_embed/embedders/onthefly_agrifm.py`
- Shared temporal fetch/coercion helpers: `src/rs_embed/embedders/runtime_utils.py`

