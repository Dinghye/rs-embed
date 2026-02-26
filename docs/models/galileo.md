# Galileo (`galileo`)

> Multi-frame Sentinel-2 adapter that constructs Galileo encoder inputs (including `months`) from a temporal window and returns pooled vectors or S2-group patch-token grids.

## Quick Facts

| Field | Value |
|---|---|
| Model ID | `galileo` |
| Family / Backbone | Galileo `Encoder` from official `single_file_galileo.py` |
| Adapter type | `on-the-fly` |
| Typical backend | provider backend (`gee` via public API) |
| Primary input | S2 10-band time series (`T,C,H,W`) |
| Temporal mode | `range` in practice (adapter normalizes via shared helper) |
| Output modes | `pooled`, `grid` |
| Extra side inputs | **required** `months` (per-frame month tokens), plus Galileo masks/tensors built by adapter |
| Training alignment (adapter path) | Medium (depends on `FRAMES`, `IMG`, `PATCH`, normalization, NDVI inclusion) |

---

## When To Use This Model

### Good fit for

- temporal S2 sequence modeling with explicit month tokens
- comparisons against other multi-frame adapters (`anysat`, `agrifm`)
- feature-grid analysis over Galileo S2-related token groups

### Be careful when

- comparing to single-composite models without matching temporal assumptions
- changing `image_size` / `patch_size` inconsistently (`image_size` must divide by `patch_size`)
- changing month handling (`RS_EMBED_GALILEO_MONTH`) without documenting it

---

## Input Contract (Current Adapter Path)

### Spatial / temporal

- `SpatialSpec`: `BBox` or `PointBuffer`
- `TemporalSpec`: normalized to range via shared helper (`range` recommended for reproducibility)
- Adapter builds `T` frames by splitting the temporal window into equal sub-windows and compositing each frame
- Month side input:
  - default: derived from frame-bin midpoints
  - optional override: `RS_EMBED_GALILEO_MONTH` (constant month for all frames)

### Sensor / channels

Default `SensorSpec` if omitted:

- Collection: `COPERNICUS/S2_SR_HARMONIZED`
- Bands: 10-band S2 set used by adapter (`B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12`)
- `scale_m=10`, `cloudy_pct=30`, `composite="median"`, `fill_value=0.0`

`input_chw` contract:

- accepts `CHW` (`C=10`) or `TCHW` (`C=10`) through shared coercion helper
- `CHW` repeats to `T`; `TCHW` pads/truncates to exact `T`
- values are clipped to raw-SR range `0..10000`

---

## Preprocessing Pipeline (Current rs-embed Path)

1. Fetch S2 10-band `raw_tchw` or coerce user `input_chw` to `TCHW`
2. Resolve months sequence from frame bins (or `RS_EMBED_GALILEO_MONTH`)
3. Resize frames to `RS_EMBED_GALILEO_IMG` (default `64`) if needed
4. Normalize S2 series with `RS_EMBED_GALILEO_NORM` (default `unit_scale`)
5. Build Galileo encoder tensors and masks:
   - `s_t_x`, `sp_x`, `t_x`, `st_x`
   - masks `s_t_m`, `sp_m`, `t_m`, `st_m`
   - `months`
6. Optional NDVI channel injection when `RS_EMBED_GALILEO_INCLUDE_NDVI=1` and model space supports NDVI
7. Forward Galileo encoder (`patch_size=RS_EMBED_GALILEO_PATCH`, default `8`)
8. Outputs:
   - pooled vector from visible tokens (`encoder.average_tokens(...)`)
   - grid from S2-related space-time groups, averaged over time/group dimensions

Constraint:

- `image_size % patch_size == 0` is required

---

## Environment Variables / Tuning Knobs

| Env var | Default | Effect |
|---|---|---|
| `RS_EMBED_GALILEO_MODEL_SIZE` | `nano` | Galileo model size selector |
| `RS_EMBED_GALILEO_MODEL_PATH` | unset | Local model/checkpoint path override |
| `RS_EMBED_GALILEO_REPO_PATH` | unset | Local Galileo repo path |
| `RS_EMBED_GALILEO_REPO_URL` | upstream Galileo repo | Repo clone source |
| `RS_EMBED_GALILEO_REPO_CACHE` | `~/.cache/rs_embed/galileo` | Repo cache root |
| `RS_EMBED_GALILEO_AUTO_DOWNLOAD_REPO` | `1` | Auto-clone repo if missing |
| `RS_EMBED_GALILEO_IMG` | `64` | Frame resize target |
| `RS_EMBED_GALILEO_PATCH` | `8` | Encoder patch size |
| `RS_EMBED_GALILEO_FRAMES` | `8` | Number of temporal frames `T` |
| `RS_EMBED_GALILEO_NORM` | `unit_scale` | S2 normalization mode |
| `RS_EMBED_GALILEO_ADD_LN` | `1` | Add layer norm on encoder exit |
| `RS_EMBED_GALILEO_INCLUDE_NDVI` | `1` | Include NDVI channel when supported |
| `RS_EMBED_GALILEO_MONTH` | unset | Force a constant month (1..12) for all frames |
| `RS_EMBED_GALILEO_FETCH_WORKERS` | `8` | Prefetch workers for batch APIs |

---

## Output Semantics

### `OutputSpec.pooled()`

- Default pooled path uses Galileo pooled token output (`token_mean` semantics in metadata)
- If `OutputSpec.pooling="max"`, adapter max-pools the produced grid instead (`grid_max`)

### `OutputSpec.grid()`

- Returns S2-related Galileo space-time-group patch tokens as `xarray.DataArray` `(D,H,W)`
- Grid is produced by selecting S2 groups and averaging over time + channel-group axes
- This is model token structure, not georeferenced raster pixels

---

## Examples

### Minimal example

```python
from rs_embed import get_embedding, PointBuffer, TemporalSpec, OutputSpec

emb = get_embedding(
    "galileo",
    spatial=PointBuffer(lon=121.5, lat=31.2, buffer_m=2048),
    temporal=TemporalSpec.range("2022-01-01", "2023-01-01"),
    output=OutputSpec.pooled(),
    backend="gee",
)
```

### Example tuning temporal packaging (env-controlled)

```python
# Example (shell):
# export RS_EMBED_GALILEO_FRAMES=8
# export RS_EMBED_GALILEO_IMG=64
# export RS_EMBED_GALILEO_PATCH=8
# export RS_EMBED_GALILEO_NORM=unit_scale
# export RS_EMBED_GALILEO_INCLUDE_NDVI=1
```

---

## Common Failure Modes / Debugging

- backend is not provider-compatible
- `image_size` not divisible by `patch_size`
- wrong `input_chw` channel count (must be 10)
- unexpected effects from `RS_EMBED_GALILEO_MONTH` forcing a constant month
- missing local repo/model path when auto-download is disabled

Recommended first checks:

- verify temporal window/frame count and month sequence in metadata
- inspect raw inputs before changing normalization or model settings

---

## Reproducibility Notes

Record and keep fixed:

- temporal window and `RS_EMBED_GALILEO_FRAMES`
- `RS_EMBED_GALILEO_IMG`, `RS_EMBED_GALILEO_PATCH`
- normalization mode / NDVI inclusion
- month override (if any)
- model source (repo/model path, model size)

---

## Source of Truth (Code Pointers)

- Registration/catalog: `src/rs_embed/embedders/catalog.py`
- Adapter implementation: `src/rs_embed/embedders/onthefly_galileo.py`
- Shared TCHW coercion helper: `src/rs_embed/embedders/runtime_utils.py`
