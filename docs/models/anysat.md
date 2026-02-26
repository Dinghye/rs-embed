# AnySat (`anysat`)

> Multi-frame Sentinel-2 time-series adapter that builds AnySat inputs (`s2` + `s2_dates`) from a temporal window and returns patch-grid features or pooled vectors.

## Quick Facts

| Field | Value |
|---|---|
| Model ID | `anysat` |
| Family / Backbone | AnySat (loaded from upstream `hubconf.py`) |
| Adapter type | `on-the-fly` |
| Typical backend | provider backend (`gee` via public API) |
| Primary input | S2 10-band time series (`T,C,H,W`) |
| Temporal mode | `range` in practice (adapter normalizes `year`/`None` to range) |
| Output modes | `pooled`, `grid` |
| Extra side inputs | **required** `s2_dates` (per-frame DOY values) |
| Training alignment (adapter path) | Medium (depends on frame count, normalization mode, and image size) |

---

## When To Use This Model

### Good fit for

- temporal S2 sequence modeling (not just single composites)
- experiments where day-of-year context matters
- comparing multi-frame adapters (`anysat`, `galileo`, `agrifm`)

### Be careful when

- comparing to single-composite models without matching temporal design assumptions
- changing frame count / normalization mode without recording it
- assuming `grid` is a georeferenced raster (it is patch output mapped to `(D,H,W)`)

---

## Input Contract (Current Adapter Path)

### Spatial / temporal

- `SpatialSpec`: `BBox` or `PointBuffer`
- `TemporalSpec`: normalized to range via shared helper (`range` preferred for reproducibility)
- Adapter splits the temporal window into `T` sub-windows and composites one frame per bin
- Frame dates are converted to AnySat-style day-of-year values (`0..364`) and passed as `s2_dates`

### Sensor / channels

Default `SensorSpec` if omitted:

- Collection: `COPERNICUS/S2_SR_HARMONIZED`
- Bands: `("B2","B3","B4","B5","B6","B7","B8","B8A","B11","B12")`
- `scale_m=10`, `cloudy_pct=30`, `composite="median"`, `fill_value=0.0`

`input_chw` contract:

- accepts `CHW` (`C=10`) or `TCHW` (`C=10`)
- `CHW` is repeated to `T` frames
- `TCHW` is padded/truncated to exact `T`
- values are clipped to raw-SR range `0..10000`

---

## Preprocessing Pipeline (Current rs-embed Path)

1. Fetch S2 10-band time series `raw_tchw` in `[T,C,H,W]` (or coerce `input_chw` to `TCHW`)
2. Optionally resize all frames to square `RS_EMBED_ANYSAT_IMG` (default `24`)
3. Normalize series using `RS_EMBED_ANYSAT_NORM`:
   - `per_tile_zscore` (default)
   - `unit_scale` / `reflectance` (`/10000 -> [0,1]`)
   - `raw` / `none`
4. Build AnySat side input dict:
   - `s2`: `[1,T,10,H,W]`
   - `s2_dates`: `[1,T]` (DOY values from frame-bin midpoints)
5. Forward with `output="patch"` and `patch_size=output.scale_m`
6. Map AnySat patch output `[B,H,W,D]` -> rs-embed grid `[D,H,W]`
7. Optionally spatial-pool grid to vector (`mean` or `max`)

Important constraint:

- `output.scale_m` / patch size must be a **positive multiple of 10 meters**

---

## Environment Variables / Tuning Knobs

| Env var | Default | Effect |
|---|---|---|
| `RS_EMBED_ANYSAT_FRAMES` | `8` | Number of temporal frames `T` |
| `RS_EMBED_ANYSAT_IMG` | `24` | Per-frame resize target (square) |
| `RS_EMBED_ANYSAT_NORM` | `per_tile_zscore` | Series normalization mode |
| `RS_EMBED_ANYSAT_MODEL_SIZE` | `base` | AnySat model size |
| `RS_EMBED_ANYSAT_FLASH_ATTN` | `0` | Enable flash attention path if supported |
| `RS_EMBED_ANYSAT_PRETRAINED` | `1` | Use pretrained model |
| `RS_EMBED_ANYSAT_CKPT` | unset | Local checkpoint override |
| `RS_EMBED_ANYSAT_REPO_PATH` | unset | Local AnySat repo path (`hubconf.py`) |
| `RS_EMBED_ANYSAT_REPO_URL` | upstream AnySat Git URL | Repo clone source when auto-downloading |
| `RS_EMBED_ANYSAT_REPO_CACHE` | `~/.cache/rs_embed/anysat` | Repo cache root |
| `RS_EMBED_ANYSAT_AUTO_DOWNLOAD_REPO` | `1` | Auto-clone AnySat repo if missing |
| `RS_EMBED_ANYSAT_FETCH_WORKERS` | `8` | Provider prefetch workers for batch APIs |

---

## Output Semantics

### `OutputSpec.pooled()`

- Pools patch grid over spatial dims `(H,W)`
- Supports `mean` and `max` pooling (`patch_mean` / `patch_max` semantics)

### `OutputSpec.grid()`

- Returns patch features as `xarray.DataArray` `(D,H,W)`
- Metadata includes `grid_hw`, `grid_shape`, patch output info, and temporal packaging details (`n_frames`, `doy0_values`)
- This is model patch output structure, not guaranteed georeferenced raster pixels

---

## Examples

### Minimal example

```python
from rs_embed import get_embedding, PointBuffer, TemporalSpec, OutputSpec

emb = get_embedding(
    "anysat",
    spatial=PointBuffer(lon=121.5, lat=31.2, buffer_m=2048),
    temporal=TemporalSpec.range("2022-01-01", "2023-01-01"),
    output=OutputSpec.pooled(),
    backend="gee",
)
```

### Example with temporal/frame tuning (env-controlled)

```python
# Example (shell):
# export RS_EMBED_ANYSAT_FRAMES=8
# export RS_EMBED_ANYSAT_NORM=per_tile_zscore
# export RS_EMBED_ANYSAT_IMG=24
```

---

## Common Failure Modes / Debugging

- backend is not provider-compatible
- wrong channel count for `input_chw` (must be 10 channels)
- `output.scale_m` is not a positive multiple of 10
- AnySat repo unavailable and auto-download disabled (`RS_EMBED_ANYSAT_AUTO_DOWNLOAD_REPO=0`)
- local checkpoint missing / too small / invalid format
- inconsistent results from untracked changes to `FRAMES`, `NORM`, or image size

Recommended first check:

- Inspect input patches and confirm temporal window/frame construction is what you intended.

---

## Reproducibility Notes

For fair comparisons and stable reruns, record:

- temporal window (`TemporalSpec.range(...)`)
- `RS_EMBED_ANYSAT_FRAMES`
- `RS_EMBED_ANYSAT_NORM`
- `RS_EMBED_ANYSAT_IMG`
- `output.scale_m` (patch size)
- AnySat repo/ckpt source (`repo_path`, repo URL/cache, pretrained vs local ckpt)

---

## Source of Truth (Code Pointers)

- Registration/catalog: `src/rs_embed/embedders/catalog.py`
- Adapter implementation: `src/rs_embed/embedders/onthefly_anysat.py`
- TCHW coercion helper: `src/rs_embed/embedders/runtime_utils.py`
