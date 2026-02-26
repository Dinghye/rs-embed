# Prithvi-EO v2 (`prithvi_eo_v2_s2_6b`)

> TerraTorch-backed Prithvi adapter for Sentinel-2 6-band inputs, with required temporal/location coordinate side inputs derived by rs-embed.

## Quick Facts

| Field | Value |
|---|---|
| Model ID | `prithvi_eo_v2_s2_6b` |
| Family / Backbone | Prithvi-EO v2 via TerraTorch `BACKBONE_REGISTRY` |
| Adapter type | `on-the-fly` |
| Typical backend | provider backend (`gee` via public API) |
| Primary input | S2 6-band (`BLUE,GREEN,RED,NIR_NARROW,SWIR_1,SWIR_2`) |
| Temporal mode | `range` preferred; adapter normalizes `year`/`None` to a range |
| Output modes | `pooled`, `grid` |
| Extra side inputs | **required** temporal coords + location coords (derived by adapter) |
| Training alignment (adapter path) | Medium (depends on preprocessing mode and resize/pad choices) |

---

## When To Use This Model

### Good fit for

- multispectral S2 experiments beyond RGB
- token/grid feature inspection with a ViT-like backbone
- comparisons that need explicit time + location conditioning in the forward path

### Be careful when

- comparing to models without side inputs (time/location signals can change behavior)
- changing preprocessing mode (`resize` vs `pad`) without documenting it
- assuming `grid` is georeferenced pixel space (it is token grid)

---

## Input Contract (Current Adapter Path)

### Spatial / temporal

- `SpatialSpec`: `BBox` or `PointBuffer`
- `TemporalSpec`:
  - `range`: used directly
  - `year`: normalized to full-year half-open range `[YYYY-01-01, (YYYY+1)-01-01)`
  - `None`: normalized to adapter default range via shared helper (not recommended for reproducible experiments)
- Adapter derives:
  - temporal coordinates from temporal midpoint date
  - location coordinates from ROI center `(lon, lat)` in EPSG:4326

### Sensor / channels

Default `SensorSpec` if omitted:

- Collection: `COPERNICUS/S2_SR_HARMONIZED`
- Bands: `("BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2")`
- `scale_m=30`, `cloudy_pct=30`, `composite="median"`, `fill_value=0.0`

`input_chw` contract:

- must be `CHW` with 6 bands (raw S2 SR values expected, `0..10000`)
- adapter normalizes to `[0,1]`, clips, and replaces non-finite values

---

## Preprocessing Pipeline (Current rs-embed Path)

1. Fetch 6-band S2 patch from provider (or reuse `input_chw`)
2. Normalize raw SR -> `[0,1]` (`/10000`, clip, `nan_to_num`)
3. Optional input checks and quicklook RGB export (`bands=(2,1,0)`)
4. Apply Prithvi input prep (`_prepare_prithvi_chw`):
   - `resize` to `RS_EMBED_PRITHVI_IMG` (default `224`), or
   - `pad` H/W to multiple of `RS_EMBED_PRITHVI_PATCH_MULT` (default `16`)
5. Compute temporal/date and location side inputs
6. Forward pass to token sequence
7. Convert to pooled vector or patch-token grid

---

## Environment Variables / Tuning Knobs

| Env var | Default | Effect |
|---|---|---|
| `RS_EMBED_PRITHVI_KEY` | `prithvi_eo_v2_100_tl` | TerraTorch model key |
| `RS_EMBED_PRITHVI_PRETRAINED` | `1` | Use pretrained weights vs random init |
| `RS_EMBED_PRITHVI_PREP` | `resize` | Input prep mode: `resize` or `pad` |
| `RS_EMBED_PRITHVI_IMG` | `224` | Target square size for `resize` mode |
| `RS_EMBED_PRITHVI_PATCH_MULT` | `16` | Pad multiple for `pad` mode |
| `RS_EMBED_PRITHVI_FETCH_WORKERS` | `8` | Provider prefetch workers for batch APIs |
| `RS_EMBED_PRITHVI_BATCH_SIZE` | CPU:`4`, CUDA:`16` | Inference batch size for batch APIs |

---

## Output Semantics

### `OutputSpec.pooled()`

- Pools patch tokens using `mean`/`max` according to `OutputSpec.pooling`
- CLS token is removed if present (recorded in metadata)

### `OutputSpec.grid()`

- Returns patch-token grid `(D,H,W)` as `xarray.DataArray`
- Grid metadata includes token grid shape and whether CLS was removed
- Token grid is model-internal spatial structure, not guaranteed georeferenced raster pixels

---

## Examples

### Minimal example (explicit temporal window)

```python
from rs_embed import get_embedding, PointBuffer, TemporalSpec, OutputSpec

emb = get_embedding(
    "prithvi_eo_v2_s2_6b",
    spatial=PointBuffer(lon=121.5, lat=31.2, buffer_m=2048),
    temporal=TemporalSpec.range("2022-06-01", "2022-09-01"),
    output=OutputSpec.pooled(),
    backend="gee",
)
```

### With custom preprocessing mode (env-controlled)

```python
# Example (shell):
# export RS_EMBED_PRITHVI_PREP=pad
# export RS_EMBED_PRITHVI_PATCH_MULT=16
# export RS_EMBED_PRITHVI_PRETRAINED=1
```

---

## Common Failure Modes / Debugging

- non-provider backend passed to adapter path
- wrong `input_chw` channel count (must be 6)
- missing TerraTorch/torch dependencies
- inconsistent comparisons due to hidden changes in `RS_EMBED_PRITHVI_PREP` / `RS_EMBED_PRITHVI_IMG`
- confusion about `year` input semantics (adapter converts to full-year range)

Recommended first check:

- Inspect raw fetched patch and verify 6-band order/value range before debugging model output quality.

---

## Reproducibility Notes

Document and keep fixed:

- temporal specification (prefer explicit `TemporalSpec.range(...)`)
- `sensor.scale_m` (default here is `30`, unlike many S2 adapters using `10`)
- preprocessing mode (`resize` vs `pad`) and related env vars
- output mode (`pooled` vs `grid`) and pooling method
- model key / pretrained flag

---

## Source of Truth (Code Pointers)

- Registration/catalog: `src/rs_embed/embedders/catalog.py`
- Adapter implementation: `src/rs_embed/embedders/onthefly_prithvi.py`
- Shared temporal normalization helper: `src/rs_embed/embedders/meta_utils.py`
