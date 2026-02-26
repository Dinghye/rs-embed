# THOR 1.0 Base (`thor_1_0_base`)

> TerraTorch-backed THOR adapter for Sentinel-2 SR 10-band inputs, with THOR-specific normalization and flexible group-grid aggregation (`mean` / `sum` / `concat`).

## Quick Facts

| Field | Value |
|---|---|
| Model ID | `thor_1_0_base` |
| Family / Backbone | THOR via TerraTorch + `thor_terratorch_ext` (`thor_v1_base`) |
| Adapter type | `on-the-fly` |
| Typical backend | provider backend (`gee`) |
| Primary input | S2 SR 10-band `CHW` |
| Temporal mode | `range` in practice (composite window) |
| Output modes | `pooled`, `grid` |
| Extra side inputs | none required in current adapter |
| Training alignment (adapter path) | High when `thor_stats` normalization and default S2 SR setup are preserved |

---

## When To Use This Model

### Good fit for

- strong S2 SR baselines with THOR pretrained weights
- experiments needing both pooled and token-grid outputs
- studies where group-wise THOR token aggregation (`group_merge`) is relevant

### Be careful when

- changing normalization away from `thor_stats` for benchmark comparisons
- changing `patch_size` / `image_size` without logging `ground_cover_m`
- assuming `grid` is always available (some configs may fall back to pooled-only usability)

---

## Input Contract (Current Adapter Path)

### Spatial / temporal

- Provider backend only (`backend="gee"` / provider-compatible backend)
- `TemporalSpec` is normalized to range via shared helper (`TemporalSpec.range(...)` recommended)
- Temporal window is used for compositing, not scene identity locking

### Sensor / channels

Default `SensorSpec` if omitted:

- Collection: `COPERNICUS/S2_SR_HARMONIZED`
- Bands (adapter order): `B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12`
- `scale_m=10`, `cloudy_pct=30`, `composite="median"`, `fill_value=0.0`

`input_chw` contract:

- must be `CHW` with `C=10`
- raw S2 SR values expected in `0..10000`
- adapter clips NaN/Inf and clamps to `0..10000` before normalization

---

## Preprocessing Pipeline (Current rs-embed Path)

1. Fetch S2 SR 10-band composite patch (provider path) or accept `input_chw`
2. Optional input inspection on raw values (`expected_channels=10`, range `0..10000`)
3. Normalize with `RS_EMBED_THOR_NORMALIZE`:
   - `thor_stats` (default): `/10000` then THOR z-score stats
   - `unit_scale`: `/10000` and clip `[0,1]`
   - `none` / `raw`: keep raw `0..10000` (clipped)
4. Resize to `RS_EMBED_THOR_IMG` (default `288`)
5. Build/load THOR backbone via TerraTorch + THOR extension
   - `ground_cover_m = sensor.scale_m * image_size`
   - `patch_size` passed into THOR build params
6. Forward model, extract token sequence `[N,D]`
7. Try building `grid`:
   - first via THOR group-aware token layout (`group_merge`)
   - fallback to generic square patch-token reshape
   - if both fail, `grid` is unavailable

---

## Environment Variables / Tuning Knobs

| Env var | Default | Effect |
|---|---|---|
| `RS_EMBED_THOR_MODEL_KEY` | `thor_v1_base` | THOR backbone key for TerraTorch registry |
| `RS_EMBED_THOR_CKPT` | unset | Local checkpoint path override |
| `RS_EMBED_THOR_PRETRAINED` | `1` | Use pretrained weights (HF default path) |
| `RS_EMBED_THOR_IMG` | `288` | Resize target image size |
| `RS_EMBED_THOR_NORMALIZE` | `thor_stats` | `thor_stats`, `unit_scale`, or `none` |
| `RS_EMBED_THOR_GROUP_MERGE` | `mean` | THOR group-grid aggregation: `mean`, `sum`, `concat` |
| `RS_EMBED_THOR_PATCH_SIZE` | `16` | THOR flexi patch size parameter |
| `RS_EMBED_THOR_FETCH_WORKERS` | `8` | Provider prefetch workers for batch APIs |

Notes:

- `RS_EMBED_THOR_PATCH_SIZE` and `RS_EMBED_THOR_IMG` jointly affect token layout and `ground_cover_m`.
- Changing `group_merge` changes grid channel semantics and dimensionality (especially `concat`).

---

## Output Semantics

### `OutputSpec.pooled()`

- Pools token sequence using `_pool_thor_tokens(...)`
- Uses expected THOR patch-token count when available to avoid pooling non-patch tokens incorrectly
- Metadata records pooling mode and `cls_removed`

### `OutputSpec.grid()`

- Preferred path: THOR group-aware grid (`grid_kind="thor_group_grid"`) built from channel groups
- Fallback path: generic ViT patch-token reshape (`grid_kind="patch_tokens"`)
- Can fail for some model/config/token layouts; adapter raises a clear error suggesting pooled output

---

## Examples

### Minimal provider-backed example

```python
from rs_embed import get_embedding, PointBuffer, TemporalSpec, OutputSpec

emb = get_embedding(
    "thor_1_0_base",
    spatial=PointBuffer(lon=121.5, lat=31.2, buffer_m=2048),
    temporal=TemporalSpec.range("2022-06-01", "2022-09-01"),
    output=OutputSpec.pooled(),
    backend="gee",
)
```

### Example THOR tuning (env-controlled)

```python
# Example (shell):
# export RS_EMBED_THOR_NORMALIZE=thor_stats
# export RS_EMBED_THOR_GROUP_MERGE=mean
# export RS_EMBED_THOR_IMG=288
# export RS_EMBED_THOR_PATCH_SIZE=16
```

---

## Common Failure Modes / Debugging

- missing optional deps (`terratorch`, `thor_terratorch_ext`)
- wrong `input_chw` shape (`C` must be `10`)
- invalid `RS_EMBED_THOR_GROUP_MERGE` (must be `mean` / `sum` / `concat`)
- grid unavailable for chosen config (token layout not square and group parsing failed)
- normalization mismatch causing unstable comparison across runs

Recommended first checks:

- verify metadata fields: `model_key`, `normalization`, `group_merge`, `patch_size`, `ground_cover_m`
- try `OutputSpec.pooled()` to isolate grid-layout issues
- revert to default `thor_stats` + `group_merge=mean` before benchmarking

---

## Reproducibility Notes

Keep fixed and record:

- `RS_EMBED_THOR_MODEL_KEY`, `RS_EMBED_THOR_PRETRAINED`, and local `CKPT` (if used)
- normalization mode
- `image_size` and `patch_size`
- `group_merge` (affects grid representation)
- temporal window and provider compositing settings

---

## Source of Truth (Code Pointers)

- Registration/catalog: `src/rs_embed/embedders/catalog.py`
- Adapter implementation: `src/rs_embed/embedders/onthefly_thor.py`
- Token/grid helpers: `src/rs_embed/embedders/_vit_mae_utils.py`

