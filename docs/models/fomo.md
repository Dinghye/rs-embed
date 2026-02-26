# FoMo (`fomo`)

> Multispectral Sentinel-2 adapter for FoMo-Bench (MultiSpectralViT), with explicit S2 modality-key mapping and token/grid outputs averaged across spectral modalities.

## Quick Facts

| Field | Value |
|---|---|
| Model ID | `fomo` |
| Family / Backbone | FoMo-Bench `MultiSpectralViT` (repo + checkpoint loader) |
| Adapter type | `on-the-fly` |
| Typical backend | provider backend (`gee`) |
| Primary input | S2 SR 12-band (`B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B11,B12`) |
| Temporal mode | `range` in practice (normalized via shared helper) |
| Output modes | `pooled`, `grid` |
| Extra side inputs | **required modality keys** (adapter provides S2 defaults, configurable via env) |
| Training alignment (adapter path) | Medium-High when `S2_KEYS`, normalization, and model config match checkpoint assumptions |

---

## When To Use This Model

### Good fit for

- strict multispectral S2 experiments with spectral-token modeling
- testing modality-aware token layouts beyond plain ViT RGB pipelines
- analyses where spectral-token averaging into a spatial grid is acceptable

### Be careful when

- changing `RS_EMBED_FOMO_S2_KEYS` without understanding FoMo modality indexing
- comparing to raster-like models and treating FoMo grid as georeferenced pixels
- modifying model architecture envs (`DIM/DEPTH/HEADS/...`) while using incompatible checkpoints

---

## Input Contract (Current Adapter Path)

### Spatial / temporal

- Provider backend only (`backend="gee"` / provider-compatible backend)
- `TemporalSpec` normalized via shared helper; use `TemporalSpec.range(...)`

### Sensor / channels

Default `SensorSpec` if omitted:

- Collection: `COPERNICUS/S2_SR_HARMONIZED`
- Bands: `B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B11,B12`
- `scale_m=10`, `cloudy_pct=30`, `composite="median"`, `fill_value=0.0`

`input_chw` contract:

- must be `CHW` with `C=12` in the adapter S2 band order above
- expected raw S2 SR values in `0..10000`

### Modality keys

- FoMo forward path requires a list of modality keys for each channel
- adapter default S2 mapping is encoded via `_DEFAULT_S2_MODALITY_KEYS`
- override with `RS_EMBED_FOMO_S2_KEYS` (must provide exactly 12 comma-separated integers)

---

## Preprocessing Pipeline (Current rs-embed Path)

1. Fetch S2 SR 12-band patch (provider path) or use `input_chw`
2. Normalize with `RS_EMBED_FOMO_NORM`:
   - `unit_scale` (default): divide by `10000`
   - `per_tile_minmax`: per-channel tile min-max after unit scaling
   - `none` / `raw`: keep raw `0..10000` (clipped)
3. Resize to `RS_EMBED_FOMO_IMG` (default `64`)
4. Build/load FoMo model from:
   - FoMo-Bench repo code (local / auto-clone)
   - checkpoint (local / auto-download)
5. Forward with `(x, spectral_keys)` and `pool=False` to get tokens `[N,D]`
6. Compute outputs:
   - pooled vector: mean/max over all tokens
   - grid: average tokens across modalities and reshape patch grid when possible
   - fallback: `1x1` vector grid if token layout is not grid-compatible

---

## Environment Variables / Tuning Knobs

### Core model / preprocessing

| Env var | Default | Effect |
|---|---|---|
| `RS_EMBED_FOMO_IMG` | `64` | Resize target image size |
| `RS_EMBED_FOMO_PATCH` | `16` | Patch size (used for model config + grid expectations) |
| `RS_EMBED_FOMO_NORM` | `unit_scale` | `unit_scale`, `per_tile_minmax`, or `none` |
| `RS_EMBED_FOMO_S2_KEYS` | adapter S2 default mapping | 12 comma-separated modality keys |
| `RS_EMBED_FOMO_FETCH_WORKERS` | `8` | Provider prefetch workers for batch APIs |

### FoMo model config (advanced; keep aligned with checkpoint)

| Env var | Default | Effect |
|---|---|---|
| `RS_EMBED_FOMO_DIM` | `768` | Model dim |
| `RS_EMBED_FOMO_DEPTH` | `12` | Transformer depth |
| `RS_EMBED_FOMO_HEADS` | `12` | Attention heads |
| `RS_EMBED_FOMO_MLP_DIM` | `2048` | MLP dim |
| `RS_EMBED_FOMO_NUM_CLASSES` | `1000` | Class head size (config compatibility) |

### Checkpoint / repo loading

| Env var | Default | Effect |
|---|---|---|
| `RS_EMBED_FOMO_CKPT` | unset | Local checkpoint path |
| `RS_EMBED_FOMO_AUTO_DOWNLOAD` | `1` | Allow checkpoint auto-download |
| `RS_EMBED_FOMO_CACHE_DIR` | repo cache default | Checkpoint cache dir |
| `RS_EMBED_FOMO_CKPT_FILE` | default FoMo checkpoint filename | Cached ckpt filename |
| `RS_EMBED_FOMO_CKPT_URL` | default Dropbox URL | Checkpoint download URL |
| `RS_EMBED_FOMO_CKPT_MIN_BYTES` | adapter threshold | Download size sanity check |
| `RS_EMBED_FOMO_REPO_PATH` | unset | Local FoMo-Bench repo path |
| `RS_EMBED_FOMO_REPO_URL` | FoMo-Bench GitHub URL | Repo clone source |
| `RS_EMBED_FOMO_REPO_CACHE` | `~/.cache/rs_embed/fomo` | Repo cache root |
| `RS_EMBED_FOMO_AUTO_DOWNLOAD_REPO` | `1` | Auto-clone FoMo-Bench repo |

---

## Output Semantics

### `OutputSpec.pooled()`

- Pools FoMo token sequence across tokens:
  - `mean` -> `token_mean`
  - `max` -> `token_max`
- Metadata records `token_count`, `token_dim`, and pooling mode

### `OutputSpec.grid()`

- Preferred path:
  - interpret tokens as `[modalities, H, W, D]`
  - average over modalities
  - return `(D,H,W)` grid with `grid_kind="spectral_mean_patch_tokens"`
- Fallback path:
  - if token layout is incompatible with expected modality/grid layout, return `1x1` vector grid (`grid_kind="vector_as_1x1"`)

Grid is model token-derived structure, not georeferenced raster pixels.

---

## Examples

### Minimal provider-backed example

```python
from rs_embed import get_embedding, PointBuffer, TemporalSpec, OutputSpec

emb = get_embedding(
    "fomo",
    spatial=PointBuffer(lon=121.5, lat=31.2, buffer_m=2048),
    temporal=TemporalSpec.range("2022-06-01", "2022-09-01"),
    output=OutputSpec.pooled(),
    backend="gee",
)
```

### Example FoMo tuning (env-controlled)

```python
# Example (shell):
# export RS_EMBED_FOMO_IMG=64
# export RS_EMBED_FOMO_PATCH=16
# export RS_EMBED_FOMO_NORM=unit_scale
# export RS_EMBED_FOMO_S2_KEYS=6,7,8,9,10,11,12,13,14,15,17,18
```

---

## Common Failure Modes / Debugging

- backend mismatch (`fomo` is provider-only)
- wrong `input_chw` shape (`C` must be `12`)
- `RS_EMBED_FOMO_S2_KEYS` length mismatch (must be 12 values)
- checkpoint/repo download/import failures
- changed model config envs incompatible with checkpoint architecture
- grid returns `1x1` fallback because token layout is not modality-grid compatible

Recommended first checks:

- inspect metadata `spectral_keys`, `token_count`, `grid_kind`, `grid_expected_tokens`
- revert architecture envs to defaults before benchmarking
- verify normalization mode and image/patch sizes

---

## Reproducibility Notes

Keep fixed and record:

- checkpoint source/path and FoMo-Bench repo source/path
- `IMG`, `PATCH`, normalization mode
- `S2_KEYS` mapping
- model config envs (`DIM/DEPTH/HEADS/MLP_DIM/NUM_CLASSES`)
- temporal window + compositing settings

---

## Source of Truth (Code Pointers)

- Registration/catalog: `src/rs_embed/embedders/catalog.py`
- Adapter implementation: `src/rs_embed/embedders/onthefly_fomo.py`

