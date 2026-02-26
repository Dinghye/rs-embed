# WildSAT (`wildsat`)

> Sentinel-2 RGB on-the-fly adapter for WildSAT checkpoints, supporting multiple backbone architectures (ViT/ResNet/Swin), optional decoder image-head features, and token-grid fallback behavior.

## Quick Facts

| Field | Value |
|---|---|
| Model ID | `wildsat` |
| Family / Backbone | WildSAT checkpoint loader + torchvision backbones (`vitb16`, `vitl16`, `resnet50`, `swint`) |
| Adapter type | `on-the-fly` |
| Typical backend | provider backend (`gee`) |
| Primary input | S2 RGB (`B4,B3,B2`) |
| Temporal mode | `range` in practice (normalized via shared helper) |
| Output modes | `pooled`, `grid` |
| Extra side inputs | none (but checkpoint/arch/image-head settings matter) |
| Training alignment (adapter path) | Medium (depends on checkpoint source, arch inference, normalization mode, and feature source) |

---

## When To Use This Model

### Good fit for

- experimenting with WildSAT checkpoints and different feature extraction targets
- comparing backbone features vs image-head features (`RS_EMBED_WILDSAT_FEATURE`)
- RGB workflows where you want a single adapter covering ViT/ResNet/Swin variants

### Be careful when

- checkpoint architecture is inferred incorrectly (set `RS_EMBED_WILDSAT_ARCH` explicitly if needed)
- comparing runs with different normalization modes (`minmax` vs `unit_scale`)
- assuming `grid` is always a ViT patch grid (non-ViT or token-disabled paths can return `1x1` vector grid)

---

## Input Contract (Current Adapter Path)

### Spatial / temporal

- Provider backend only (`backend="gee"` / provider-compatible backend)
- `TemporalSpec` normalized via shared helper; use `TemporalSpec.range(...)` for reproducibility

### Sensor / channels

Default `SensorSpec` if omitted:

- Collection: `COPERNICUS/S2_SR_HARMONIZED`
- Bands: `("B4", "B3", "B2")`
- `scale_m=10`, `cloudy_pct=30`, `composite="median"`

`input_chw` contract:

- must be `CHW` with `C=3` in `(B4,B3,B2)` order
- expected raw values in `0..10000`
- adapter clips NaN/Inf and converts to `uint8` RGB after normalization

---

## Preprocessing Pipeline (Current rs-embed Path)

1. Resolve checkpoint path (local or auto-download)
2. Fetch S2 RGB patch (provider path) or use `input_chw`
3. Normalize `raw_chw` (`0..10000` -> `[0,1]`) with `RS_EMBED_WILDSAT_NORM`:
   - `minmax` (default): per-tile min-max stretch after unit scaling
   - `unit_scale` / `none`: keep unit-scaled values (no extra per-tile stretch)
4. Convert to `uint8 HWC` and resize to `RS_EMBED_WILDSAT_IMG` (default `224`)
5. Load torchvision backbone + optional decoder image head from checkpoint
6. Forward pass:
   - choose feature source (`image_head`, `backbone`, or `auto`)
   - optional token extraction for ViT backbones (used for token pooling / grid)
7. Return pooled vector or grid

Important behavior:

- If `grid` is requested but ViT tokens are unavailable (e.g., non-ViT arch or token extraction disabled), adapter returns a `1x1` grid (`grid_kind="vector_as_1x1"`) instead of failing.

---

## Environment Variables / Tuning Knobs

### Runtime / feature extraction

| Env var | Default | Effect |
|---|---|---|
| `RS_EMBED_WILDSAT_ARCH` | `auto` | Backbone arch hint (`vitb16`, `vitl16`, `resnet50`, `swint`, or `auto`) |
| `RS_EMBED_WILDSAT_IMG` | `224` | Resize target image size |
| `RS_EMBED_WILDSAT_NORM` | `minmax` | `minmax`, `unit_scale`, or `none` |
| `RS_EMBED_WILDSAT_FEATURE` | `image_head` | Feature source: `auto`, `image_head`, `backbone` |
| `RS_EMBED_WILDSAT_IMAGE_BRANCH` | `3` | Preferred decoder branch for image head extraction |
| `RS_EMBED_WILDSAT_POOLED_FROM_TOKENS` | `0` | If true and ViT tokens available, pooled output uses token pooling |
| `RS_EMBED_WILDSAT_GRID_FROM_TOKENS` | `1` | Enable ViT token extraction for grid/token-based pooling |
| `RS_EMBED_WILDSAT_FETCH_WORKERS` | `8` | Provider prefetch workers for batch APIs |

### Checkpoint path / download

| Env var | Default | Effect |
|---|---|---|
| `RS_EMBED_WILDSAT_CKPT` | unset | Local checkpoint path |
| `RS_EMBED_WILDSAT_AUTO_DOWNLOAD` | `1` | Allow auto-download if `CKPT` not set |
| `RS_EMBED_WILDSAT_CACHE_DIR` | `~/.cache/rs_embed/wildsat` | Checkpoint cache dir |
| `RS_EMBED_WILDSAT_CKPT_MIN_BYTES` | adapter threshold | Download size sanity check |
| `RS_EMBED_WILDSAT_GDRIVE_ID` | official sample file id | Google Drive source (default auto-download path) |
| `RS_EMBED_WILDSAT_CKPT_FILE` | `vitb16-imagenet-bnfc.pth` | Local cached filename for GDrive path |
| `RS_EMBED_WILDSAT_HF_REPO` | unset | Optional HF repo override (must pair with `HF_FILE`) |
| `RS_EMBED_WILDSAT_HF_FILE` | unset | Optional HF file override (must pair with `HF_REPO`) |

---

## Output Semantics

### `OutputSpec.pooled()`

- Default pooled output usually comes from selected feature source vector (`image_head` preferred by default)
- If `RS_EMBED_WILDSAT_POOLED_FROM_TOKENS=1` and ViT tokens are available, pooled output uses token pooling (`mean` / `max`)
- Metadata records:
  - `feature_source`
  - `tokens_available`
  - `pooled_from_tokens`

### `OutputSpec.grid()`

- If ViT tokens are available and token-grid extraction is enabled:
  - returns ViT patch-token grid (`grid_kind="vit_patch_tokens"`)
- Otherwise:
  - returns `1x1` vector grid (`grid_kind="vector_as_1x1"`)
- Grid is model feature layout, not georeferenced raster pixels

---

## Examples

### Minimal provider-backed example

```python
from rs_embed import get_embedding, PointBuffer, TemporalSpec, OutputSpec

emb = get_embedding(
    "wildsat",
    spatial=PointBuffer(lon=121.5, lat=31.2, buffer_m=2048),
    temporal=TemporalSpec.range("2022-06-01", "2022-09-01"),
    output=OutputSpec.pooled(),
    backend="gee",
)
```

### Example feature/arch tuning (env-controlled)

```python
# Example (shell):
# export RS_EMBED_WILDSAT_ARCH=vitb16
# export RS_EMBED_WILDSAT_FEATURE=image_head
# export RS_EMBED_WILDSAT_NORM=minmax
# export RS_EMBED_WILDSAT_GRID_FROM_TOKENS=1
```

---

## Common Failure Modes / Debugging

- backend mismatch (`wildsat` is provider-only)
- missing/invalid checkpoint path or auto-download failure
- unsupported / mis-inferred architecture (`RS_EMBED_WILDSAT_ARCH`)
- invalid `RS_EMBED_WILDSAT_FEATURE` value
- misunderstanding `grid` fallback (`1x1` vector grid is expected in some configs)
- token-based pooling requested but tokens unavailable (adapter silently falls back to vector path and records metadata)

Recommended first checks:

- inspect metadata for `arch`, `feature_source`, `tokens_available`, `grid_kind`
- set `RS_EMBED_WILDSAT_ARCH` explicitly if checkpoint inference is ambiguous
- fix checkpoint source first, then tune feature branch / normalization

---

## Reproducibility Notes

Keep fixed and record:

- checkpoint path/source and file hash if possible
- `RS_EMBED_WILDSAT_ARCH`
- normalization mode (`RS_EMBED_WILDSAT_NORM`)
- feature source / image branch / token-grid toggles
- temporal window and provider compositing settings

---

## Source of Truth (Code Pointers)

- Registration/catalog: `src/rs_embed/embedders/catalog.py`
- Adapter implementation: `src/rs_embed/embedders/onthefly_wildsat.py`
- Shared token/grid helpers: `src/rs_embed/embedders/_vit_mae_utils.py`

