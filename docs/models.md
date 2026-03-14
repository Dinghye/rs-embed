# Supported Models (Overview)

This page is the model selection entry point.
Use it to answer one question quickly:

- Which model IDs should I shortlist for this task?

After you have a shortlist:

- use [Advanced Model Reference](models_reference.md) for side-by-side preprocessing and temporal details
- open the linked detail page for the exact contract, caveats, and examples

---

## How To Read This Page

1. Pick a shortlist from the quick chooser
2. Scan the catalog table for input and temporal fit
3. Open the detail page before benchmarking or production use

Canonical model IDs use the short public names shown on this page, such as `remoteclip`, `prithvi`, `terrafm`, and `thor`.
Some detail-page filenames still use older names for compatibility, but the canonical IDs above are the names users should copy into code.

---

## Quick Chooser by Goal

| Goal | Good starting models | Why |
|---|---|---|
| Fast baseline / simple pipeline | `tessera`, `gse`, `copernicus` | Precomputed embeddings, fewer runtime dependencies |
| Simple S2 RGB on-the-fly experiments | `remoteclip`, `satmae`, `satmaepp`, `scalemae` | Straightforward RGB input paths |
| Time-series temporal modeling | `agrifm`, `anysat`, `galileo` | Native multi-frame temporal packaging |
| Multispectral / strict spectral semantics | `satmaepp_s2_10b`, `dofa`, `terramind`, `thor`, `satvision` | Strong channel/schema assumptions |
| Mixed-modality experiments (S1/S2) | `terrafm` | Supports S2 or S1 path (per call) |

## Model Catalog Snapshot

### Precomputed Embeddings

| Model ID | Type | Primary Input / Source | Outputs | Temporal mode | Notes | Detail |
|---|---|---|---|---|---|---|
| `tessera` | Precomputed | GeoTessera embedding tiles | `pooled`, `grid` | yearly coverage product | Fast baseline, source-fixed precomputed workflow | [detail](models/tessera.md) |
| `gse` | Precomputed | Google Satellite Embedding (annual) | `pooled`, `grid` | `TemporalSpec.year(...)` | Annual product via provider path | [detail](models/gse.md) |
| `copernicus` | Precomputed | Copernicus embeddings | `pooled`, `grid` | limited (2021) | Coarse resolution product | [detail](models/copernicus.md) |

### On-the-fly Foundation Models

| Model ID | Primary Input | Temporal style | Outputs | Notable requirements | Detail |
|---|---|---|---|---|---|
| `remoteclip` | S2 RGB (`B4,B3,B2`) | single composite window | `pooled`, `grid` | provider backend; RGB preprocessing | [detail](models/remoteclip.md) |
| `satmae` | S2 RGB (`B4,B3,B2`) | single composite window | `pooled`, `grid` | RGB path; ViT token/grid behavior | [detail](models/satmae.md) |
| `satmaepp` | S2 RGB (`B4,B3,B2`) | single composite window | `pooled`, `grid` | SatMAE++ fMoW-style eval preprocessing; channel order control | [detail](models/satmaepp.md) |
| `satmaepp_s2_10b` | S2 SR 10-band (`B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12`) | single composite window | `pooled`, `grid` | strict 10-band order; grouped-channel token handling | [detail](models/satmaepp.md) |
| `scalemae` | S2 RGB + scale | single composite window | `pooled`, `grid` | requires `sensor.scale_m` / `input_res_m` | [detail](models/scalemae.md) |
| `wildsat` | S2 RGB | single composite window | `pooled`, `grid` | normalization options | [detail](models/wildsat.md) |
| `prithvi` | S2 6-band | single composite window | `pooled`, `grid` | required temporal + location side inputs | [detail](models/prithvi.md) |
| `terrafm` | S2 12-band or S1 VV/VH | single composite window | `pooled`, `grid` | choose modality per call | [detail](models/terrafm.md) |
| `terramind` | S2 SR 12-band | single composite window | `pooled`, `grid` | strict normalization/channel semantics | [detail](models/terramind.md) |
| `dofa` | Multispectral + wavelengths | single composite window | `pooled`, `grid` | wavelength vector required/inferred | [detail](models/dofa.md) |
| `fomo` | S2 12-band | single composite window | `pooled`, `grid` | normalization mode choices | [detail](models/fomo.md) |
| `thor` | S2 SR 10-band | single composite window | `pooled`, `grid` | strict stats-based normalization | [detail](models/thor.md) |
| `satvision` | TOA 14-channel | single composite window | `pooled`, `grid` | strict channel order + calibration | [detail](models/satvision.md) |
| `anysat` | S2 10-band time series | multi-frame | `pooled`, `grid` | frame dates (`s2_dates`) side input | [detail](models/anysat.md) |
| `galileo` | S2 10-band time series | multi-frame | `pooled`, `grid` | month tokens + grouped tensors | [detail](models/galileo.md) |
| `agrifm` | S2 10-band time series | multi-frame | `pooled`, `grid` | fixed `T` frame stack behavior | [detail](models/agrifm.md) |

---

## Temporal and Comparison Notes (What People Usually Miss)

- `TemporalSpec.range(start, end)` is usually a **window for compositing**, not a single-scene acquisition selector.
- `OutputSpec.grid()` may be a **token/patch grid**, not a georeferenced raster grid (especially for ViT-like backbones).
- Cross-model comparisons are easiest with `OutputSpec.pooled()` and fixed ROI/temporal/compositing settings.
- Multi-frame models (`agrifm`, `anysat`, `galileo`) need extra attention to frame count and temporal side inputs.

Read the details in [Supported Models (Advanced Reference)](models_reference.md).

---

## More Detail

- [Advanced Model Reference](models_reference.md): cross-model tables for preprocessing, temporal packaging, and env knobs
- [Extending](extending.md): add a new model adapter and document it consistently
