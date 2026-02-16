# Supported Models

This page lists supported model backends and their I/O characteristics.



## 🧊 **Precomputed Embeddings**

|**Model**|**ID**|**Output**|**Resolution**|**Dim**|**Time Coverage**|**Notes**|
|---|---|---|---|---|---|---|
|**Tessera**|tessera|pooled / grid|0.1°|128|2017–2025|GeoTessera global tile embeddings|
|**Google Satellite Embedding (Alpha Earth)**|gse_annual|pooled / grid|10 m|64|2017–2024|Annual embeddings via GEE|
|**Copernicus Embed**|copernicus_embed|pooled / grid|0.25°|768|2021|Official Copernicus embeddings|

---
## 🔥  **On-the-fly Foundation Models**

| **Model**  | **ID**          | **Architecture**                                | **Input**       | **Preprocessing / Normalization**                        | **Raw output dimension(s)**                                                        | **Note**        ｜
|------------|------------------|-------------------------------------------------|---------------------------|------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------|------------------|
| RemoteClip | remoteclip_s2rgb | CLIP-style Vision Transformer (ViT-B/32)        | S2 → RGB composite        | RGB → uint8 + CLIP image normalization (mean/std); optional GEE patch checks + quicklook                                         | pooled: [512] (projection dim) ; grid: [768, 7, 7] (ViT width + 224/32 tokens)     |                  |
| SatMAE     | satmae_rgb     | ViT-L/16 (Masked Autoencoder-style pretraining) | S2 → RGB composite        | RGB → uint8 + CLIP image normalization; optional GEE patch checks + quicklook                                                   | pooled: [1024] ; grid: [1024, 14, 14]                                              |                  |
| ScaleMAE   | scalemae_rgb   | ViT-L/16 (Masked Autoencoder-style)             | S2 → RGB composite        | RGB → uint8 + CLIP image normalization; optional GEE patch checks + quicklook                                                | pooled: [1024] ; grid: [1024, 14, 14]                                              |                  |
| AnySat     | anysat         | Multi-modal EO transformer (AnySat base)        | S2 10 bands (single-step composite) | S2 SR raw → optional per-tile normalization; uses AnySat patch output as embedding grid                                        | pooled: [D] (patch mean/max) ; grid: [D, H, W]                                     | From AnySat      |
| DynamicVis | dynamicvis     | Dynamic sparse visual backbone (Mamba-based)    | S2 → RGB composite        | RGB → ImageNet normalization; requires DynamicVis/OpenMMLab runtime deps + HF checkpoint                                        | pooled: last-stage feature mean/max ; grid: last-stage feature map [D, H, W]        | From DynamicVis  |
| Galileo    | galileo        | Multi-modal EO transformer encoder (Galileo)    | S2 10 bands (single-step composite) | S2 SR raw → unit/minmax normalization; auto-loads official Galileo repo + nano weights                                          | pooled: token-mean vector [D] ; grid: S2-group patch tokens [D, H, W]               | From Galileo     |
| WildSAT    | wildsat        | Wildlife-guided satellite encoder (WildSAT checkpoint) | S2 → RGB composite   | S2 SR raw → unit/minmax normalization; auto-downloads default WildSAT ckpt if `RS_EMBED_WILDSAT_CKPT` not set (configurable via env) | pooled: image-head/backbone vector [D] ; grid: ViT patch tokens [D, H, W] or [D,1,1] | From WildSAT     |
| Prithvi    | prithvi_eo_v2_s2_6b   | Temporal ViT encoder (Prithvi ViT-100 style)    | S2 6 bands         | Scales S2 SR by /10000, then clamps to [0,1]; optional GEE patch checks + quicklook                                                 | pooled: [768] ; grid: [768, 14, 14]                                                | From TerraTorch  |
| DOFA       | dofa          | ViT (Base/Large variants)                       | S2 bands            | Scales by /10000, clamps to [0,1]; optional GEE patch checks + quicklook                                                       | base: pooled [768], grid [768, 14, 14] ; large: pooled [1024], grid [1024, 14, 14] | From TorchGeo    |
| THOR       | thor_1_0_base | THOR ViT backbone (v1 base)                     | S2 10 bands         | Uses S2 SR DN (0–10000) with configurable normalization (`thor_stats` default), then resize; loads via terratorch + thor_terratorch_ext | pooled: token mean/max [D] ; grid: THOR grouped token grid [D, H, W]               | From FM4CS/THOR  |
| TerraFM-B  | terrafm_b        | ViT-Base style (embed dim 768, patch 16)        | S1/S2 depending on config | Applies TerraFM-specific normalization (including scaling/clamping and per-sensor handling); optional GEE patch checks + quicklook | pooled: [768] ; grid: [768, 14, 14]                                                |                  |
| TerraMind  | terramind        | Multi-modal ViT backbone (TerraMind v1 family) | S2 L2A 12 bands     | Uses raw S2 SR DN (0–10000) + TerraMind S2L2A z-score stats, then 224 resize; optional GEE patch checks                              | pooled: variant dependent (tiny/small/base/large) ; grid: [D, 14, 14]              | From TerraTorch  |
| AgriFM     | agrifm           | Multi-source temporal Video Swin encoder (AgriFM) | S2 10 bands multi-temporal [T, C, H, W] | Uses S2 raw DN (0–10000), default `agrifm_stats` z-score normalization, then resize to 256; supports temporal binning from GEE | pooled: spatial mean/max of encoder feature map [D] ; grid: encoder feature map [D, H, W] | From AgriFM      |
| SatVision-TOA | satvision_toa | Swin Transformer V2 (SatVision TOA giant, default) | MODIS TOA 14 bands (`1,2,3,26,6,20,7,27,28,29,31,32,33,34`) | Raw mode: reflectance/emissive channel-wise scaling to [0,1]; unit mode: clip to [0,1]; auto mode picks by value range. If `MODIS/061/MOD021KM` is unavailable, auto-fallback uses `MODIS/061/MOD09GA` + `MODIS/061/MOD21A1D` proxy channels. | pooled: token mean/max [D] ; grid: patch tokens/feature map [D, H, W] (default giant: D=4096) | From NASA CISTO SatVision |
| FoMo       | fomo             | MultiSpectral ViT (FoMo-Net_1 style)          | S2 SR 12 bands      | Uses S2 raw DN (0-10000), default unit-scale normalization, 64 resize; loads FoMo-Bench code + FoMo-Net checkpoint                | pooled: token mean/max [768] ; grid: spectral-averaged patch tokens [768, H, W]    | From FoMo-Bench  |
