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
| Prithvi    | prithvi_eo_v2_s2_6b   | Temporal ViT encoder (Prithvi ViT-100 style)    | S2 6 bands         | Scales S2 SR by /10000, then clamps to [0,1]; optional GEE patch checks + quicklook                                                 | pooled: [768] ; grid: [768, 14, 14]                                                | From TerraTorch  |
| DOFA       | dofa          | ViT (Base/Large variants)                       | S2 bands            | Scales by /10000, clamps to [0,1]; optional GEE patch checks + quicklook                                                       | base: pooled [768], grid [768, 14, 14] ; large: pooled [1024], grid [1024, 14, 14] | From TorchGeo    |
| TerraFM-B  | terrafm_b        | ViT-Base style (embed dim 768, patch 16)        | S1/S2 depending on config | Applies TerraFM-specific normalization (including scaling/clamping and per-sensor handling); optional GEE patch checks + quicklook | pooled: [768] ; grid: [768, 14, 14]                                                |                  |

