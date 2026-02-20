from __future__ import annotations

from typing import Dict, Tuple

# model_id -> (module_name, class_name)
MODEL_SPECS: Dict[str, Tuple[str, str]] = {
    "gse_annual": ("precomputed_gse_annual", "GSEAnnualEmbedder"),
    "remoteclip_s2rgb": ("onthefly_remoteclip", "RemoteCLIPS2RGBEmbedder"),
    "copernicus_embed": ("precomputed_copernicus_embed", "CopernicusEmbedder"),
    "tessera": ("precomputed_tessera", "TesseraEmbedder"),
    "satmae_rgb": ("onthefly_satmae", "SatMAERGBEmbedder"),
    "scalemae_rgb": ("onthefly_scalemae", "ScaleMAERGBEmbedder"),
    "anysat": ("onthefly_anysat", "AnySatEmbedder"),
    "dynamicvis": ("onthefly_dynamicvis", "DynamicVisEmbedder"),
    "galileo": ("onthefly_galileo", "GalileoEmbedder"),
    "wildsat": ("onthefly_wildsat", "WildSATEmbedder"),
    "prithvi_eo_v2_s2_6b": ("onthefly_prithvi", "PrithviEOV2S2_6B_Embedder"),
    "terrafm_b": ("onthefly_terrafm", "TerraFMBEmbedder"),
    "terramind": ("onthefly_terramind", "TerraMindEmbedder"),
    "dofa": ("onthefly_dofa", "DOFAEmbedder"),
    "fomo": ("onthefly_fomo", "FoMoEmbedder"),
    "thor_1_0_base": ("onthefly_thor", "THORBaseEmbedder"),
    "agrifm": ("onthefly_agrifm", "AgriFMEmbedder"),
    "satvision_toa": ("onthefly_satvision_toa", "SatVisionTOAEmbedder"),
}

# Optional convenience map for lazy class access from rs_embed.embedders
CLASS_TO_MODULE: Dict[str, str] = {class_name: module for module, class_name in MODEL_SPECS.values()}
