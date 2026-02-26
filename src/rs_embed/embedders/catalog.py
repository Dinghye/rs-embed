from __future__ import annotations

from typing import Dict, Tuple

# Canonical model_id -> (module_name, class_name)
MODEL_SPECS: Dict[str, Tuple[str, str]] = {
    "gse": ("precomputed_gse_annual", "GSEAnnualEmbedder"),
    "remoteclip": ("onthefly_remoteclip", "RemoteCLIPS2RGBEmbedder"),
    "copernicus": ("precomputed_copernicus_embed", "CopernicusEmbedder"),
    "tessera": ("precomputed_tessera", "TesseraEmbedder"),
    "satmae": ("onthefly_satmae", "SatMAERGBEmbedder"),
    "scalemae": ("onthefly_scalemae", "ScaleMAERGBEmbedder"),
    "anysat": ("onthefly_anysat", "AnySatEmbedder"),
    "dynamicvis": ("onthefly_dynamicvis", "DynamicVisEmbedder"),
    "galileo": ("onthefly_galileo", "GalileoEmbedder"),
    "wildsat": ("onthefly_wildsat", "WildSATEmbedder"),
    "prithvi": ("onthefly_prithvi", "PrithviEOV2S2_6B_Embedder"),
    "terrafm": ("onthefly_terrafm", "TerraFMBEmbedder"),
    "terramind": ("onthefly_terramind", "TerraMindEmbedder"),
    "dofa": ("onthefly_dofa", "DOFAEmbedder"),
    "fomo": ("onthefly_fomo", "FoMoEmbedder"),
    "thor": ("onthefly_thor", "THORBaseEmbedder"),
    "agrifm": ("onthefly_agrifm", "AgriFMEmbedder"),
    "satvision": ("onthefly_satvision_toa", "SatVisionTOAEmbedder"),
}

MODEL_ALIASES: Dict[str, str] = {
    # Legacy IDs kept for backward compatibility.
    "gse_annual": "gse",
    "remoteclip_s2rgb": "remoteclip",
    "copernicus_embed": "copernicus",
    "satmae_rgb": "satmae",
    "scalemae_rgb": "scalemae",
    "prithvi_eo_v2_s2_6b": "prithvi",
    "terrafm_b": "terrafm",
    "thor_1_0_base": "thor",
    "satvision_toa": "satvision",
}


def canonical_model_id(name: str) -> str:
    k = str(name).strip().lower()
    return MODEL_ALIASES.get(k, k)


# Optional convenience map for lazy class access from rs_embed.embedders
CLASS_TO_MODULE: Dict[str, str] = {class_name: module for module, class_name in MODEL_SPECS.values()}
