# Import embedders so they register themselves
from .precomputed_gse_annual import GSEAnnualEmbedder  # noqa: F401
from .onthefly_remoteclip import RemoteCLIPS2RGBEmbedder  # noqa: F401
from .precomputed_copernicus_embed import CopernicusEmbedder  # noqa: F401
from .precomputed_tessera import TesseraEmbedder  # noqa: F401
from .onthefly_satmae import SatMAERGBEmbedder  # noqa: F401
from .onthefly_scalemae import ScaleMAERGBEmbedder  # noqa: F401
from .onthefly_prithvi import PrithviEOV2S2_6B_Embedder  # noqa: F401
# from .onthefly_presto import PrestoEmbedder  # noqa: F401