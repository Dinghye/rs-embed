# Import embedders so they register themselves
from .precomputed_gse_annual import GSEAnnualEmbedder  # noqa: F401
from .onthefly_remoteclip import RemoteCLIPS2RGBEmbedder  # noqa: F401
from .precomputed_copernicus_embed import CopernicusEmbedder  # noqa: F401
from .precomputed_tessera import TesseraEmbedder  # noqa: F401
from .onthefly_satmae import SatMAERGBEmbedder  # noqa: F401
from .onthefly_scalemae import ScaleMAERGBEmbedder  # noqa: F401
from .onthefly_anysat import AnySatEmbedder  # noqa: F401
from .onthefly_dynamicvis import DynamicVisEmbedder  # noqa: F401
from .onthefly_galileo import GalileoEmbedder  # noqa: F401
from .onthefly_wildsat import WildSATEmbedder  # noqa: F401
from .onthefly_prithvi import PrithviEOV2S2_6B_Embedder  # noqa: F401
# from .onthefly_presto import PrestoEmbedder  # noqa: F401
from .onthefly_terrafm import TerraFMBEmbedder  # noqa: F401
from .onthefly_terramind import TerraMindEmbedder  # noqa: F401
from .onthefly_dofa import DOFAEmbedder  # noqa: F401
from .onthefly_fomo import FoMoEmbedder  # noqa: F401
from .onthefly_thor import THORBaseEmbedder  # noqa: F401
