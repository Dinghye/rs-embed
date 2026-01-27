from rs_embed import BBox, PointBuffer, TemporalSpec, OutputSpec, get_embedding
from plot_utils import *

spatial = PointBuffer(lon=121.5, lat=31.2, buffer_m=2048)
temporal = TemporalSpec.year(2021)

# # # Tessera: grid
emb_tes_g = get_embedding(
    "tessera",
    spatial=spatial,
    temporal=temporal,#TemporalSpec.range("2021-06-01", "2021-08-31"),  # 用 start 年份当 preferred_year
    output=OutputSpec.grid(),
    backend="local",
)
pca_tes = plot_embedding_pseudocolor(emb_tes_g, title="Tessera PCA pseudocolor")
print("tessera grid meta:", emb_tes_g.meta)

# # print("tessera grid:", emb_tes_g.data.shape)

# # Tessera: pooled
# emb_tes = get_embedding(
#     "tessera",
#     spatial=spatial,
#     temporal=TemporalSpec.range("2021-06-01", "2021-08-31"),
#     output=OutputSpec.pooled(),
#     backend="local",
# )
# print("tessera pooled:", emb_tes.data.shape)

# Precomputed: GEE annual embedding
# emb = get_embedding("gse_annual", spatial=spatial, temporal=temporal, output=OutputSpec.grid(scale_m=10))
# print(emb.data.shape, emb.meta["source"])
# pca_tes = plot_embedding_pseudocolor(emb, title="Alpha Earth PCA pseudocolor")


# On-the-fly: RemoteCLIP from S2 RGB
# emb2 = get_embedding("remoteclip_s2rgb", spatial=spatial, temporal=TemporalSpec.range("2022-06-01","2022-09-01"),
#                      output=OutputSpec.pooled(pooling="mean"))
# print(emb2.data.shape, emb2.meta["model"])

# embg = get_embedding(
#   "remoteclip_s2rgb",
#   spatial=spatial,
#   temporal=TemporalSpec.range("2022-06-01","2022-09-01"),
#   output=OutputSpec.grid()
# )
# print(embg.data.shape)  # (512, 7, 7) 
# print(embg.meta["grid_hw"], embg.meta["cls_removed"])


# CopernicusEmbed: grid
# emb_cop_g = get_embedding(
#     "copernicus_embed",
#     spatial=spatial,
#     temporal = temporal,                     
#     output=OutputSpec.grid(),
#     backend="local",
# )
# print("cop grid:", emb_cop_g.data.shape)
# print("cop meta:", emb_cop_g.meta)
# plot_embedding_pseudocolor(emb_cop_g, title="CopernicusEmbed PCA pseudocolor")

# # CopernicusEmbed: pooled
# emb_cop = get_embedding(
#     "copernicus_embed",
#     spatial=spatial,
#     output=OutputSpec.pooled(),
#     backend="local",
# )
# print("cop pooled:", emb_cop.data.shape)



# emb = get_embedding(
#     "satmae_rgb",
#     spatial=spatial,
#     temporal=TemporalSpec.range("2022-06-01","2022-09-01"),
#     output=OutputSpec.pooled("mean"),
#     backend="gee",
# )
# print(emb.data.shape)

# embg = get_embedding(
#     "scalemae_rgb",
#     spatial=spatial,
#     temporal=TemporalSpec.range("2022-06-01","2022-09-01"),
#     output=OutputSpec.grid(),
#     backend="gee",
    
# )
# print(embg.data.shape, embg.meta["grid_hw"])

# emb = get_embedding(
#     "prithvi_eo_v2_s2_6b",
#     spatial=spatial,
#     temporal=TemporalSpec.range("2022-06-01","2022-09-01"),
#     output=OutputSpec.pooled("mean"),
#     backend="gee",
# )
# print(emb.data.shape, emb.meta["tokens_shape"])