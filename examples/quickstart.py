from rs_embed import BBox, PointBuffer, TemporalSpec, OutputSpec, get_embedding
from plot_utils import *

# spatial = PointBuffer(lon=121.5, lat=31.2, buffer_m=2048)
# temporal = TemporalSpec.year(2024)


# bbox = BBox(
#     minlon=121.45, minlat=31.15,
#     maxlon=121.55, maxlat=31.25,
# )
# # # # Tessera: grid
# emb_tes_g = get_embedding(
#     "tessera",
#     spatial=bbox,
#     temporal=temporal,#TemporalSpec.range("2021-06-01", "2021-08-31"),
#     output=OutputSpec.grid(),
#     backend="local",
# )
# pca_tes = plot_embedding_pseudocolor(emb_tes_g, title="Tessera PCA pseudocolor")
# print("tessera grid meta:", emb_tes_g.meta)

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
# emb = get_embedding("gse_annual", spatial=bbox, temporal=temporal, output=OutputSpec.grid(scale_m=100))
# print(emb.data.shape, emb.meta["source"])
# pca_tes = plot_embedding_pseudocolor(emb, title="Alpha Earth PCA pseudocolor")


# On-the-fly: RemoteCLIP from S2 RGB
# emb2 = get_embedding("remoteclip_s2rgb", spatial=spatial, temporal=TemporalSpec.range("2022-06-01","2022-09-01"),
#                      output=OutputSpec.pooled(pooling="mean"))
# print(emb2.data.shape, emb2.meta["model"])
# print(emb2.meta)

# embg = get_embedding(
#   "remoteclip_s2rgb",
#   spatial=spatial,
#   temporal=TemporalSpec.range("2022-06-01","2022-09-01"),
#   output=OutputSpec.grid()
# )
# print(embg.data.shape)  # (512, 7, 7) 
# print(embg.meta)


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
# print(emb.data.shape)
# print(emb.meta)

# emb = get_embedding(
#     "presto",
#     spatial=spatial,
#     temporal=TemporalSpec.year(2022),
#     output=OutputSpec.pooled(),
#     backend="gee",
# )
# print(emb.data.shape, emb.meta["dim"])

# emb = get_embedding(
#     "dofa",
#     spatial=spatial,
#     temporal=TemporalSpec.range("2021-06-01", "2021-08-31"),
#     output=OutputSpec.pooled(),
#     backend="gee",
# )
# print(emb.data.shape, emb.meta)

from rs_embed import PointBuffer, TemporalSpec, OutputSpec
from rs_embed.export import export_npz

points = [
    ("p1", PointBuffer(lon=120.10, lat=30.20, buffer_m=256)),
    ("p2", PointBuffer(lon=120.30, lat=30.10, buffer_m=256)),
]

for name, spatial in points:
    export_npz(
        out_path=f"exports/{name}_2022_summer.npz",
        models=["remoteclip_s2rgb", "prithvi_eo_v2_s2_6b"],
        spatial=spatial,
        temporal=TemporalSpec.range("2022-06-01", "2022-09-01"),
        output=OutputSpec.pooled(),
        backend="gee",
        device="auto",
        save_inputs=True,
        save_embeddings=True,
        save_manifest=True,
    )