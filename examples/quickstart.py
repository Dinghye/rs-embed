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


# from rs_embed import BBox, get_embeddings_batch

# points = [
#     PointBuffer(lon=121.5, lat=31.2, buffer_m=100),
#     # PointBuffer(lon=121.6, lat=31.3, buffer_m=100),
#     # PointBuffer(lon=120.0, lat=30.0, buffer_m=100),
# ]

# embeddings = get_embeddings_batch(
#     "skysense_plus_s2",
#     spatials=points,
#     temporal=TemporalSpec.range("2021-06-01", "2021-08-31"),
#     backend="gee"
# )

# for i, emb in enumerate(embeddings):
#     print(f"Embedding {i} shape: {emb.data.shape}")


import json
import os
from pathlib import Path

# 你这个包里导出的 API（按我给你的 package 结构）
from rs_embed import BBox, TemporalSpec, SensorSpec, inspect_gee_patch


def _pretty(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)


def main():
    # save quicklook
    save_dir = Path(os.getenv("RS_EMBED_CHECK_SAVE_DIR", "./_gee_checks")).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)


    # bbox = BBox(minlon=-122.52, minlat=37.70, maxlon=-122.36, maxlat=37.83)
    spatial = PointBuffer(lon=121.5, lat=31.2, buffer_m=2048)
    temporal = TemporalSpec.range("2022-06-01", "2022-09-01")

    # 3) Sentinel-2 SR RGB
    sensor = SensorSpec(
        collection="COPERNICUS/S2_SR_HARMONIZED",
        bands=("B4", "B3", "B2"),   # R,G,B
        scale_m=10,
        cloudy_pct=10,
        composite="median",
        fill_value=0.0,
        check_save_dir=str(save_dir),  # save quicklook
    )

    # 4) run check
    # value_range：如果你希望强校验输入值域（S2_SR 常见 0..10000）
    out = inspect_gee_patch(
        spatial=spatial,
        temporal=temporal,
        sensor=sensor,
        value_range=(0, 10000),
        # quantiles=(0.01, 0.5, 0.99),
        # hist_bins=32,
    )

    # 5) 打印关键信息
    print("\n=== inspect_gee_patch result ===")
    print("ok:", out.get("ok"))
    if out.get("issues"):
        print("issues:", out["issues"])

    report = out.get("report", {})
    print("\n--- basic ---")
    for k in ["shape", "dtype", "nan_frac", "inf_frac", "fill_frac", "const_frac"]:
        if k in report:
            print(f"{k}: {report[k]}")

    # 分位数（p01/p50/p99）
    print("\n--- band_quantiles ---")
    bq = report.get("band_quantiles")
    if bq is None:
        print("(missing) band_quantiles not found in report")
    else:
        # 只打印前 3 个 band 的前几个数，避免太长
        print(_pretty({k: v[:3] for k, v in bq.items()}))

    # 直方图
    print("\n--- histogram ---")
    hist = report.get("hist")
    if hist is None:
        print("(missing) hist not found in report")
    else:
        # 直方图 bins + 第一条 band 的 counts 预览
        preview = {
            "range": hist.get("range"),
            "bins_len": len(hist.get("bins", [])),
            "counts_band0_len": len(hist.get("counts", [[]])[0]) if hist.get("counts") else 0,
            "counts_band0_head": (hist.get("counts", [[]])[0][:8] if hist.get("counts") else []),
        }
        print(_pretty(preview))

    # quicklook 路径（若保存成功）
    artifacts = out.get("artifacts") or {}
    print("\n--- artifacts ---")
    print(_pretty(artifacts))

    # 6) 如果 ok=False，给出更明显的提示
    if not out.get("ok", False):
        raise SystemExit("❌ Input checks failed. See issues/report above.")

    print("\n✅ Done. Input checks passed.")


if __name__ == "__main__":
    main()