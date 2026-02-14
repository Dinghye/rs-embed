import numpy as np

from rs_embed.core.embedding import Embedding
from rs_embed.core.specs import OutputSpec, PointBuffer, TemporalSpec
from rs_embed.embedders.onthefly_anysat import AnySatEmbedder
from rs_embed.embedders.onthefly_prithvi import PrithviEOV2S2_6B_Embedder
from rs_embed.embedders.onthefly_remoteclip import RemoteCLIPS2RGBEmbedder
from rs_embed.embedders.onthefly_scalemae import ScaleMAERGBEmbedder
from rs_embed.embedders.onthefly_dynamicvis import DynamicVisEmbedder
from rs_embed.embedders.onthefly_galileo import GalileoEmbedder
from rs_embed.embedders.onthefly_wildsat import WildSATEmbedder
from rs_embed.embedders.onthefly_terrafm import TerraFMBEmbedder
from rs_embed.embedders.onthefly_terramind import TerraMindEmbedder
from rs_embed.embedders.precomputed_copernicus_embed import CopernicusEmbedder
from rs_embed.embedders.precomputed_gse_annual import GSEAnnualEmbedder
from rs_embed.embedders.precomputed_tessera import TesseraEmbedder


def _spatials(n: int) -> list[PointBuffer]:
    return [PointBuffer(lon=float(i), lat=0.0, buffer_m=256) for i in range(n)]


def test_remoteclip_batch_prefetch_passes_input_chw(monkeypatch):
    import rs_embed.embedders.onthefly_remoteclip as rc

    emb = RemoteCLIPS2RGBEmbedder()
    monkeypatch.setenv("RS_EMBED_REMOTECLIP_FETCH_WORKERS", "1")
    monkeypatch.setattr(emb, "_get_provider", lambda: object())
    monkeypatch.setattr(
        rc,
        "_fetch_s2_rgb_chw",
        lambda provider, spatial, temporal, **kw: np.full((3, 8, 8), 0.5, dtype=np.float32),
    )

    seen = []

    def _fake_get_embedding(**kw):
        arr = kw["input_chw"]
        seen.append(float(arr.max()))
        return Embedding(data=np.array([kw["spatial"].lon], dtype=np.float32), meta={})

    monkeypatch.setattr(emb, "get_embedding", _fake_get_embedding)

    out = emb.get_embeddings_batch(
        spatials=_spatials(3),
        temporal=TemporalSpec.range("2020-06-01", "2020-08-31"),
        output=OutputSpec.pooled(),
        backend="gee",
    )

    assert len(out) == 3
    assert all(v >= 4999.0 for v in seen)  # 0.5 * 10000


def test_scalemae_batch_prefetch_and_single_model_load(monkeypatch):
    import rs_embed.embedders.onthefly_scalemae as sm

    emb = ScaleMAERGBEmbedder()
    monkeypatch.setenv("RS_EMBED_SCALEMAE_FETCH_WORKERS", "1")
    monkeypatch.setattr(emb, "_get_provider", lambda: object())

    calls = {"load": 0}

    def _fake_fetch(*, spatial, temporal, sensor, out_size, provider):
        return np.full((out_size, out_size, 3), int(spatial.lon) + 10, dtype=np.uint8)

    def _fake_load(*, model_id, device):
        calls["load"] += 1
        return object(), {"device": "cpu"}

    def _fake_forward(model, rgb_u8, *, image_size, device, input_res_m):
        val = float(rgb_u8[0, 0, 0])
        return np.full((4, 2), val, dtype=np.float32), {"tokens_kind": "tokens_forward"}

    monkeypatch.setattr(sm, "fetch_s2_rgb_u8_from_gee", _fake_fetch)
    monkeypatch.setattr(sm, "_load_scalemae", _fake_load)
    monkeypatch.setattr(sm, "_scalemae_forward_tokens_or_vec", _fake_forward)

    out = emb.get_embeddings_batch(
        spatials=_spatials(4),
        temporal=TemporalSpec.year(2020),
        output=OutputSpec.pooled(),
        backend="gee",
    )

    assert len(out) == 4
    assert calls["load"] == 1
    assert [float(e.data[0]) for e in out] == [10.0, 11.0, 12.0, 13.0]


def test_prithvi_batch_prefetch_passes_raw_input(monkeypatch):
    import rs_embed.embedders.onthefly_prithvi as pr

    emb = PrithviEOV2S2_6B_Embedder()
    monkeypatch.setenv("RS_EMBED_PRITHVI_FETCH_WORKERS", "1")
    monkeypatch.setattr(emb, "_get_provider", lambda: object())
    monkeypatch.setattr(
        pr,
        "_fetch_s2_prithvi6_chw",
        lambda provider, spatial, temporal, **kw: np.full((6, 8, 8), 0.25, dtype=np.float32),
    )

    seen = []

    def _fake_get_embedding(**kw):
        seen.append(float(kw["input_chw"].max()))
        return Embedding(data=np.array([kw["spatial"].lon], dtype=np.float32), meta={})

    monkeypatch.setattr(emb, "get_embedding", _fake_get_embedding)

    out = emb.get_embeddings_batch(
        spatials=_spatials(3),
        temporal=TemporalSpec.year(2020),
        output=OutputSpec.pooled(),
        backend="gee",
    )

    assert len(out) == 3
    assert all(v >= 2499.0 for v in seen)  # 0.25 * 10000


def test_terrafm_batch_prefetch_passes_raw_input(monkeypatch):
    import rs_embed.embedders.onthefly_terrafm as tf

    emb = TerraFMBEmbedder()
    monkeypatch.setenv("RS_EMBED_TERRAFM_FETCH_WORKERS", "1")
    monkeypatch.setattr(emb, "_get_provider", lambda: object())
    monkeypatch.setattr(
        tf,
        "_fetch_s2_sr_12_chw",
        lambda provider, spatial, temporal, **kw: np.full((12, 8, 8), 0.1, dtype=np.float32),
    )

    seen = []

    def _fake_get_embedding(**kw):
        arr = kw["input_chw"]
        seen.append((arr.shape[0], float(arr.max())))
        return Embedding(data=np.array([kw["spatial"].lon], dtype=np.float32), meta={})

    monkeypatch.setattr(emb, "get_embedding", _fake_get_embedding)

    out = emb.get_embeddings_batch(
        spatials=_spatials(2),
        temporal=TemporalSpec.range("2020-06-01", "2020-08-31"),
        output=OutputSpec.pooled(),
        backend="gee",
    )

    assert len(out) == 2
    assert seen[0][0] == 12
    assert seen[0][1] >= 999.0  # 0.1 * 10000


def test_terramind_batch_prefetch_passes_raw_input(monkeypatch):
    import rs_embed.embedders.onthefly_terramind as tm

    emb = TerraMindEmbedder()
    monkeypatch.setenv("RS_EMBED_TERRAMIND_FETCH_WORKERS", "1")
    monkeypatch.setattr(emb, "_get_provider", lambda: object())
    monkeypatch.setattr(
        tm,
        "_fetch_s2_sr_12_raw_chw",
        lambda provider, spatial, temporal, **kw: np.full((12, 8, 8), 1234.0, dtype=np.float32),
    )

    seen = []

    def _fake_get_embedding(**kw):
        arr = kw["input_chw"]
        seen.append((arr.shape[0], float(arr.max())))
        return Embedding(data=np.array([kw["spatial"].lon], dtype=np.float32), meta={})

    monkeypatch.setattr(emb, "get_embedding", _fake_get_embedding)

    out = emb.get_embeddings_batch(
        spatials=_spatials(2),
        temporal=TemporalSpec.range("2020-06-01", "2020-08-31"),
        output=OutputSpec.pooled(),
        backend="gee",
    )

    assert len(out) == 2
    assert seen[0][0] == 12
    assert seen[0][1] >= 1234.0


def test_dynamicvis_batch_prefetch_passes_raw_input(monkeypatch):
    import rs_embed.embedders.onthefly_dynamicvis as dv

    emb = DynamicVisEmbedder()
    monkeypatch.setenv("RS_EMBED_DYNAMICVIS_FETCH_WORKERS", "1")
    monkeypatch.setattr(emb, "_get_provider", lambda: object())
    monkeypatch.setattr(
        dv,
        "_fetch_s2_rgb_chw",
        lambda provider, spatial, temporal, **kw: np.full((3, 8, 8), 0.3, dtype=np.float32),
    )

    seen = []

    def _fake_get_embedding(**kw):
        arr = kw["input_chw"]
        seen.append((arr.shape[0], float(arr.max())))
        return Embedding(data=np.array([kw["spatial"].lon], dtype=np.float32), meta={})

    monkeypatch.setattr(emb, "get_embedding", _fake_get_embedding)

    out = emb.get_embeddings_batch(
        spatials=_spatials(2),
        temporal=TemporalSpec.range("2020-06-01", "2020-08-31"),
        output=OutputSpec.pooled(),
        backend="gee",
    )

    assert len(out) == 2
    assert seen[0][0] == 3
    assert seen[0][1] >= 2999.0  # 0.3 * 10000


def test_anysat_batch_prefetch_passes_raw_input(monkeypatch):
    import rs_embed.embedders.onthefly_anysat as anysat

    emb = AnySatEmbedder()
    monkeypatch.setenv("RS_EMBED_ANYSAT_FETCH_WORKERS", "1")
    monkeypatch.setattr(emb, "_get_provider", lambda: object())
    monkeypatch.setattr(
        anysat,
        "_fetch_s2_10_raw_chw",
        lambda provider, spatial, temporal, **kw: np.full((10, 8, 8), 4321.0, dtype=np.float32),
    )

    seen = []

    def _fake_get_embedding(**kw):
        arr = kw["input_chw"]
        seen.append((arr.shape[0], float(arr.max())))
        return Embedding(data=np.array([kw["spatial"].lon], dtype=np.float32), meta={})

    monkeypatch.setattr(emb, "get_embedding", _fake_get_embedding)

    out = emb.get_embeddings_batch(
        spatials=_spatials(2),
        temporal=TemporalSpec.range("2020-06-01", "2020-08-31"),
        output=OutputSpec.pooled(),
        backend="gee",
    )

    assert len(out) == 2
    assert seen[0][0] == 10
    assert seen[0][1] >= 4321.0


def test_wildsat_batch_prefetch_passes_raw_input(monkeypatch):
    import rs_embed.embedders.onthefly_wildsat as ws

    emb = WildSATEmbedder()
    monkeypatch.setenv("RS_EMBED_WILDSAT_FETCH_WORKERS", "1")
    monkeypatch.setattr(emb, "_get_provider", lambda: object())
    monkeypatch.setattr(
        ws,
        "_fetch_s2_rgb_chw",
        lambda provider, spatial, temporal, **kw: np.full((3, 8, 8), 0.4, dtype=np.float32),
    )

    seen = []

    def _fake_get_embedding(**kw):
        arr = kw["input_chw"]
        seen.append((arr.shape[0], float(arr.max())))
        return Embedding(data=np.array([kw["spatial"].lon], dtype=np.float32), meta={})

    monkeypatch.setattr(emb, "get_embedding", _fake_get_embedding)

    out = emb.get_embeddings_batch(
        spatials=_spatials(2),
        temporal=TemporalSpec.range("2020-06-01", "2020-08-31"),
        output=OutputSpec.pooled(),
        backend="gee",
    )

    assert len(out) == 2
    assert seen[0][0] == 3
    assert seen[0][1] >= 3999.0  # 0.4 * 10000


def test_galileo_batch_prefetch_passes_raw_input(monkeypatch):
    import rs_embed.embedders.onthefly_galileo as gal

    emb = GalileoEmbedder()
    monkeypatch.setenv("RS_EMBED_GALILEO_FETCH_WORKERS", "1")
    monkeypatch.setattr(emb, "_get_provider", lambda: object())
    monkeypatch.setattr(
        gal,
        "_fetch_s2_10_raw_chw",
        lambda provider, spatial, temporal, **kw: np.full((10, 8, 8), 2222.0, dtype=np.float32),
    )

    seen = []

    def _fake_get_embedding(**kw):
        arr = kw["input_chw"]
        seen.append((arr.shape[0], float(arr.max())))
        return Embedding(data=np.array([kw["spatial"].lon], dtype=np.float32), meta={})

    monkeypatch.setattr(emb, "get_embedding", _fake_get_embedding)

    out = emb.get_embeddings_batch(
        spatials=_spatials(2),
        temporal=TemporalSpec.range("2020-06-01", "2020-08-31"),
        output=OutputSpec.pooled(),
        backend="gee",
    )

    assert len(out) == 2
    assert seen[0][0] == 10
    assert seen[0][1] >= 2222.0


def test_precomputed_batch_overrides_call_single_embedding(monkeypatch):
    # gse_annual
    gse = GSEAnnualEmbedder()
    monkeypatch.setenv("RS_EMBED_GSE_BATCH_WORKERS", "1")
    monkeypatch.setattr(
        gse,
        "get_embedding",
        lambda **kw: Embedding(data=np.array([kw["spatial"].lon], dtype=np.float32), meta={}),
    )
    out_gse = gse.get_embeddings_batch(
        spatials=_spatials(2),
        temporal=TemporalSpec.year(2020),
        output=OutputSpec.pooled(),
        backend="gee",
    )
    assert len(out_gse) == 2

    # copernicus_embed
    cop = CopernicusEmbedder()
    monkeypatch.setenv("RS_EMBED_COPERNICUS_BATCH_WORKERS", "1")
    monkeypatch.setattr(
        cop,
        "get_embedding",
        lambda **kw: Embedding(data=np.array([kw["spatial"].lon], dtype=np.float32), meta={}),
    )
    out_cop = cop.get_embeddings_batch(
        spatials=_spatials(2),
        temporal=TemporalSpec.year(2021),
        output=OutputSpec.pooled(),
        backend="local",
    )
    assert len(out_cop) == 2

    # tessera
    tes = TesseraEmbedder()
    monkeypatch.setenv("RS_EMBED_TESSERA_BATCH_WORKERS", "1")
    monkeypatch.setattr(
        tes,
        "get_embedding",
        lambda **kw: Embedding(data=np.array([kw["spatial"].lon], dtype=np.float32), meta={}),
    )
    out_tes = tes.get_embeddings_batch(
        spatials=_spatials(2),
        temporal=TemporalSpec.year(2021),
        output=OutputSpec.pooled(),
        backend="local",
    )
    assert len(out_tes) == 2
