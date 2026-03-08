import sys
import types

import numpy as np

from rs_embed.core.specs import BBox, OutputSpec, PointBuffer, TemporalSpec
from rs_embed.embedders.precomputed_copernicus_embed import CopernicusEmbedder
from rs_embed.embedders.precomputed_gse_annual import GSEAnnualEmbedder
from rs_embed.embedders.precomputed_tessera import TesseraEmbedder


class _FakeRegistry:
    def __init__(self, n_tiles: int):
        self._n_tiles = n_tiles

    def load_blocks_for_region(self, bounds, year):
        return list(range(self._n_tiles))


class _FakeGeoTessera:
    def __init__(self):
        self.registry = _FakeRegistry(1)

    def fetch_embeddings(self, tiles):
        for i in tiles:
            yield {"tile": i}


class _FakeTorchTensor:
    def __init__(self, arr: np.ndarray):
        self._arr = arr

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


class _FakeCopernicusDataset:
    def __getitem__(self, key):
        return {
            "image": _FakeTorchTensor(
                np.array(
                    [
                        [[1.0, 3.0], [5.0, 7.0]],
                        [[2.0, 4.0], [6.0, 8.0]],
                    ],
                    dtype=np.float32,
                ),
            )
        }


def test_precomputed_custom_init_preserves_base_state():
    tessera = TesseraEmbedder()
    copernicus = CopernicusEmbedder()

    assert tessera._providers == {}
    assert copernicus._providers == {}


def test_gse_get_embedding_ignores_input_chw(monkeypatch):
    import rs_embed.embedders.precomputed_gse_annual as gse_mod

    embedder = GSEAnnualEmbedder()
    embedder.model_name = "gse"
    monkeypatch.setattr(embedder, "_get_provider", lambda _backend: object())
    monkeypatch.setattr(
        gse_mod,
        "_fetch_collection_patch_all_bands_chw",
        lambda provider, **kw: (
            np.array(
                [
                    [[1.0, 3.0], [5.0, 7.0]],
                    [[2.0, 4.0], [6.0, 8.0]],
                ],
                dtype=np.float32,
            ),
            ["b0", "b1"],
        ),
    )

    emb = embedder.get_embedding(
        spatial=PointBuffer(lon=0.0, lat=0.0, buffer_m=256),
        temporal=TemporalSpec.year(2020),
        sensor=None,
        output=OutputSpec.pooled(),
        backend="gee",
        input_chw=np.ones((3, 8, 8), dtype=np.float32),
    )

    np.testing.assert_allclose(emb.data, np.array([4.0, 5.0], dtype=np.float32))


def test_tessera_get_embedding_ignores_input_chw(monkeypatch):
    import rs_embed.embedders.precomputed_tessera as tessera_mod

    embedder = TesseraEmbedder()
    embedder.model_name = "tessera"
    monkeypatch.setattr(embedder, "_get_gt", lambda _cache: _FakeGeoTessera())
    monkeypatch.setattr(
        tessera_mod,
        "_mosaic_and_crop_strict_roi",
        lambda tiles_fn, bbox_4326: (
            np.full((64, 1, 1), 1.0, dtype=np.float32),
            {"mosaic_hw": (1, 1), "crop_hw": (1, 1)},
        ),
    )

    emb = embedder.get_embedding(
        spatial=BBox(minlon=0.2, minlat=0.2, maxlon=0.8, maxlat=0.8, crs="EPSG:4326"),
        temporal=TemporalSpec.year(2021),
        sensor=None,
        output=OutputSpec.pooled(),
        backend="auto",
        input_chw=np.ones((3, 8, 8), dtype=np.float32),
    )

    np.testing.assert_allclose(emb.data, np.full((64,), 1.0, dtype=np.float32))


def test_copernicus_get_embedding_ignores_input_chw(monkeypatch):
    embedder = CopernicusEmbedder()
    embedder.model_name = "copernicus"
    monkeypatch.setitem(sys.modules, "torchgeo", types.ModuleType("torchgeo"))
    monkeypatch.setattr(
        embedder,
        "_get_dataset",
        lambda *, data_dir, download: _FakeCopernicusDataset(),
    )

    emb = embedder.get_embedding(
        spatial=PointBuffer(lon=0.0, lat=0.0, buffer_m=256),
        temporal=TemporalSpec.year(2021),
        sensor=None,
        output=OutputSpec.pooled(),
        backend="auto",
        input_chw=np.ones((3, 8, 8), dtype=np.float32),
    )

    np.testing.assert_allclose(emb.data, np.array([4.0, 5.0], dtype=np.float32))
