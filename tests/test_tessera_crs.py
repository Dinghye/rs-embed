"""Tests for mixed-CRS mosaic in precomputed_tessera (GitHub issue #6).

When an ROI sits on a UTM zone boundary (e.g. lon = 120°) the tile
store can return tiles in *two* EPSG codes (e.g. 32650 / 32651).  The
mosaic helper must reproject them to a common CRS instead of rejecting
the operation.
"""
from __future__ import annotations

import numpy as np
import pytest

from affine import Affine
from pyproj import Transformer

from rs_embed.core.specs import BBox
from rs_embed.core.errors import ModelError
from rs_embed.embedders.precomputed_tessera import (
    _mosaic_and_crop_strict_roi,
    _reproject_tile,
)

rasterio = pytest.importorskip("rasterio", reason="rasterio needed for mixed-CRS tests")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

D = 64  # embedding dimension (must be in _to_hwc's allowlist)
TILE_HW = 8  # small tiles for fast tests
PX_M = 500.0  # 500 m pixels


def _make_tile(lon: float, lat: float, epsg: str, value: float = 1.0):
    """Create a synthetic (year, lon, lat, embedding, crs, transform) tuple."""
    tfm = Transformer.from_crs("EPSG:4326", epsg, always_xy=True)
    x, y = tfm.transform(lon, lat)

    # north-up transform: origin at top-left of tile
    transform = Affine(PX_M, 0.0, x, 0.0, -PX_M, y + TILE_HW * PX_M)
    emb = np.full((TILE_HW, TILE_HW, D), value, dtype=np.float32)
    return (2021, lon, lat, emb, epsg, transform)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMixedCRSMosaic:
    """Regression tests for issue #6 – UTM zone boundary."""

    def test_same_crs_still_works(self):
        """Sanity: single-CRS mosaic must still succeed."""
        tile_a = _make_tile(119.5, 30.0, "EPSG:32650", value=1.0)
        tile_b = _make_tile(119.6, 30.0, "EPSG:32650", value=2.0)

        bbox = BBox(minlon=119.45, minlat=29.95, maxlon=119.65, maxlat=30.05)
        chw, meta = _mosaic_and_crop_strict_roi([tile_a, tile_b], bbox_4326=bbox)

        assert chw.ndim == 3
        assert chw.shape[0] == D
        assert meta["tile_crs"] == "EPSG:32650"

    def test_mixed_crs_at_utm_boundary(self):
        """Two tiles in EPSG:32650 / 32651 at lon ≈ 120° must mosaic."""
        tile_a = _make_tile(119.95, 30.0, "EPSG:32650", value=1.0)
        tile_b = _make_tile(120.05, 30.0, "EPSG:32651", value=2.0)

        bbox = BBox(minlon=119.90, minlat=29.95, maxlon=120.10, maxlat=30.05)
        chw, meta = _mosaic_and_crop_strict_roi([tile_a, tile_b], bbox_4326=bbox)

        assert chw.ndim == 3
        assert chw.shape[0] == D
        # target CRS should be one of the two (whichever has more tiles, or
        # first in case of tie)
        assert meta["tile_crs"] in ("EPSG:32650", "EPSG:32651")

    def test_mixed_crs_majority_wins(self):
        """Target CRS is the most-common one among tiles."""
        tile_a = _make_tile(119.8, 30.0, "EPSG:32650", value=1.0)
        tile_b = _make_tile(119.9, 30.0, "EPSG:32650", value=1.5)
        tile_c = _make_tile(120.1, 30.0, "EPSG:32651", value=2.0)

        bbox = BBox(minlon=119.75, minlat=29.95, maxlon=120.15, maxlat=30.05)
        chw, meta = _mosaic_and_crop_strict_roi(
            [tile_a, tile_b, tile_c], bbox_4326=bbox,
        )

        assert chw.ndim == 3
        assert chw.shape[0] == D
        # 2 tiles in 32650, 1 in 32651 → target should be 32650
        assert meta["tile_crs"] == "EPSG:32650"

    def test_empty_tiles_raises(self):
        """No tiles at all should still raise."""
        bbox = BBox(minlon=119.0, minlat=29.0, maxlon=120.0, maxlat=30.0)
        with pytest.raises(ModelError, match="No tiles"):
            _mosaic_and_crop_strict_roi([], bbox_4326=bbox)


class TestReprojectTile:
    """Unit tests for _reproject_tile helper."""

    def test_identity_reproject(self):
        """Reprojecting to the same CRS should return data of similar shape."""
        hwc = np.ones((4, 4, D), dtype=np.float32)
        transform = Affine(PX_M, 0.0, 500_000.0, 0.0, -PX_M, 3_500_000.0)

        out_hwc, out_tf = _reproject_tile(
            hwc, transform, "EPSG:32650", "EPSG:32650",
        )

        assert out_hwc.ndim == 3
        assert out_hwc.shape[-1] == D
        np.testing.assert_allclose(out_hwc, 1.0, atol=1e-5)

    def test_cross_zone_reproject(self):
        """Reproject from zone 50 to zone 51 preserves data values."""
        hwc = np.full((4, 4, D), 42.0, dtype=np.float32)
        transform = Affine(PX_M, 0.0, 800_000.0, 0.0, -PX_M, 3_500_000.0)

        out_hwc, out_tf = _reproject_tile(
            hwc, transform, "EPSG:32650", "EPSG:32651",
        )

        assert out_hwc.ndim == 3
        assert out_hwc.shape[-1] == D
        # nearest-neighbour: non-zero pixels should carry the original value
        nonzero = out_hwc[out_hwc != 0]
        if nonzero.size:
            np.testing.assert_allclose(nonzero, 42.0, atol=1e-5)

    def test_target_res_snaps_resolution(self):
        """Providing target_res should lock the output pixel size."""
        hwc = np.ones((4, 4, D), dtype=np.float32)
        transform = Affine(PX_M, 0.0, 500_000.0, 0.0, -PX_M, 3_500_000.0)

        target_res = (PX_M, PX_M)
        out_hwc, out_tf = _reproject_tile(
            hwc, transform, "EPSG:32650", "EPSG:32651", target_res=target_res,
        )

        assert abs(abs(float(out_tf.a)) - PX_M) < 1e-3
        assert abs(abs(float(out_tf.e)) - PX_M) < 1e-3
