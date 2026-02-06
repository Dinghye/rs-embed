import pytest

from rs_embed.core.errors import SpecError
from rs_embed.core.specs import BBox, PointBuffer, TemporalSpec, OutputSpec


def test_bbox_validate_ok():
    bbox = BBox(minlon=0.0, minlat=0.0, maxlon=1.0, maxlat=1.0)
    bbox.validate()


def test_bbox_validate_invalid_bounds():
    bbox = BBox(minlon=1.0, minlat=0.0, maxlon=0.0, maxlat=1.0)
    with pytest.raises(SpecError, match="Invalid bbox bounds"):
        bbox.validate()


def test_pointbuffer_validate_ok():
    pb = PointBuffer(lon=1.0, lat=2.0, buffer_m=100.0)
    pb.validate()


def test_pointbuffer_validate_invalid():
    pb = PointBuffer(lon=1.0, lat=2.0, buffer_m=0.0)
    with pytest.raises(SpecError, match="buffer_m must be positive"):
        pb.validate()


def test_temporal_spec_year_and_range():
    TemporalSpec.year(2024).validate()
    TemporalSpec.range("2022-01-01", "2022-02-01").validate()


def test_temporal_spec_invalid_mode():
    ts = TemporalSpec(mode="oops")
    with pytest.raises(SpecError, match="Unknown TemporalSpec mode"):
        ts.validate()


def test_output_spec_factories():
    grid = OutputSpec.grid(scale_m=20)
    pooled = OutputSpec.pooled(pooling="max")
    assert grid.mode == "grid"
    assert grid.scale_m == 20
    assert pooled.mode == "pooled"
    assert pooled.pooling == "max"
