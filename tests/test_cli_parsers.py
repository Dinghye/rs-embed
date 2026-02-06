import pytest

from rs_embed import cli


def test_parse_bands_and_models():
    assert cli._parse_bands("B4, B3, B2") == ("B4", "B3", "B2")
    assert cli._parse_models("m1, m2") == ["m1", "m2"]


def test_parse_bands_empty():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["inspect-gee", "--bands", "", "--collection", "c", "--bbox", "0", "0", "1", "1"])


def test_parse_value_range():
    assert cli._parse_value_range("1,2") == (1.0, 2.0)
    with pytest.raises(Exception):
        cli._parse_value_range("bad")


def test_parse_spatial_bbox():
    args = cli.build_parser().parse_args(
        [
            "inspect-gee",
            "--collection",
            "c",
            "--bands",
            "B1",
            "--bbox",
            "0",
            "0",
            "1",
            "1",
        ]
    )
    spatial = cli._parse_spatial(args)
    assert spatial.minlon == 0.0
    assert spatial.maxlon == 1.0


def test_parse_spatial_pointbuffer():
    args = cli.build_parser().parse_args(
        [
            "inspect-gee",
            "--collection",
            "c",
            "--bands",
            "B1",
            "--pointbuffer",
            "1",
            "2",
            "128",
        ]
    )
    spatial = cli._parse_spatial(args)
    assert spatial.lon == 1.0
    assert spatial.buffer_m == 128.0
