"""Tests for GEE provider helpers that don't require actual GEE auth.

Band alias resolution is pure logic — no network calls needed.
"""
import pytest

from rs_embed.providers.gee import _resolve_band_aliases


# ══════════════════════════════════════════════════════════════════════
# Sentinel-2 aliases
# ══════════════════════════════════════════════════════════════════════

def test_s2_rgb_aliases():
    result = _resolve_band_aliases(
        "COPERNICUS/S2_SR_HARMONIZED", ("RED", "GREEN", "BLUE")
    )
    assert result == ("B4", "B3", "B2")


def test_s2_nir_aliases():
    result = _resolve_band_aliases(
        "COPERNICUS/S2_SR_HARMONIZED", ("NIR", "NIR_NARROW")
    )
    assert result == ("B8", "B8A")


def test_s2_swir_aliases():
    result = _resolve_band_aliases(
        "COPERNICUS/S2_SR_HARMONIZED", ("SWIR1", "SWIR2")
    )
    assert result == ("B11", "B12")


def test_s2_red_edge_aliases():
    result = _resolve_band_aliases(
        "COPERNICUS/S2_SR_HARMONIZED", ("RE1", "RE2", "RE3", "RE4")
    )
    assert result == ("B5", "B6", "B7", "B8A")


def test_s2_passthrough_real_bands():
    result = _resolve_band_aliases(
        "COPERNICUS/S2_SR_HARMONIZED", ("B4", "B3", "B2")
    )
    assert result == ("B4", "B3", "B2")


def test_s2_toa_also_resolves():
    result = _resolve_band_aliases(
        "COPERNICUS/S2", ("RED", "GREEN", "BLUE")
    )
    assert result == ("B4", "B3", "B2")


# ══════════════════════════════════════════════════════════════════════
# Landsat 8/9 aliases
# ══════════════════════════════════════════════════════════════════════

def test_landsat89_rgb():
    result = _resolve_band_aliases(
        "LANDSAT/LC08/C02/T1_L2", ("RED", "GREEN", "BLUE")
    )
    assert result == ("SR_B4", "SR_B3", "SR_B2")


def test_landsat89_nir_swir():
    result = _resolve_band_aliases(
        "LANDSAT/LC09/C02/T1_L2", ("NIR", "SWIR1", "SWIR2")
    )
    assert result == ("SR_B5", "SR_B6", "SR_B7")


# ══════════════════════════════════════════════════════════════════════
# Landsat 4/5/7 aliases
# ══════════════════════════════════════════════════════════════════════

def test_landsat457_rgb():
    result = _resolve_band_aliases(
        "LANDSAT/LE07/C02/T1_L2", ("RED", "GREEN", "BLUE")
    )
    assert result == ("SR_B3", "SR_B2", "SR_B1")


def test_landsat5_nir():
    result = _resolve_band_aliases(
        "LANDSAT/LT05/C02/T1_L2", ("NIR",)
    )
    assert result == ("SR_B4",)


# ══════════════════════════════════════════════════════════════════════
# Unknown collection — no aliasing
# ══════════════════════════════════════════════════════════════════════

def test_unknown_collection_passthrough():
    result = _resolve_band_aliases(
        "SOME/OTHER/COLLECTION", ("RED", "GREEN", "BLUE")
    )
    # No mapping → returned as-is
    assert result == ("RED", "GREEN", "BLUE")


def test_empty_bands():
    result = _resolve_band_aliases("COPERNICUS/S2_SR_HARMONIZED", ())
    assert result == ()


# ══════════════════════════════════════════════════════════════════════
# Case insensitivity of aliases
# ══════════════════════════════════════════════════════════════════════

def test_alias_case_insensitive():
    result = _resolve_band_aliases(
        "COPERNICUS/S2_SR_HARMONIZED", ("red", "green", "blue")
    )
    assert result == ("B4", "B3", "B2")


def test_mixed_alias_and_real():
    result = _resolve_band_aliases(
        "COPERNICUS/S2_SR_HARMONIZED", ("RED", "B3", "BLUE")
    )
    assert result == ("B4", "B3", "B2")
