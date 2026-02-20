from rs_embed.core.specs import TemporalSpec
from rs_embed.embedders.meta_utils import temporal_to_dict, temporal_to_range


def test_temporal_to_range_year_is_end_exclusive():
    t = temporal_to_range(TemporalSpec.year(2022))
    assert t.mode == "range"
    assert t.start == "2022-01-01"
    assert t.end == "2023-01-01"


def test_temporal_to_dict_year_matches_end_exclusive_convention():
    d = temporal_to_dict(TemporalSpec.year(2022))
    assert d["mode"] == "year"
    assert d["start"] == "2022-01-01"
    assert d["end"] == "2023-01-01"
