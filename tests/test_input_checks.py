import numpy as np

from rs_embed.core.input_checks import inspect_chw, maybe_inspect_chw
from rs_embed.core.input_checks import checks_enabled, checks_should_raise, checks_save_dir


def test_inspect_chw_non_array():
    report = inspect_chw("not-array")
    assert report["ok"] is False
    assert any("not a numpy array" in s for s in report["issues"])


def test_inspect_chw_flags_fill_and_range():
    x = np.zeros((2, 4, 4), dtype=np.float32)
    report = inspect_chw(x, value_range=(1.0, 2.0), fill_value=0.0)
    assert report["ok"] is False
    assert report.get("fill_frac", 0.0) > 0.98
    assert report.get("outside_range_frac", 0.0) > 0.9


def test_maybe_inspect_chw_disabled(monkeypatch):
    monkeypatch.delenv("RS_EMBED_CHECK_INPUT", raising=False)
    report = maybe_inspect_chw(np.zeros((1, 2, 2), dtype=np.float32))
    assert report is None


def test_maybe_inspect_chw_enabled_meta(monkeypatch):
    monkeypatch.setenv("RS_EMBED_CHECK_INPUT", "1")
    meta = {}
    report = maybe_inspect_chw(
        np.zeros((1, 2, 2), dtype=np.float32),
        name="test",
        meta=meta,
    )
    assert report is not None
    assert "input_checks" in meta
    assert "test" in meta["input_checks"]
    assert meta.get("input_checks_config", {}).get("enabled") is True


def test_checks_flags_env_override(monkeypatch):
    monkeypatch.setenv("RS_EMBED_CHECK_INPUT", "1")
    monkeypatch.setenv("RS_EMBED_CHECK_RAISE", "0")
    assert checks_enabled() is True
    assert checks_should_raise() is False


def test_checks_save_dir_env(monkeypatch):
    monkeypatch.setenv("RS_EMBED_CHECK_SAVE_DIR", "/tmp/rs_embed_checks")
    assert checks_save_dir() == "/tmp/rs_embed_checks"
