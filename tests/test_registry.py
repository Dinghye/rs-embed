import pytest

from rs_embed.core import registry


# ── fixture to isolate registry between tests ──────────────────────

@pytest.fixture(autouse=True)
def clean_registry():
    """Clear registry before and after every test in this module."""
    registry._REGISTRY.clear()
    if hasattr(registry, "_REGISTRY_IMPORT_ERROR"):
        registry._REGISTRY_IMPORT_ERROR = None
    yield
    registry._REGISTRY.clear()
    if hasattr(registry, "_REGISTRY_IMPORT_ERROR"):
        registry._REGISTRY_IMPORT_ERROR = None


# ══════════════════════════════════════════════════════════════════════
# register + get_embedder_cls
# ══════════════════════════════════════════════════════════════════════

def test_register_and_get_embedder_cls():
    @registry.register("TestModel")
    class DummyEmbedder:
        pass

    assert DummyEmbedder.model_name == "testmodel"
    assert registry.get_embedder_cls("testmodel") is DummyEmbedder
    assert registry.get_embedder_cls("TESTMODEL") is DummyEmbedder
    assert "testmodel" in registry.list_models()


def test_get_embedder_cls_missing():
    from rs_embed.core.errors import ModelError
    with pytest.raises(ModelError, match="Unknown model"):
        registry.get_embedder_cls("missing-model")


# ── case insensitivity ─────────────────────────────────────────────

def test_register_case_insensitive():
    @registry.register("MiXeD_CaSe")
    class M:
        pass

    assert registry.get_embedder_cls("mixed_case") is M
    assert registry.get_embedder_cls("MIXED_CASE") is M


# ── multiple registrations ─────────────────────────────────────────

def test_register_multiple():
    @registry.register("alpha")
    class A:
        pass

    @registry.register("beta")
    class B:
        pass

    assert registry.get_embedder_cls("alpha") is A
    assert registry.get_embedder_cls("beta") is B
    assert registry.list_models() == ["alpha", "beta"]


# ── overwrite same name ────────────────────────────────────────────

def test_register_overwrite():
    @registry.register("dup")
    class First:
        pass

    @registry.register("dup")
    class Second:
        pass

    assert registry.get_embedder_cls("dup") is Second


# ── empty registry ─────────────────────────────────────────────────

def test_list_models_empty():
    assert registry.list_models() == []


def test_get_embedder_cls_empty_shows_available():
    from rs_embed.core.errors import ModelError
    with pytest.raises(ModelError, match="Available: \\[\\]"):
        registry.get_embedder_cls("anything")


def test_get_embedder_cls_includes_last_import_error(monkeypatch):
    from rs_embed.core.errors import ModelError

    registry._REGISTRY.clear()
    registry._REGISTRY_IMPORT_ERROR = RuntimeError("boom")
    monkeypatch.setattr(registry, "_ensure_registry_loaded", lambda: None)

    with pytest.raises(ModelError) as ei:
        registry.get_embedder_cls("anything")
    msg = str(ei.value)
    assert "Last embedder import error" in msg
    assert "RuntimeError: boom" in msg
