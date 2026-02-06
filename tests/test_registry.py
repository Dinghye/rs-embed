import pytest

from rs_embed.core import registry


def test_register_and_get_embedder_cls():
    registry._REGISTRY.clear()

    @registry.register("TestModel")
    class DummyEmbedder:
        pass

    assert DummyEmbedder.model_name == "testmodel"
    assert registry.get_embedder_cls("testmodel") is DummyEmbedder
    assert registry.get_embedder_cls("TESTMODEL") is DummyEmbedder
    assert "testmodel" in registry.list_models()


def test_get_embedder_cls_missing():
    registry._REGISTRY.clear()
    with pytest.raises(KeyError, match="Unknown model"):
        registry.get_embedder_cls("missing-model")
