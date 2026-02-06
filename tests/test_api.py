"""Tests for the public API (get_embedding, get_embeddings_batch).

These use a mock embedder registered in the test so they don't require
GEE, torch, or any real model weights.
"""
import numpy as np
import pytest

from rs_embed.core import registry
from rs_embed.core.embedding import Embedding
from rs_embed.core.specs import PointBuffer, TemporalSpec, OutputSpec
from rs_embed.embedders.base import EmbedderBase


# ── mock embedder ──────────────────────────────────────────────────

class _MockEmbedder(EmbedderBase):
    """Returns a deterministic embedding without any I/O."""

    def describe(self):
        return {"type": "mock", "dim": 8}

    def get_embedding(self, *, spatial, temporal, sensor, output, backend, device="auto"):
        vec = np.arange(8, dtype=np.float32)
        return Embedding(data=vec, meta={"model": self.model_name, "output": output.mode})


@pytest.fixture(autouse=True)
def register_mock():
    registry._REGISTRY.clear()
    registry.register("mock_model")(_MockEmbedder)
    yield
    registry._REGISTRY.clear()


# ── helpers ────────────────────────────────────────────────────────

_SPATIAL = PointBuffer(lon=0.0, lat=0.0, buffer_m=512)
_TEMPORAL = TemporalSpec.year(2024)


# ══════════════════════════════════════════════════════════════════════
# get_embedding
# ══════════════════════════════════════════════════════════════════════

def test_get_embedding_returns_embedding():
    from rs_embed.api import get_embedding

    emb = get_embedding("mock_model", spatial=_SPATIAL, temporal=_TEMPORAL)
    assert isinstance(emb, Embedding)
    assert emb.data.shape == (8,)
    assert emb.meta["model"] == "mock_model"


def test_get_embedding_pooled_mode():
    from rs_embed.api import get_embedding

    emb = get_embedding("mock_model", spatial=_SPATIAL, output=OutputSpec.pooled())
    assert emb.meta["output"] == "pooled"


def test_get_embedding_grid_mode():
    from rs_embed.api import get_embedding

    emb = get_embedding("mock_model", spatial=_SPATIAL, output=OutputSpec.grid())
    assert emb.meta["output"] == "grid"


def test_get_embedding_unknown_model():
    from rs_embed.api import get_embedding
    from rs_embed.core.errors import ModelError

    with pytest.raises(ModelError, match="Unknown model"):
        get_embedding("nonexistent", spatial=_SPATIAL)


# ══════════════════════════════════════════════════════════════════════
# get_embeddings_batch
# ══════════════════════════════════════════════════════════════════════

def test_get_embeddings_batch():
    from rs_embed.api import get_embeddings_batch

    spatials = [
        PointBuffer(lon=0.0, lat=0.0, buffer_m=256),
        PointBuffer(lon=1.0, lat=1.0, buffer_m=256),
        PointBuffer(lon=2.0, lat=2.0, buffer_m=256),
    ]
    results = get_embeddings_batch("mock_model", spatials=spatials, temporal=_TEMPORAL)
    assert len(results) == 3
    for emb in results:
        assert isinstance(emb, Embedding)


def test_get_embeddings_batch_empty():
    from rs_embed.api import get_embeddings_batch
    from rs_embed.core.errors import ModelError

    with pytest.raises(ModelError, match="non-empty"):
        get_embeddings_batch("mock_model", spatials=[], temporal=_TEMPORAL)
