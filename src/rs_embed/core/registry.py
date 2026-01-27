# src/rs_embed/core/registry.py
from __future__ import annotations
from typing import Dict, Type, Any

_REGISTRY: Dict[str, Type[Any]] = {}

def register(name: str):
    """Decorator to register an embedder class by name."""
    def deco(cls: Type[Any]):
        _REGISTRY[name.lower()] = cls
        setattr(cls, "model_name", name.lower())
        return cls
    return deco

def get_embedder_cls(name: str) -> Type[Any]:
    k = name.lower()
    if k not in _REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(_REGISTRY.keys())}")
    return _REGISTRY[k]

def list_models():
    return sorted(_REGISTRY.keys())