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

def _ensure_registry_loaded() -> None:
    """Populate registry on first use via a lightweight side-effect import."""
    if _REGISTRY:
        return
    try:
        import rs_embed.embedders  # noqa: F401
    except Exception:
        # Keep it silent here; if model still not found we raise a clear error below.
        return

def get_embedder_cls(name: str) -> Type[Any]:
    _ensure_registry_loaded()
    k = name.lower()
    if k not in _REGISTRY:
        raise KeyError(
            f"Unknown model '{name}'. Available: {sorted(_REGISTRY.keys())}. "
            f"If this list is empty, ensure embedders are importable "
            f"(e.g. optional deps like torch/ee are installed)."
        )
    return _REGISTRY[k]

def list_models():
    _ensure_registry_loaded()
    return sorted(_REGISTRY.keys())