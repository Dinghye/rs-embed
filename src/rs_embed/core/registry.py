# src/rs_embed/core/registry.py
from __future__ import annotations
from typing import Dict, Type, Any, Optional

from .errors import ModelError

_REGISTRY: Dict[str, Type[Any]] = {}
_REGISTRY_IMPORT_ERROR: Optional[BaseException] = None

def register(name: str):
    """Decorator to register an embedder class by name."""
    def deco(cls: Type[Any]):
        _REGISTRY[name.lower()] = cls
        setattr(cls, "model_name", name.lower())
        return cls
    return deco

def _ensure_registry_loaded() -> None:
    """Populate registry on first use via a lightweight side-effect import."""
    global _REGISTRY_IMPORT_ERROR
    if _REGISTRY:
        return
    try:
        import rs_embed.embedders  # noqa: F401
        _REGISTRY_IMPORT_ERROR = None
    except Exception as e:
        _REGISTRY_IMPORT_ERROR = e
        return

def get_embedder_cls(name: str) -> Type[Any]:
    _ensure_registry_loaded()
    k = name.lower()
    if k not in _REGISTRY:
        msg = (
            f"Unknown model '{name}'. Available: {sorted(_REGISTRY.keys())}. "
            f"If this list is empty, ensure embedders are importable "
            f"(e.g. optional deps like torch/ee are installed)."
        )
        if (not _REGISTRY) and (_REGISTRY_IMPORT_ERROR is not None):
            msg += (
                f" Last embedder import error: "
                f"{type(_REGISTRY_IMPORT_ERROR).__name__}: {_REGISTRY_IMPORT_ERROR}"
            )
        raise ModelError(
            msg
        )
    return _REGISTRY[k]

def list_models():
    _ensure_registry_loaded()
    return sorted(_REGISTRY.keys())
