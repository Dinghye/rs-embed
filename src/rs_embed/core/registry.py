# src/rs_embed/core/registry.py
from __future__ import annotations
import importlib
from typing import Dict, Type, Any, Optional

from rs_embed.embedders.catalog import MODEL_SPECS, canonical_model_id

from .errors import ModelError

_REGISTRY: Dict[str, Type[Any]] = {}
_REGISTRY_IMPORT_ERROR: Optional[BaseException] = None

def register(name: str):
    """Decorator to register an embedder class by name."""
    def deco(cls: Type[Any]):
        model_id = canonical_model_id(name)
        _REGISTRY[model_id] = cls
        setattr(cls, "model_name", model_id)
        return cls
    return deco

def _ensure_registry_loaded() -> None:
    """Backward-compatible hook; registry loading is now per-model lazy import."""
    return


def _try_lazy_load_model(name: str) -> None:
    """Load only the module that owns `name`, then backfill registration if needed."""
    global _REGISTRY_IMPORT_ERROR

    model_id = canonical_model_id(name)
    if model_id in _REGISTRY:
        return
    spec = MODEL_SPECS.get(model_id)
    if spec is None:
        return
    module_name, class_name = spec
    fqmn = f"rs_embed.embedders.{module_name}"
    try:
        mod = importlib.import_module(fqmn)
    except Exception as e:
        _REGISTRY_IMPORT_ERROR = e
        return

    try:
        cls = getattr(mod, class_name)
    except Exception as e:
        _REGISTRY_IMPORT_ERROR = e
        return

    # If decorators did not run in this process state (e.g. registry was cleared),
    # repopulate from the imported module class symbol.
    _REGISTRY[model_id] = cls
    setattr(cls, "model_name", model_id)
    _REGISTRY_IMPORT_ERROR = None

def get_embedder_cls(name: str) -> Type[Any]:
    _ensure_registry_loaded()
    k = canonical_model_id(name)
    if k not in _REGISTRY:
        _try_lazy_load_model(k)
    if k not in _REGISTRY:
        msg = (
            f"Unknown model '{name}'. Available: {sorted(_REGISTRY.keys())}. "
            f"If this list is empty, ensure requested embedder module is importable "
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
