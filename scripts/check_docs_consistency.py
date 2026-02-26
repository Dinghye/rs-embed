#!/usr/bin/env python3
from __future__ import annotations

"""Lightweight checks to keep docs aligned with code.

Current checks:
- All registered model IDs in `MODEL_SPECS` appear in `docs/models.md`
- All registered model IDs in `MODEL_SPECS` appear in `README.md`
- Selected public APIs exported from `rs_embed.__init__` are documented in `docs/api.md`
"""

from pathlib import Path
import ast
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def _parse_model_specs_keys(catalog_src: str) -> list[str]:
    tree = ast.parse(catalog_src)
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "MODEL_SPECS":
            value = node.value
            if not isinstance(value, ast.Dict):
                raise ValueError("MODEL_SPECS is not a dict literal.")
            keys: list[str] = []
            for key in value.keys:
                if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                    raise ValueError("MODEL_SPECS contains a non-string key.")
                keys.append(key.value)
            return keys
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "MODEL_SPECS":
                    value = node.value
                    if not isinstance(value, ast.Dict):
                        raise ValueError("MODEL_SPECS is not a dict literal.")
                    keys = []
                    for key in value.keys:
                        if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                            raise ValueError("MODEL_SPECS contains a non-string key.")
                        keys.append(key.value)
                    return keys
    raise ValueError("MODEL_SPECS not found in catalog.py")


def _parse_dunder_all(init_src: str) -> set[str]:
    tree = ast.parse(init_src)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "__all__":
                    if not isinstance(node.value, (ast.List, ast.Tuple)):
                        raise ValueError("__all__ is not a list/tuple literal.")
                    out: set[str] = set()
                    for elt in node.value.elts:
                        if not isinstance(elt, ast.Constant) or not isinstance(elt.value, str):
                            raise ValueError("__all__ contains a non-string item.")
                        out.add(elt.value)
                    return out
    raise ValueError("__all__ not found in src/rs_embed/__init__.py")


def _check_model_mentions(errors: list[str]) -> None:
    keys = sorted(_parse_model_specs_keys(_read("src/rs_embed/embedders/catalog.py")))
    model_docs_paths = [
        "docs/models.md",
        "docs/models_reference.md",
    ]
    docs_models = "\n".join(_read(p) for p in model_docs_paths)
    readme = _read("README.md")

    missing_docs_models = [k for k in keys if k not in docs_models]
    missing_readme = [k for k in keys if k not in readme]

    if missing_docs_models:
        errors.append(
            "model docs are missing registered model IDs: " + ", ".join(missing_docs_models)
        )
    if missing_readme:
        errors.append(
            "README.md is missing registered model IDs: " + ", ".join(missing_readme)
        )


def _check_api_docs(errors: list[str]) -> None:
    exported = _parse_dunder_all(_read("src/rs_embed/__init__.py"))
    api_doc_paths = [
        "docs/api.md",
        "docs/api_specs.md",
        "docs/api_embedding.md",
        "docs/api_export.md",
        "docs/api_inspect.md",
    ]
    api_doc = "\n".join(_read(p) for p in api_doc_paths)

    # Public APIs that should have sections in api.md if exported.
    section_requirements = {
        "get_embedding": "get_embedding(",
        "get_embeddings_batch": "get_embeddings_batch(",
        "export_batch": "export_batch(",
        "export_npz": "export_npz(",
        "inspect_provider_patch": "inspect_provider_patch(",
        "inspect_gee_patch": "inspect_gee_patch(",
    }

    for name, marker in section_requirements.items():
        if name in exported and marker not in api_doc:
            errors.append(f"docs/api.md is missing documentation marker for exported API `{name}` ({marker!r}).")


def main() -> int:
    errors: list[str] = []
    try:
        _check_model_mentions(errors)
        _check_api_docs(errors)
    except Exception as e:
        print(f"[docs-check] Internal error: {e}", file=sys.stderr)
        return 2

    if errors:
        print("[docs-check] FAIL")
        for err in errors:
            print(f"- {err}")
        return 1

    print("[docs-check] OK: docs consistency checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
