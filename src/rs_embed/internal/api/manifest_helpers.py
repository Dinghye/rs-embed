from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from ...core.export_helpers import jsonable as _jsonable
from ...core.export_helpers import utc_ts as _utc_ts
from ...core.specs import OutputSpec, SpatialSpec, TemporalSpec


def load_json_dict(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            return payload
    except Exception:
        return None
    return None


def point_resume_manifest(
    *,
    point_index: int,
    spatial: SpatialSpec,
    temporal: Optional[TemporalSpec],
    output: OutputSpec,
    backend: str,
    device: str,
    out_file: str,
) -> Dict[str, Any]:
    json_path = os.path.splitext(out_file)[0] + ".json"
    manifest = load_json_dict(json_path)
    if manifest is None:
        manifest = {
            "created_at": _utc_ts(),
            "point_index": int(point_index),
            "status": "skipped",
            "stage": "resume",
            "reason": "output_exists",
            "backend": backend,
            "device": device,
            "models": [],
            "spatial": _jsonable(spatial),
            "temporal": _jsonable(temporal),
            "output": _jsonable(output),
        }
    manifest["resume_skipped"] = True
    manifest["resume_output_path"] = out_file
    manifest.setdefault("point_index", int(point_index))
    manifest.setdefault("status", "ok")
    return manifest


def combined_resume_manifest(
    *,
    spatials: List[SpatialSpec],
    temporal: Optional[TemporalSpec],
    output: OutputSpec,
    backend: str,
    device: str,
    out_file: str,
) -> Dict[str, Any]:
    json_path = os.path.splitext(out_file)[0] + ".json"
    manifest = load_json_dict(json_path)
    if manifest is None:
        manifest = {
            "created_at": _utc_ts(),
            "status": "skipped",
            "stage": "resume",
            "reason": "output_exists",
            "backend": backend,
            "device": device,
            "n_items": len(spatials),
            "temporal": _jsonable(temporal),
            "output": _jsonable(output),
            "spatials": [_jsonable(s) for s in spatials],
            "models": [],
        }
    manifest["resume_skipped"] = True
    manifest["resume_output_path"] = out_file
    manifest.setdefault("status", "ok")
    return manifest


def point_failure_manifest(
    *,
    point_index: int,
    spatial: SpatialSpec,
    temporal: Optional[TemporalSpec],
    output: OutputSpec,
    backend: str,
    device: str,
    stage: str,
    error: Exception,
) -> Dict[str, Any]:
    return {
        "created_at": _utc_ts(),
        "point_index": int(point_index),
        "status": "failed",
        "stage": stage,
        "error": repr(error),
        "backend": backend,
        "device": device,
        "models": [],
        "spatial": _jsonable(spatial),
        "temporal": _jsonable(temporal),
        "output": _jsonable(output),
    }
