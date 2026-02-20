from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ...core.export_helpers import sensor_cache_key as _sensor_cache_key
from ...core.specs import SensorSpec
from ...providers.gee import _resolve_band_aliases


def sensor_fetch_group_key(sensor: SensorSpec) -> Tuple[str, int, int, float, str]:
    """Fetch identity excluding bands; used to build reusable band supersets."""
    return (
        str(sensor.collection),
        int(sensor.scale_m),
        int(sensor.cloudy_pct),
        float(sensor.fill_value),
        str(sensor.composite),
    )


def select_prefetched_channels(x_chw: np.ndarray, idx: Tuple[int, ...]) -> np.ndarray:
    if len(idx) == x_chw.shape[0] and all(i == j for j, i in enumerate(idx)):
        return x_chw
    return x_chw[list(idx), :, :]


def build_gee_prefetch_plan(
    *,
    models: List[str],
    resolved_sensor: Dict[str, Optional[SensorSpec]],
    model_type: Dict[str, str],
) -> Tuple[
    Dict[str, SensorSpec],  # sensor_by_key
    Dict[str, SensorSpec],  # fetch_sensor_by_key
    Dict[str, Tuple[str, Tuple[int, ...]]],  # sensor_key -> (fetch_key, channel_idx)
    Dict[str, List[str]],  # sensor_models
    Dict[str, List[str]],  # fetch_members
]:
    sensor_by_key: Dict[str, SensorSpec] = {}
    sensor_models: Dict[str, List[str]] = {}
    for m in models:
        sspec = resolved_sensor.get(m)
        if sspec is None or "precomputed" in (model_type.get(m) or ""):
            continue
        skey = _sensor_cache_key(sspec)
        sensor_by_key.setdefault(skey, sspec)
        sensor_models.setdefault(skey, []).append(m)

    groups: Dict[Tuple[str, int, int, float, str], List[Tuple[str, SensorSpec, Tuple[str, ...]]]] = {}
    for skey, sspec in sensor_by_key.items():
        gkey = sensor_fetch_group_key(sspec)
        groups.setdefault(gkey, []).append((skey, sspec, _resolve_band_aliases(sspec.collection, sspec.bands)))

    fetch_sensor_by_key: Dict[str, SensorSpec] = {}
    sensor_to_fetch: Dict[str, Tuple[str, Tuple[int, ...]]] = {}
    fetch_members: Dict[str, List[str]] = {}

    for members in groups.values():
        union_bands: List[str] = []
        seen: set[str] = set()
        for _, _, rbands in members:
            for b in rbands:
                if b not in seen:
                    seen.add(b)
                    union_bands.append(b)
        if not union_bands:
            continue

        base = members[0][1]
        fetch_sensor = SensorSpec(
            collection=str(base.collection),
            bands=tuple(union_bands),
            scale_m=int(base.scale_m),
            cloudy_pct=int(base.cloudy_pct),
            fill_value=float(base.fill_value),
            composite=str(base.composite),
            check_input=bool(getattr(base, "check_input", False)),
            check_raise=bool(getattr(base, "check_raise", True)),
            check_save_dir=getattr(base, "check_save_dir", None),
        )
        fetch_key = _sensor_cache_key(fetch_sensor)
        fetch_sensor_by_key[fetch_key] = fetch_sensor
        fetch_members.setdefault(fetch_key, [])

        band_pos = {b: i for i, b in enumerate(fetch_sensor.bands)}
        for member_key, _member_sensor, member_bands in members:
            idx = tuple(band_pos[b] for b in member_bands)
            sensor_to_fetch[member_key] = (fetch_key, idx)
            if member_key not in fetch_members[fetch_key]:
                fetch_members[fetch_key].append(member_key)

    return sensor_by_key, fetch_sensor_by_key, sensor_to_fetch, sensor_models, fetch_members
