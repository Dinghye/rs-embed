from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np

from ..tools.serialization import jsonable
from ..tools.runtime import run_with_retry
from .point_payload import build_one_point_payload  # re-export


def write_one_payload(
    *,
    out_path: str,
    arrays: Dict[str, np.ndarray],
    manifest: Dict[str, Any],
    save_manifest: bool,
    fmt: str,
    max_retries: int,
    retry_backoff_s: float,
) -> Dict[str, Any]:
    from ..writers import write_arrays

    return run_with_retry(
        lambda: write_arrays(
            fmt=fmt,
            out_path=out_path,
            arrays=arrays,
            manifest=jsonable(manifest),
            save_manifest=save_manifest,
        ),
        retries=max_retries,
        backoff_s=retry_backoff_s,
    )
