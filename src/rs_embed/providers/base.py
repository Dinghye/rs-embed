from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np

class ProviderBase:
    name: str = "base"

    def ensure_ready(self) -> None:
        raise NotImplementedError

    def fetch_array_chw(self, *, image: Any, bands: Tuple[str, ...], region: Any,
                        scale_m: int, fill_value: float) -> np.ndarray:
        raise NotImplementedError