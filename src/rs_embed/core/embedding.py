from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Union

import numpy as np
import xarray as xr

@dataclass
class Embedding:
    data: Union[np.ndarray, xr.DataArray]
    meta: Dict[str, Any]