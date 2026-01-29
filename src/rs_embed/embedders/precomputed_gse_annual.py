from __future__ import annotations
from typing import Any, Dict, Optional, Tuple

import numpy as np
import xarray as xr

from ..core.registry import register
from ..core.embedding import Embedding
from ..core.errors import ModelError
from ..core.specs import SpatialSpec, TemporalSpec, SensorSpec, OutputSpec
from ..providers.gee import GEEProvider
from ..ops.pooling import pool_chw_to_vec
from .base import EmbedderBase
from .meta_utils import build_meta, temporal_midpoint_str

@register("gse_annual")
class GSEAnnualEmbedder(EmbedderBase):
    """
    Precomputed embeddings on GEE:
      ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
    """

    def describe(self) -> Dict[str, Any]:
        return {
            "type": "precomputed",
            "backend": ["gee"],
            "temporal": {"mode": "year"},
            "output": ["grid", "pooled"],
            "source": "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL",
            "notes": "Uses sampleRectangle in EPSG:3857; returns [C,H,W] or pooled [C].",
        }

    def get_embedding(
        self,
        *,
        spatial: SpatialSpec,
        temporal: Optional[TemporalSpec],
        sensor: Optional[SensorSpec],
        output: OutputSpec,
        backend: str,
        device: str = "auto",
    ) -> Embedding:
        if backend.lower() != "gee":
            raise ModelError("gse_annual only supports backend='gee' in v0.1.")
        if temporal is None:
            raise ModelError("gse_annual requires TemporalSpec.year(year=...).")
        temporal.validate()
        if temporal.mode != "year":
            raise ModelError("gse_annual only supports TemporalSpec.year in v0.1.")

        import ee
        provider = GEEProvider(auto_auth=True)
        region = provider.get_region_3857(spatial)
        print(region.getInfo())

        col = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
        img = col.filterDate(f"{temporal.year}-01-01", f"{temporal.year+1}-01-01").mosaic()
        img = img.reproject(crs="EPSG:3857", scale=output.scale_m)

        # We don't know band names until sampling; simplest: sample all bands by reading properties keys.
        rect = img.sampleRectangle(region=region, defaultValue=-9999).getInfo()
        props = rect["properties"]
        band_names = tuple(props.keys())
        arrs = [np.array(props[b], dtype=np.float32) for b in band_names]
        emb_chw = np.stack(arrs, axis=0).astype(np.float32)
        emb_chw[emb_chw == -9999] = np.nan

        meta = build_meta(
            model=self.model_name,
            kind="precomputed",
            backend="gee",
            source="GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL",
            sensor=None,
            temporal=temporal,
            image_size=None,
            input_time=temporal_midpoint_str(temporal),
            extra={
                "year": temporal.year,
                "scale_m": output.scale_m,
                "bands": band_names,
            },
        )

        if output.mode == "pooled":
            vec = pool_chw_to_vec(emb_chw, method=output.pooling)
            return Embedding(data=vec, meta={**meta, "pooling": output.pooling})

        # grid: return xarray with dims (band,y,x)
        da = xr.DataArray(
            emb_chw,
            dims=("band", "y", "x"),
            coords={"band": list(band_names)},
            name="embedding",
            attrs=meta,
        )
        return Embedding(data=da, meta=meta)
