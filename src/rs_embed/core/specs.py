from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Tuple

from .errors import SpecError

@dataclass(frozen=True)
class BBox:
    """EPSG:4326 bbox."""
    minlon: float
    minlat: float
    maxlon: float
    maxlat: float
    crs: str = "EPSG:4326"

    def validate(self) -> None:
        if self.crs != "EPSG:4326":
            raise SpecError("BBox currently must be EPSG:4326 (v0.1).")
        if not (self.minlon < self.maxlon and self.minlat < self.maxlat):
            raise SpecError("Invalid bbox bounds.")

@dataclass(frozen=True)
class PointBuffer:
    lon: float
    lat: float
    buffer_m: float
    crs: str = "EPSG:4326"

    def validate(self) -> None:
        if self.crs != "EPSG:4326":
            raise SpecError("PointBuffer currently must be EPSG:4326 (v0.1).")
        if self.buffer_m <= 0:
            raise SpecError("buffer_m must be positive.")

SpatialSpec = BBox | PointBuffer

@dataclass(frozen=True)
class TemporalSpec:
    """Either year-based (for annual products) or start/end range."""
    mode: Literal["year", "range"]
    year: Optional[int] = None
    start: Optional[str] = None
    end: Optional[str] = None

    @staticmethod
    def year(y: int) -> "TemporalSpec":
        return TemporalSpec(mode="year", year=y)

    @staticmethod
    def range(start: str, end: str) -> "TemporalSpec":
        return TemporalSpec(mode="range", start=start, end=end)

    def validate(self) -> None:
        if self.mode == "year":
            if self.year is None:
                raise SpecError("TemporalSpec.year requires year.")
        elif self.mode == "range":
            if not self.start or not self.end:
                raise SpecError("TemporalSpec.range requires start and end.")
        else:
            raise SpecError(f"Unknown TemporalSpec mode: {self.mode}")

@dataclass(frozen=True)
class SensorSpec:
    """For on-the-fly models: what imagery to pull and how."""
    collection: str
    bands: Tuple[str, ...]
    scale_m: int = 10
    cloudy_pct: int = 30
    fill_value: float = 0.0
    composite: Literal["median", "mosaic"] = "median"

    # Optional: on-the-fly input inspection for GEE downloads.
    # If enabled, embedders can attach a compact stats report into Embedding.meta
    # (and optionally raise if issues are detected).
    check_input: bool = False
    check_raise: bool = True
    check_save_dir: Optional[str] = None

@dataclass(frozen=True)
class OutputSpec:
    mode: Literal["grid", "pooled"]
    scale_m: int = 10
    pooling: Literal["mean", "max"] = "mean"

    @staticmethod
    def grid(scale_m: int = 10) -> "OutputSpec":
        return OutputSpec(mode="grid", scale_m=scale_m)

    @staticmethod
    def pooled(pooling: Literal["mean","max"]="mean") -> "OutputSpec":
        return OutputSpec(mode="pooled", scale_m=10, pooling=pooling)