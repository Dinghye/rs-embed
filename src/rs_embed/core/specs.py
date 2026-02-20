from __future__ import annotations
from dataclasses import dataclass
from datetime import date
from typing import Literal, Optional, Tuple

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
            try:
                y = int(self.year)
            except Exception as e:
                raise SpecError("TemporalSpec.year requires an integer year.") from e
            if y < 1 or y > 9999:
                raise SpecError("TemporalSpec.year must be in [1, 9999].")
        elif self.mode == "range":
            if not self.start or not self.end:
                raise SpecError("TemporalSpec.range requires start and end.")
            try:
                start_d = date.fromisoformat(str(self.start))
                end_d = date.fromisoformat(str(self.end))
            except Exception as e:
                raise SpecError("TemporalSpec.range expects ISO dates 'YYYY-MM-DD'.") from e
            if start_d >= end_d:
                raise SpecError("TemporalSpec.range requires start < end.")
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
    # Grid orientation policy:
    # - north_up: normalize y-axis to north->south when metadata permits.
    # - native: keep model/provider native orientation.
    grid_orientation: Literal["north_up", "native"] = "north_up"

    @staticmethod
    def grid(
        scale_m: int = 10,
        *,
        grid_orientation: Literal["north_up", "native"] = "north_up",
    ) -> "OutputSpec":
        return OutputSpec(mode="grid", scale_m=scale_m, grid_orientation=grid_orientation)

    @staticmethod
    def pooled(pooling: Literal["mean","max"]="mean") -> "OutputSpec":
        return OutputSpec(mode="pooled", scale_m=10, pooling=pooling, grid_orientation="north_up")
