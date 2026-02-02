from __future__ import annotations

import argparse
import json
import sys
from typing import Optional, Tuple

from .core.specs import BBox, PointBuffer, SensorSpec, TemporalSpec
from .inspect import inspect_gee_patch


def _parse_bands(s: str) -> Tuple[str, ...]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("--bands must be a comma-separated list, e.g. 'B4,B3,B2'")
    return tuple(parts)


def _parse_value_range(s: Optional[str]) -> Optional[Tuple[float, float]]:
    if not s:
        return None
    try:
        lo, hi = s.split(",")
        return (float(lo), float(hi))
    except Exception as e:
        raise argparse.ArgumentTypeError("--value-range must be 'lo,hi' (floats)") from e


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rs-embed", description="rs-embed utilities")
    sub = p.add_subparsers(dest="cmd", required=True)

    ig = sub.add_parser(
        "inspect-gee",
        help="Download a patch from Google Earth Engine and output an input-inspection report (no model run).",
    )

    ig.add_argument("--collection", required=True, help="GEE ImageCollection (or Image) id")
    ig.add_argument("--bands", required=True, type=_parse_bands, help="Comma-separated band list")
    ig.add_argument("--scale-m", type=int, default=10, help="Pixel scale (meters)")
    ig.add_argument("--cloudy-pct", type=int, default=30, help="Best-effort cloud filter (CLOUDY_PIXEL_PERCENTAGE)")
    ig.add_argument("--fill-value", type=float, default=0.0, help="Default fill value used by sampleRectangle")
    ig.add_argument("--composite", choices=["median", "mosaic"], default="median", help="How to composite collection")

    # Spatial (either bbox or pointbuffer)
    sp = ig.add_mutually_exclusive_group(required=True)
    sp.add_argument(
        "--bbox",
        metavar=("MINLON", "MINLAT", "MAXLON", "MAXLAT"),
        nargs=4,
        type=float,
        help="EPSG:4326 bbox",
    )
    sp.add_argument(
        "--pointbuffer",
        metavar=("LON", "LAT", "BUFFER_M"),
        nargs=3,
        type=float,
        help="EPSG:4326 pointbuffer (meters)",
    )

    # Temporal
    tg = ig.add_mutually_exclusive_group(required=False)
    tg.add_argument("--year", type=int, help="Year mode (will use [year-01-01, year+1-01-01)")
    tg.add_argument("--range", metavar=("START", "END"), nargs=2, help="Date range, e.g. 2022-06-01 2022-09-01")

    ig.add_argument("--value-range", default=None, help="Optional sanity range 'lo,hi' for values")
    ig.add_argument(
        "--save-dir",
        default=None,
        help="Optional directory to save a quicklook PNG (first 3 bands)",
    )

    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)

    if args.cmd == "inspect-gee":
        sensor = SensorSpec(
            collection=args.collection,
            bands=args.bands,
            scale_m=args.scale_m,
            cloudy_pct=args.cloudy_pct,
            fill_value=args.fill_value,
            composite=args.composite,
            check_input=True,
            check_raise=False,
            check_save_dir=args.save_dir,
        )

        if args.bbox is not None:
            spatial = BBox(*args.bbox)
        else:
            lon, lat, buf = args.pointbuffer
            spatial = PointBuffer(lon=lon, lat=lat, buffer_m=buf)

        temporal = None
        if args.year is not None:
            temporal = TemporalSpec.year(int(args.year))
        elif args.range is not None:
            temporal = TemporalSpec.range(args.range[0], args.range[1])

        value_range = _parse_value_range(args.value_range)
        out = inspect_gee_patch(spatial=spatial, temporal=temporal, sensor=sensor, value_range=value_range)

        # JSON output (stdout)
        json.dump(out, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
        return

    raise SystemExit(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
