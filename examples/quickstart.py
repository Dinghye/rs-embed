"""Quick smoke test for any registered on-the-fly model.

Pass ``--model`` after registering a new embedder to verify it runs end-to-end
without editing this script. Default values mimic the playground notebook
demo (temporal range + three sample points for batch mode).

How to run:

- Default (matches notebook): quickstart.py → pooled DOFA, 2021-06-01 to 2021-08-31 around (121.5, 31.2) buffer 2048 m.
- Different model: quickstart.py --model remoteclip_s2rgb --output grid --grid-scale 20
- AnySat: quickstart.py --model anysat --output pooled
- DynamicVis: quickstart.py --model dynamicvis --output pooled
- Galileo: quickstart.py --model galileo --output pooled
- WildSAT: quickstart.py --model wildsat --output pooled
  (auto-download enabled by default; optional env: RS_EMBED_WILDSAT_CKPT=/path/to/wildsat_checkpoint.pth)
- Batch sample: quickstart.py --batch
- Custom points: python examples/quickstart.py --point 120 30 --point 121.6 31.3
- BBox: python examples/quickstart.py --bbox 121.45 31.15 121.55 31.25


"""

import argparse
from typing import List, Sequence, Tuple

from rs_embed import (
    BBox,
    PointBuffer,
    TemporalSpec,
    OutputSpec,
    get_embedding,
    get_embeddings_batch,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-m",
        "--model",
        default="dofa",
        help="Model id registered in rs_embed (e.g., dofa, remoteclip_s2rgb, anysat, dynamicvis, galileo, wildsat)",
    )
    parser.add_argument(
        "--backend",
        default="gee",
        help="Backend to use; on-the-fly models typically use 'gee'",
    )
    parser.add_argument(
        "--start",
        default="2021-06-01",
        help="Start date for TemporalSpec.range (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        default="2021-08-31",
        help="End date for TemporalSpec.range (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--year",
        type=int,
        help="Optional single year (overrides start/end) for models that use TemporalSpec.year",
    )
    parser.add_argument(
        "--output",
        choices=["pooled", "grid"],
        default="pooled",
        help="Choose pooled vector or grid tokens",
    )
    parser.add_argument(
        "--pooling",
        default="mean",
        help="Pooling strategy when output=pooled",
    )
    parser.add_argument(
        "--grid-scale",
        type=int,
        default=10,
        help="Scale (meters) when output=grid",
    )
    parser.add_argument(
        "--lon",
        type=float,
        default=121.5,
        help="Longitude for the single-point example",
    )
    parser.add_argument(
        "--lat",
        type=float,
        default=31.2,
        help="Latitude for the single-point example",
    )
    parser.add_argument(
        "--buffer-m",
        type=int,
        default=2048,
        help="Buffer size in meters around the point",
    )
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("MINLON", "MINLAT", "MAXLON", "MAXLAT"),
        help="Optional bbox (uses bbox instead of point buffer)",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Also run get_embeddings_batch on a small list of points",
    )
    parser.add_argument(
        "--point",
        action="append",
        nargs=2,
        type=float,
        metavar=("LON", "LAT"),
        help="Extra lon/lat pairs for batch mode (implies --batch)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device selection forwarded to the embedder (auto/cpu/cuda)",
    )
    return parser.parse_args()


def _build_temporal(args: argparse.Namespace) -> TemporalSpec:
    if args.year:
        return TemporalSpec.year(args.year)
    return TemporalSpec.range(args.start, args.end)


def _build_output_spec(args: argparse.Namespace) -> OutputSpec:
    if args.output == "grid":
        return OutputSpec.grid(scale_m=args.grid_scale)
    return OutputSpec.pooled(pooling=args.pooling)


def _default_batch_points(buffer_m: int) -> List[PointBuffer]:
    return [
        PointBuffer(lon=121.5, lat=31.2, buffer_m=buffer_m),
        PointBuffer(lon=121.6, lat=31.3, buffer_m=buffer_m),
        PointBuffer(lon=120.0, lat=30.0, buffer_m=buffer_m),
    ]


def _collect_batch_points(args: argparse.Namespace) -> List[PointBuffer]:
    if not args.point:
        return _default_batch_points(args.buffer_m)
    return [PointBuffer(lon=lon, lat=lat, buffer_m=args.buffer_m) for lon, lat in args.point]


def _run_single(args: argparse.Namespace, output_spec: OutputSpec, temporal: TemporalSpec) -> None:
    if args.bbox:
        minlon, minlat, maxlon, maxlat = args.bbox
        spatial = BBox(minlon=minlon, minlat=minlat, maxlon=maxlon, maxlat=maxlat)
    else:
        spatial = PointBuffer(lon=args.lon, lat=args.lat, buffer_m=args.buffer_m)

    emb = get_embedding(
        args.model,
        spatial=spatial,
        temporal=temporal,
        output=output_spec,
        backend=args.backend,
        device=args.device,
    )
    print(f"[single] model={args.model}, output={args.output}, backend={args.backend}, shape={emb.data.shape}")
    print("meta:", emb.meta)


def _run_batch(args: argparse.Namespace, output_spec: OutputSpec, temporal: TemporalSpec) -> None:
    points = _collect_batch_points(args)
    embeddings = get_embeddings_batch(
        args.model,
        spatials=points,
        temporal=temporal,
        output=output_spec,
        backend=args.backend,
        device=args.device,
    )
    for i, emb in enumerate(embeddings):
        print(f"[batch] idx={i}, shape={emb.data.shape}")
    if embeddings:
        print("meta (first):", embeddings[0].meta)


def main() -> None:
    args = _parse_args()
    if args.point:
        args.batch = True

    temporal = _build_temporal(args)
    output_spec = _build_output_spec(args)

    _run_single(args, output_spec, temporal)

    if args.batch:
        _run_batch(args, output_spec, temporal)


if __name__ == "__main__":
    main()
