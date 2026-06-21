#!/usr/bin/env python3
"""Download a small paired IR↔RGB demo set via GEE or STAC/COG.

CLI front-end over :class:`irchroma.data.gee.GEEIngestor` (Earth Engine, Recipe 1:
Landsat-TIRS thermal ↔ Sentinel-2 RGB) and :class:`irchroma.data.stac_cog.STACCOGReader`
(STAC + COG range reads). Heavy/credentialed dependencies are optional — the script
imports and prints clear guidance when they are missing rather than crashing.

Examples
--------
GEE export (requires `earthengine-api` + `earthengine authenticate`)::

    python scripts/download_data.py --source gee \
        --bbox 77.4 12.8 77.8 13.1 --dates 2024-01-01 2024-03-31 --out data/demo

STAC/COG streaming read (requires `pystac-client` + `rio-tiler`/`rasterio`)::

    python scripts/download_data.py --source stac --endpoint earth_search \
        --bbox 77.4 12.8 77.8 13.1 --dates 2024-01-01 2024-03-31 \
        --zoom 12 --out data/demo
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional, Sequence


def _build_parser() -> argparse.ArgumentParser:
    """Construct the argparse CLI parser."""
    p = argparse.ArgumentParser(
        prog="download_data.py",
        description="Fetch a small paired IR↔RGB demo set (GEE or STAC/COG).",
    )
    p.add_argument(
        "--source",
        choices=["gee", "stac"],
        required=True,
        help="Access backend: 'gee' (Earth Engine export) or 'stac' (COG range reads).",
    )
    p.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("MINX", "MINY", "MAXX", "MAXY"),
        required=True,
        help="Area of interest in WGS-84 degrees: minx miny maxx maxy.",
    )
    p.add_argument(
        "--dates",
        type=str,
        nargs=2,
        metavar=("START", "END"),
        required=True,
        help="Date window: START END as YYYY-MM-DD.",
    )
    p.add_argument(
        "--out",
        type=str,
        default="data/demo",
        help="Output directory for the downloaded/exported tiles.",
    )
    p.add_argument(
        "--endpoint",
        type=str,
        default="earth_search",
        help="STAC endpoint name ('earth_search'|'planetary_computer') or URL "
        "(used only with --source stac).",
    )
    p.add_argument(
        "--sensor",
        type=str,
        default="landsat",
        help="STAC sensor alias ('landsat'|'sentinel2') or collection id.",
    )
    p.add_argument(
        "--zoom",
        type=int,
        default=12,
        help="Web-mercator zoom for the STAC tile read (used with --source stac).",
    )
    p.add_argument(
        "--max-cloud",
        type=float,
        default=20.0,
        help="Maximum cloud cover percent for STAC/GEE filtering.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Maximum number of STAC items / pairs to fetch.",
    )
    return p


def _run_gee(args: argparse.Namespace) -> int:
    """Drive the GEE Recipe-1 export; print guidance if EE is unavailable."""
    try:
        from irchroma.config import DataConfig
        from irchroma.data.gee import GEEIngestor, _HAS_EE
    except Exception as exc:  # pragma: no cover - import surface
        print(f"[error] could not import irchroma.data.gee: {exc}", file=sys.stderr)
        return 2

    if not _HAS_EE:
        print(
            "[gee] earthengine-api is not installed.\n"
            "      Install it and authenticate, then re-run:\n"
            "        pip install earthengine-api\n"
            "        earthengine authenticate\n"
            f"      Would export Landsat-TIRS↔S2 pair for bbox={args.bbox} "
            f"dates={args.dates} -> {args.out}",
            file=sys.stderr,
        )
        return 1

    os.makedirs(args.out, exist_ok=True)
    ingestor = GEEIngestor(cfg=DataConfig())
    try:
        ingestor.initialize()
    except Exception as exc:  # pragma: no cover - credential surface
        print(
            f"[gee] Earth Engine initialization failed: {exc}\n"
            "      Run `earthengine authenticate` (or configure a service account).",
            file=sys.stderr,
        )
        return 1

    pair = ingestor.landsat_s2_pair(
        bbox=args.bbox, start=args.dates[0], end=args.dates[1], max_cloud=args.max_cloud
    )
    tasks = ingestor.export_to_drive(pair, prefix="irchroma_demo")
    print(
        f"[gee] started {len(tasks)} Drive export task(s) for the IR/RGB pair "
        f"(CRS={pair.get('crs')}). Monitor at https://code.earthengine.google.com/tasks"
    )
    return 0


def _run_stac(args: argparse.Namespace) -> int:
    """Drive the STAC search + COG tile read; print guidance if deps are unavailable."""
    try:
        from irchroma.config import DataConfig
        from irchroma.data.catalog import lonlat_to_tile
        from irchroma.data.stac_cog import (
            STACCOGReader,
            _HAS_PYSTAC,
            _HAS_RIO_TILER,
            _HAS_RASTERIO,
        )
    except Exception as exc:  # pragma: no cover - import surface
        print(f"[error] could not import irchroma.data.stac_cog: {exc}", file=sys.stderr)
        return 2

    if not _HAS_PYSTAC:
        print(
            "[stac] pystac-client is not installed.\n"
            "       Install the STAC/COG stack and re-run:\n"
            "         pip install pystac-client rio-tiler rasterio planetary-computer\n"
            f"       Would search '{args.sensor}' over bbox={args.bbox} "
            f"dates={args.dates} on endpoint '{args.endpoint}'.",
            file=sys.stderr,
        )
        return 1
    if not (_HAS_RIO_TILER or _HAS_RASTERIO):
        print(
            "[stac] need rio-tiler or rasterio to read COG tiles.\n"
            "         pip install rio-tiler rasterio",
            file=sys.stderr,
        )
        return 1

    os.makedirs(args.out, exist_ok=True)
    reader = STACCOGReader(endpoint=args.endpoint, cfg=DataConfig())
    datetime_str = f"{args.dates[0]}/{args.dates[1]}"
    items = reader.search(
        bbox=args.bbox,
        datetime=datetime_str,
        sensor=args.sensor,
        max_cloud=args.max_cloud,
        limit=args.limit,
    )
    if not items:
        print("[stac] no items matched the query.", file=sys.stderr)
        return 1

    # Center of the bbox -> the tile to read at the requested zoom.
    cx = (args.bbox[0] + args.bbox[2]) / 2.0
    cy = (args.bbox[1] + args.bbox[3]) / 2.0
    x, y, z = lonlat_to_tile(cx, cy, args.zoom)
    print(f"[stac] {len(items)} item(s) found; reading tile z/x/y = {z}/{x}/{y}")

    try:
        if args.sensor == "landsat":
            out = reader.read_rgb_ir(items[0], x, y, z)
            ir_shape = getattr(out["ir"], "shape", None)
            rgb_shape = getattr(out["rgb"], "shape", None)
            print(f"[stac] read IR{ir_shape} + RGB{rgb_shape} for {items[0].id}")
        else:
            arr = reader.read_tile_xyz(items[0], "red", x, y, z)
            print(f"[stac] read tile {getattr(arr, 'shape', None)} for {items[0].id}")
    except Exception as exc:  # pragma: no cover - network/asset surface
        print(f"[stac] tile read failed: {exc}", file=sys.stderr)
        return 1
    print(f"[stac] demo read complete (out dir: {args.out}).")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Parse args and dispatch to the selected source. Returns a process exit code."""
    # Make the package importable when run from a source checkout.
    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(os.path.dirname(here), "src")
    if os.path.isdir(src) and src not in sys.path:
        sys.path.insert(0, src)

    args = _build_parser().parse_args(argv)
    if args.source == "gee":
        return _run_gee(args)
    return _run_stac(args)


if __name__ == "__main__":
    raise SystemExit(main())
