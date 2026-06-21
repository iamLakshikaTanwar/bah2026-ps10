"""irchroma.data.stac_cog — STAC discovery + O(1) COG range reads.

Implements the ⭐⭐⭐ streaming access tier from docs/research/01 §15 and
ARCHITECTURE.md §4.4: query a STAC API (Earth Search / Microsoft Planetary
Computer) for Landsat/Sentinel-2 Cloud-Optimized GeoTIFFs, then read a **single
tile window** via an HTTP **range request** — only the bytes you need, one GET per
tile → the O(1)-per-tile read (docs/research/05 §A1–A2).

Honest complexity (docs/research/05): the STAC *search* is O(log n) (spatial
index / partition pruning), done rarely and cached; the per-tile COG **read** is
O(1) (one warm-header + one bounded range read). This class separates the two.

All heavy deps (``pystac-client``, ``planetary-computer``, ``rio-tiler``,
``rasterio``, ``numpy``) are import-guarded so the module always imports; methods
raise a clear :class:`RuntimeError` naming the missing package only when called.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..config import DataConfig

# --------------------------------------------------------------------------- #
# Guarded optional deps (module must import without any of them).
# --------------------------------------------------------------------------- #
try:  # pragma: no cover
    import numpy as np

    _HAS_NUMPY = True
except Exception:  # pragma: no cover
    np = None  # type: ignore
    _HAS_NUMPY = False

try:  # pragma: no cover
    import pystac_client  # type: ignore

    _HAS_PYSTAC = True
except Exception:  # pragma: no cover
    pystac_client = None  # type: ignore
    _HAS_PYSTAC = False

try:  # pragma: no cover
    import planetary_computer  # type: ignore

    _HAS_PC = True
except Exception:  # pragma: no cover
    planetary_computer = None  # type: ignore
    _HAS_PC = False

try:  # pragma: no cover - rio-tiler is the cleanest per-tile COG reader
    from rio_tiler.io import COGReader  # type: ignore

    _HAS_RIO_TILER = True
except Exception:  # pragma: no cover
    COGReader = None  # type: ignore
    _HAS_RIO_TILER = False

try:  # pragma: no cover - rasterio windowed read is the fallback path
    import rasterio  # type: ignore
    from rasterio.windows import from_bounds as _window_from_bounds  # type: ignore

    _HAS_RASTERIO = True
except Exception:  # pragma: no cover
    rasterio = None  # type: ignore
    _window_from_bounds = None  # type: ignore
    _HAS_RASTERIO = False


__all__ = [
    "STAC_ENDPOINTS",
    "STACCOGReader",
]


#: Public STAC API roots for the supported access tiers (docs/research/01 §1.2, §15).
STAC_ENDPOINTS: Dict[str, str] = {
    "earth_search": "https://earth-search.aws.element84.com/v1",
    "planetary_computer": "https://planetarycomputer.microsoft.com/api/stac/v1",
}

#: Per-endpoint default collection ids for the two anchor sensors.
_DEFAULT_COLLECTIONS: Dict[str, Dict[str, str]] = {
    "earth_search": {
        "landsat": "landsat-c2-l2",
        "sentinel2": "sentinel-2-l2a",
    },
    "planetary_computer": {
        "landsat": "landsat-c2-l2",
        "sentinel2": "sentinel-2-l2a",
    },
}


class STACCOGReader:
    """Query a STAC API and read single COG tiles via HTTP range requests.

    Args:
      endpoint: one of :data:`STAC_ENDPOINTS` keys (``"earth_search"`` or
                ``"planetary_computer"``) or a full STAC API URL.
      cfg:      :class:`DataConfig` (uses ``source_ids`` / cloud + date defaults).
                Defaults to ``DataConfig()``.
      sign:     whether to sign asset hrefs with ``planetary_computer.sign`` (auto-
                enabled for the Planetary-Computer endpoint when the package is present).
    """

    def __init__(
        self,
        endpoint: str = "earth_search",
        cfg: Optional[DataConfig] = None,
        sign: Optional[bool] = None,
    ) -> None:
        self.cfg = cfg if cfg is not None else DataConfig()
        self.endpoint_name = endpoint
        self.url = STAC_ENDPOINTS.get(endpoint, endpoint)
        is_pc = endpoint == "planetary_computer" or "planetarycomputer" in self.url
        self.sign = (is_pc and _HAS_PC) if sign is None else bool(sign)
        self._client = None  # lazily opened

    # ---- discovery (O(log n), cached upstream) ----------------------------- #
    def _open_client(self) -> Any:
        """Open (and cache) the ``pystac_client`` connection; guarded."""
        if not _HAS_PYSTAC:
            raise RuntimeError(
                "STAC discovery requires pystac-client (`pip install pystac-client`)."
            )
        if self._client is None:
            self._client = pystac_client.Client.open(self.url)
        return self._client

    def _collection_id(self, sensor: str) -> str:
        """Resolve a sensor alias (``landsat``/``sentinel2``) to a collection id."""
        mapping = _DEFAULT_COLLECTIONS.get(self.endpoint_name, {})
        return mapping.get(sensor, sensor)

    def search(
        self,
        bbox: Sequence[float],
        datetime: str,
        sensor: str = "landsat",
        max_cloud: float = 20.0,
        limit: int = 10,
    ) -> List[Any]:
        """Search the STAC API for items intersecting ``bbox`` in the ``datetime`` window.

        Args:
          bbox:     ``[minx, miny, maxx, maxy]`` in WGS-84 degrees.
          datetime: an RFC-3339 interval, e.g. ``"2024-01-01/2024-03-31"``.
          sensor:   ``"landsat"`` / ``"sentinel2"`` alias, or an explicit collection id.
          max_cloud: maximum ``eo:cloud_cover`` (percent).
          limit:    maximum number of items to return.

        Returns:
          A list of STAC ``Item`` objects (signed if ``self.sign``). This is the
          O(log n) discovery step; cache the returned hrefs and reuse for O(1) reads.
        """
        client = self._open_client()
        collection = self._collection_id(sensor)
        search = client.search(
            collections=[collection],
            bbox=list(bbox),
            datetime=datetime,
            query={"eo:cloud_cover": {"lt": max_cloud}},
            max_items=limit,
        )
        items = list(search.items())
        if self.sign and _HAS_PC:
            items = [planetary_computer.sign(it) for it in items]
        return items

    # ---- O(1) per-tile read ------------------------------------------------ #
    @staticmethod
    def _asset_href(item: Any, asset_key: str) -> str:
        """Return the href for ``asset_key`` on a STAC item (raises if absent)."""
        assets = getattr(item, "assets", None)
        if assets is None or asset_key not in assets:
            available = sorted(assets.keys()) if assets else []
            raise KeyError(
                f"Asset {asset_key!r} not found on item; available: {available}"
            )
        return assets[asset_key].href

    def read_tile_xyz(
        self,
        item: Any,
        asset_key: str,
        x: int,
        y: int,
        z: int,
        tilesize: int = 256,
    ) -> Any:
        """Read one web-mercator ``z/x/y`` tile from a COG asset via a range request.

        Uses ``rio-tiler``'s ``COGReader.tile`` which issues a single HTTP range GET
        for the requested tile from the COG's internal tiling/overviews → the
        O(1)-per-tile read (docs/research/05 §A1). Requires ``rio-tiler``.

        Args:
          item:      a STAC item (its ``asset_key`` href is read).
          asset_key: asset name (e.g. ``"lwir11"`` thermal, ``"red"``/``"green"``/``"blue"``).
          x, y, z:   slippy-map tile coordinate.
          tilesize:  output tile edge in pixels.

        Returns:
          A numpy array ``[bands, tilesize, tilesize]`` of the tile's pixels.
        """
        if not _HAS_RIO_TILER:
            raise RuntimeError(
                "read_tile_xyz() requires rio-tiler (`pip install rio-tiler`)."
            )
        href = self._asset_href(item, asset_key)
        with COGReader(href) as cog:
            img = cog.tile(x, y, z, tilesize=tilesize)
        return img.data  # ImageData.data -> ndarray [bands, H, W]

    def read_window_bounds(
        self,
        href: str,
        bounds: Tuple[float, float, float, float],
        bounds_crs: str = "EPSG:4326",
        out_shape: Optional[Tuple[int, int]] = None,
    ) -> Any:
        """Read a COG window for geographic ``bounds`` via a rasterio range read.

        Fallback path (used when ``rio-tiler`` is unavailable): opens the COG with
        ``rasterio`` and reads only the windowed byte ranges for ``bounds``. Still an
        O(1)-per-window read for a COG (internal tiling + overviews + HTTP range).

        Args:
          href:       the COG URL/path.
          bounds:     ``(minx, miny, maxx, maxy)`` in ``bounds_crs``.
          bounds_crs: CRS of ``bounds`` (default WGS-84); reprojected to the COG CRS.
          out_shape:  optional ``(height, width)`` to resample the window to.

        Returns:
          A numpy array ``[bands, H, W]`` of the windowed pixels.
        """
        if not _HAS_RASTERIO:
            raise RuntimeError(
                "read_window_bounds() requires rasterio (`pip install rasterio`)."
            )
        from rasterio.warp import transform_bounds  # local import (guarded by above)

        with rasterio.open(href) as src:
            # Reproject the request bounds into the dataset CRS if needed.
            if bounds_crs and str(src.crs) != str(bounds_crs):
                ds_bounds = transform_bounds(bounds_crs, src.crs, *bounds)
            else:
                ds_bounds = bounds
            window = _window_from_bounds(*ds_bounds, transform=src.transform)
            read_kwargs: Dict[str, Any] = {"window": window}
            if out_shape is not None:
                read_kwargs["out_shape"] = (src.count,) + tuple(out_shape)
            data = src.read(**read_kwargs)
        return data

    def read_rgb_ir(
        self,
        item: Any,
        x: int,
        y: int,
        z: int,
        ir_asset: str = "lwir11",
        rgb_assets: Sequence[str] = ("red", "green", "blue"),
        tilesize: int = 256,
    ) -> Dict[str, Any]:
        """Read co-located IR and RGB tiles for one item → ``{"ir": arr, "rgb": arr}``.

        Convenience wrapper that calls :meth:`read_tile_xyz` for the thermal asset
        and each RGB asset and stacks the RGB bands. For Landsat C2 L2 on Earth
        Search / Planetary Computer the thermal asset is ``"lwir11"`` (ST_B10) and
        RGB are ``"red"``/``"green"``/``"blue"``.

        Returns:
          ``{"ir": ndarray[1,H,W], "rgb": ndarray[3,H,W]}`` (numpy required).
        """
        if not _HAS_NUMPY:
            raise RuntimeError("read_rgb_ir() requires numpy to stack RGB bands.")
        ir = self.read_tile_xyz(item, ir_asset, x, y, z, tilesize=tilesize)
        rgb_planes = [
            self.read_tile_xyz(item, a, x, y, z, tilesize=tilesize)[0]
            for a in rgb_assets
        ]
        rgb = np.stack(rgb_planes, axis=0)  # [3, H, W]
        return {"ir": ir, "rgb": rgb}
