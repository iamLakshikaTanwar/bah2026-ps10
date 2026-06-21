"""irchroma.data.catalog — O(1) tile addressing + an in-memory LRU tile cache.

Implements the constant-time spatial-addressing primitives from
docs/research/05 §A4–A7 and ARCHITECTURE.md §9:

  * **Slippy-map / web-mercator tiling** — ``lonlat_to_tile(lon, lat, z)`` converts a
    geographic coordinate to a ``z/x/y`` tile, and ``quadkey(x, y, z)`` interleaves
    the x/y bits into the Bing base-4 **quadkey** string. Both are pure bit/arith
    operations independent of dataset size → **genuinely O(1)**.
  * :class:`TileCatalog` — wraps these into ``tile_key(lat, lon, zoom)`` (the cache
    key) plus an optional **H3** cell path (guarded ``h3``), backed by an in-memory
    **LRU cache** keyed by the tile key (``functools``/``OrderedDict``-based, O(1)
    get/put). This is the "has this tile already been rendered?" lookup.

Honest complexity (docs/research/05 §A): the *encode* (lat/lon→cell, x/y→quadkey)
and a cache **hit** are O(1). A k-ring / radius / range **query** over cells is
O(k)/O(log n), and a STAC *search* is O(log n) — those are not done here. ``h3`` is
optional and import-guarded; the quadkey path is pure-stdlib and always available.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Any, Dict, Generic, Hashable, List, Optional, Tuple, TypeVar

# --------------------------------------------------------------------------- #
# Optional H3 (guarded). The quadkey path needs no third-party dep.
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - present only if installed
    import h3  # type: ignore

    _HAS_H3 = True
except Exception:  # pragma: no cover
    h3 = None  # type: ignore
    _HAS_H3 = False


__all__ = [
    "lonlat_to_tile",
    "tile_to_lonlat",
    "quadkey",
    "quadkey_to_tile",
    "LRUCache",
    "TileCatalog",
]

_V = TypeVar("_V")
TileXYZ = Tuple[int, int, int]  # (x, y, z)


# --------------------------------------------------------------------------- #
# Pure-python slippy-map / quadkey math (all O(1)).
# --------------------------------------------------------------------------- #
def lonlat_to_tile(lon: float, lat: float, z: int) -> TileXYZ:
    """Convert a lon/lat (WGS-84 degrees) to a web-mercator ``(x, y, z)`` tile.

    Standard slippy-map formula (OSM/Bing/Google XYZ). Latitude is clamped to the
    web-mercator valid range ``±85.0511°``. Returns integer tile indices at zoom
    ``z`` (so ``0 <= x, y < 2**z``). Pure arithmetic → **O(1)**.
    """
    if z < 0:
        raise ValueError(f"zoom must be >= 0, got {z}.")
    lat = max(-85.05112878, min(85.05112878, float(lat)))
    lon = ((float(lon) + 180.0) % 360.0) - 180.0  # wrap to [-180, 180)
    n = 1 << z  # 2**z
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int(
        (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    )
    # Clamp to valid range (guards float edge cases at the antimeridian/poles).
    x = min(n - 1, max(0, x))
    y = min(n - 1, max(0, y))
    return (x, y, z)


def tile_to_lonlat(x: int, y: int, z: int) -> Tuple[float, float]:
    """Return the (lon, lat) of the **north-west corner** of tile ``(x, y, z)``.

    Inverse of :func:`lonlat_to_tile` (to the tile corner). O(1).
    """
    n = 1 << z
    lon = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n)))
    return (lon, math.degrees(lat_rad))


def quadkey(x: int, y: int, z: int) -> str:
    """Return the Bing **quadkey** (base-4 string) for tile ``(x, y, z)``.

    The quadkey interleaves the bits of ``x`` and ``y`` from the most-significant
    bit (zoom 1) down, giving a hierarchical string of length ``z`` over the digits
    ``{0,1,2,3}``. Pure bit manipulation → **O(z)** ≈ O(1) (z is a tiny constant,
    ≤ ~22); the canonical web-mercator cache key (docs/research/05 §A4).
    """
    if z < 0:
        raise ValueError(f"zoom must be >= 0, got {z}.")
    digits: List[str] = []
    for i in range(z, 0, -1):
        digit = 0
        mask = 1 << (i - 1)
        if (x & mask) != 0:
            digit += 1
        if (y & mask) != 0:
            digit += 2
        digits.append(str(digit))
    return "".join(digits)


def quadkey_to_tile(qk: str) -> TileXYZ:
    """Decode a quadkey string back to ``(x, y, z)`` (inverse of :func:`quadkey`). O(1)."""
    x = y = 0
    z = len(qk)
    for i in range(z, 0, -1):
        mask = 1 << (i - 1)
        digit = qk[z - i]
        if digit == "1":
            x |= mask
        elif digit == "2":
            y |= mask
        elif digit == "3":
            x |= mask
            y |= mask
        elif digit != "0":
            raise ValueError(f"Invalid quadkey digit {digit!r} in {qk!r}.")
    return (x, y, z)


# --------------------------------------------------------------------------- #
# In-memory LRU cache (O(1) get/put). Stdlib OrderedDict-backed.
# --------------------------------------------------------------------------- #
class LRUCache(Generic[_V]):
    """A bounded least-recently-used cache with O(1) ``get``/``put``.

    Backed by an ``OrderedDict`` (move-to-end on access). Used by
    :class:`TileCatalog` to hold already-rendered/fetched tiles keyed by their
    O(1) tile key — turning repeat reads into O(1) hits (docs/research/05 §A7).

    Args:
      capacity: maximum number of entries to retain (LRU eviction beyond this).
    """

    def __init__(self, capacity: int = 1024) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be >= 1, got {capacity}.")
        self.capacity = int(capacity)
        self._store: "OrderedDict[Hashable, _V]" = OrderedDict()
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, key: Hashable) -> bool:
        return key in self._store

    def get(self, key: Hashable, default: Optional[_V] = None) -> Optional[_V]:
        """Return the cached value (marking it most-recent) or ``default`` on miss."""
        if key in self._store:
            self._store.move_to_end(key)
            self.hits += 1
            return self._store[key]
        self.misses += 1
        return default

    def put(self, key: Hashable, value: _V) -> None:
        """Insert/update ``key`` → ``value``, evicting the LRU entry if over capacity."""
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = value
        if len(self._store) > self.capacity:
            self._store.popitem(last=False)  # evict least-recently-used

    def clear(self) -> None:
        """Drop all entries and reset hit/miss counters."""
        self._store.clear()
        self.hits = 0
        self.misses = 0

    def stats(self) -> Dict[str, Any]:
        """Return cache statistics (size, capacity, hits, misses, hit-rate)."""
        total = self.hits + self.misses
        return {
            "size": len(self._store),
            "capacity": self.capacity,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": (self.hits / total) if total else 0.0,
        }


class TileCatalog:
    """O(1) tile addressing + an in-memory LRU tile cache.

    Provides the cache **key** for a geographic location at a zoom level — either a
    web-mercator **quadkey** (default; ``key_scheme="quadkey"``) or an **H3** cell id
    (``key_scheme="h3"``, requires the optional ``h3`` package) — and an attached
    :class:`LRUCache` so callers can do "have we already rendered this tile?" in O(1)
    (ARCHITECTURE.md §9.2, docs/research/05 §A5–A7).

    Args:
      zoom:        default web-mercator zoom for ``tile_key`` (slippy-map level).
      key_scheme:  ``"quadkey"`` (default, pure-stdlib) or ``"h3"`` (guarded).
      h3_resolution: H3 resolution (0–15) when ``key_scheme == "h3"``.
      cache_capacity: size of the in-memory LRU tile cache.
    """

    def __init__(
        self,
        zoom: int = 12,
        key_scheme: str = "quadkey",
        h3_resolution: int = 9,
        cache_capacity: int = 1024,
    ) -> None:
        if key_scheme not in ("quadkey", "h3"):
            raise ValueError(
                f"key_scheme must be 'quadkey' or 'h3', got {key_scheme!r}."
            )
        self.zoom = int(zoom)
        self.key_scheme = key_scheme
        self.h3_resolution = int(h3_resolution)
        self.cache: LRUCache[Any] = LRUCache(capacity=cache_capacity)

    # ---- addressing -------------------------------------------------------- #
    def lonlat_to_tile(self, lon: float, lat: float, zoom: Optional[int] = None) -> TileXYZ:
        """Convenience: ``(x, y, z)`` for a lon/lat at ``zoom`` (default ``self.zoom``)."""
        return lonlat_to_tile(lon, lat, self.zoom if zoom is None else zoom)

    def quadkey(self, x: int, y: int, z: int) -> str:
        """Convenience wrapper around the module-level :func:`quadkey`."""
        return quadkey(x, y, z)

    def h3_cell(self, lat: float, lon: float, resolution: Optional[int] = None) -> str:
        """Return the H3 cell id for ``(lat, lon)`` (requires the ``h3`` package).

        Uses ``h3.latlng_to_cell`` (v4) or ``h3.geo_to_h3`` (v3) — both O(1) encodes.
        Raises a clear error if ``h3`` is not installed.
        """
        if not _HAS_H3:
            raise RuntimeError(
                "H3 tile keys require the 'h3' package (`pip install h3`). Use "
                "key_scheme='quadkey' for a pure-stdlib O(1) key instead."
            )
        res = self.h3_resolution if resolution is None else int(resolution)
        # Support both the v4 and v3 H3 Python APIs.
        if hasattr(h3, "latlng_to_cell"):
            return h3.latlng_to_cell(float(lat), float(lon), res)  # h3 v4
        return h3.geo_to_h3(float(lat), float(lon), res)  # type: ignore[attr-defined]  # h3 v3

    def tile_key(self, lat: float, lon: float, zoom: Optional[int] = None) -> str:
        """Return the O(1) cache key for a location (quadkey or H3, per ``key_scheme``).

        Args:
          lat, lon: geographic coordinate (WGS-84 degrees).
          zoom:     web-mercator zoom for the quadkey scheme (ignored for H3;
                    defaults to ``self.zoom``).

        Returns:
          A string key: a quadkey (``"quadkey"`` scheme) or an H3 cell id (``"h3"``).
        """
        if self.key_scheme == "h3":
            return self.h3_cell(lat, lon, self.h3_resolution)
        z = self.zoom if zoom is None else int(zoom)
        x, y, _ = lonlat_to_tile(lon, lat, z)
        return quadkey(x, y, z)

    # ---- cache facade ------------------------------------------------------ #
    def get_cached(self, lat: float, lon: float, zoom: Optional[int] = None) -> Optional[Any]:
        """O(1) cache lookup for a tile at ``(lat, lon)`` (``None`` on miss)."""
        return self.cache.get(self.tile_key(lat, lon, zoom))

    def put_cached(
        self, lat: float, lon: float, value: Any, zoom: Optional[int] = None
    ) -> str:
        """O(1) cache insert for a rendered/fetched tile; returns the tile key used."""
        key = self.tile_key(lat, lon, zoom)
        self.cache.put(key, value)
        return key

    def cache_stats(self) -> Dict[str, Any]:
        """Return the underlying LRU cache statistics."""
        return self.cache.stats()
