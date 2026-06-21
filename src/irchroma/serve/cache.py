"""irchroma.serve.cache — O(1) tile cache (in-memory LRU + optional Redis).

The cache turns repeat tile requests into **true O(1)** hits (docs/research/05 §A7;
ARCHITECTURE.md §9.2 #2, #6). The cache **key** is a web-mercator **quadkey** (or an
H3 cell id) computed by pure bit/arithmetic from ``z/x/y`` — independent of archive
size, an O(1) encode — and the lookup is a single hash ``get``:

  * **In-memory LRU** (default) — reuses :class:`irchroma.data.catalog.LRUCache`
    (``OrderedDict``-backed, O(1) ``get``/``put`` with LRU eviction).
  * **Redis** (optional, ``backend="redis"``) — a distributed/shared tile cache keyed
    by the same quadkey, so multiple serving replicas share rendered tiles; guarded
    behind ``try/except import redis`` and only required when actually selected.

``key_for(z, x, y)`` builds the quadkey via :class:`irchroma.data.catalog.TileCatalog`
(the single source of truth for slippy-map ↔ quadkey math). Bytes (e.g. an encoded
PNG) are the natural cached value, but any value is accepted by the in-memory path.
"""

from __future__ import annotations

from typing import Any, Optional

from ..data.catalog import LRUCache, TileCatalog, quadkey

# --------------------------------------------------------------------------- #
# Optional Redis backend (guarded — only needed when backend="redis").
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - present only if installed
    import redis as _redis  # type: ignore

    _HAS_REDIS = True
except Exception:  # pragma: no cover
    _redis = None  # type: ignore
    _HAS_REDIS = False


__all__ = ["TileCache"]


class TileCache:
    """O(1) get/put tile cache keyed by a web-mercator quadkey (or H3 cell).

    Wraps an in-memory LRU (default) or a Redis backend behind a uniform
    ``get`` / ``put`` API keyed by :meth:`key_for` (``z/x/y`` → quadkey). The quadkey
    encode is O(1) and the lookup is an O(1) hash get, so a cache hit short-circuits
    the entire render pipeline (docs/research/05 §A7, §C5).

    Args:
      backend:        ``"memory"`` (default, in-process LRU) or ``"redis"``
                      (shared/distributed; requires the ``redis`` package).
      capacity:       max entries for the in-memory LRU (ignored for Redis, which has
                      its own eviction policy).
      key_scheme:     ``"quadkey"`` (default) or ``"h3"`` for the cache key (forwarded
                      to :class:`~irchroma.data.catalog.TileCatalog`).
      namespace:      key prefix (useful to version cache contents / separate layers).
      redis_url:      Redis connection URL (``redis://host:port/db``) when ``backend="redis"``.
      redis_client:   an already-constructed Redis client to use instead of ``redis_url``.
      ttl_seconds:    optional expiry for Redis entries (``None`` = no expiry).
      h3_resolution:  H3 resolution when ``key_scheme="h3"``.

    Notes:
      * The in-memory backend accepts **any** value; the Redis backend stores **bytes**
        (encode your tile to PNG/COG bytes before :meth:`put` on that path).
      * Selecting ``backend="redis"`` without the ``redis`` package raises a clear,
        friendly error — the in-memory path needs no optional dependency.
    """

    def __init__(
        self,
        backend: str = "memory",
        capacity: int = 1024,
        key_scheme: str = "quadkey",
        namespace: str = "irchroma",
        redis_url: str = "redis://localhost:6379/0",
        redis_client: Optional[Any] = None,
        ttl_seconds: Optional[int] = None,
        h3_resolution: int = 9,
    ) -> None:
        self.backend = str(backend or "memory").lower()
        self.namespace = str(namespace)
        self.ttl_seconds = ttl_seconds
        # TileCatalog is the O(1) addressing source of truth (quadkey / H3).
        self.catalog = TileCatalog(
            key_scheme=key_scheme,
            h3_resolution=h3_resolution,
            cache_capacity=capacity,
        )

        self._lru: Optional[LRUCache] = None
        self._redis: Optional[Any] = None
        if self.backend == "redis":
            self._redis = self._connect_redis(redis_url, redis_client)
        elif self.backend in ("memory", "lru"):
            self._lru = LRUCache(capacity=capacity)
        elif self.backend in ("none", "off"):
            self._lru = None  # caching disabled; get always misses
        else:
            raise ValueError(
                f"Unknown cache backend {backend!r}; use 'memory', 'redis', or 'none'."
            )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _connect_redis(redis_url: str, redis_client: Optional[Any]) -> Any:
        """Open a Redis connection (guarded), or return the provided client."""
        if redis_client is not None:
            return redis_client
        if not _HAS_REDIS or _redis is None:
            raise RuntimeError(
                "TileCache(backend='redis') requires the 'redis' package "
                "(`pip install redis`). Use backend='memory' for the pure-stdlib O(1) "
                "in-process LRU instead."
            )
        return _redis.Redis.from_url(redis_url)

    # ---- key construction (O(1) quadkey/H3 encode) ------------------------- #
    def key_for(self, z: int, x: int, y: int) -> str:
        """Return the namespaced cache key (quadkey by default) for tile ``z/x/y``.

        Computes the Bing **quadkey** for ``(x, y, z)`` via the catalog's pure
        bit/arithmetic encoder (O(1)); when ``key_scheme="h3"`` the tile's NW-corner
        lon/lat is encoded to an H3 cell instead. The key is prefixed by ``namespace``
        so cache contents can be versioned / layered.
        """
        if self.catalog.key_scheme == "h3":
            from ..data.catalog import tile_to_lonlat  # local import (H3 path only)

            lon, lat = tile_to_lonlat(int(x), int(y), int(z))
            raw = self.catalog.h3_cell(lat, lon)
        else:
            raw = quadkey(int(x), int(y), int(z))
        return f"{self.namespace}:{raw}"

    # ---- O(1) get / put ---------------------------------------------------- #
    def get(self, z: int, x: int, y: int) -> Optional[Any]:
        """O(1) cache lookup for tile ``z/x/y``; returns the value or ``None`` on miss."""
        key = self.key_for(z, x, y)
        if self._redis is not None:
            try:
                return self._redis.get(key)
            except Exception:  # pragma: no cover - treat backend hiccup as a miss
                return None
        if self._lru is not None:
            return self._lru.get(key)
        return None  # caching disabled

    def put(self, z: int, x: int, y: int, value: Any) -> str:
        """O(1) cache insert for tile ``z/x/y``; returns the key used.

        On the Redis path the value should be ``bytes`` (e.g. an encoded PNG); a
        configured ``ttl_seconds`` sets per-key expiry. On the in-memory path any
        value is accepted. Caching-disabled (``backend="none"``) is a no-op.
        """
        key = self.key_for(z, x, y)
        if self._redis is not None:
            try:
                if self.ttl_seconds is not None:
                    self._redis.setex(key, int(self.ttl_seconds), value)
                else:
                    self._redis.set(key, value)
            except Exception:  # pragma: no cover - never let a cache write break serving
                pass
            return key
        if self._lru is not None:
            self._lru.put(key, value)
        return key

    def get_by_key(self, key: str) -> Optional[Any]:
        """Low-level get by an already-computed (namespaced) key."""
        if self._redis is not None:
            try:
                return self._redis.get(key)
            except Exception:  # pragma: no cover
                return None
        if self._lru is not None:
            return self._lru.get(key)
        return None

    def put_by_key(self, key: str, value: Any) -> None:
        """Low-level put by an already-computed (namespaced) key."""
        if self._redis is not None:
            try:
                if self.ttl_seconds is not None:
                    self._redis.setex(key, int(self.ttl_seconds), value)
                else:
                    self._redis.set(key, value)
            except Exception:  # pragma: no cover
                pass
            return
        if self._lru is not None:
            self._lru.put(key, value)

    def __contains__(self, key: str) -> bool:
        if self._redis is not None:
            try:
                return bool(self._redis.exists(key))
            except Exception:  # pragma: no cover
                return False
        if self._lru is not None:
            return key in self._lru
        return False

    def clear(self) -> None:
        """Drop all cached tiles (flush the in-memory LRU; ``flushdb`` on Redis)."""
        if self._redis is not None:
            try:
                self._redis.flushdb()
            except Exception:  # pragma: no cover
                pass
        if self._lru is not None:
            self._lru.clear()

    def stats(self) -> dict:
        """Return cache statistics (LRU hit/miss/size; backend identity for Redis)."""
        if self._lru is not None:
            s = dict(self._lru.stats())
            s["backend"] = "memory"
            return s
        return {"backend": self.backend, "enabled": self._redis is not None}
