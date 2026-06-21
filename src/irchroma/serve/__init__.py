"""irchroma.serve — FastAPI / TiTiler-style tile-serving layer.

Owner: Builder-7 (InferServeBuilder). Implements the tile-serving layer of
docs/research/05 §A4–A7 / ARCHITECTURE.md §9.2 #6:

  * :class:`~irchroma.serve.cache.TileCache` — O(1) get/put tile cache keyed by a
    web-mercator **quadkey** (or H3 cell), with an in-memory LRU (default; reuses
    :class:`irchroma.data.catalog.LRUCache`) or an optional **Redis** backend. A cache
    hit short-circuits the entire render pipeline.
  * :func:`~irchroma.serve.api.create_app` — a FastAPI app exposing the slippy-map
    endpoint ``GET /tiles/{z}/{x}/{y}.png`` (quadkey cache key → cache → COG range read
    → ``InferenceEngine.predict`` + O(1) color-LUT → PNG → write-back), plus
    ``GET /health`` and ``GET /info``; ``run(host, port)`` serves it via uvicorn.

Import resilience: the cache (pure-stdlib + guarded Redis) always imports; the API
imports even without ``fastapi``/``uvicorn``/``pillow``/``rasterio`` — ``create_app``
raises a clear, friendly error only when FastAPI is genuinely missing. ``__all__`` is
built dynamically to reflect what is importable.
"""

from __future__ import annotations

from typing import List

__all__: List[str] = []

# --- Tile cache (pure-stdlib LRU + guarded Redis; always importable) ------- #
try:
    from .cache import TileCache  # noqa: F401

    __all__ += ["TileCache"]
except Exception:  # pragma: no cover - never expected (no hard deps)
    TileCache = None  # type: ignore

# --- Tile-serving API (guarded fastapi/uvicorn/pillow; import-safe) -------- #
try:
    from .api import TileServer, create_app, run  # noqa: F401

    __all__ += ["create_app", "TileServer", "run"]
except Exception:  # pragma: no cover
    create_app = None  # type: ignore
    TileServer = None  # type: ignore
    run = None  # type: ignore
