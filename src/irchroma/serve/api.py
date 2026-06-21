"""irchroma.serve.api — FastAPI XYZ/WMTS tile server (TiTiler-style), guarded.

The serving layer of the fast platform (docs/research/05 §A4, §C5; ARCHITECTURE.md
§9.2 #6). :func:`create_app` builds a FastAPI app exposing a slippy-map tile
endpoint ``GET /tiles/{z}/{x}/{y}.png`` whose hot path is dominated by genuinely
O(1) primitives:

    z/x/y  ─►  quadkey cache key (O(1) encode)  ─►  TileCache.get  ──HIT──►  PNG bytes
                                                          │ MISS
                                                          ▼
        STACCOGReader range read (1 tile, O(1))  ─►  InferenceEngine.predict
            ─►  apply_color_lut (O(1)/pixel)  ─►  raised-cosine blend  ─►  encode PNG
            ─►  TileCache.put (write-back, next request = O(1) hit)  ─►  return PNG

plus ``GET /health`` (liveness) and ``GET /info`` (runtime/config introspection).
``run(host, port)`` serves it via uvicorn.

Every heavy/optional dependency is import-guarded: ``fastapi``/``starlette``,
``uvicorn``, ``pillow`` (PNG encode), ``rasterio``/``rio-tiler`` (COG reads). The
module **imports** with none of them installed; :func:`create_app` raises a clear,
friendly error if FastAPI is missing, and individual endpoints raise informative
HTTP errors if a feature they need (COG reads, PNG encode) is unavailable. The
inference engine and pipeline are imported **lazily** so this module is import-safe
during concurrent development.
"""

from __future__ import annotations

import io
from typing import Any, Optional

from ..config import Config, InferConfig
from .cache import TileCache

# --------------------------------------------------------------------------- #
# Guarded optional deps. The module must import with none of these present.
# --------------------------------------------------------------------------- #
try:  # FastAPI / Starlette (the web framework).
    from fastapi import FastAPI, HTTPException, Query  # type: ignore
    from fastapi.responses import JSONResponse, Response  # type: ignore

    _HAS_FASTAPI = True
except Exception:  # pragma: no cover - no fastapi installed
    FastAPI = None  # type: ignore
    HTTPException = None  # type: ignore
    Query = None  # type: ignore
    Response = None  # type: ignore
    JSONResponse = None  # type: ignore
    _HAS_FASTAPI = False

try:  # uvicorn (ASGI server) — only needed by run().
    import uvicorn  # type: ignore

    _HAS_UVICORN = True
except Exception:  # pragma: no cover
    uvicorn = None  # type: ignore
    _HAS_UVICORN = False

try:  # numpy (array glue between COG reads, torch, and PNG encode).
    import numpy as np  # type: ignore

    _HAS_NUMPY = True
except Exception:  # pragma: no cover
    np = None  # type: ignore
    _HAS_NUMPY = False

try:  # Pillow (PNG encode). rio-tiler can also render; Pillow is the simple path.
    from PIL import Image  # type: ignore

    _HAS_PIL = True
except Exception:  # pragma: no cover
    Image = None  # type: ignore
    _HAS_PIL = False

try:  # torch (only used to bridge engine outputs to numpy for encoding).
    import torch  # type: ignore

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    _HAS_TORCH = False


__all__ = ["create_app", "TileServer"]


def _as_infer_config(config: Any) -> InferConfig:
    """Coerce a ``Config`` / ``InferConfig`` / ``None`` into an :class:`InferConfig`."""
    if config is None:
        return InferConfig()
    if isinstance(config, InferConfig):
        return config
    if isinstance(config, Config):
        return config.infer
    inner = getattr(config, "infer", None)
    if isinstance(inner, InferConfig):
        return inner
    return InferConfig()


def _to_uint8_chw_rgb(arr: Any) -> Any:
    """Coerce a model RGB output to an ``HxWx3`` ``uint8`` numpy array for PNG encode.

    Accepts a torch tensor or numpy array in ``[B,3,H,W]`` / ``[3,H,W]`` / ``[H,W,3]``
    layout in ``[0, 1]`` (or already ``uint8``); returns ``HxWx3 uint8``.
    """
    if _HAS_TORCH and torch is not None and torch.is_tensor(arr):
        arr = arr.detach().cpu().float().numpy()
    a = np.asarray(arr)
    # Drop a leading batch dim if present.
    if a.ndim == 4:
        a = a[0]
    # CHW -> HWC.
    if a.ndim == 3 and a.shape[0] in (1, 3) and a.shape[-1] not in (1, 3):
        a = np.transpose(a, (1, 2, 0))
    if a.ndim == 2:
        a = np.stack([a, a, a], axis=-1)
    if a.shape[-1] == 1:
        a = np.repeat(a, 3, axis=-1)
    a = a[..., :3]
    if a.dtype != np.uint8:
        a = np.clip(a, 0.0, 1.0) * 255.0
        a = a.astype(np.uint8)
    return a


def _encode_png(rgb_uint8: Any) -> bytes:
    """Encode an ``HxWx3 uint8`` array to PNG bytes (Pillow), guarded."""
    if not _HAS_PIL or Image is None:
        raise RuntimeError(
            "PNG encoding requires Pillow (`pip install pillow`)."
        )
    buf = io.BytesIO()
    Image.fromarray(rgb_uint8, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


class TileServer:
    """Stateful holder for the tile-serving app: engine, cache, COG reader, config.

    Bundles the moving parts so the FastAPI route handlers (closures in
    :func:`create_app`) can share them, and so the same logic is unit-testable without
    spinning up the web server. The COG reader and color-LUT are built lazily on first
    use to keep construction cheap and import-safe.

    Args:
      engine:     an :class:`irchroma.infer.engine.InferenceEngine` (or ``None`` to run
                  in "cache + COG passthrough" mode without a model).
      config:     :class:`~irchroma.config.InferConfig`/:class:`~irchroma.config.Config`.
      cache:      a :class:`~irchroma.serve.cache.TileCache` (built from config if ``None``).
      cog_endpoint: STAC endpoint name for :class:`~irchroma.data.stac_cog.STACCOGReader`
                  (``"earth_search"`` / ``"planetary_computer"``); used on cache miss.
      apply_lut:  whether to run the O(1) class→Lab color clamp on the model output.
    """

    def __init__(
        self,
        engine: Optional[Any] = None,
        config: Any = None,
        cache: Optional[TileCache] = None,
        cog_endpoint: str = "earth_search",
        apply_lut: bool = True,
    ) -> None:
        self.cfg: InferConfig = _as_infer_config(config)
        self.engine = engine
        self.cog_endpoint = cog_endpoint
        self.apply_lut = bool(apply_lut)
        # Cache backend honors config unless an explicit cache was supplied.
        if cache is not None:
            self.cache = cache
        else:
            backend = "memory"
            if bool(getattr(self.cfg, "enable_cache", True)):
                backend = str(getattr(self.cfg, "cache_backend", "memory"))
                # Fall back to in-memory if redis is requested but unavailable, so the
                # server still starts (cache misses just recompute).
                if backend == "redis":
                    from .cache import _HAS_REDIS  # local import (guard probe)

                    if not _HAS_REDIS:
                        backend = "memory"
            else:
                backend = "none"
            self.cache = TileCache(
                backend=backend,
                key_scheme=str(getattr(self.cfg, "cache_key", "quadkey")),
                h3_resolution=int(getattr(self.cfg, "h3_resolution", 9)),
            )
        self.write_back = bool(getattr(self.cfg, "write_back", True))
        self._cog_reader: Optional[Any] = None
        self._class_lut: Optional[Any] = None

    # ------------------------------------------------------------------ #
    def _get_cog_reader(self) -> Any:
        """Lazily construct the STAC/COG reader (guarded import; friendly error)."""
        if self._cog_reader is None:
            try:
                from ..data.stac_cog import STACCOGReader  # lazy import
            except Exception as exc:
                raise RuntimeError(
                    "Reading source tiles needs irchroma.data.stac_cog.STACCOGReader "
                    "(rio-tiler / rasterio / pystac-client). Install those to serve "
                    f"live COG tiles. Original error: {exc}"
                ) from exc
            self._cog_reader = STACCOGReader(endpoint=self.cog_endpoint, cfg=None)
        return self._cog_reader

    def _get_class_lut(self) -> Optional[Any]:
        """Lazily build a reusable ClassColorLUT (cached so the table is built once)."""
        if self._class_lut is None and self.apply_lut:
            try:
                from ..infer.color_refine import build_class_lut  # lazy import

                self._class_lut = build_class_lut()
            except Exception:  # pragma: no cover - torch/Wave-A missing -> skip LUT
                self._class_lut = None
        return self._class_lut

    def read_source_tile(
        self,
        z: int,
        x: int,
        y: int,
        item: Any = None,
        asset_key: str = "lwir11",
        tilesize: Optional[int] = None,
    ) -> Any:
        """Read one source IR tile for ``z/x/y`` from a COG (O(1) range read).

        Delegates to :meth:`STACCOGReader.read_tile_xyz` (a single HTTP range GET on the
        COG's internal tiling/overviews). ``item`` is a STAC item whose ``asset_key``
        href is read; in a real deployment the item is resolved once via discovery
        (O(log n)) and cached. Returns a numpy ``[bands, H, W]`` tile.
        """
        reader = self._get_cog_reader()
        ts = int(tilesize if tilesize is not None else getattr(self.cfg, "tile_size", 256))
        if item is None:
            raise RuntimeError(
                "read_source_tile needs a STAC `item` (resolve it once via the reader's "
                "search() discovery step and cache it)."
            )
        return reader.read_tile_xyz(item, asset_key, int(x), int(y), int(z), tilesize=ts)

    def render_tile(
        self,
        z: int,
        x: int,
        y: int,
        ir_tile: Any,
        semantic_tile: Any = None,
    ) -> bytes:
        """Render a PNG tile from an IR array: model predict → color-LUT → PNG bytes.

        Args:
          z, x, y:       slippy-map tile coordinate (for write-back keying).
          ir_tile:       IR source tile (numpy ``[bands,H,W]`` or torch ``[C,H,W]``/``[B,C,H,W]``).
          semantic_tile: optional LULC labels for the class-LUT clamp.

        Returns:
          PNG-encoded RGB ``bytes`` for the tile. Requires torch (model) + Pillow (encode).
        """
        if not _HAS_TORCH or torch is None:
            raise RuntimeError("render_tile requires PyTorch to run the model.")
        if self.engine is None:
            raise RuntimeError("render_tile requires an InferenceEngine (engine=None).")

        ir = ir_tile if torch.is_tensor(ir_tile) else torch.as_tensor(np.asarray(ir_tile))
        ir = ir.float()
        if ir.dim() == 2:
            ir = ir.unsqueeze(0)  # [1, H, W]
        if ir.dim() == 3:
            ir = ir.unsqueeze(0)  # [1, C, H, W]

        sem = None
        if semantic_tile is not None:
            sem = (
                semantic_tile
                if torch.is_tensor(semantic_tile)
                else torch.as_tensor(np.asarray(semantic_tile))
            ).long()
            if sem.dim() == 2:
                sem = sem.unsqueeze(0)

        rgb = self.engine.predict(ir, semantic=sem)

        # O(1)/pixel color refinement (class→Lab chroma clamp) if labels are available.
        if sem is not None and self.apply_lut:
            lut = self._get_class_lut()
            if lut is not None:
                try:
                    from ..infer.color_refine import apply_color_lut  # lazy import

                    rgb = apply_color_lut(rgb, semantic=sem, lut=lut)
                except Exception:  # pragma: no cover - never fail the render on LUT
                    pass

        png = _encode_png(_to_uint8_chw_rgb(rgb))
        if self.write_back:
            self.cache.put(z, x, y, png)
        return png

    def info(self) -> dict:
        """Return a serving-runtime info dict for the ``/info`` endpoint."""
        engine_info = {}
        if self.engine is not None and hasattr(self.engine, "info"):
            try:
                engine_info = self.engine.info()
            except Exception:  # pragma: no cover
                engine_info = {}
        return {
            "service": "irchroma-tile-server",
            "tile_size": int(getattr(self.cfg, "tile_size", 512)),
            "cache": self.cache.stats(),
            "cache_key": str(getattr(self.cfg, "cache_key", "quadkey")),
            "write_back": self.write_back,
            "engine": engine_info,
            "backends": {
                "fastapi": _HAS_FASTAPI,
                "uvicorn": _HAS_UVICORN,
                "pillow": _HAS_PIL,
                "numpy": _HAS_NUMPY,
                "torch": _HAS_TORCH,
            },
        }


def create_app(
    engine: Optional[Any] = None,
    config: Any = None,
    cache: Optional[TileCache] = None,
    cog_endpoint: str = "earth_search",
    item_resolver: Optional[Any] = None,
) -> Any:
    """Build the FastAPI tile-serving app (guarded by ``fastapi``).

    Routes:
      * ``GET /tiles/{z}/{x}/{y}.png`` — the slippy-map tile endpoint. Computes the
        quadkey cache key (O(1)), returns the cached PNG on a hit, else reads the source
        COG tile (range read), runs ``InferenceEngine.predict`` + the O(1) color-LUT,
        encodes a PNG, writes it back to the cache, and returns it. A STAC ``item`` must
        be resolvable for live COG reads — supply an ``item_resolver(z, x, y) -> item``
        (e.g. a MosaicJSON quadkey→item lookup); without one, misses return HTTP 503 with
        guidance (the cache/health/info routes still work).
      * ``GET /health`` — liveness probe (``{"status": "ok"}``).
      * ``GET /info`` — runtime/config + cache-stats introspection.

    Args:
      engine:       an :class:`~irchroma.infer.engine.InferenceEngine` (or ``None`` for a
                    cache-only / passthrough server).
      config:       :class:`~irchroma.config.InferConfig`/:class:`~irchroma.config.Config`.
      cache:        a :class:`~irchroma.serve.cache.TileCache` (built from config if ``None``).
      cog_endpoint: STAC endpoint for source reads on cache miss.
      item_resolver: optional callable ``(z, x, y) -> STAC item`` to resolve which COG
                    serves a tile (the discovery result; cache it). ``None`` ⇒ live
                    reads are disabled and misses return a friendly 503.

    Returns:
      A configured ``fastapi.FastAPI`` instance.

    Raises:
      RuntimeError: if FastAPI is not installed (with install guidance).
    """
    if not _HAS_FASTAPI or FastAPI is None:
        raise RuntimeError(
            "irchroma.serve.api.create_app requires FastAPI (`pip install fastapi "
            "uvicorn`). The cache layer (irchroma.serve.cache.TileCache) works without it."
        )

    server = TileServer(
        engine=engine, config=config, cache=cache, cog_endpoint=cog_endpoint
    )
    app = FastAPI(
        title="IRChroma Tile Server",
        description="O(1)-per-tile IR→RGB super-resolution + colorization tile serving.",
        version="0.1.0",
    )
    # Stash the server so tests/integrations can reach it via app.state.
    app.state.server = server

    @app.get("/health")
    def health() -> Any:  # noqa: D401 - tiny liveness probe
        """Liveness probe."""
        return {"status": "ok"}

    @app.get("/info")
    def info() -> Any:
        """Runtime / config / cache introspection."""
        return server.info()

    @app.get("/tiles/{z}/{x}/{y}.png")
    def tile(z: int, x: int, y: int, asset: str = Query("lwir11")) -> Any:
        """Serve one ``z/x/y`` PNG tile (O(1) cache hit, else render + write-back)."""
        # 1) O(1) cache lookup keyed by quadkey.
        cached = server.cache.get(z, x, y)
        if cached is not None:
            data = cached if isinstance(cached, (bytes, bytearray)) else bytes(cached)
            return Response(content=data, media_type="image/png")

        # 2) Cache miss: need a model + a way to resolve the source COG.
        if server.engine is None:
            raise HTTPException(
                status_code=503,
                detail="Cache miss and no inference engine attached; cannot render tile.",
            )
        if item_resolver is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Cache miss and no item_resolver configured: cannot resolve which "
                    "COG serves this tile. Provide create_app(item_resolver=...) (e.g. a "
                    "MosaicJSON quadkey→item lookup) to enable live COG reads."
                ),
            )
        try:
            item = item_resolver(z, x, y)
            ir_tile = server.read_source_tile(z, x, y, item=item, asset_key=asset)
            png = server.render_tile(z, x, y, ir_tile)
        except RuntimeError as exc:
            # Missing optional backend (COG reader / Pillow / torch) -> 503 with guidance.
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - unexpected render failure
            raise HTTPException(status_code=500, detail=f"Tile render failed: {exc}") from exc
        return Response(content=png, media_type="image/png")

    return app


def run(
    app_or_engine: Any = None,
    host: str = "0.0.0.0",
    port: int = 8000,
    config: Any = None,
    **uvicorn_kwargs: Any,
) -> None:
    """Serve the tile app with uvicorn (guarded by ``uvicorn``).

    Convenience launcher: pass either an already-built FastAPI app or an
    :class:`InferenceEngine` (in which case :func:`create_app` is called for you).

    Args:
      app_or_engine: a FastAPI app, or an ``InferenceEngine`` (or ``None`` for a
                     cache-only server).
      host, port:    bind address.
      config:        config forwarded to :func:`create_app` when building from an engine.
      uvicorn_kwargs: extra keyword args forwarded to ``uvicorn.run``.

    Raises:
      RuntimeError: if uvicorn (or FastAPI) is not installed.
    """
    if not _HAS_UVICORN or uvicorn is None:
        raise RuntimeError(
            "Serving requires uvicorn (`pip install uvicorn`). FastAPI is also needed "
            "(`pip install fastapi`)."
        )
    # Decide whether we already have an app or need to build one.
    if _HAS_FASTAPI and FastAPI is not None and isinstance(app_or_engine, FastAPI):
        app = app_or_engine
    else:
        app = create_app(engine=app_or_engine, config=config)
    uvicorn.run(app, host=host, port=int(port), **uvicorn_kwargs)
