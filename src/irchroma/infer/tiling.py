"""irchroma.infer.tiling — patch-tiled inference with raised-cosine overlap-blend.

This is the **scalability / large-scene path** of the fast platform
(docs/research/05 §C4; ARCHITECTURE.md §9.2 #5). A whole scene is split into
fixed-size tiles (default ``512×512``) with an overlap margin, each tile is run
through the model (or a callable), and the per-tile outputs are stitched with a
**2-D raised-cosine weight window** that down-weights tile edges and normalizes by
the summed weights — giving a **seamless, seam-free** mosaic.

Honest complexity (docs/research/05): per-tile cost is **O(1)** (the tile is a
fixed size, independent of the scene); whole-scene cost is ``O(#tiles)`` and is
**embarrassingly parallel** (and batched here to saturate the accelerator). The
window blending and the overlap accumulation are the only "extra" work and are
themselves O(pixels) over a fixed tile.

Tensor conventions (see :mod:`irchroma.interfaces`): images are ``FloatTensor``
``[B, C, H, W]`` (channels-first). The model/callable may upscale by an integer
``scale`` — the stitched canvas is allocated at ``H*scale × W*scale`` and the
windows are placed at the scaled tile positions, so super-resolution tiling is
handled natively. A single-image ``[C, H, W]`` input is also accepted (a batch dim
is added and removed transparently).

``torch`` is imported lazily/guarded so this module *imports* even on a torch-less
box; :func:`tile_inference` raises a clear error if actually called without torch.
A pure-``numpy`` fallback for :func:`raised_cosine_window` is provided so the window
math is inspectable without torch.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Optional, Tuple, Union

# --------------------------------------------------------------------------- #
# Guarded torch / numpy imports — the module must import without either.
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - exercised on the real (torch) runtime
    import torch
    from torch import Tensor

    _HAS_TORCH = True
except Exception:  # pragma: no cover - torch-less environments (docs/CI)
    torch = None  # type: ignore
    Tensor = object  # type: ignore
    _HAS_TORCH = False

try:  # pragma: no cover
    import numpy as np

    _HAS_NUMPY = True
except Exception:  # pragma: no cover
    np = None  # type: ignore
    _HAS_NUMPY = False


__all__ = ["raised_cosine_window", "tile_inference"]

# A model is either an ``nn.Module``/any callable taking a tile tensor -> tile
# tensor, or a plain function. We only require ``out = fn(tile)``.
ModelOrFn = Union[Callable[["Tensor"], "Tensor"], Any]


def _require_torch() -> None:
    """Raise a clear error if torch is unavailable on the call path."""
    if not _HAS_TORCH or torch is None:  # pragma: no cover
        raise RuntimeError(
            "irchroma.infer.tiling.tile_inference requires PyTorch. Install torch "
            "to run patch-tiled inference (`pip install torch`)."
        )


# =========================================================================== #
# Raised-cosine (Hann) 2-D overlap-blend window.
# =========================================================================== #
def _raised_cosine_1d_values(length: int, overlap: int) -> "list[float]":
    """Return a length-``length`` 1-D raised-cosine taper as a Python float list.

    The window is **flat (== 1.0) in the interior** and tapers smoothly to ~0 at
    each edge over an ``overlap``-pixel ramp using a raised-cosine (Hann) half-period::

        w(i) = 0.5 * (1 - cos(pi * (i + 0.5) / overlap))   for the rising edge,

    mirrored on the falling edge. With this taper two tiles overlapping by exactly
    ``overlap`` pixels have complementary weights that sum to ~1 across the seam, so
    the normalized blend is seamless. A tiny positive floor keeps every weight > 0 so
    the weight-sum normalization never divides by zero at a covered pixel.
    """
    length = int(length)
    overlap = max(0, int(overlap))
    if length <= 0:
        return []
    # Clamp the ramp so the two edge ramps never overlap (need 2*ramp <= length).
    ramp = min(overlap, length // 2)
    w = [1.0] * length
    if ramp > 0:
        for i in range(ramp):
            # Half-cosine rising edge in (0, 1); +0.5 centers the sample in its cell.
            val = 0.5 * (1.0 - math.cos(math.pi * (i + 0.5) / ramp))
            w[i] = val
            w[length - 1 - i] = val
    # Positive floor so no covered pixel has zero accumulated weight.
    eps = 1e-4
    return [max(v, eps) for v in w]


def raised_cosine_window(tile: int, overlap: int) -> "Tensor":
    """Build a 2-D raised-cosine (Hann) overlap-blend weight window.

    The returned ``tile × tile`` weight map is the **outer product** of two 1-D
    raised-cosine tapers (one per axis): ~1.0 in the tile interior, tapering to ~0
    over an ``overlap``-pixel margin at every edge. Stitching tiles weighted by this
    window and dividing by the summed weights removes visible seams
    (docs/research/05 §C4 "overlap-blend"; ARCHITECTURE.md §9.2 #5).

    Args:
      tile:    tile edge length in pixels (the window is ``[tile, tile]``).
      overlap: taper width in pixels (the raised-cosine ramp at each edge). ``0``
               yields an all-ones (box) window.

    Returns:
      ``FloatTensor [tile, tile]`` in ``(0, 1]`` (torch). If torch is unavailable a
      ``numpy.ndarray`` is returned instead so the window math stays inspectable.
    """
    if tile <= 0:
        raise ValueError(f"tile must be >= 1, got {tile}.")
    vals = _raised_cosine_1d_values(tile, overlap)
    if _HAS_TORCH and torch is not None:
        w1d = torch.as_tensor(vals, dtype=torch.float32)  # [tile]
        win = torch.outer(w1d, w1d)  # [tile, tile]
        return win
    if _HAS_NUMPY and np is not None:  # pragma: no cover - torch-less fallback
        w1d = np.asarray(vals, dtype=np.float32)
        return np.outer(w1d, w1d)
    # Last resort: nested Python lists (no array backend at all).  # pragma: no cover
    return [[a * b for b in vals] for a in vals]  # type: ignore[return-value]


# =========================================================================== #
# Patch-tiled inference with overlap-blend stitching.
# =========================================================================== #
def _tile_starts(extent: int, tile: int, stride: int) -> "list[int]":
    """Compute tile start indices covering ``[0, extent)`` with the last tile flush.

    Steps by ``stride`` and always includes a final start of ``extent - tile`` so the
    bottom/right border is fully covered (clamped to ``>= 0`` for small extents).
    """
    if extent <= tile:
        return [0]
    starts = list(range(0, extent - tile + 1, stride))
    last = extent - tile
    if not starts or starts[-1] != last:
        starts.append(last)
    return starts


def tile_inference(
    model_or_fn: ModelOrFn,
    image: "Tensor",
    tile: int = 512,
    overlap: int = 64,
    scale: int = 1,
    device: Optional[Any] = None,
    batch_size: int = 1,
    out_channels: Optional[int] = None,
    window: Optional[str] = "raised_cosine",
    autocast: bool = False,
) -> "Tensor":
    """Run patch-tiled inference over a (possibly large) image with overlap-blend.

    The image is partitioned into ``tile × tile`` patches stepping by
    ``tile - overlap``; each patch is run through ``model_or_fn`` (batched
    ``batch_size`` patches at a time to saturate the accelerator), and the outputs
    are accumulated on a canvas weighted by a 2-D **raised-cosine** window
    (:func:`raised_cosine_window`) then divided by the summed weights → a seamless
    mosaic. Borders are handled by flushing the final row/column tile to the edge,
    and arbitrary input sizes (including smaller than one tile) are supported.

    Super-resolution: if ``model_or_fn`` upscales spatially by ``scale`` (an integer),
    the stitched canvas is allocated at ``H*scale × W*scale`` and every window is
    placed at the *scaled* tile position, so SR tiling needs no extra bookkeeping.

    Args:
      model_or_fn: an ``nn.Module`` or any callable ``tile_tensor -> tile_tensor``
                   mapping ``[b, C_in, tile, tile] -> [b, C_out, tile*scale,
                   tile*scale]``. Put the model in ``.eval()`` yourself if desired.
      image:       ``FloatTensor [B, C, H, W]`` (or ``[C, H, W]`` — a batch dim is
                   added then removed transparently).
      tile:        tile edge in pixels (fixed size ⇒ O(1) per tile).
      overlap:     overlap margin in pixels between adjacent tiles (raised-cosine ramp).
      scale:       integer spatial upscale factor applied by the model (default 1).
      device:      device to run on (``"cuda"``/``"cpu"``/``torch.device``); inputs
                   are moved there per-batch and outputs returned on the input device.
      batch_size:  number of tiles per forward pass (GPU saturation; default 1).
      out_channels: output channel count, if known (inferred from the first tile
                   otherwise — one cheap probe forward).
      window:      blend window name: ``"raised_cosine"`` (default) or ``"none"``/
                   ``None`` for a box (all-ones) window.
      autocast:    if ``True`` run each forward under ``torch.autocast`` on CUDA
                   (FP16 mixed precision) — a constant-factor speedup (docs/research/05 §C2).

    Returns:
      ``FloatTensor`` on the input's device: ``[B, C_out, H*scale, W*scale]`` (or
      ``[C_out, H*scale, W*scale]`` if the input was unbatched).

    Notes:
      * Per-tile work is **O(1)** (fixed tile size); whole-image work is ``O(#tiles)``
        and embarrassingly parallel (docs/research/05 §C4).
      * ``overlap`` is clamped to ``< tile``; ``overlap = 0`` is a hard (non-blended)
        tiling. Use a non-trivial overlap (e.g. 32–64) to avoid visible seams.
    """
    _require_torch()
    if tile <= 0:
        raise ValueError(f"tile must be >= 1, got {tile}.")
    scale = int(scale)
    if scale < 1:
        raise ValueError(f"scale must be >= 1, got {scale}.")
    overlap = max(0, min(int(overlap), tile - 1))
    stride = tile - overlap
    if stride <= 0:  # pragma: no cover - guarded above, defensive
        raise ValueError("overlap must be < tile so the stride is positive.")

    # ---- Normalize input to [B, C, H, W] ---------------------------------- #
    if not torch.is_tensor(image):
        image = torch.as_tensor(image)
    squeeze_batch = False
    if image.dim() == 3:
        image = image.unsqueeze(0)
        squeeze_batch = True
    if image.dim() != 4:
        raise ValueError(
            f"tile_inference expects image [B, C, H, W] or [C, H, W]; got {tuple(image.shape)}."
        )
    src_device = image.device
    run_device = (
        torch.device(device) if device is not None else src_device
    )
    b, c, h, w = image.shape
    out_h, out_w = h * scale, w * scale

    # ---- Move the model to the run device if it is an nn.Module ----------- #
    if hasattr(model_or_fn, "to") and device is not None:
        try:
            model_or_fn = model_or_fn.to(run_device)  # type: ignore[assignment]
        except Exception:  # pragma: no cover - non-movable callables are fine
            pass

    # ---- Build the blend window (at the OUTPUT/scaled tile size) ----------- #
    out_tile = tile * scale
    use_window = window not in (None, "none", "box")
    if use_window:
        win = raised_cosine_window(out_tile, overlap * scale).to(
            device=run_device, dtype=torch.float32
        )  # [out_tile, out_tile]
    else:
        win = torch.ones((out_tile, out_tile), device=run_device, dtype=torch.float32)
    win = win.view(1, 1, out_tile, out_tile)  # broadcast over [b, C_out]

    # ---- Tile coverage (flush last tile to the border) -------------------- #
    ys = _tile_starts(h, tile, stride)
    xs = _tile_starts(w, tile, stride)
    coords = [(yy, xx) for yy in ys for xx in xs]

    def _forward(batch_tiles: "Tensor") -> "Tensor":
        """Run the model/callable on a [n, C, tile, tile] batch, guarded by autocast."""
        if autocast and run_device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                return model_or_fn(batch_tiles)
        return model_or_fn(batch_tiles)

    # ---- Infer output channel count (one cheap probe if unknown) ---------- #
    if out_channels is None:
        probe_y, probe_x = coords[0]
        probe = image[:, :, probe_y : probe_y + tile, probe_x : probe_x + tile].to(run_device)
        with torch.no_grad():
            probe_out = _forward(probe)
        if not torch.is_tensor(probe_out):  # pragma: no cover - defensive
            probe_out = torch.as_tensor(probe_out)
        out_channels = int(probe_out.shape[1])
        _probe_cache = {(probe_y, probe_x): probe_out.detach()}
    else:
        _probe_cache = {}

    # ---- Allocate accumulation canvases ----------------------------------- #
    acc = torch.zeros((b, out_channels, out_h, out_w), device=run_device, dtype=torch.float32)
    wsum = torch.zeros((b, 1, out_h, out_w), device=run_device, dtype=torch.float32)

    # ---- Iterate tiles in batches ----------------------------------------- #
    with torch.no_grad():
        i = 0
        n = len(coords)
        while i < n:
            chunk = coords[i : i + max(1, int(batch_size))]
            # Gather this chunk's input tiles, reusing the probe if present.
            need_forward = [(yy, xx) for (yy, xx) in chunk if (yy, xx) not in _probe_cache]
            forwarded: dict = {}
            if need_forward:
                in_tiles = torch.stack(
                    [
                        image[:, :, yy : yy + tile, xx : xx + tile]
                        for (yy, xx) in need_forward
                    ],
                    dim=0,
                )  # [k, b, C, tile, tile]
                k = in_tiles.shape[0]
                in_tiles = in_tiles.reshape(k * b, c, tile, tile).to(run_device)
                out_tiles = _forward(in_tiles)
                if not torch.is_tensor(out_tiles):  # pragma: no cover
                    out_tiles = torch.as_tensor(out_tiles)
                out_tiles = out_tiles.reshape(k, b, out_channels, out_tile, out_tile)
                for j, (yy, xx) in enumerate(need_forward):
                    forwarded[(yy, xx)] = out_tiles[j]

            for (yy, xx) in chunk:
                out_tile_t = _probe_cache.pop((yy, xx), None)
                if out_tile_t is None:
                    out_tile_t = forwarded[(yy, xx)]
                out_tile_t = out_tile_t.to(device=run_device, dtype=torch.float32)
                oy, ox = yy * scale, xx * scale
                acc[:, :, oy : oy + out_tile, ox : ox + out_tile] += out_tile_t * win
                wsum[:, :, oy : oy + out_tile, ox : ox + out_tile] += win
            i += len(chunk)

    # ---- Normalize by the summed weights (seamless blend) ----------------- #
    out = acc / wsum.clamp_min(1e-8)
    out = out.to(src_device)
    if squeeze_batch:
        out = out[0]
    return out
