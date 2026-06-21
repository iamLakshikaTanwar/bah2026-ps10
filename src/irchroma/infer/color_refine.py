"""irchroma.infer.color_refine — O(1)-per-pixel color refinement (class-LUT + 3D-LUT).

The **last** step of the per-tile hot path (docs/research/05 §B; ARCHITECTURE.md
§9.2 #3): apply the genuinely *O(1)-per-pixel* color operators from Wave A to the
network's raw RGB output —

  1. **Class → CIE-Lab chroma clamp** (:class:`irchroma.models.semantic.color_lut.ClassColorLUT`):
     a single gather of the per-class Lab box by the per-pixel semantic label + an
     elementwise clip toward the class palette (water→blue, veg→green, built→gray).
     O(1)/pixel; this is the anti-hallucination color guarantee.
  2. **Learned image-adaptive 3D LUT** (:class:`...color_lut.AdaIntLUT`, optional):
     a tiny CNN predicts blend weights over basis 3D LUTs (once per tile); the
     per-pixel transform is a single trilinear interpolation (8-tap) → O(1)/pixel,
     <2 ms @4K. Applied *after* the class clamp for controllable, scene-adaptive color.

Both are imported **lazily** from ``irchroma.models.semantic.color_lut`` (inside the
function) so this module imports/compiles even while that Wave A file is unavailable
or torch is absent; a clear error is raised only when refinement is actually called
without torch.

Tensor conventions (see :mod:`irchroma.interfaces`):
  * ``rgb``      : ``FloatTensor [B, 3, H, W]`` in ``[0, 1]`` (sRGB, R, G, B).
  * ``semantic`` : ``LongTensor  [B, H, W]`` (or ``[B, 1, H, W]``) — LULC indices.
"""

from __future__ import annotations

from typing import Any, Optional

# --------------------------------------------------------------------------- #
# Guarded torch import — module must import without torch.
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - exercised on the real (torch) runtime
    import torch
    import torch.nn.functional as F
    from torch import Tensor

    _HAS_TORCH = True
except Exception:  # pragma: no cover - torch-less environments (docs/CI)
    torch = None  # type: ignore
    F = None  # type: ignore
    Tensor = object  # type: ignore
    _HAS_TORCH = False


__all__ = ["apply_color_lut", "build_class_lut", "build_adaint_lut"]


def _require_torch() -> None:
    """Raise a clear error if torch is unavailable on the call path."""
    if not _HAS_TORCH or torch is None:  # pragma: no cover
        raise RuntimeError(
            "irchroma.infer.color_refine requires PyTorch. Install torch to apply "
            "the O(1) color-LUT refinement (`pip install torch`)."
        )


def _lazy_color_lut_module() -> Any:
    """Lazily import :mod:`irchroma.models.semantic.color_lut` (guarded, friendly error)."""
    try:
        from irchroma.models.semantic import color_lut as _cl  # local import on purpose
    except Exception as exc:  # pragma: no cover - only if Wave A file/torch missing
        raise RuntimeError(
            "Color refinement needs irchroma.models.semantic.color_lut "
            "(ClassColorLUT / AdaIntLUT). Ensure torch is installed and the semantic "
            f"color-LUT module is importable. Original error: {exc}"
        ) from exc
    return _cl


def build_class_lut(cfg: Optional[Any] = None, strength: float = 1.0, **kwargs: Any) -> Any:
    """Construct a :class:`ClassColorLUT` (lazy import). Returns an ``nn.Module``.

    Args:
      cfg:      an :class:`irchroma.config.SemanticConfig` (clamp sigmas / flags);
                defaults to ``SemanticConfig()`` inside the LUT.
      strength: blend factor in ``[0, 1]`` (``1.0`` = hard clamp, ``0.0`` = passthrough).
      kwargs:   forwarded to ``ClassColorLUT`` (``lab_palette``, ``chroma_margin``, ...).
    """
    _require_torch()
    cl = _lazy_color_lut_module()
    return cl.ClassColorLUT(cfg=cfg, strength=strength, **kwargs)


def build_adaint_lut(cfg: Optional[Any] = None, **kwargs: Any) -> Any:
    """Construct an :class:`AdaIntLUT` learned 3D-LUT (lazy import). Returns an ``nn.Module``.

    Args:
      cfg:    an :class:`irchroma.config.SemanticConfig` (reads ``lut3d_dim`` /
              ``lut3d_n_basis`` / ``lut3d_adaint``); defaults inside the module.
      kwargs: forwarded to ``AdaIntLUT`` (``thumb_size``, ``hidden``).
    """
    _require_torch()
    cl = _lazy_color_lut_module()
    return cl.AdaIntLUT(cfg=cfg, **kwargs)


def apply_color_lut(
    rgb: "Tensor",
    semantic: Optional["Tensor"] = None,
    lut: Optional[Any] = None,
    adaint: Optional[Any] = None,
    strength: float = 1.0,
    cfg: Optional[Any] = None,
) -> "Tensor":
    """Apply the O(1)-per-pixel color refinement: class-LUT chroma clamp + optional 3D-LUT.

    This wraps :meth:`ClassColorLUT.apply` (class-conditioned CIE-Lab chroma clamp,
    O(1)/pixel) and, when supplied, the learned :class:`AdaIntLUT` 3D-LUT (trilinear,
    O(1)/pixel) — the closing step of the per-tile hot path (docs/research/05 §B,
    ARCHITECTURE.md §9.2 #3). Order matters: the class clamp runs **first** (enforces
    per-class palette plausibility → anti-hallucination), then the learned 3D LUT does
    scene-adaptive grading on top.

    Both operators are imported **lazily** so this function is import-safe; it raises a
    friendly :class:`RuntimeError` only if called without torch or without the Wave A
    color-LUT module.

    Args:
      rgb:      ``FloatTensor [B, 3, H, W]`` in ``[0, 1]`` — raw predicted color.
      semantic: ``LongTensor [B, H, W]`` (or ``[B, 1, H, W]``) of LULC class indices.
                If ``None`` the class clamp is **skipped** (no labels to gather), but a
                supplied ``adaint`` 3D-LUT is still applied.
      lut:      an existing :class:`ClassColorLUT` instance to reuse (built per-call
                from ``cfg`` if ``None`` and ``semantic`` is provided). Passing a cached
                instance avoids rebuilding the table every tile.
      adaint:   an optional :class:`AdaIntLUT` instance (learned 3D-LUT) applied after
                the class clamp. ``None`` ⇒ no learned refinement.
      strength: class-clamp blend factor in ``[0, 1]`` used when ``lut`` is built here
                (``1.0`` = hard clamp). Ignored if a prebuilt ``lut`` is passed.
      cfg:      :class:`irchroma.config.SemanticConfig` used to build a ``ClassColorLUT``
                when one is not supplied.

    Returns:
      ``FloatTensor [B, 3, H, W]`` in ``[0, 1]`` — color-refined RGB.
    """
    _require_torch()
    if not torch.is_tensor(rgb):
        rgb = torch.as_tensor(rgb)
    if rgb.dim() != 4 or rgb.shape[1] != 3:
        raise ValueError(
            f"apply_color_lut expects rgb [B, 3, H, W]; got {tuple(rgb.shape)}."
        )

    out = rgb

    # ---- 1) Class-conditioned CIE-Lab chroma clamp (needs labels) --------- #
    if semantic is not None:
        sem = semantic if torch.is_tensor(semantic) else torch.as_tensor(semantic)
        # Accept [B, 1, H, W] by squeezing the singleton channel.
        if sem.dim() == 4 and sem.shape[1] == 1:
            sem = sem[:, 0]
        sem = sem.to(device=out.device).long()
        # Resize labels to the RGB grid if they came in at a coarser (IR) resolution.
        if sem.shape[-2:] != out.shape[-2:]:
            sem = (
                F.interpolate(
                    sem.unsqueeze(1).float(), size=out.shape[-2:], mode="nearest"
                )
                .long()
                .squeeze(1)
            )
        class_lut = lut if lut is not None else build_class_lut(cfg=cfg, strength=strength)
        # Move buffers to the RGB device (ClassColorLUT registers Lab tables as buffers).
        if hasattr(class_lut, "to"):
            try:
                class_lut = class_lut.to(out.device)
            except Exception:  # pragma: no cover - defensive
                pass
        out = class_lut.apply(out, sem)

    # ---- 2) Optional learned image-adaptive 3D LUT (AdaInt), O(1)/pixel ---- #
    if adaint is not None:
        if hasattr(adaint, "to"):
            try:
                adaint = adaint.to(out.device)
            except Exception:  # pragma: no cover - defensive
                pass
        out = adaint(out)

    return out.clamp(0.0, 1.0)
