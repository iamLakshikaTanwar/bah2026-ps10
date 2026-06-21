"""irchroma.models.semantic.uncertainty — honest low-confidence flagging & desaturation.

Dynamic-World-style uncertainty handling (docs/research/04 mechanisms §3; ARCHITECTURE.md
§7.5): rather than emit confident color where the model is guessing, high-uncertainty
pixels are **flagged** and **desaturated** so the product is *honest* instead of
hallucinated. The canonical uncertainty source is ``U = 1 - max(Dynamic World prob)`` but
any per-pixel confidence map (MC-dropout / ensemble variance) works.

Two pure functions (no module state, fully vectorized, differentiable where it matters):
  * :func:`flag_uncertain` — boolean mask of pixels whose uncertainty exceeds a threshold.
  * :func:`desaturate_low_confidence` — blend low-confidence pixels toward grayscale
    (luminance) so they read as "uncertain" in the final RGB.

Tensor conventions (see :mod:`irchroma.interfaces`):
  * ``rgb``         : ``FloatTensor [B, 3, H, W]`` in ``[0, 1]``.
  * confidence maps : ``FloatTensor [B, 1, H, W]`` (or ``[B, H, W]``) in ``[0, 1]``.
  * uncertainty     : ``FloatTensor`` in ``[0, 1]`` (``= 1 - confidence``); the
    :data:`PipelineOutput.uncertainty` convention is **high = least reliable**.

The module imports even without torch installed.
"""

from __future__ import annotations

from typing import Optional

from irchroma.interfaces import TORCH_AVAILABLE

# --------------------------------------------------------------------------- #
# Guarded torch import.
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - exercised only when torch is present
    import torch
    import torch.nn.functional as F
    from torch import Tensor
except Exception:  # pragma: no cover - torch-less environments (docs/CI)
    torch = None  # type: ignore
    F = None  # type: ignore
    Tensor = object  # type: ignore


__all__ = [
    "flag_uncertain",
    "desaturate_low_confidence",
    "uncertainty_from_probs",
]

# ITU-R BT.601 luminance weights (R, G, B) for desaturation toward gray.
_LUMA = (0.299, 0.587, 0.114)


def _as_bchw(x: "Tensor", name: str) -> "Tensor":
    """Coerce a ``[B, H, W]`` or ``[B, 1, H, W]`` map to ``[B, 1, H, W]``."""
    if x.dim() == 3:
        return x.unsqueeze(1)
    if x.dim() == 4 and x.shape[1] == 1:
        return x
    raise ValueError(f"{name} must be [B, H, W] or [B, 1, H, W]; got {tuple(x.shape)}.")


def uncertainty_from_probs(prob_or_logits: "Tensor", is_logits: bool = False) -> "Tensor":
    """Dynamic-World uncertainty ``U = 1 - max_class_prob`` from class scores.

    Args:
      prob_or_logits: ``FloatTensor [B, K, H, W]`` per-class probabilities (or logits if
                      ``is_logits``); ``K`` = number of classes.
      is_logits:      if ``True``, softmax over the class dim first.

    Returns:
      ``FloatTensor [B, 1, H, W]`` uncertainty in ``[0, 1]`` (high = least reliable).
    """
    if prob_or_logits.dim() != 4:
        raise ValueError(
            f"uncertainty_from_probs expects [B, K, H, W]; got {tuple(prob_or_logits.shape)}."
        )
    probs = F.softmax(prob_or_logits, dim=1) if is_logits else prob_or_logits
    max_prob = probs.max(dim=1, keepdim=True).values  # [B, 1, H, W]
    return (1.0 - max_prob).clamp(0.0, 1.0)


def flag_uncertain(prob_or_logits: "Tensor", threshold: float = 0.5) -> "Tensor":
    """Boolean mask of low-confidence (high-uncertainty) pixels.

    Accepts either a per-class score tensor ``[B, K, H, W]`` (``K > 1`` -> derive
    ``U = 1 - max prob``) or an already-computed single-channel uncertainty map
    ``[B, 1, H, W]`` / ``[B, H, W]`` in ``[0, 1]``. A pixel is flagged when its
    uncertainty **exceeds** ``threshold``.

    Args:
      prob_or_logits: ``[B, K, H, W]`` class probabilities/logits **or** a ``[B, 1, H, W]``
                      / ``[B, H, W]`` uncertainty map in ``[0, 1]``.
      threshold:      uncertainty above which a pixel is flagged (default
                      ``SemanticConfig.uncertainty_threshold`` = ``0.5``).

    Returns:
      ``BoolTensor [B, 1, H, W]`` — ``True`` where the pixel is uncertain/unreliable.
    """
    if not TORCH_AVAILABLE:  # pragma: no cover
        raise RuntimeError("flag_uncertain requires torch.")
    if prob_or_logits.dim() == 4 and prob_or_logits.shape[1] > 1:
        # Multi-class scores: logits if outside [0,1] OR not summing ~1 -> treat softmax.
        is_logits = bool(
            (prob_or_logits.min() < 0.0) or (prob_or_logits.max() > 1.0)
        )
        uncertainty = uncertainty_from_probs(prob_or_logits, is_logits=is_logits)
    else:
        uncertainty = _as_bchw(prob_or_logits, "uncertainty").clamp(0.0, 1.0)
    return uncertainty > float(threshold)


def desaturate_low_confidence(
    rgb: "Tensor",
    confidence: "Tensor",
    threshold: float = 0.5,
    min_saturation: float = 0.0,
    soft: bool = True,
    confidence_is_uncertainty: bool = False,
) -> "Tensor":
    """Desaturate (toward grayscale) pixels the model is unsure about — honest output.

    For each pixel we compute its luminance ``Y`` (BT.601) and blend the color toward
    ``Y`` by a per-pixel factor driven by confidence: confident pixels keep full color;
    low-confidence pixels are pulled toward gray. With ``soft=True`` the blend is a smooth
    ramp of the (clamped) confidence; with ``soft=False`` it is a hard switch at
    ``threshold`` (fully gray below, full color above). ``min_saturation`` keeps a floor of
    color even for the most uncertain pixels (``0`` = full gray).

    Args:
      rgb:        ``FloatTensor [B, 3, H, W]`` in ``[0, 1]``.
      confidence: ``FloatTensor [B, 1, H, W]`` / ``[B, H, W]`` in ``[0, 1]``. By default
                  interpreted as **confidence** (1 = reliable); pass
                  ``confidence_is_uncertainty=True`` to feed an uncertainty map instead.
      threshold:  confidence below which (or, for uncertainty, above which) pixels are
                  desaturated. Used as the ramp midpoint (soft) or switch (hard).
      min_saturation: lower bound on the kept color fraction in ``[0, 1]`` (default ``0``).
      soft:       smooth ramp (``True``) vs hard switch at ``threshold`` (``False``).
      confidence_is_uncertainty: treat ``confidence`` as ``U = 1 - confidence``.

    Returns:
      ``FloatTensor [B, 3, H, W]`` in ``[0, 1]`` with low-confidence pixels desaturated.
    """
    if not TORCH_AVAILABLE:  # pragma: no cover
        raise RuntimeError("desaturate_low_confidence requires torch.")
    if rgb.dim() != 4 or rgb.shape[1] != 3:
        raise ValueError(f"desaturate_low_confidence expects rgb [B, 3, H, W]; got {tuple(rgb.shape)}.")

    conf = _as_bchw(confidence, "confidence").to(dtype=rgb.dtype).clamp(0.0, 1.0)
    if confidence_is_uncertainty:
        conf = 1.0 - conf  # convert uncertainty -> confidence

    # Per-pixel "keep color" fraction in [0, 1].
    thr = float(threshold)
    if soft:
        if thr <= 0.0:
            keep = conf
        elif thr >= 1.0:
            keep = torch.zeros_like(conf)
        else:
            # Piecewise-linear ramp: 0 at conf<=0, 1 at conf>=1, crossing ~0.5 at thr.
            below = conf / thr  # confidence below threshold ramps 0..1 over [0, thr]
            above = 0.5 + 0.5 * (conf - thr) / (1.0 - thr)  # [thr, 1] -> [0.5, 1]
            keep = torch.where(conf < thr, 0.5 * below, above)
    else:
        keep = (conf >= thr).to(dtype=rgb.dtype)  # hard switch

    # Apply the min-saturation floor.
    floor = float(min_saturation)
    if floor > 0.0:
        keep = floor + (1.0 - floor) * keep
    keep = keep.clamp(0.0, 1.0)

    # Luminance (gray) target, broadcast to 3 channels.
    w = torch.as_tensor(_LUMA, dtype=rgb.dtype, device=rgb.device).view(1, 3, 1, 1)
    luma = (rgb * w).sum(dim=1, keepdim=True)  # [B, 1, H, W]
    gray = luma.expand_as(rgb)  # [B, 3, H, W]

    out = gray + keep * (rgb - gray)  # blend color toward gray by (1 - keep)
    return out.clamp(0.0, 1.0)
