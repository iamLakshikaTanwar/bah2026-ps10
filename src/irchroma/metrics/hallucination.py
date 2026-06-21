"""irchroma.metrics.hallucination — Family E: hallucination / semantic faithfulness.

Owner: Builder-6 (MetricsBuilder). Implements docs/research/06 §E.

This is the make-or-break family for PS-10 ("ensure the process does not distort or
misrepresent ground-truth objects" / "no hallucinations"). Pixel and perceptual
metrics **cannot** catch a plausible-but-fake vehicle, so we need *task-level* and
*structural* consistency checks that quantify fabricated content:

  * :class:`SegConsistencyMetric` — run a frozen, INDEPENDENT segmenter on the
    generated RGB vs the GT-RGB (or input-derived labels) and return **mIoU**. High
    agreement => colorization preserved semantics; drops localize where the model
    *changed the scene's meaning*. The quantified no-hallucination number
    (``EvalConfig.seg_consistency_min`` gate). Higher better.
  * :class:`EdgeIoUMetric` — IoU of binarized edge maps (Sobel/Canny) between pred
    and the target structure. Catches *added / removed* structures. Higher better.
  * :class:`GradientCorrelationMetric` — Pearson correlation of gradient-magnitude
    maps; the output's edges must derive from the input, not appear from nowhere
    (``EvalConfig.edge_correlation_min`` gate). Higher better.

:class:`EdgeIoUMetric` and :class:`GradientCorrelationMetric` are **self-contained**
(numpy Sobel in ``_common``); OpenCV/skimage are used only as optional accelerators
for Canny. :class:`SegConsistencyMetric` needs a segmenter (from ``ctx`` or a lazily
imported ``FrozenSegmenter``) and raises a friendly error if none is available.

Tensor convention: ``pred`` / ``target`` are RGB ``[B, 3, H, W]`` in ``[0, 1]``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..interfaces import TORCH_AVAILABLE, Metric
from . import _common as _c

try:
    import numpy as np  # type: ignore

    _NUMPY = True
except Exception:  # pragma: no cover
    np = None  # type: ignore
    _NUMPY = False

if TORCH_AVAILABLE:  # pragma: no cover - exercised only with torch installed
    import torch  # type: ignore
else:  # pragma: no cover
    torch = None  # type: ignore

try:  # OpenCV — optional accelerator for Canny edges.
    import cv2 as _cv2  # type: ignore

    _HAS_CV2 = True
except Exception:  # pragma: no cover
    _cv2 = None  # type: ignore
    _HAS_CV2 = False


# --------------------------------------------------------------------------- #
# Shared structural helpers (numpy).
# --------------------------------------------------------------------------- #
def _to_gray_bhw(bchw: "np.ndarray") -> "np.ndarray":
    """Reduce a ``[B, C, H, W]`` array to grayscale ``[B, H, W]`` (BT.601 if RGB)."""
    if bchw.shape[1] == 1:
        return bchw[:, 0, :, :]
    if bchw.shape[1] >= 3:
        r, g, b = bchw[:, 0, :, :], bchw[:, 1, :, :], bchw[:, 2, :, :]
        return 0.299 * r + 0.587 * g + 0.114 * b
    return bchw.mean(axis=1)


def _grad_mag_bhw(gray_bhw: "np.ndarray") -> "np.ndarray":
    """Per-image Sobel gradient magnitude for a ``[B, H, W]`` stack."""
    return np.stack(
        [_c.sobel_gradient_magnitude(gray_bhw[i]) for i in range(gray_bhw.shape[0])],
        axis=0,
    )


def _canny_edges(gray_hw_01: "np.ndarray", low: float, high: float) -> "np.ndarray":
    """Boolean edge map for a single ``[H, W]`` image in [0,1].

    Uses OpenCV Canny when available (8-bit input); otherwise a self-contained
    Sobel-magnitude threshold at the high ratio (a robust fallback that needs no
    optional dep).
    """
    if _HAS_CV2:
        u8 = np.clip(gray_hw_01 * 255.0, 0, 255).astype(np.uint8)
        edges = _cv2.Canny(u8, int(low * 255), int(high * 255))  # type: ignore[union-attr]
        return edges > 0
    mag = _c.sobel_gradient_magnitude(gray_hw_01)
    if mag.max() <= 1e-12:
        return np.zeros_like(mag, dtype=bool)
    thr = high * float(mag.max())
    return mag >= thr


def _binary_iou(a: "np.ndarray", b: "np.ndarray") -> float:
    """IoU between two boolean masks (empty-vs-empty -> 1.0)."""
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(inter) / float(union)


def _logits_to_labels(x: Any) -> "Any":
    """Coerce a segmenter output to a ``[B, H, W]`` integer-label torch tensor.

    Accepts class logits ``[B, K, H, W]`` (argmax over K) or an already-discrete
    label map ``[B, H, W]`` / ``[H, W]``.
    """
    t = x
    if not _c.is_torch_tensor(t):
        if TORCH_AVAILABLE:
            t = torch.as_tensor(_c.to_numpy(x))  # type: ignore[union-attr]
        else:  # pragma: no cover
            raise RuntimeError("SegConsistencyMetric requires torch for label handling.")
    if t.dim() == 4:  # [B, K, H, W] logits
        return t.argmax(dim=1)
    if t.dim() == 2:  # [H, W]
        return t[None]
    return t  # [B, H, W]


class SegConsistencyMetric(Metric):
    """Segmentation-consistency mIoU — the quantified no-hallucination metric. Higher better.

    Runs a **frozen, independent** semantic segmenter on the generated RGB and on
    the reference (GT-RGB), then returns the **mean IoU** between the two label maps
    (doc 06 §E.29). High agreement => colorization preserved semantics; a large drop
    localizes where the model changed the scene's meaning (an invented object gets a
    different label). This feeds the ``EvalConfig.seg_consistency_min`` (≥ 0.70) gate.

    Segmenter resolution order:
      1. ``ctx["segmenter"]`` — a shared frozen segmenter (preferred; reused across
         metrics/losses). Must be callable ``rgb[B,3,H,W] -> logits[B,K,H,W]``.
      2. ``irchroma.models.semantic.landcover.FrozenSegmenter`` — lazily imported and
         constructed if available (Builder-4's module).
    If neither is available a friendly ``RuntimeError`` is raised (and the suite
    records NaN / skips).

    ``target`` may be either reference RGB (segmented to derive reference labels) or
    a precomputed label map ``[B, H, W]`` (e.g. the input-derived LULC map ``L``); a
    4-D non-3-channel target is treated as logits.
    """

    name: str = "seg_consistency_miou"
    higher_is_better: bool = True
    family: str = "faithfulness"

    def __init__(
        self,
        num_classes: Optional[int] = None,
        ignore_index: int = -1,
        device: Optional[str] = None,
    ) -> None:
        self.num_classes = num_classes
        self.ignore_index = int(ignore_index)
        self.device = device
        self._segmenter = None  # lazily resolved

    def _resolve_segmenter(self, ctx: Optional[Dict[str, Any]]) -> Any:
        if ctx is not None and ctx.get("segmenter") is not None:
            return ctx["segmenter"]
        if self._segmenter is not None:
            return self._segmenter
        # Lazy import of Builder-4's frozen checker (may not exist yet).
        try:
            from ..models.semantic.landcover import FrozenSegmenter  # type: ignore
        except Exception as exc:  # ImportError or AttributeError
            raise RuntimeError(
                "SegConsistencyMetric needs a frozen segmenter. Provide one via "
                "ctx['segmenter'] (callable rgb[B,3,H,W] -> logits[B,K,H,W]), or "
                "install/implement irchroma.models.semantic.landcover.FrozenSegmenter."
            ) from exc
        seg = FrozenSegmenter()
        if self.device is not None and hasattr(seg, "to"):
            seg = seg.to(self.device)
        if hasattr(seg, "eval"):
            seg.eval()
        self._segmenter = seg
        return seg

    def _segment(self, segmenter: Any, rgb: Any) -> "Any":
        """Run the segmenter and return a ``[B, H, W]`` label tensor."""
        t = rgb
        if not _c.is_torch_tensor(t):
            if not TORCH_AVAILABLE:  # pragma: no cover
                raise RuntimeError("SegConsistencyMetric requires torch to run a segmenter.")
            t = torch.as_tensor(_c.as_bchw(rgb), dtype=torch.float32)  # type: ignore[union-attr]
        if t.dim() == 3:
            t = t[None]
        if self.device is not None:
            t = t.to(self.device)
        if TORCH_AVAILABLE:
            with torch.no_grad():  # type: ignore[union-attr]
                out = segmenter(t)
        else:  # pragma: no cover
            out = segmenter(t)
        return _logits_to_labels(out)

    def _miou(self, pred_lbl: Any, ref_lbl: Any) -> float:
        """Mean IoU between two ``[B, H, W]`` label tensors (numpy under the hood)."""
        a = _c.to_numpy(pred_lbl).astype(np.int64)
        b = _c.to_numpy(ref_lbl).astype(np.int64)
        if a.shape != b.shape:
            raise ValueError(
                f"Segmentation maps differ in shape: {a.shape} vs {b.shape}."
            )
        valid = b != self.ignore_index
        if self.num_classes is not None:
            classes = range(int(self.num_classes))
        else:
            present = np.unique(np.concatenate([a[valid].ravel(), b[valid].ravel()]))
            classes = [int(c) for c in present if c != self.ignore_index]
        ious = []
        for c in classes:
            pa = (a == c) & valid
            pb = (b == c) & valid
            union = np.logical_or(pa, pb).sum()
            if union == 0:
                continue  # class absent in both -> excluded from the mean
            inter = np.logical_and(pa, pb).sum()
            ious.append(float(inter) / float(union))
        if not ious:
            return 1.0  # identical / empty-valid -> perfect agreement
        return float(np.mean(ious))

    def __call__(
        self,
        pred: Any,
        target: Any = None,
        ctx: Optional[Dict[str, Any]] = None,
    ) -> float:
        if target is None:
            raise ValueError("SegConsistencyMetric requires a target (GT-RGB or labels).")
        segmenter = self._resolve_segmenter(ctx)
        pred_lbl = self._segment(segmenter, pred)

        # Decide whether the target is RGB (segment it) or precomputed labels.
        t_arr = _c.as_bchw(target) if not _c.is_torch_tensor(target) else None
        is_rgb_target = False
        if _c.is_torch_tensor(target):
            is_rgb_target = target.dim() == 4 and target.shape[1] == 3
        elif t_arr is not None:
            is_rgb_target = t_arr.shape[1] == 3
        if is_rgb_target:
            ref_lbl = self._segment(segmenter, target)
        else:
            ref_lbl = _logits_to_labels(target)
        return self._miou(pred_lbl, ref_lbl)


class EdgeIoUMetric(Metric):
    """Edge IoU — IoU of binarized edge maps between pred and target. Higher better.

    Binarizes Canny (or Sobel-threshold fallback) edges of the generated RGB and the
    reference structure, then computes the IoU of edge pixels — catching *added /
    removed* structures (doc 06 §E.32). Self-contained: OpenCV's Canny is used when
    present, otherwise a numpy Sobel-magnitude threshold (no optional dep required).

    Args:
      low, high: Canny hysteresis thresholds as fractions of the 0–1 intensity
                 range (mapped to 0–255 for OpenCV). For the Sobel fallback, ``high``
                 is the gradient-magnitude threshold ratio.
    """

    name: str = "edge_iou"
    higher_is_better: bool = True
    family: str = "faithfulness"

    def __init__(self, low: float = 0.1, high: float = 0.2, border: int = 0) -> None:
        self.low = float(low)
        self.high = float(high)
        self.border = int(border)

    def __call__(
        self,
        pred: Any,
        target: Any = None,
        ctx: Optional[Dict[str, Any]] = None,
    ) -> float:
        if target is None:
            raise ValueError("EdgeIoUMetric requires a target (reference structure).")
        p = _c.as_bchw(pred)
        t = _c.as_bchw(target)
        if self.border > 0:
            p = _c.shave_border(p, self.border)
            t = _c.shave_border(t, self.border)
        gp = _to_gray_bhw(p)
        gt = _to_gray_bhw(t)
        n = min(gp.shape[0], gt.shape[0])
        ious = []
        for i in range(n):
            ep = _canny_edges(gp[i], self.low, self.high)
            et = _canny_edges(gt[i], self.low, self.high)
            ious.append(_binary_iou(ep, et))
        return float(np.mean(ious)) if ious else 1.0


class GradientCorrelationMetric(Metric):
    """Gradient-magnitude correlation — structure-fabrication detector. Higher better.

    Pearson correlation between the Sobel gradient-magnitude maps of the prediction
    and the reference structure (doc 06 §E.31). SR/colorization must *not* invent
    edges: the output's structural gradients should track the input/reference, not
    appear from nowhere. Feeds the ``EvalConfig.edge_correlation_min`` (≥ 0.80) gate.

    Fully self-contained (numpy Sobel); no optional dependency. The ``target`` is
    typically the GT-RGB, but may be the upsampled IR input (its structure should be
    reproduced) — anything convertible to ``[B, C, H, W]``.
    """

    name: str = "gradient_correlation"
    higher_is_better: bool = True
    family: str = "faithfulness"

    def __init__(self, border: int = 0) -> None:
        self.border = int(border)

    def __call__(
        self,
        pred: Any,
        target: Any = None,
        ctx: Optional[Dict[str, Any]] = None,
    ) -> float:
        if target is None:
            raise ValueError("GradientCorrelationMetric requires a target (reference).")
        p = _c.as_bchw(pred)
        t = _c.as_bchw(target)
        if self.border > 0:
            p = _c.shave_border(p, self.border)
            t = _c.shave_border(t, self.border)
        gp = _grad_mag_bhw(_to_gray_bhw(p))
        gt = _grad_mag_bhw(_to_gray_bhw(t))
        if gp.shape != gt.shape:
            raise ValueError(
                f"Gradient maps differ in shape: {gp.shape} vs {gt.shape} "
                "(pred and target must be the same spatial size)."
            )
        return _c.pearson_corr(gp, gt)


__all__ = [
    "SegConsistencyMetric",
    "EdgeIoUMetric",
    "GradientCorrelationMetric",
]
