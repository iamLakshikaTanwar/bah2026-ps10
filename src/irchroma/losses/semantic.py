"""irchroma.losses.semantic — faithfulness / no-hallucination + smoothness terms.

These terms implement the anti-hallucination training signals and a light smoothness
prior (ARCHITECTURE §8, §11; docs/research/04):

  * :class:`SegConsistencyLoss` — run a FROZEN, independent segmenter on the generated
                                  RGB and require its prediction to match the
                                  input-derived label map (``target['semantic']``), or
                                  the segmenter's own prediction on the GT RGB. If the
                                  colorizer invents an object, the segmenter mislabels
                                  it and the loss rises (``seg_consistency``). The key
                                  no-hallucination signal (docs/research/04 §7.4).
  * :class:`SpectralAngleLoss`  — Spectral Angle Mapper (SAM) over multi-band tensors;
                                  preserves spectral relationships (``color_sam``).
  * :class:`TVLoss`             — total-variation smoothness, a small denoise prior
                                  (``color_tv``).

Every term subclasses :class:`irchroma.interfaces.LossTerm`, returns a **scalar**, and
is robust to missing optional inputs (e.g. ``SegConsistencyLoss`` returns a zero scalar
when neither labels nor a segmenter are available).
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, Optional

from ..config import IGNORE_INDEX, NUM_LULC_CLASSES
from ..interfaces import LossTerm, PipelineOutput, Sample, TORCH_AVAILABLE, Tensor

try:  # Real torch at runtime; guarded so the module always imports.
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
    import torch.nn.functional as F  # type: ignore
except Exception:  # pragma: no cover - torch-less docs/CI box
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore


def _zero_scalar(ref: Optional[Tensor] = None) -> Tensor:
    """Return a 0-dim zero tensor matching ``ref``'s device/dtype when possible."""
    if not TORCH_AVAILABLE:  # pragma: no cover
        return 0.0  # type: ignore[return-value]
    if ref is not None and isinstance(ref, torch.Tensor):  # type: ignore[attr-defined]
        return torch.zeros((), dtype=ref.dtype, device=ref.device)  # type: ignore[union-attr]
    return torch.zeros((), dtype=torch.float32)  # type: ignore[union-attr]


def _to_3ch(x: Tensor) -> Tensor:
    """Repeat single-channel to 3ch; collapse multi-band (>3) to a 3ch mean."""
    if x.shape[1] == 3:
        return x
    if x.shape[1] == 1:
        return x.repeat(1, 3, 1, 1)
    return x.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)


class SegConsistencyLoss(LossTerm):
    """Frozen-segmenter segmentation-consistency loss (the no-hallucination signal).

    Runs a frozen, independent segmenter ``S`` on the colorized output ``output['rgb']``
    and compares its prediction to a reference label distribution:

      * if ``target['semantic']`` (LULC label map) is available -> cross-entropy of
        ``S(rgb_pred)`` against those hard labels (``ignore_index`` honored);
      * else if the GT RGB ``target['rgb']`` is available -> KL divergence between
        ``S(rgb_pred)`` and ``S(rgb_gt)`` (self-referential consistency).

    The segmenter is sourced, in order, from:
      1. ``ctx['segmenter']``  (a shared frozen ``S`` provided by the training loop);
      2. a lazily-built ``irchroma.models.semantic.landcover.FrozenSegmenter``
         (Builder-4). If that module is unavailable, the loss returns a zero scalar and
         warns once (so the stack still runs before the segmenter lands).

    ``S`` is expected to return class logits ``[B, K, H, W]`` from RGB ``[B, 3, H, W]``
    in ``[0, 1]`` (the segmenter/checker convention in ARCHITECTURE §14).

    Args:
      mode:         ``'auto'`` (default; labels if present else KL-vs-GT), ``'ce'``
                    (force cross-entropy vs labels), or ``'kl'`` (force KL vs GT).
      ignore_index: label value to ignore in the cross-entropy (default config value).
      build_lazy:   if ``True`` (default) attempt to build a ``FrozenSegmenter`` when
                    ``ctx['segmenter']`` is absent.
    """

    name = "seg_consistency"

    def __init__(
        self,
        mode: str = "auto",
        ignore_index: int = IGNORE_INDEX,
        build_lazy: bool = True,
    ) -> None:
        super().__init__()  # type: ignore[misc]
        mode = str(mode).lower()
        if mode not in ("auto", "ce", "kl"):
            raise ValueError(
                f"SegConsistencyLoss: mode must be 'auto' | 'ce' | 'kl', got {mode!r}."
            )
        self.mode = mode
        self.ignore_index = int(ignore_index)
        self.build_lazy = bool(build_lazy)
        self._lazy_segmenter: Any = None
        self._lazy_failed = False
        self._warned = False

    def _resolve_segmenter(self, ctx: Dict[str, Any]) -> Optional[Callable[..., Any]]:
        """Return a callable frozen segmenter from ctx or a lazily-built one."""
        seg = ctx.get("segmenter") if ctx else None
        if seg is not None:
            return seg
        if not self.build_lazy or self._lazy_failed:
            return None
        if self._lazy_segmenter is not None:
            return self._lazy_segmenter
        try:  # Builder-4 territory; import lazily and guard hard.
            from ..models.semantic.landcover import FrozenSegmenter  # type: ignore

            seg = FrozenSegmenter()  # type: ignore[call-arg]
            # Best-effort freeze + eval (FrozenSegmenter should already be frozen).
            try:
                for p in seg.parameters():  # type: ignore[attr-defined]
                    p.requires_grad_(False)
                seg.eval()  # type: ignore[attr-defined]
            except Exception:
                pass
            self._lazy_segmenter = seg
            return seg
        except Exception:
            self._lazy_failed = True
            return None

    @staticmethod
    def _logits(seg_out: Any) -> Optional[Tensor]:
        """Extract class-logits ``[B, K, H, W]`` from a segmenter output."""
        if seg_out is None:
            return None
        if TORCH_AVAILABLE and isinstance(seg_out, torch.Tensor):  # type: ignore[attr-defined]
            return seg_out if seg_out.dim() == 4 else None
        if isinstance(seg_out, dict):
            for key in ("logits", "out", "semantic_pred"):
                v = seg_out.get(key)
                if TORCH_AVAILABLE and isinstance(v, torch.Tensor) and v.dim() == 4:  # type: ignore[attr-defined]
                    return v
        if isinstance(seg_out, (list, tuple)) and seg_out:
            first = seg_out[0]
            if TORCH_AVAILABLE and isinstance(first, torch.Tensor) and first.dim() == 4:  # type: ignore[attr-defined]
                return first
        return None

    def forward(
        self,
        output: PipelineOutput,
        target: Sample,
        ctx: Dict[str, Any],
    ) -> Tensor:
        ctx = ctx or {}
        rgb = output.get("rgb", None) if hasattr(output, "get") else None
        if rgb is None:
            return _zero_scalar(None)

        semantic = target.get("semantic", None) if hasattr(target, "get") else None
        rgb_gt = target.get("rgb", None) if hasattr(target, "get") else None
        # Decide the effective comparison mode given what's available.
        use_ce = (self.mode in ("auto", "ce")) and (semantic is not None)
        use_kl = (not use_ce) and (self.mode in ("auto", "kl")) and (rgb_gt is not None)
        if not use_ce and not use_kl:
            return _zero_scalar(rgb)

        segmenter = self._resolve_segmenter(ctx)
        if segmenter is None:
            if not self._warned:
                warnings.warn(
                    "SegConsistencyLoss: no segmenter in ctx and "
                    "irchroma.models.semantic.landcover.FrozenSegmenter unavailable; "
                    "returning zero (term inactive until a segmenter is provided).",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._warned = True
            return _zero_scalar(rgb)

        logits_pred = self._logits(segmenter(_to_3ch(rgb.clamp(0.0, 1.0))))
        if logits_pred is None:
            return _zero_scalar(rgb)

        if use_ce:
            sem = semantic
            if sem.dim() == 4:  # tolerate [B,1,H,W]
                sem = sem[:, 0]
            # Resize labels (nearest) to the logits' spatial size.
            if sem.shape[-2:] != logits_pred.shape[-2:]:
                sem = F.interpolate(  # type: ignore[union-attr]
                    sem.unsqueeze(1).float(), size=logits_pred.shape[-2:], mode="nearest"
                )[:, 0]
            sem = sem.long()
            return F.cross_entropy(  # type: ignore[union-attr]
                logits_pred, sem, ignore_index=self.ignore_index
            )

        # KL consistency vs frozen segmenter on the GT RGB (GT path detached).
        with torch.no_grad():  # type: ignore[union-attr]
            logits_gt = self._logits(segmenter(_to_3ch(rgb_gt.clamp(0.0, 1.0))))
        if logits_gt is None:
            return _zero_scalar(rgb)
        if logits_gt.shape[-2:] != logits_pred.shape[-2:]:
            logits_gt = F.interpolate(  # type: ignore[union-attr]
                logits_gt, size=logits_pred.shape[-2:], mode="bilinear", align_corners=False
            )
        log_p = F.log_softmax(logits_pred, dim=1)  # type: ignore[union-attr]
        q = F.softmax(logits_gt, dim=1)  # type: ignore[union-attr]
        # KL(q || p) summed over classes, averaged over pixels/batch.
        return F.kl_div(log_p, q, reduction="batchmean")  # type: ignore[union-attr]


class SpectralAngleLoss(LossTerm):
    """Spectral Angle Mapper (SAM) loss for multi-band tensors (``color_sam``).

    Treats each pixel as a spectral vector across channels and penalizes the average
    angle between the predicted and target spectra
    ``mean( arccos( <p, g> / (||p|| ||g||) ) )``. Physically meaningful for multi-band
    IR / RGB consistency (docs/research/02 §7, 04). Operates on ``output['rgb']`` vs
    ``target['rgb']`` by default; the angle is scale-invariant (insensitive to overall
    brightness), complementing the pixel terms.

    Args:
      pred_key/target_key: which tensors to compare (defaults to RGB pair).
      eps:                 numerical floor for norms / arccos clamping.
    """

    name = "spectral_angle"

    def __init__(
        self,
        pred_key: str = "rgb",
        target_key: str = "rgb",
        eps: float = 1e-7,
    ) -> None:
        super().__init__()  # type: ignore[misc]
        self.pred_key = pred_key
        self.target_key = target_key
        self.eps = float(eps)

    def forward(
        self,
        output: PipelineOutput,
        target: Sample,
        ctx: Dict[str, Any],
    ) -> Tensor:
        pred = output.get(self.pred_key, None) if hasattr(output, "get") else None
        gt = target.get(self.target_key, None) if hasattr(target, "get") else None
        if pred is None or gt is None:
            return _zero_scalar(pred if pred is not None else gt)
        if pred.shape[1] != gt.shape[1]:
            # Reconcile channel counts (e.g. 1ch SR vs multi-band) before the angle.
            if pred.shape[1] == 1:
                pred = pred.repeat(1, gt.shape[1], 1, 1)
            elif gt.shape[1] == 1:
                gt = gt.repeat(1, pred.shape[1], 1, 1)
            else:
                return _zero_scalar(pred)
        if pred.shape[-2:] != gt.shape[-2:]:
            gt = F.interpolate(  # type: ignore[union-attr]
                gt, size=pred.shape[-2:], mode="bilinear", align_corners=False
            )
        # Dot product / norms along the channel (spectral) dim.
        dot = (pred * gt).sum(dim=1)
        np_ = torch.sqrt((pred * pred).sum(dim=1) + self.eps)  # type: ignore[union-attr]
        ng = torch.sqrt((gt * gt).sum(dim=1) + self.eps)  # type: ignore[union-attr]
        cos = (dot / (np_ * ng)).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        angle = torch.acos(cos)  # type: ignore[union-attr]  # radians, [B,H,W]
        return angle.mean()


class TVLoss(LossTerm):
    """Total-variation smoothness loss (light denoise prior; ``color_tv``).

    Penalizes the mean absolute spatial gradient of ``output['rgb']`` (anisotropic TV).
    Kept at a tiny weight so it only removes speckle without blurring real edges
    (docs/research/02 §7: "mild smoothing only"). Reference-free (no target needed).

    Args:
      pred_key: which output tensor to smooth (default ``"rgb"``).
    """

    name = "tv"

    def __init__(self, pred_key: str = "rgb") -> None:
        super().__init__()  # type: ignore[misc]
        self.pred_key = pred_key

    def forward(
        self,
        output: PipelineOutput,
        target: Sample,
        ctx: Dict[str, Any],
    ) -> Tensor:
        x = output.get(self.pred_key, None) if hasattr(output, "get") else None
        if x is None:
            return _zero_scalar(None)
        if x.dim() != 4 or x.shape[-1] < 2 or x.shape[-2] < 2:
            return _zero_scalar(x)
        dh = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
        dw = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
        return dh + dw


__all__ = [
    "SegConsistencyLoss",
    "SpectralAngleLoss",
    "TVLoss",
]
