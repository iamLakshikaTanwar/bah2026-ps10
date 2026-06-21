"""irchroma.losses.color — CIE-Lab chroma / palette / colorfulness color terms.

Color-space losses enforce *correct, in-class, sufficiently-vivid* color while curbing
sepia and wrong hues (ARCHITECTURE §8, docs/research/02 §7, 04 "losses"):

  * :class:`ChrominanceLoss`        — L1 on the CIE-Lab ``a*,b*`` chroma channels
                                      (``color_lab_chroma``). Curbs sepia / wrong hues.
  * :class:`ColorHistogramLoss`     — differentiable color-distribution match in Lab
                                      (soft-histogram L2 / 1-D Wasserstein;
                                      ``color_histogram``).
  * :class:`ColorfulnessLoss`       — Hasler-Süsstrunk colorfulness steered toward the
                                      target's colorfulness, bounded (``color_colorfulness``).
  * :class:`PaletteConsistencyLoss` — penalize predicted chroma that falls OUTSIDE the
                                      per-class Lab palette range (``color_lut_outofclass``;
                                      the "no-wrong-color" term), indexed by
                                      ``target['semantic']`` and the config
                                      ``DEFAULT_LAB_PALETTE``.

All terms operate on RGB in ``[0, 1]`` and convert to CIE-Lab (D65) in torch via a
differentiable sRGB->linear->XYZ->Lab pipeline implemented here. Every term returns a
**scalar** and is robust to a missing paired target / missing semantic labels
(returns a zero scalar on the right device).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ..config import (
    DEFAULT_LAB_PALETTE,
    HARD_PRIOR_CLASSES,
    LULC_CLASSES,
    NUM_LULC_CLASSES,
)
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


# --------------------------------------------------------------------------- #
# Differentiable sRGB -> CIE-Lab (D65) conversion in torch.
# --------------------------------------------------------------------------- #
#: sRGB (linear) -> XYZ matrix (D65), row-major.
_RGB2XYZ: Tuple[Tuple[float, float, float], ...] = (
    (0.4124564, 0.3575761, 0.1804375),
    (0.2126729, 0.7151522, 0.0721750),
    (0.0193339, 0.1191920, 0.9503041),
)
#: D65 reference white (Xn, Yn, Zn), scaled so Yn = 1.0.
_WHITE_D65: Tuple[float, float, float] = (0.95047, 1.0, 1.08883)


# Strictly-positive floor for the *input* of every fractional power, so the power and
# its gradient are finite for ALL elements before selection. ``torch.where`` (and the
# ``mask*a + (1-mask)*b`` blend) differentiate BOTH branches for every element, so the
# unselected branch's power must never see 0/negative inputs (grad inf/NaN -> NaN).
_POW_EPS = 1e-12


def _srgb_to_linear(c: Tensor) -> Tensor:
    """Inverse-companding sRGB -> linear-RGB (differentiable, NaN-gradient-safe).

    Uses the safe-input ``torch.where`` trick: the ``** 2.4`` branch is evaluated on a
    strictly-positive sanitized base for every element so its gradient is finite, then
    selected against the linear segment (the old ``mask*a + (1-mask)*b`` blend evaluated
    the power's gradient at unsafe inputs for the unselected pixels).
    """
    c = c.clamp(0.0, 1.0)
    low = c / 12.92
    base = ((c + 0.055) / 1.055).clamp(min=_POW_EPS)
    high = base ** 2.4
    return torch.where(c > 0.04045, high, low)  # type: ignore[union-attr]


def _lab_f(t: Tensor) -> Tensor:
    """The CIE-Lab nonlinearity ``f(t)`` (with the linear segment near 0).

    NaN-gradient-safe: the cube-root branch sees a strictly-positive sanitized input for
    every element. ``x ** (1/3)`` has gradient ``(1/3) x**(-2/3)`` which is inf at 0 and
    NaN for x<0; clamping the *input* (not the result) keeps the gradient finite
    everywhere, and ``torch.where`` then selects the correct branch per element.
    """
    delta = 6.0 / 29.0
    delta3 = delta ** 3
    t_safe = t.clamp(min=_POW_EPS)
    cube_root = t_safe ** (1.0 / 3.0)
    linear = t / (3.0 * delta * delta) + 4.0 / 29.0
    return torch.where(t > delta3, cube_root, linear)  # type: ignore[union-attr]


def rgb_to_lab(rgb: Tensor) -> Tensor:
    """Convert sRGB ``[B, 3, H, W]`` in ``[0, 1]`` to CIE-Lab ``[B, 3, H, W]``.

    Output channels are ``(L, a, b)`` with ``L in [0, 100]`` and ``a, b`` roughly in
    ``[-128, 127]``. Fully differentiable (used inside color losses). D65 white point,
    sRGB primaries (matches ``DEFAULT_LAB_PALETTE`` which were computed sRGB/D65).
    """
    lin = _srgb_to_linear(rgb)  # [B,3,H,W]
    m = torch.tensor(_RGB2XYZ, dtype=lin.dtype, device=lin.device)  # type: ignore[union-attr]
    # [B,3,H,W] -> apply 3x3 over the channel dim.
    b, _, h, w = lin.shape
    flat = lin.reshape(b, 3, h * w)  # [B,3,N]
    xyz = torch.matmul(m, flat)  # type: ignore[union-attr]  # [B,3,N]
    xyz = xyz.reshape(b, 3, h, w)
    xn, yn, zn = _WHITE_D65
    x = xyz[:, 0:1] / xn
    y = xyz[:, 1:2] / yn
    z = xyz[:, 2:3] / zn
    fx = _lab_f(x)
    fy = _lab_f(y)
    fz = _lab_f(z)
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    bb = 200.0 * (fy - fz)
    return torch.cat([L, a, bb], dim=1)  # type: ignore[union-attr]


def _resolve_rgb_pair(output: PipelineOutput, target: Sample) -> Any:
    """Return ``(rgb_pred, rgb_gt)`` or ``(None, ref)`` when the GT RGB is missing."""
    pred = output.get("rgb", None) if hasattr(output, "get") else None
    gt = target.get("rgb", None) if hasattr(target, "get") else None
    if pred is None or gt is None:
        return None, (pred if pred is not None else gt)
    if pred.shape[-2:] != gt.shape[-2:]:
        gt = F.interpolate(  # type: ignore[union-attr]
            gt, size=pred.shape[-2:], mode="bilinear", align_corners=False
        )
    return pred, gt


class ChrominanceLoss(LossTerm):
    """L1 chrominance loss on the CIE-Lab ``a*,b*`` channels (``color_lab_chroma``).

    Converts predicted and target RGB to Lab and L1-matches only the chroma channels
    (``a*``, ``b*``), leaving luminance ``L`` to the structural/pixel terms. Directly
    targets sepia bias and wrong hues (docs/research/02 §7, 04 "losses"). Optionally
    normalizes by a nominal chroma scale so the magnitude is comparable to pixel terms.

    Args:
      include_l: if ``True`` also match the ``L`` channel (default ``False``).
      scale:     divide the Lab differences by this before averaging (default ``1.0``;
                 set e.g. ``128.0`` to put chroma error on a ~[0,1]-ish scale).
    """

    name = "lab_chroma"

    def __init__(self, include_l: bool = False, scale: float = 1.0) -> None:
        super().__init__()  # type: ignore[misc]
        self.include_l = bool(include_l)
        self.scale = float(scale) if scale else 1.0

    def forward(
        self,
        output: PipelineOutput,
        target: Sample,
        ctx: Dict[str, Any],
    ) -> Tensor:
        pred, gt = _resolve_rgb_pair(output, target)
        if pred is None:
            return _zero_scalar(gt)
        lab_p = rgb_to_lab(pred.clamp(0.0, 1.0))
        lab_g = rgb_to_lab(gt.clamp(0.0, 1.0))
        if self.include_l:
            diff = lab_p - lab_g
        else:
            diff = lab_p[:, 1:3] - lab_g[:, 1:3]  # a*, b* only
        return (diff.abs() / self.scale).mean()


class ColorHistogramLoss(LossTerm):
    """Differentiable color-distribution loss in CIE-Lab (soft-histogram match).

    Builds **soft** (differentiable) marginal histograms of the ``a*`` and ``b*``
    chroma channels for prediction and target via Gaussian soft-binning, then matches
    them. Two modes:
      * ``'l2'``         : squared-error between the (normalized) soft histograms.
      * ``'wasserstein'``: 1-D Wasserstein-1 via the L1 distance of the histogram CDFs
                           (the per-channel Earth-Mover distance over the chroma axis).
    This is the palette / color-distribution match term (``color_histogram``;
    docs/research/02 §7, 04 §"histogram (Wasserstein)").

    Args:
      bins:    number of soft-histogram bins per channel (default ``64``).
      mode:    ``'wasserstein'`` (default) or ``'l2'``.
      lab_min/lab_max: chroma axis range covered by the bins (default ``[-110, 110]``).
      sigma_scale: soft-bin Gaussian width as a multiple of the bin spacing (default 1).
    """

    name = "color_histogram"

    def __init__(
        self,
        bins: int = 64,
        mode: str = "wasserstein",
        lab_min: float = -110.0,
        lab_max: float = 110.0,
        sigma_scale: float = 1.0,
    ) -> None:
        super().__init__()  # type: ignore[misc]
        self.bins = int(bins)
        mode = str(mode).lower()
        if mode not in ("wasserstein", "l2"):
            raise ValueError(
                f"ColorHistogramLoss: mode must be 'wasserstein' or 'l2', got {mode!r}."
            )
        self.mode = mode
        self.lab_min = float(lab_min)
        self.lab_max = float(lab_max)
        self.sigma_scale = float(sigma_scale)
        if TORCH_AVAILABLE:
            centers = torch.linspace(self.lab_min, self.lab_max, self.bins)  # type: ignore[union-attr]
            self.register_buffer("_centers", centers, persistent=False)  # type: ignore[attr-defined]

    def _soft_hist(self, vals: Tensor) -> Tensor:
        """Soft histogram of ``vals`` (``[B, N]``) -> ``[B, bins]`` (sums to 1/row)."""
        centers = self._centers.to(vals.dtype)  # type: ignore[attr-defined]
        spacing = (self.lab_max - self.lab_min) / max(1, self.bins - 1)
        sigma = spacing * self.sigma_scale + 1e-6
        # [B, N, 1] - [1, 1, bins] -> Gaussian soft assignment.
        d = vals.unsqueeze(-1) - centers.view(1, 1, -1)
        w = torch.exp(-0.5 * (d / sigma) ** 2)  # type: ignore[union-attr]  # [B,N,bins]
        hist = w.sum(dim=1)  # [B, bins]
        hist = hist / (hist.sum(dim=1, keepdim=True) + 1e-8)
        return hist

    def _channel_loss(self, vp: Tensor, vg: Tensor) -> Tensor:
        """Histogram distance for one chroma channel (flattened ``[B, N]``)."""
        hp = self._soft_hist(vp)
        hg = self._soft_hist(vg)
        if self.mode == "l2":
            return ((hp - hg) ** 2).sum(dim=1).mean()
        # Wasserstein-1 over the 1-D bin axis == L1 of the CDFs.
        cp = torch.cumsum(hp, dim=1)  # type: ignore[union-attr]
        cg = torch.cumsum(hg, dim=1)  # type: ignore[union-attr]
        return (cp - cg).abs().sum(dim=1).mean()

    def forward(
        self,
        output: PipelineOutput,
        target: Sample,
        ctx: Dict[str, Any],
    ) -> Tensor:
        pred, gt = _resolve_rgb_pair(output, target)
        if pred is None:
            return _zero_scalar(gt)
        lab_p = rgb_to_lab(pred.clamp(0.0, 1.0))
        lab_g = rgb_to_lab(gt.clamp(0.0, 1.0))
        b = lab_p.shape[0]
        ap = lab_p[:, 1].reshape(b, -1)
        bp = lab_p[:, 2].reshape(b, -1)
        ag = lab_g[:, 1].reshape(b, -1)
        bg = lab_g[:, 2].reshape(b, -1)
        return 0.5 * (self._channel_loss(ap, ag) + self._channel_loss(bp, bg))


class ColorfulnessLoss(LossTerm):
    """Hasler-Süsstrunk colorfulness loss, steered toward the target (bounded).

    Computes the Hasler-Süsstrunk colorfulness metric
    ``M = sqrt(std_rg^2 + std_yb^2) + 0.3 * sqrt(mean_rg^2 + mean_yb^2)`` (with
    ``rg = R - G`` and ``yb = 0.5*(R + G) - B`` on ``[0, 255]``-scaled RGB) for both
    prediction and target, and penalizes the squared difference. This *counters
    desaturation* without unbounded saturation pumping (``color_colorfulness``;
    docs/research/02 §7 DDColor colorfulness). If the paired target is missing it
    degrades to a one-sided hinge that nudges colorfulness up toward ``target_value``.

    Args:
      target_value: fallback target colorfulness when no GT RGB is present (default 25).
      scale:        divide the colorfulness difference by this before squaring
                    (default ``50.0`` to keep the term ``O(1)``).
    """

    name = "colorfulness"

    def __init__(self, target_value: float = 25.0, scale: float = 50.0) -> None:
        super().__init__()  # type: ignore[misc]
        self.target_value = float(target_value)
        self.scale = float(scale) if scale else 1.0

    @staticmethod
    def _colorfulness(rgb: Tensor) -> Tensor:
        """Per-image Hasler-Süsstrunk colorfulness ``[B]`` (RGB in [0,1])."""
        x = rgb.clamp(0.0, 1.0) * 255.0
        r, g, b = x[:, 0], x[:, 1], x[:, 2]
        rg = r - g
        yb = 0.5 * (r + g) - b
        rg_f = rg.reshape(rg.shape[0], -1)
        yb_f = yb.reshape(yb.shape[0], -1)
        std_rg = rg_f.std(dim=1)
        std_yb = yb_f.std(dim=1)
        mean_rg = rg_f.mean(dim=1)
        mean_yb = yb_f.mean(dim=1)
        std_root = torch.sqrt(std_rg ** 2 + std_yb ** 2 + 1e-8)  # type: ignore[union-attr]
        mean_root = torch.sqrt(mean_rg ** 2 + mean_yb ** 2 + 1e-8)  # type: ignore[union-attr]
        return std_root + 0.3 * mean_root

    def forward(
        self,
        output: PipelineOutput,
        target: Sample,
        ctx: Dict[str, Any],
    ) -> Tensor:
        pred = output.get("rgb", None) if hasattr(output, "get") else None
        if pred is None:
            return _zero_scalar(None)
        gt = target.get("rgb", None) if hasattr(target, "get") else None
        c_pred = self._colorfulness(pred)
        if gt is not None:
            c_gt = self._colorfulness(gt)
            diff = (c_pred - c_gt) / self.scale
            return (diff ** 2).mean()
        # No GT: one-sided hinge nudging colorfulness UP toward target_value.
        deficit = (self.target_value - c_pred).clamp(min=0.0) / self.scale
        return (deficit ** 2).mean()


class PaletteConsistencyLoss(LossTerm):
    """Out-of-class color penalty against the per-class Lab palette range.

    Implements the constrained / out-of-class penalty
    ``L = mean_p dist(chroma_p, range[class_p])`` (docs/research/04 §"Constrained /
    out-of-class penalty loss"): for each pixel, look up its class's allowed CIE-Lab
    chroma box ``[mu - k*sigma, mu + k*sigma]`` (seeded by ``DEFAULT_LAB_PALETTE`` from
    config) and penalize the predicted ``a*,b*`` only where they fall **outside** the
    box (a one-sided hinge). Hard-prior classes (``HARD_PRIOR_CLASSES`` = water, snow)
    get a tighter box. This is the ``color_lut_outofclass`` "no-wrong-color" term.

    It needs ``target['semantic']`` (the per-pixel label map). If absent, returns a
    zero scalar. The semantic map is bilinearly index-resized (nearest) to the RGB
    resolution when needed.

    Args:
      sigma:        half-width of the chroma box in Lab units for normal classes.
      hard_sigma:   tighter half-width for hard-prior classes.
      ignore_index: label value to skip (no-data / ignore).
    """

    name = "palette_consistency"

    def __init__(
        self,
        sigma: float = 25.0,
        hard_sigma: float = 12.0,
        ignore_index: int = -1,
    ) -> None:
        super().__init__()  # type: ignore[misc]
        self.sigma = float(sigma)
        self.hard_sigma = float(hard_sigma)
        self.ignore_index = int(ignore_index)
        # Precompute per-class (a_lo, a_hi, b_lo, b_hi) boxes as plain Python lists;
        # converted to a tensor lazily on the right device at first forward.
        self._boxes_py: List[List[float]] = self._build_boxes()
        self._boxes_t: Any = None

    def _build_boxes(self) -> List[List[float]]:
        """Per-class ``[a_lo, a_hi, b_lo, b_hi]`` chroma boxes from the Lab palette."""
        boxes: List[List[float]] = []
        hard = set(HARD_PRIOR_CLASSES)
        for cls in LULC_CLASSES:
            _l, a, b = DEFAULT_LAB_PALETTE.get(cls, (50.0, 0.0, 0.0))
            s = self.hard_sigma if cls in hard else self.sigma
            boxes.append([a - s, a + s, b - s, b + s])
        return boxes

    def _boxes(self, device: Any, dtype: Any) -> Tensor:
        """Return (and cache) the ``[K, 4]`` chroma-box tensor on ``device``."""
        if self._boxes_t is None or self._boxes_t.device != device:
            self._boxes_t = torch.tensor(  # type: ignore[union-attr]
                self._boxes_py, dtype=dtype, device=device
            )
        return self._boxes_t

    def forward(
        self,
        output: PipelineOutput,
        target: Sample,
        ctx: Dict[str, Any],
    ) -> Tensor:
        pred = output.get("rgb", None) if hasattr(output, "get") else None
        if pred is None:
            return _zero_scalar(None)
        semantic = target.get("semantic", None) if hasattr(target, "get") else None
        if semantic is None:
            return _zero_scalar(pred)

        rgb = pred.clamp(0.0, 1.0)
        lab = rgb_to_lab(rgb)  # [B,3,H,W]
        b, _, h, w = lab.shape

        # Align semantic [B,H,W] (Long) to the RGB spatial size via nearest resize.
        sem = semantic
        if sem.dim() == 4:  # tolerate [B,1,H,W]
            sem = sem[:, 0]
        if sem.shape[-2:] != (h, w):
            sem = F.interpolate(  # type: ignore[union-attr]
                sem.unsqueeze(1).float(), size=(h, w), mode="nearest"
            )[:, 0]
        sem = sem.long()

        # Valid mask: in-range class indices and not the ignore index.
        valid = (sem != self.ignore_index) & (sem >= 0) & (sem < NUM_LULC_CLASSES)
        if valid.sum() == 0:
            return _zero_scalar(pred)
        sem_clamped = sem.clamp(0, NUM_LULC_CLASSES - 1)

        boxes = self._boxes(lab.device, lab.dtype)  # [K,4]
        gathered = boxes[sem_clamped]  # [B,H,W,4]
        a_lo = gathered[..., 0]
        a_hi = gathered[..., 1]
        b_lo = gathered[..., 2]
        b_hi = gathered[..., 3]

        a = lab[:, 1]  # [B,H,W]
        bb = lab[:, 2]
        # One-sided hinge: distance outside [lo, hi] (0 when inside the box).
        a_pen = (a_lo - a).clamp(min=0.0) + (a - a_hi).clamp(min=0.0)
        b_pen = (b_lo - bb).clamp(min=0.0) + (bb - b_hi).clamp(min=0.0)
        pen = a_pen + b_pen  # [B,H,W]
        valid_f = valid.to(pen.dtype)
        return (pen * valid_f).sum() / (valid_f.sum() + 1e-8)


__all__ = [
    "ChrominanceLoss",
    "ColorHistogramLoss",
    "ColorfulnessLoss",
    "PaletteConsistencyLoss",
    "rgb_to_lab",
]
