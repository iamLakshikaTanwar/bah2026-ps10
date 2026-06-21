"""irchroma.losses.reconstruction — pixel / structure / frequency fidelity terms.

These are the **fidelity-dominant** anchors of the composite loss (ARCHITECTURE §8,
docs/research/02 §7, 03 §9). They pin the prediction to the ground truth in the
spatial, structural, and Fourier domains so the model cannot relocate or invent
structure without paying a large penalty:

  * :class:`CharbonnierLoss`  — robust L1 ``sqrt((x - y)^2 + eps^2)`` (the SR pixel
                                anchor ``sr_charbonnier`` and the colorization pixel
                                anchor ``color_l1``).
  * :class:`MSSSIMLoss`       — multi-scale SSIM structure loss ``1 - MS-SSIM``,
                                implemented from scratch with Gaussian windows
                                (``color_ms_ssim``). Works on RGB or single-channel.
  * :class:`GradientLoss`     — Sobel edge-gradient L1, the SPSR / SGA structure term
                                (``sr_gradient`` / ``color_edge``).
  * :class:`FFTLoss`          — Fourier-domain L1, the SwinFIR frequency term
                                (``sr_fft``).

Every term subclasses :class:`irchroma.interfaces.LossTerm`, implements
``forward(output, target, ctx) -> Tensor`` returning a **scalar**, and is robust to a
missing paired target (returns a zero scalar on the right device/dtype).

The terms read a configurable pair of keys from ``output`` / ``target`` so the same
class serves both stages:

  * SR terms    : ``output['sr']``  vs ``target['ir']`` *(or a high-res IR GT — see
                  ``target_key``)*.
  * Color terms : ``output['rgb']`` vs ``target['rgb']``.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from ..interfaces import LossTerm, PipelineOutput, Sample, TORCH_AVAILABLE, Tensor

try:  # Real torch at runtime; guarded so the module always imports.
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
    import torch.nn.functional as F  # type: ignore
except Exception:  # pragma: no cover - torch-less docs/CI box
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _zero_scalar(ref: Optional[Tensor] = None) -> Tensor:
    """Return a 0-dim zero tensor matching ``ref``'s device/dtype when possible.

    Used as the graceful fallback whenever a paired target is unavailable, so the
    same loss stack runs paired (training) and unpaired (inference) without raising.
    """
    if not TORCH_AVAILABLE:  # pragma: no cover - defensive only
        return 0.0  # type: ignore[return-value]
    if ref is not None and isinstance(ref, torch.Tensor):  # type: ignore[attr-defined]
        return torch.zeros((), dtype=ref.dtype, device=ref.device)  # type: ignore[union-attr]
    return torch.zeros((), dtype=torch.float32)  # type: ignore[union-attr]


def _resolve_pair(
    output: PipelineOutput,
    target: Sample,
    pred_key: str,
    target_key: str,
) -> Any:
    """Return ``(pred, gt)`` tensors for ``pred_key``/``target_key`` or ``(None, ref)``.

    ``ref`` is whichever tensor *is* present (preferring ``pred``) so callers can
    build a correctly-placed zero scalar when the GT is missing.
    """
    pred = output.get(pred_key, None) if hasattr(output, "get") else None
    gt = target.get(target_key, None) if hasattr(target, "get") else None
    if pred is None or gt is None:
        ref = pred if pred is not None else gt
        return None, ref
    return pred, gt


def _match_channels(pred: Tensor, gt: Tensor) -> Any:
    """Best-effort align channel counts of ``pred``/``gt`` (handles 1<->3 channel).

    If one side is single-channel and the other has ``C`` channels, the single
    channel is broadcast/repeated to ``C`` so e.g. an SR (1-ch) prediction can be
    compared against a 1-ch IR GT, or a gray GT against an RGB prediction. When the
    counts already match (or cannot be reconciled) the inputs are returned as-is.
    """
    if pred.shape[1] == gt.shape[1]:
        return pred, gt
    if pred.shape[1] == 1 and gt.shape[1] > 1:
        return pred.repeat(1, gt.shape[1], 1, 1), gt
    if gt.shape[1] == 1 and pred.shape[1] > 1:
        return pred, gt.repeat(1, pred.shape[1], 1, 1)
    return pred, gt


class CharbonnierLoss(LossTerm):
    """Robust L1 (Charbonnier) pixel-fidelity loss ``mean(sqrt((x - y)^2 + eps^2))``.

    The smooth-L1 variant used across the SR stack (``sr_charbonnier``) and as the
    dominant colorization pixel anchor (``color_l1``). More robust to IR noise /
    outliers than L2 while remaining differentiable at zero (docs/research/03 §9).

    Args:
      eps:        Charbonnier epsilon (``LossConfig.charbonnier_eps``; default ``1e-3``).
      pred_key:   key into ``output`` for the prediction (``"rgb"`` or ``"sr"``).
      target_key: key into ``target`` for the ground truth (``"rgb"`` or ``"ir"``).
    """

    name = "charbonnier"

    def __init__(
        self,
        eps: float = 1e-3,
        pred_key: str = "rgb",
        target_key: str = "rgb",
    ) -> None:
        super().__init__()  # type: ignore[misc]
        self.eps = float(eps)
        self.pred_key = pred_key
        self.target_key = target_key

    def forward(
        self,
        output: PipelineOutput,
        target: Sample,
        ctx: Dict[str, Any],
    ) -> Tensor:
        pred, gt = _resolve_pair(output, target, self.pred_key, self.target_key)
        if pred is None:
            return _zero_scalar(gt)
        pred, gt = _match_channels(pred, gt)
        diff = pred - gt
        loss = torch.sqrt(diff * diff + self.eps * self.eps)  # type: ignore[union-attr]
        return loss.mean()


class GradientLoss(LossTerm):
    """Sobel edge-gradient L1 loss (the SPSR / SGA structure-preservation term).

    Computes horizontal/vertical Sobel gradients of prediction and target and takes
    the L1 distance of their gradient magnitudes. Penalizes object-boundary drift and
    invented edges (``sr_gradient`` for SR, ``color_edge`` for colorization;
    docs/research/02 §7, 03 §9).

    Args:
      pred_key/target_key: which tensors to compare (defaults to RGB pair).
    """

    name = "gradient"

    def __init__(self, pred_key: str = "rgb", target_key: str = "rgb") -> None:
        super().__init__()  # type: ignore[misc]
        self.pred_key = pred_key
        self.target_key = target_key
        if TORCH_AVAILABLE:
            kx = torch.tensor(  # type: ignore[union-attr]
                [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
            )
            ky = kx.t().contiguous()
            # Registered as buffers so .to(device)/.half() move them with the module.
            self.register_buffer("kx", kx.view(1, 1, 3, 3), persistent=False)  # type: ignore[attr-defined]
            self.register_buffer("ky", ky.view(1, 1, 3, 3), persistent=False)  # type: ignore[attr-defined]

    def _grad_mag(self, x: Tensor) -> Tensor:
        """Per-channel Sobel gradient magnitude of ``x`` (``[B, C, H, W]``)."""
        c = x.shape[1]
        kx = self.kx.to(dtype=x.dtype).repeat(c, 1, 1, 1)  # type: ignore[attr-defined]
        ky = self.ky.to(dtype=x.dtype).repeat(c, 1, 1, 1)  # type: ignore[attr-defined]
        x = F.pad(x, (1, 1, 1, 1), mode="replicate")  # type: ignore[union-attr]
        gx = F.conv2d(x, kx, groups=c)  # type: ignore[union-attr]
        gy = F.conv2d(x, ky, groups=c)  # type: ignore[union-attr]
        return torch.sqrt(gx * gx + gy * gy + 1e-12)  # type: ignore[union-attr]

    def forward(
        self,
        output: PipelineOutput,
        target: Sample,
        ctx: Dict[str, Any],
    ) -> Tensor:
        pred, gt = _resolve_pair(output, target, self.pred_key, self.target_key)
        if pred is None:
            return _zero_scalar(gt)
        pred, gt = _match_channels(pred, gt)
        gp = self._grad_mag(pred)
        gg = self._grad_mag(gt)
        return (gp - gg).abs().mean()


class FFTLoss(LossTerm):
    """Fourier-domain L1 loss (the SwinFIR / high-frequency-emphasis term).

    Takes the 2-D real FFT of prediction and target and L1-matches the complex
    spectra (real & imaginary parts, equivalently magnitude+phase via stacking).
    IR detail lives in specific frequency bands, so this complements the spatial
    Charbonnier/gradient terms (``sr_fft``; docs/research/03 §9, SRConfig.use_fft_block).

    Args:
      pred_key/target_key: which tensors to compare (defaults to SR pair).
    """

    name = "fft"

    def __init__(self, pred_key: str = "sr", target_key: str = "ir") -> None:
        super().__init__()  # type: ignore[misc]
        self.pred_key = pred_key
        self.target_key = target_key

    def forward(
        self,
        output: PipelineOutput,
        target: Sample,
        ctx: Dict[str, Any],
    ) -> Tensor:
        pred, gt = _resolve_pair(output, target, self.pred_key, self.target_key)
        if pred is None:
            return _zero_scalar(gt)
        pred, gt = _match_channels(pred, gt)
        # If spatial sizes differ (e.g. SR vs LR IR), align via bilinear resize.
        if pred.shape[-2:] != gt.shape[-2:]:
            gt = F.interpolate(  # type: ignore[union-attr]
                gt, size=pred.shape[-2:], mode="bilinear", align_corners=False
            )
        # rfft2 over the last two dims; compare real+imag as a stacked real tensor.
        fp = torch.fft.rfft2(pred.float(), norm="ortho")  # type: ignore[union-attr]
        fg = torch.fft.rfft2(gt.float(), norm="ortho")  # type: ignore[union-attr]
        fp_ri = torch.stack([fp.real, fp.imag], dim=-1)  # type: ignore[union-attr]
        fg_ri = torch.stack([fg.real, fg.imag], dim=-1)  # type: ignore[union-attr]
        return (fp_ri - fg_ri).abs().mean()


# --------------------------------------------------------------------------- #
# MS-SSIM (implemented from scratch in torch)
# --------------------------------------------------------------------------- #
def _gaussian_window(window_size: int, sigma: float, channels: int, dtype: Any, device: Any) -> Tensor:
    """Build a separable 2-D Gaussian window of shape ``[C, 1, k, k]`` (groups=C)."""
    coords = torch.arange(window_size, dtype=dtype, device=device)  # type: ignore[union-attr]
    coords = coords - (window_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma))  # type: ignore[union-attr]
    g = g / g.sum()
    window_2d = g[:, None] * g[None, :]  # [k, k]
    return window_2d.expand(channels, 1, window_size, window_size).contiguous()


def _ssim_map(
    x: Tensor,
    y: Tensor,
    window: Tensor,
    window_size: int,
    channels: int,
    data_range: float,
) -> Any:
    """Return ``(mean_ssim, mean_cs)`` for one scale (luminance·contrast·structure)."""
    pad = window_size // 2
    mu_x = F.conv2d(x, window, padding=pad, groups=channels)  # type: ignore[union-attr]
    mu_y = F.conv2d(y, window, padding=pad, groups=channels)  # type: ignore[union-attr]
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y
    sigma_x2 = F.conv2d(x * x, window, padding=pad, groups=channels) - mu_x2  # type: ignore[union-attr]
    sigma_y2 = F.conv2d(y * y, window, padding=pad, groups=channels) - mu_y2  # type: ignore[union-attr]
    sigma_xy = F.conv2d(x * y, window, padding=pad, groups=channels) - mu_xy  # type: ignore[union-attr]

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    cs_map = (2.0 * sigma_xy + c2) / (sigma_x2 + sigma_y2 + c2)
    ssim_map = ((2.0 * mu_xy + c1) / (mu_x2 + mu_y2 + c1)) * cs_map
    # Clamp for numerical stability before the geometric-mean power steps.
    ssim_map = ssim_map.clamp(min=1e-6)
    cs_map = cs_map.clamp(min=1e-6)
    return ssim_map.mean(), cs_map.mean()


class MSSSIMLoss(LossTerm):
    """Multi-scale SSIM structure loss ``1 - MS-SSIM`` (from-scratch torch impl).

    Implements MS-SSIM with Gaussian windows and per-scale average pooling, combining
    contrast-structure (``cs``) terms across scales with the standard MS-SSIM
    exponents and the luminance term at the coarsest scale. Works on RGB **and**
    single-channel inputs (the window adapts to the channel count). This is the
    ``color_ms_ssim`` structure-preservation term (ARCHITECTURE §8;
    docs/research/02 §7).

    Args:
      window_size: Gaussian window side length (odd; default ``11``).
      sigma:       Gaussian sigma (default ``1.5``).
      data_range:  dynamic range of the inputs (default ``1.0`` for [0,1] images).
      weights:     per-scale MS-SSIM weights (default the canonical 5-scale set).
      pred_key/target_key: which tensors to compare (defaults to RGB pair).
    """

    name = "ms_ssim"
    #: Canonical Wang et al. 2003 MS-SSIM scale weights.
    _DEFAULT_WEIGHTS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)

    def __init__(
        self,
        window_size: int = 11,
        sigma: float = 1.5,
        data_range: float = 1.0,
        weights: Optional[List[float]] = None,
        pred_key: str = "rgb",
        target_key: str = "rgb",
    ) -> None:
        super().__init__()  # type: ignore[misc]
        self.window_size = int(window_size)
        self.sigma = float(sigma)
        self.data_range = float(data_range)
        self.weights = list(weights) if weights is not None else list(self._DEFAULT_WEIGHTS)
        self.pred_key = pred_key
        self.target_key = target_key

    def _msssim(self, x: Tensor, y: Tensor) -> Tensor:
        """Compute scalar MS-SSIM in ``[0, 1]`` for ``[B, C, H, W]`` inputs."""
        channels = x.shape[1]
        weights = torch.tensor(self.weights, dtype=x.dtype, device=x.device)  # type: ignore[union-attr]
        # Renormalize so the geometric-mean exponents sum to 1 even when the scale
        # set was trimmed for a small tile (keeps MS-SSIM well-scaled in [0, 1]).
        weights = weights / weights.sum().clamp(min=1e-8)
        levels = weights.numel()
        window = _gaussian_window(
            self.window_size, self.sigma, channels, x.dtype, x.device
        )
        mcs: List[Tensor] = []
        ssim_val: Tensor = x.new_tensor(1.0)  # type: ignore[attr-defined]
        for i in range(levels):
            ssim_val, cs = _ssim_map(
                x, y, window, self.window_size, channels, self.data_range
            )
            if i < levels - 1:
                mcs.append(cs)
                # Average-pool by 2 for the next (coarser) scale; pad odd sizes.
                pad_h = x.shape[-2] % 2
                pad_w = x.shape[-1] % 2
                if pad_h or pad_w:
                    x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")  # type: ignore[union-attr]
                    y = F.pad(y, (0, pad_w, 0, pad_h), mode="replicate")  # type: ignore[union-attr]
                x = F.avg_pool2d(x, kernel_size=2)  # type: ignore[union-attr]
                y = F.avg_pool2d(y, kernel_size=2)  # type: ignore[union-attr]
        # MS-SSIM = prod_{i<L} cs_i^{w_i} * ssim_L^{w_L}.
        mcs_stack = torch.stack(mcs + [ssim_val], dim=0)  # type: ignore[union-attr]
        mcs_stack = mcs_stack.clamp(min=1e-6)
        msssim = torch.prod(mcs_stack ** weights)  # type: ignore[union-attr]
        return msssim

    def forward(
        self,
        output: PipelineOutput,
        target: Sample,
        ctx: Dict[str, Any],
    ) -> Tensor:
        pred, gt = _resolve_pair(output, target, self.pred_key, self.target_key)
        if pred is None:
            return _zero_scalar(gt)
        pred, gt = _match_channels(pred, gt)
        # MS-SSIM needs enough spatial extent for the downsampling pyramid; if the
        # tile is too small for all scales, trim the weight set to what fits.
        min_side = min(pred.shape[-2], pred.shape[-1])
        max_levels = max(1, int(math.floor(math.log2(max(1, min_side) / (self.window_size - 1) + 1e-9))) + 1)
        if max_levels < len(self.weights):
            saved = self.weights
            self.weights = saved[:max_levels]
            try:
                msssim = self._msssim(pred.float(), gt.float())
            finally:
                self.weights = saved
        else:
            msssim = self._msssim(pred.float(), gt.float())
        return (1.0 - msssim).to(pred.dtype)


__all__ = [
    "CharbonnierLoss",
    "MSSSIMLoss",
    "GradientLoss",
    "FFTLoss",
]
