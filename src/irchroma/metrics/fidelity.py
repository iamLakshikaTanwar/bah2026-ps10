"""irchroma.metrics.fidelity — Family A: full-reference reconstruction fidelity.

Owner: Builder-6 (MetricsBuilder). Implements docs/research/06 §A.

These metrics compare a predicted RGB (or single-channel SR) tile against an
aligned ground-truth reference and quantify *pixel / structural / spectral*
correctness. They are meaningful only on the **paired** split (never on real IR
with no RGB GT — docs/research/06 §A "Library note").

Design priority (per the task contract): **PSNR, SSIM, MS-SSIM, RMSE, MAE, SAM,
and ERGAS are implemented directly in numpy and therefore NEVER require any
optional dependency.** They always work as long as numpy is importable. The
``torchmetrics`` / ``sewar`` / ``piq`` packages are *not* used here — a direct
implementation removes a whole class of layout/normalization bugs and guarantees
availability.

SR conventions honoured (docs/research/06 §6, §10 pitfalls):
  * **Y-channel option** — :class:`PSNRMetric` / :class:`SSIMMetric` can compute on
    the BT.601 luminance channel (the BasicSR/RCAN standard) via ``y_channel=True``.
  * **Border shave** — a configurable border (≈ the scale factor, 4–6 px) is
    cropped before computing the metric (``border``), read from
    ``EvalConfig.border_shave`` when assembled by the suite.

Tensor convention: ``pred`` / ``target`` are ``[B, 3, H, W]`` (or ``[B, 1, H, W]``)
torch tensors / numpy arrays in ``[0, 1]``. All metrics return Python ``float``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..interfaces import Metric
from . import _common as _c

try:  # numpy is a near-universal base dep but guarded so the module always imports.
    import numpy as np  # type: ignore

    _NUMPY = True
except Exception:  # pragma: no cover - numpy-less environment
    np = None  # type: ignore
    _NUMPY = False


# --------------------------------------------------------------------------- #
# Internal numerics (numpy only). Operate on [B, C, H, W] float64 arrays.
# --------------------------------------------------------------------------- #
def _prep_pair(
    pred: Any,
    target: Any,
    border: int = 0,
    y_channel: bool = False,
) -> "np.ndarray":
    """Convert/validate a (pred, target) pair to matched ``[B, C, H, W]`` arrays.

    Applies optional border-shave and Y-channel reduction (in that order), then
    returns a stacked array of shape ``[2, B, C, H, W]`` (index 0 = pred, 1 = tgt).
    """
    p = _c.as_bchw(pred)
    t = _c.as_bchw(target)
    _c.check_same_shape(p, t)
    if border > 0:
        p = _c.shave_border(p, border)
        t = _c.shave_border(t, border)
    if y_channel:
        p = _c.rgb_to_y(p)
        t = _c.rgb_to_y(t)
    return np.stack([p, t], axis=0)


def _psnr(pred: "np.ndarray", target: "np.ndarray", data_range: float = 1.0) -> float:
    """Peak signal-to-noise ratio in dB over a ``[B, C, H, W]`` pair."""
    mse = float(np.mean((pred - target) ** 2))
    if mse <= 1e-12:
        return float("inf")
    return float(10.0 * np.log10((data_range ** 2) / mse))


def _gaussian_window(size: int, sigma: float) -> "np.ndarray":
    """1-D Gaussian kernel of length ``size`` (normalized to sum 1)."""
    coords = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
    g = np.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    return g / g.sum()


def _largest_odd_leq(n: int) -> int:
    """Largest odd integer ``<= n`` (so a window never exceeds the image size)."""
    n = int(n)
    if n < 1:
        return 1
    return n if (n % 2 == 1) else n - 1


def _filter2d_separable(img: "np.ndarray", kernel1d: "np.ndarray") -> "np.ndarray":
    """Apply a separable Gaussian (valid convolution) to a ``[B, C, H, W]`` array.

    Implemented with stride-tricks sliding windows so no scipy/torch is needed.
    """
    k = kernel1d
    ksz = k.shape[0]
    # Horizontal pass (valid) over width.
    win_w = np.lib.stride_tricks.sliding_window_view(img, ksz, axis=3)
    tmp = np.tensordot(win_w, k, axes=([4], [0]))  # [B, C, H, W-ksz+1]
    # Vertical pass (valid) over height.
    win_h = np.lib.stride_tricks.sliding_window_view(tmp, ksz, axis=2)
    out = np.tensordot(win_h, k, axes=([4], [0]))  # [B, C, H-ksz+1, W-ksz+1]
    return out


def _ssim_map(
    pred: "np.ndarray",
    target: "np.ndarray",
    data_range: float = 1.0,
    win_size: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
) -> "np.ndarray":
    """Per-window SSIM map for a ``[B, C, H, W]`` pair (Wang et al. 2004).

    Uses a Gaussian window (the standard SSIM kernel). Returns the SSIM map of
    shape ``[B, C, H', W']`` (valid region); callers take its mean.
    """
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    win = _gaussian_window(win_size, sigma)

    mu1 = _filter2d_separable(pred, win)
    mu2 = _filter2d_separable(target, win)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = _filter2d_separable(pred * pred, win) - mu1_sq
    sigma2_sq = _filter2d_separable(target * target, win) - mu2_sq
    sigma12 = _filter2d_separable(pred * target, win) - mu1_mu2

    num = (2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)
    den = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    return num / den


def _ssim(
    pred: "np.ndarray",
    target: "np.ndarray",
    data_range: float = 1.0,
    win_size: int = 11,
    sigma: float = 1.5,
) -> float:
    """Mean Gaussian-window SSIM over a ``[B, C, H, W]`` pair."""
    if min(pred.shape[-2], pred.shape[-1]) < win_size:
        # Image smaller than the window: shrink the window to fit (odd, <= dim).
        win_size = max(3, _largest_odd_leq(min(pred.shape[-2], pred.shape[-1])))
        sigma = max(0.6, sigma * win_size / 11.0)
    smap = _ssim_map(pred, target, data_range, win_size, sigma)
    return float(np.mean(smap))


# MS-SSIM scale weights (Wang, Simoncelli & Bovik 2003 — the canonical 5 values).
_MSSSIM_WEIGHTS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)


def _downsample_2x(img: "np.ndarray") -> "np.ndarray":
    """2x average-pool downsample of a ``[B, C, H, W]`` array (drops odd tail)."""
    h, w = img.shape[-2], img.shape[-1]
    h2, w2 = h - (h % 2), w - (w % 2)
    a = img[..., :h2, :w2]
    a = a.reshape(a.shape[0], a.shape[1], h2 // 2, 2, w2 // 2, 2)
    return a.mean(axis=(3, 5))


def _ms_ssim(
    pred: "np.ndarray",
    target: "np.ndarray",
    data_range: float = 1.0,
    win_size: int = 11,
    sigma: float = 1.5,
) -> float:
    """Multi-scale SSIM over a ``[B, C, H, W]`` pair (Wang et al. 2003).

    Computes the contrast-structure term at each of 5 scales and the luminance
    term at the coarsest, combined with the canonical scale weights. Falls back
    gracefully to single-scale SSIM when the image is too small to pyramid.
    """
    weights = np.asarray(_MSSSIM_WEIGHTS, dtype=np.float64)
    levels = len(weights)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    p, t = pred, target
    mcs = []
    last_ssim = None
    for i in range(levels):
        ws = win_size
        min_dim = min(p.shape[-2], p.shape[-1])
        if min_dim < ws:
            ws = _largest_odd_leq(min_dim)
            if ws < 3:
                # Too small for even a 3-tap window at this scale -> stop pyramiding.
                break
        win = _gaussian_window(ws, sigma * ws / 11.0 if ws != win_size else sigma)
        mu1 = _filter2d_separable(p, win)
        mu2 = _filter2d_separable(t, win)
        mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
        sigma1_sq = _filter2d_separable(p * p, win) - mu1_sq
        sigma2_sq = _filter2d_separable(t * t, win) - mu2_sq
        sigma12 = _filter2d_separable(p * t, win) - mu1_mu2
        cs = (2.0 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)
        ssim_full = ((2.0 * mu1_mu2 + c1) / (mu1_sq + mu2_sq + c1)) * cs
        mcs.append(float(np.mean(np.maximum(cs, 0.0))))
        last_ssim = float(np.mean(np.maximum(ssim_full, 0.0)))
        if i < levels - 1:
            p = _downsample_2x(p)
            t = _downsample_2x(t)
            if min(p.shape[-2], p.shape[-1]) < 2:
                break

    n = len(mcs)
    if n == 0 or last_ssim is None:
        return _ssim(pred, target, data_range, win_size, sigma)
    if n < levels:
        # Renormalize the available weights when we ran out of scales.
        w = weights[:n]
        w = w / w.sum()
    else:
        w = weights
    # Product of contrast-structure terms over scales 1..n-1, times luminance at n.
    cs_prod = 1.0
    for i in range(n - 1):
        cs_prod *= max(mcs[i], 1e-8) ** w[i]
    val = cs_prod * (max(last_ssim, 1e-8) ** w[n - 1])
    return float(val)


# =========================================================================== #
# PSNR
# =========================================================================== #
class PSNRMetric(Metric):
    """Peak Signal-to-Noise Ratio (dB) — pixel reconstruction error. Higher better.

    Self-contained numpy implementation (no optional deps). Typical good range for
    SR/colorization is ≈ 20–35 dB. *Pitfall (doc 06):* PSNR is nearly insensitive
    to perceptually obvious changes — a blurry image can score high — so it must
    never be a *sole* criterion. Report alongside SSIM/LPIPS/FID.

    SR conventions (doc 06 §6):
      * ``y_channel=True`` computes PSNR on the BT.601 luminance channel
        (BasicSR/RCAN convention; comparable to SR literature numbers).
      * ``border`` shaves that many pixels off every spatial edge first
        (boundary-artifact removal; ≈ the SR scale factor).
    """

    name: str = "psnr"
    higher_is_better: bool = True
    family: str = "fidelity"

    def __init__(
        self,
        data_range: float = 1.0,
        y_channel: bool = False,
        border: int = 0,
    ) -> None:
        self.data_range = float(data_range)
        self.y_channel = bool(y_channel)
        self.border = int(border)
        if y_channel:
            self.name = "psnr_y"

    def __call__(
        self,
        pred: Any,
        target: Any = None,
        ctx: Optional[Dict[str, Any]] = None,
    ) -> float:
        if target is None:
            raise ValueError("PSNRMetric requires a target (full-reference).")
        stk = _prep_pair(pred, target, border=self.border, y_channel=self.y_channel)
        return _psnr(stk[0], stk[1], data_range=self.data_range)


# =========================================================================== #
# SSIM
# =========================================================================== #
class SSIMMetric(Metric):
    """Structural Similarity Index — luminance/contrast/structure. Higher better.

    Self-contained Gaussian-window SSIM (Wang et al. 2004), 11-tap window,
    ``sigma=1.5``, ``K1=0.01``, ``K2=0.03`` — the standard configuration. No
    optional dependency. More reliable than PSNR for structure; range ≈ [0, 1].

    Supports the SR ``y_channel`` and ``border`` conventions exactly like
    :class:`PSNRMetric`.
    """

    name: str = "ssim"
    higher_is_better: bool = True
    family: str = "fidelity"

    def __init__(
        self,
        data_range: float = 1.0,
        y_channel: bool = False,
        border: int = 0,
        win_size: int = 11,
        sigma: float = 1.5,
    ) -> None:
        self.data_range = float(data_range)
        self.y_channel = bool(y_channel)
        self.border = int(border)
        self.win_size = int(win_size)
        self.sigma = float(sigma)
        if y_channel:
            self.name = "ssim_y"

    def __call__(
        self,
        pred: Any,
        target: Any = None,
        ctx: Optional[Dict[str, Any]] = None,
    ) -> float:
        if target is None:
            raise ValueError("SSIMMetric requires a target (full-reference).")
        stk = _prep_pair(pred, target, border=self.border, y_channel=self.y_channel)
        return _ssim(stk[0], stk[1], self.data_range, self.win_size, self.sigma)


# =========================================================================== #
# MS-SSIM
# =========================================================================== #
class MSSSIMMetric(Metric):
    """Multi-Scale SSIM — SSIM over a Gaussian pyramid. Higher better, ≈ [0, 1].

    Self-contained implementation of Wang, Simoncelli & Bovik (2003) with the
    canonical 5-scale weights ``(0.0448, 0.2856, 0.3001, 0.2363, 0.1333)``. More
    robust to scale/viewing conditions than single-scale SSIM. Degrades
    gracefully to fewer scales (renormalized weights) on small tiles.
    """

    name: str = "ms_ssim"
    higher_is_better: bool = True
    family: str = "fidelity"

    def __init__(self, data_range: float = 1.0, border: int = 0) -> None:
        self.data_range = float(data_range)
        self.border = int(border)

    def __call__(
        self,
        pred: Any,
        target: Any = None,
        ctx: Optional[Dict[str, Any]] = None,
    ) -> float:
        if target is None:
            raise ValueError("MSSSIMMetric requires a target (full-reference).")
        stk = _prep_pair(pred, target, border=self.border, y_channel=False)
        return _ms_ssim(stk[0], stk[1], self.data_range)


# =========================================================================== #
# RMSE / MAE
# =========================================================================== #
class RMSEMetric(Metric):
    """Root Mean Squared Error (pixel). Lower better. Self-contained (numpy)."""

    name: str = "rmse"
    higher_is_better: bool = False
    family: str = "fidelity"

    def __init__(self, border: int = 0) -> None:
        self.border = int(border)

    def __call__(
        self,
        pred: Any,
        target: Any = None,
        ctx: Optional[Dict[str, Any]] = None,
    ) -> float:
        if target is None:
            raise ValueError("RMSEMetric requires a target (full-reference).")
        stk = _prep_pair(pred, target, border=self.border)
        return float(np.sqrt(np.mean((stk[0] - stk[1]) ** 2)))


class MAEMetric(Metric):
    """Mean Absolute Error / L1 (pixel). Lower better. Self-contained (numpy).

    Less outlier-sensitive than RMSE (doc 06 §A.5).
    """

    name: str = "mae"
    higher_is_better: bool = False
    family: str = "fidelity"

    def __init__(self, border: int = 0) -> None:
        self.border = int(border)

    def __call__(
        self,
        pred: Any,
        target: Any = None,
        ctx: Optional[Dict[str, Any]] = None,
    ) -> float:
        if target is None:
            raise ValueError("MAEMetric requires a target (full-reference).")
        stk = _prep_pair(pred, target, border=self.border)
        return float(np.mean(np.abs(stk[0] - stk[1])))


# =========================================================================== #
# SAM — Spectral Angle Mapper
# =========================================================================== #
class SAMMetric(Metric):
    """Spectral Angle Mapper — per-pixel angle between spectral vectors. Lower better.

    Self-contained (numpy). The angle (radians by default) between predicted and
    reference per-pixel spectral vectors across channels (the standard
    multispectral fidelity metric; doc 06 §A.8). 0 = identical spectra. Measures
    hue/spectral *direction* independent of brightness — directly relevant to
    "is the color right?". Designed for multi-band inputs but works for RGB.

    Args:
      degrees: return the mean angle in degrees instead of radians.
    """

    name: str = "sam"
    higher_is_better: bool = False
    family: str = "fidelity"

    def __init__(self, degrees: bool = False, border: int = 0) -> None:
        self.degrees = bool(degrees)
        self.border = int(border)

    def __call__(
        self,
        pred: Any,
        target: Any = None,
        ctx: Optional[Dict[str, Any]] = None,
    ) -> float:
        if target is None:
            raise ValueError("SAMMetric requires a target (full-reference).")
        stk = _prep_pair(pred, target, border=self.border)
        p, t = stk[0], stk[1]  # [B, C, H, W]
        if p.shape[1] < 2:
            raise ValueError("SAM needs >= 2 channels (a spectral vector per pixel).")
        # Dot product / norms along the channel axis.
        dot = np.sum(p * t, axis=1)
        np_ = np.sqrt(np.sum(p * p, axis=1))
        nt = np.sqrt(np.sum(t * t, axis=1))
        denom = np_ * nt
        valid = denom > 1e-12
        cos = np.ones_like(dot)
        cos[valid] = dot[valid] / denom[valid]
        cos = np.clip(cos, -1.0, 1.0)
        ang = np.arccos(cos)  # radians, per pixel
        mean_ang = float(np.mean(ang))
        if self.degrees:
            return float(np.degrees(mean_ang))
        return mean_ang


# =========================================================================== #
# ERGAS
# =========================================================================== #
class ERGASMetric(Metric):
    """ERGAS — global relative dimensionless synthesis error. Lower better.

    Self-contained (numpy). The canonical pansharpening / RS-fusion quality index
    (doc 06 §A.9): normalizes per-band RMSE by the band mean and the resolution
    ratio::

        ERGAS = 100 * (h/l) * sqrt( mean_b [ RMSE_b^2 / mean_b^2 ] )

    where ``h/l`` is the high/low spatial-resolution ratio (the SR scale's
    reciprocal). ``< 3`` is generally "good" in fusion literature. Lower better.

    Args:
      ratio: ``h/l`` (high-res GSD / low-res GSD). For an x4 SR product the
             classic convention is ``ratio = 1/scale = 0.25``; defaults to that
             via ``scale=4``. Pass ``ratio`` directly to override.
    """

    name: str = "ergas"
    higher_is_better: bool = False
    family: str = "fidelity"

    def __init__(
        self,
        scale: int = 4,
        ratio: Optional[float] = None,
        border: int = 0,
    ) -> None:
        self.border = int(border)
        if ratio is not None:
            self.ratio = float(ratio)
        else:
            self.ratio = 1.0 / float(scale) if scale else 1.0

    def __call__(
        self,
        pred: Any,
        target: Any = None,
        ctx: Optional[Dict[str, Any]] = None,
    ) -> float:
        if target is None:
            raise ValueError("ERGASMetric requires a target (full-reference).")
        stk = _prep_pair(pred, target, border=self.border)
        p, t = stk[0], stk[1]  # [B, C, H, W]
        # Per-band RMSE and reference mean (averaged over batch + spatial dims).
        diff_sq = (p - t) ** 2
        rmse_b = np.sqrt(np.mean(diff_sq, axis=(0, 2, 3)))  # [C]
        mu_b = np.mean(t, axis=(0, 2, 3))  # [C]
        valid = np.abs(mu_b) > 1e-12
        if not np.any(valid):
            return 0.0
        term = (rmse_b[valid] ** 2) / (mu_b[valid] ** 2)
        return float(100.0 * self.ratio * np.sqrt(np.mean(term)))


__all__ = [
    "PSNRMetric",
    "SSIMMetric",
    "MSSSIMMetric",
    "RMSEMetric",
    "MAEMetric",
    "SAMMetric",
    "ERGASMetric",
]
