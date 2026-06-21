"""irchroma.models.sr.degradation — blind real-world degradation synthesizer.

Owner: Builder-2 (Stage-1 super-resolution / restoration).

This module implements :class:`BSRGANDegradation`, a **blind, real-world**
degradation pipeline used to synthesize realistic low-resolution IR (``lr``) from
high-resolution IR (``hr``) so the SR network learns to invert *plausible sensor
degradations* rather than only clean bicubic downsampling. It follows the
Real-ESRGAN (Wang et al., 2021) / BSRGAN (Zhang et al., 2021) recipe and the
remote-sensing guidance in ``docs/research/03-super-resolution.md`` §5/§12:

  * random **isotropic / anisotropic Gaussian blur** (sensor optics / motion),
  * random **downsampling** (bicubic / bilinear / area) to bridge native -> target
    GSD,
  * **Gaussian + Poisson** photon/read noise,
  * optional **JPEG** compression (guarded OpenCV; skipped if unavailable),
  * optional **second-order** repetition of the blur->resize->noise chain
    (Real-ESRGAN "high-order" degradation) for broader coverage,
  * an optional **sensor MTF/PSF Gaussian kernel** sized to mimic the Landsat TIRS
    thermal point-spread (the dominant real degradation for coarse thermal), and
  * optional **pushbroom striping** characteristic of line-scanned IR sensors.

The single biggest SR-evaluation pitfall is training/testing on bicubic pairs only
— it overstates performance (``docs/research/06`` §6). Synthesizing realistic LR
here is what makes the trained model robust on *real* native-resolution LR-HR pairs.

Tensors & fallbacks
-------------------
Operates on **torch tensors** ``[B, C, H, W]`` (or ``[C, H, W]``) of floats in
``~[0, 1]`` when torch is available, with a **NumPy fallback** so the synthesizer
remains usable in torch-less environments (e.g. a data-prep box). All randomness
draws from a seedable :class:`random.Random` / ``numpy`` generator for reproducible
training pairs.

API
---
  * ``__call__(hr) -> lr`` — degrade ``hr`` to ``lr`` at the configured ``scale``.
  * ``paired_degrade(hr, scale) -> (hr, lr)`` — convenience that returns the *aligned*
    ``(hr, lr)`` training pair at an explicit ``scale``.
"""

from __future__ import annotations

import math
import random
from typing import List, Optional, Sequence, Tuple

from ...config import SRConfig
from ...interfaces import TORCH_AVAILABLE, Tensor

# --------------------------------------------------------------------------- #
# Optional backends (all guarded so importing this module never fails).
# --------------------------------------------------------------------------- #
if TORCH_AVAILABLE:  # pragma: no cover - exercised only with torch installed
    import torch
    import torch.nn.functional as F
else:  # pragma: no cover - torch-less import shim
    torch = None  # type: ignore
    F = None  # type: ignore

try:  # NumPy is the fallback compute backend + kernel math helper.
    import numpy as np  # type: ignore

    _HAS_NUMPY = True
except Exception:  # pragma: no cover - numpy-less environment
    np = None  # type: ignore
    _HAS_NUMPY = False

try:  # OpenCV powers the optional JPEG step only; everything else avoids it.
    import cv2  # type: ignore

    _HAS_CV2 = True
except Exception:  # pragma: no cover - opencv not installed
    cv2 = None  # type: ignore
    _HAS_CV2 = False


# =========================================================================== #
# Gaussian-kernel helpers (NumPy; used to build torch conv weights too)
# =========================================================================== #
def _gaussian_kernel2d(
    ksize: int,
    sigma_x: float,
    sigma_y: Optional[float] = None,
    theta: float = 0.0,
) -> "np.ndarray":
    """Build a (possibly anisotropic, rotated) 2-D Gaussian kernel, sum-normalized.

    Args:
        ksize:   odd kernel size (rounded up to odd if even).
        sigma_x: std-dev along the (rotated) x axis, in pixels.
        sigma_y: std-dev along the (rotated) y axis; defaults to ``sigma_x`` (isotropic).
        theta:   rotation angle in radians (anisotropic kernels only).

    Returns:
        ``np.ndarray [ksize, ksize]`` float kernel summing to 1.

    Requires NumPy (raises if unavailable). This is the sensor-PSF / blur primitive.
    """
    if not _HAS_NUMPY:  # pragma: no cover - guarded by callers
        raise RuntimeError("Gaussian kernel construction requires numpy.")
    if ksize % 2 == 0:
        ksize += 1
    sigma_y = sigma_x if sigma_y is None else sigma_y
    half = ksize // 2
    ax = np.arange(-half, half + 1, dtype=np.float64)
    xx, yy = np.meshgrid(ax, ax)
    # Rotate coordinates by theta for anisotropy.
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    x_rot = cos_t * xx + sin_t * yy
    y_rot = -sin_t * xx + cos_t * yy
    sx = max(float(sigma_x), 1e-6)
    sy = max(float(sigma_y), 1e-6)
    kernel = np.exp(-0.5 * ((x_rot / sx) ** 2 + (y_rot / sy) ** 2))
    total = kernel.sum()
    if total <= 0:  # pragma: no cover - numerically degenerate
        return np.ones((ksize, ksize), dtype=np.float64) / float(ksize * ksize)
    return kernel / total


# =========================================================================== #
# BSRGANDegradation
# =========================================================================== #
class BSRGANDegradation:
    """Blind real-world degradation synthesizer (Real-ESRGAN / BSRGAN style).

    Produces a realistically-degraded low-resolution IR from a high-resolution IR,
    to train the guided-SR network for blind robustness. Not an ``nn.Module`` (it
    is a stochastic data transform), so it has no learnable parameters and is used
    inside the dataset / training loop.

    Args:
        scale:          target downscale factor (``SRConfig.scale``). ``lr`` is
                        ``scale`` x smaller than ``hr`` per side.
        blur_sigma:     ``(min, max)`` Gaussian-blur sigma range (pixels).
        kernel_size:    odd blur-kernel size.
        aniso_prob:     probability the blur kernel is anisotropic (rotated).
        downsample_modes: resize interpolations sampled uniformly
                        (``{"bicubic", "bilinear", "area", "nearest"}``).
        noise_sigma:    ``(min, max)`` additive-Gaussian-noise sigma range (in the
                        [0,1] intensity scale).
        gaussian_noise_prob / poisson_noise_prob:
                        per-call probabilities of applying each noise type.
        jpeg_prob:      probability of an OpenCV JPEG round (no-op if OpenCV missing).
        jpeg_quality:   ``(min, max)`` JPEG quality range.
        second_order_prob:
                        probability of a second blur->resize->noise round
                        (Real-ESRGAN high-order degradation).
        use_sensor_mtf: prepend a fixed sensor-PSF Gaussian blur each call
                        (``SRConfig.use_sensor_mtf``; see :data:`sensor_mtf_sigma`).
        sensor_mtf_sigma:
                        sigma (pixels) of the sensor-PSF blur approximating the
                        Landsat TIRS thermal MTF (native ~100 m sampled to a 30 m
                        grid => a broad PSF; default tuned for that ratio).
        add_striping:   add multiplicative column striping (pushbroom artifact;
                        ``SRConfig.add_striping``).
        striping_strength:
                        amplitude of the per-column gain ripple (fraction of signal).
        seed:           optional RNG seed for reproducible training pairs.

    Call:
        ``deg(hr) -> lr`` where ``hr`` is ``[B, C, H, W]`` (or ``[C, H, W]``) float in
        ``~[0, 1]``; ``lr`` is the degraded ``[B, C, H/scale, W/scale]`` tensor of the
        same dtype/device (torch) or array (numpy fallback).
    """

    def __init__(
        self,
        scale: int = 4,
        blur_sigma: Tuple[float, float] = (0.2, 2.0),
        kernel_size: int = 15,
        aniso_prob: float = 0.5,
        downsample_modes: Optional[Sequence[str]] = None,
        noise_sigma: Tuple[float, float] = (0.0, 0.06),
        gaussian_noise_prob: float = 0.7,
        poisson_noise_prob: float = 0.5,
        jpeg_prob: float = 0.3,
        jpeg_quality: Tuple[int, int] = (50, 95),
        second_order_prob: float = 0.4,
        use_sensor_mtf: bool = True,
        sensor_mtf_sigma: float = 1.5,
        add_striping: bool = True,
        striping_strength: float = 0.03,
        seed: Optional[int] = None,
    ) -> None:
        self.scale = int(scale)
        self.blur_sigma = (float(blur_sigma[0]), float(blur_sigma[1]))
        self.kernel_size = int(kernel_size) | 1  # force odd
        self.aniso_prob = float(aniso_prob)
        self.downsample_modes: List[str] = list(
            downsample_modes if downsample_modes is not None
            else ["bicubic", "bilinear", "area"]
        )
        self.noise_sigma = (float(noise_sigma[0]), float(noise_sigma[1]))
        self.gaussian_noise_prob = float(gaussian_noise_prob)
        self.poisson_noise_prob = float(poisson_noise_prob)
        self.jpeg_prob = float(jpeg_prob)
        self.jpeg_quality = (int(jpeg_quality[0]), int(jpeg_quality[1]))
        self.second_order_prob = float(second_order_prob)
        self.use_sensor_mtf = bool(use_sensor_mtf)
        self.sensor_mtf_sigma = float(sensor_mtf_sigma)
        self.add_striping = bool(add_striping)
        self.striping_strength = float(striping_strength)
        self._rng = random.Random(seed)
        # Independent numpy generator (only used on the numpy fallback path).
        self._np_rng = np.random.default_rng(seed) if _HAS_NUMPY else None

    # ------------------------------------------------------------------ #
    @classmethod
    def from_config(cls, cfg: SRConfig, seed: Optional[int] = None) -> "BSRGANDegradation":
        """Build a :class:`BSRGANDegradation` from an :class:`~irchroma.config.SRConfig`.

        Honors ``scale``, ``use_sensor_mtf`` and ``add_striping`` from the config;
        the ``degradation`` field (``"realesrgan"`` / ``"bsrgan"``) only nudges the
        second-order probability (BSRGAN shuffles a single order; Real-ESRGAN repeats),
        since both share this primitive set.
        """
        second_order = 0.5 if str(cfg.degradation) == "realesrgan" else 0.3
        return cls(
            scale=int(cfg.scale),
            second_order_prob=second_order,
            use_sensor_mtf=bool(cfg.use_sensor_mtf),
            add_striping=bool(cfg.add_striping),
            seed=seed,
        )

    # ================================================================== #
    # Public API
    # ================================================================== #
    def __call__(self, hr: Tensor) -> Tensor:
        """Degrade ``hr`` to a low-resolution ``lr`` at ``self.scale``.

        Dispatches to the torch path when torch is available and ``hr`` is a tensor,
        else to the NumPy fallback. See the class docstring for the tensor contract.
        """
        if TORCH_AVAILABLE and _is_torch_tensor(hr):
            return self._degrade_torch(hr)
        if _HAS_NUMPY:
            return self._degrade_numpy(hr)  # type: ignore[arg-type]
        raise RuntimeError(  # pragma: no cover - no compute backend at all
            "BSRGANDegradation requires either torch or numpy to run."
        )

    def paired_degrade(self, hr: Tensor, scale: Optional[int] = None) -> Tuple[Tensor, Tensor]:
        """Return an aligned ``(hr, lr)`` training pair.

        Args:
            hr:    high-resolution IR ``[B, C, H, W]`` (or ``[C, H, W]``), float ~[0,1].
            scale: optional override of ``self.scale`` for this call (restored after).

        Returns:
            ``(hr, lr)`` — ``hr`` is returned **unchanged** (the SR target) and ``lr``
            is its degraded ``scale`` x-smaller counterpart (the SR input). ``hr`` is
            center-cropped to a multiple of ``scale`` first so ``lr`` * ``scale``
            re-aligns exactly to it (no fractional-pixel offset).
        """
        use_scale = self.scale if scale is None else int(scale)
        hr = self._crop_to_multiple(hr, use_scale)
        if scale is None:
            lr = self(hr)
        else:
            prev = self.scale
            self.scale = use_scale
            try:
                lr = self(hr)
            finally:
                self.scale = prev
        return hr, lr

    # ================================================================== #
    # Torch implementation
    # ================================================================== #
    def _degrade_torch(self, hr: Tensor) -> Tensor:  # pragma: no cover - needs torch
        squeeze = False
        if hr.dim() == 3:  # [C, H, W] -> [1, C, H, W]
            hr = hr.unsqueeze(0)
            squeeze = True
        x = hr.float()
        _, _, h, w = x.shape
        target_h = max(1, h // self.scale)
        target_w = max(1, w // self.scale)

        # 0) Sensor MTF/PSF blur (Landsat TIRS-like broad PSF), applied first.
        if self.use_sensor_mtf and self.sensor_mtf_sigma > 0:
            x = self._blur_torch(x, self.sensor_mtf_sigma, self.sensor_mtf_sigma, 0.0)

        # 1) First-order blur -> resize(partial) -> noise.
        x = self._rand_blur_torch(x)
        # Resize toward (but not all the way to) the target on the first round if a
        # second order will finish the job; otherwise go straight to target.
        if self._rng.random() < self.second_order_prob:
            mid_h = max(target_h, h // max(1, self.scale // 2 or 1))
            mid_w = max(target_w, w // max(1, self.scale // 2 or 1))
            x = self._resize_torch(x, mid_h, mid_w)
            x = self._rand_noise_torch(x)
            # 2) Second-order blur -> resize(to target) -> noise.
            x = self._rand_blur_torch(x)
            x = self._resize_torch(x, target_h, target_w)
            x = self._rand_noise_torch(x)
        else:
            x = self._resize_torch(x, target_h, target_w)
            x = self._rand_noise_torch(x)

        # 3) Optional pushbroom striping (column-wise multiplicative ripple).
        if self.add_striping and self.striping_strength > 0:
            x = self._stripe_torch(x)

        # 4) Optional JPEG (OpenCV; numpy round-trip). Skipped if cv2 missing.
        if _HAS_CV2 and self._rng.random() < self.jpeg_prob:
            x = self._jpeg_torch(x)

        x = x.clamp(0.0, 1.0)
        return x.squeeze(0) if squeeze else x

    def _blur_torch(self, x: Tensor, sx: float, sy: float, theta: float) -> Tensor:  # pragma: no cover
        """Depthwise-convolve every channel with one Gaussian kernel."""
        k = _gaussian_kernel2d(self.kernel_size, sx, sy, theta)
        c = x.shape[1]
        weight = torch.as_tensor(k, dtype=x.dtype, device=x.device)
        weight = weight.view(1, 1, *k.shape).repeat(c, 1, 1, 1)
        pad = self.kernel_size // 2
        x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        return F.conv2d(x, weight, groups=c)

    def _rand_blur_torch(self, x: Tensor) -> Tensor:  # pragma: no cover - needs torch
        sx = self._rng.uniform(*self.blur_sigma)
        if self._rng.random() < self.aniso_prob:
            sy = self._rng.uniform(*self.blur_sigma)
            theta = self._rng.uniform(0.0, math.pi)
        else:
            sy, theta = sx, 0.0
        return self._blur_torch(x, sx, sy, theta)

    def _resize_torch(self, x: Tensor, out_h: int, out_w: int) -> Tensor:  # pragma: no cover
        mode = self._rng.choice(self.downsample_modes)
        if mode == "area":
            return F.interpolate(x, size=(out_h, out_w), mode="area")
        if mode == "nearest":
            return F.interpolate(x, size=(out_h, out_w), mode="nearest")
        align = False
        return F.interpolate(x, size=(out_h, out_w), mode=mode, align_corners=align)

    def _rand_noise_torch(self, x: Tensor) -> Tensor:  # pragma: no cover - needs torch
        # Poisson (shot) noise: scale to a photon count, sample, scale back.
        if self._rng.random() < self.poisson_noise_prob:
            # Higher ``peak`` => less noise; sample a plausible range.
            peak = self._rng.uniform(20.0, 200.0)
            scaled = torch.clamp(x, 0.0, 1.0) * peak
            noisy = torch.poisson(scaled) / peak
            x = noisy
        # Additive Gaussian (read) noise.
        if self._rng.random() < self.gaussian_noise_prob:
            sigma = self._rng.uniform(*self.noise_sigma)
            if sigma > 0:
                x = x + torch.randn_like(x) * sigma
        return torch.clamp(x, 0.0, 1.0)

    def _stripe_torch(self, x: Tensor) -> Tensor:  # pragma: no cover - needs torch
        b, c, h, w = x.shape
        # Per-column gain ~ 1 + strength * sinusoid with random phase/frequency.
        phase = self._rng.uniform(0.0, 2.0 * math.pi)
        freq = self._rng.uniform(0.5, 4.0)
        cols = torch.linspace(0.0, 1.0, w, dtype=x.dtype, device=x.device)
        ripple = 1.0 + self.striping_strength * torch.sin(2.0 * math.pi * freq * cols + phase)
        return x * ripple.view(1, 1, 1, w)

    def _jpeg_torch(self, x: Tensor) -> Tensor:  # pragma: no cover - needs torch + cv2
        if not _HAS_NUMPY:
            return x
        quality = self._rng.randint(*self.jpeg_quality)
        arr = x.detach().cpu().numpy()
        out = self._jpeg_numpy_array(arr, quality)
        return torch.as_tensor(out, dtype=x.dtype, device=x.device)

    # ================================================================== #
    # NumPy fallback implementation
    # ================================================================== #
    def _degrade_numpy(self, hr: "np.ndarray") -> "np.ndarray":  # pragma: no cover
        arr = np.asarray(hr, dtype=np.float64)
        squeeze = False
        if arr.ndim == 3:  # [C, H, W] -> [1, C, H, W]
            arr = arr[None, ...]
            squeeze = True
        if arr.ndim != 4:
            raise ValueError(f"Expected [B,C,H,W] or [C,H,W] array; got shape {arr.shape}.")
        _, _, h, w = arr.shape
        target_h = max(1, h // self.scale)
        target_w = max(1, w // self.scale)

        if self.use_sensor_mtf and self.sensor_mtf_sigma > 0:
            arr = self._blur_numpy(arr, self.sensor_mtf_sigma, self.sensor_mtf_sigma, 0.0)
        arr = self._rand_blur_numpy(arr)
        arr = self._resize_numpy(arr, target_h, target_w)
        arr = self._rand_noise_numpy(arr)
        if self.second_order_prob > 0 and self._rng.random() < self.second_order_prob:
            arr = self._rand_blur_numpy(arr)
            arr = self._rand_noise_numpy(arr)
        if self.add_striping and self.striping_strength > 0:
            arr = self._stripe_numpy(arr)
        if _HAS_CV2 and self._rng.random() < self.jpeg_prob:
            arr = self._jpeg_numpy_array(arr, self._rng.randint(*self.jpeg_quality))

        arr = np.clip(arr, 0.0, 1.0)
        return arr[0] if squeeze else arr

    def _blur_numpy(self, arr: "np.ndarray", sx: float, sy: float, theta: float) -> "np.ndarray":  # pragma: no cover
        k = _gaussian_kernel2d(self.kernel_size, sx, sy, theta)
        pad = self.kernel_size // 2
        out = np.empty_like(arr)
        for b in range(arr.shape[0]):
            for c in range(arr.shape[1]):
                padded = np.pad(arr[b, c], pad, mode="reflect")
                out[b, c] = _convolve2d_valid(padded, k)
        return out

    def _rand_blur_numpy(self, arr: "np.ndarray") -> "np.ndarray":  # pragma: no cover
        sx = self._rng.uniform(*self.blur_sigma)
        if self._rng.random() < self.aniso_prob:
            sy = self._rng.uniform(*self.blur_sigma)
            theta = self._rng.uniform(0.0, math.pi)
        else:
            sy, theta = sx, 0.0
        return self._blur_numpy(arr, sx, sy, theta)

    def _resize_numpy(self, arr: "np.ndarray", out_h: int, out_w: int) -> "np.ndarray":  # pragma: no cover
        # Use OpenCV resize if present (matches torch modes well); else area-average.
        if _HAS_CV2:
            mode = self._rng.choice(self.downsample_modes)
            interp = {
                "bicubic": cv2.INTER_CUBIC,
                "bilinear": cv2.INTER_LINEAR,
                "area": cv2.INTER_AREA,
                "nearest": cv2.INTER_NEAREST,
            }.get(mode, cv2.INTER_AREA)
            out = np.empty((arr.shape[0], arr.shape[1], out_h, out_w), dtype=arr.dtype)
            for b in range(arr.shape[0]):
                for c in range(arr.shape[1]):
                    out[b, c] = cv2.resize(arr[b, c], (out_w, out_h), interpolation=interp)
            return out
        return _area_resize_numpy(arr, out_h, out_w)

    def _rand_noise_numpy(self, arr: "np.ndarray") -> "np.ndarray":  # pragma: no cover
        if self._np_rng is None:
            return arr
        if self._rng.random() < self.poisson_noise_prob:
            peak = self._rng.uniform(20.0, 200.0)
            arr = self._np_rng.poisson(np.clip(arr, 0.0, 1.0) * peak) / peak
        if self._rng.random() < self.gaussian_noise_prob:
            sigma = self._rng.uniform(*self.noise_sigma)
            if sigma > 0:
                arr = arr + self._np_rng.normal(0.0, sigma, size=arr.shape)
        return np.clip(arr, 0.0, 1.0)

    def _stripe_numpy(self, arr: "np.ndarray") -> "np.ndarray":  # pragma: no cover
        w = arr.shape[-1]
        phase = self._rng.uniform(0.0, 2.0 * math.pi)
        freq = self._rng.uniform(0.5, 4.0)
        cols = np.linspace(0.0, 1.0, w)
        ripple = 1.0 + self.striping_strength * np.sin(2.0 * math.pi * freq * cols + phase)
        return arr * ripple.reshape(1, 1, 1, w)

    def _jpeg_numpy_array(self, arr: "np.ndarray", quality: int) -> "np.ndarray":  # pragma: no cover
        """JPEG-compress/decompress each channel via OpenCV (guarded)."""
        if not _HAS_CV2:
            return arr
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
        out = np.empty_like(arr)
        for b in range(arr.shape[0]):
            for c in range(arr.shape[1]):
                u8 = np.clip(arr[b, c] * 255.0, 0, 255).astype(np.uint8)
                ok, enc = cv2.imencode(".jpg", u8, encode_param)
                if not ok:
                    out[b, c] = arr[b, c]
                    continue
                dec = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
                out[b, c] = dec.astype(arr.dtype) / 255.0
        return out

    # ------------------------------------------------------------------ #
    @staticmethod
    def _crop_to_multiple(hr: Tensor, scale: int) -> Tensor:
        """Center-crop ``hr`` so H and W are exact multiples of ``scale``."""
        if scale <= 1:
            return hr
        if TORCH_AVAILABLE and _is_torch_tensor(hr):
            h, w = hr.shape[-2], hr.shape[-1]
            new_h = (h // scale) * scale
            new_w = (w // scale) * scale
            top = (h - new_h) // 2
            left = (w - new_w) // 2
            return hr[..., top:top + new_h, left:left + new_w]
        # numpy
        arr = np.asarray(hr)
        h, w = arr.shape[-2], arr.shape[-1]
        new_h = (h // scale) * scale
        new_w = (w // scale) * scale
        top = (h - new_h) // 2
        left = (w - new_w) // 2
        return arr[..., top:top + new_h, left:left + new_w]


# =========================================================================== #
# Small NumPy helpers (only used on the numpy fallback path)
# =========================================================================== #
def _is_torch_tensor(x: object) -> bool:
    """True if ``x`` is a ``torch.Tensor`` (and torch is available)."""
    return TORCH_AVAILABLE and isinstance(x, torch.Tensor)  # type: ignore[arg-type]


def _convolve2d_valid(padded: "np.ndarray", kernel: "np.ndarray") -> "np.ndarray":  # pragma: no cover
    """Naive 'valid' 2-D correlation of an already-padded image with ``kernel``.

    Implemented with a small accumulation loop over kernel taps (vectorized over the
    image), avoiding a SciPy dependency. Output size == padded size - kernel + 1.
    """
    kh, kw = kernel.shape
    out_h = padded.shape[0] - kh + 1
    out_w = padded.shape[1] - kw + 1
    out = np.zeros((out_h, out_w), dtype=np.float64)
    for i in range(kh):
        for j in range(kw):
            out += kernel[i, j] * padded[i:i + out_h, j:j + out_w]
    return out


def _area_resize_numpy(arr: "np.ndarray", out_h: int, out_w: int) -> "np.ndarray":  # pragma: no cover
    """Box / area-average downsample fallback when OpenCV is unavailable.

    Assumes a downscale (``out_h <= H`` and ``out_w <= W``); for the rare upscale
    case it falls back to nearest-neighbour indexing. Operates per [B, C].
    """
    b, c, h, w = arr.shape
    out = np.empty((b, c, out_h, out_w), dtype=arr.dtype)
    ys = (np.linspace(0, h, out_h + 1)).astype(int)
    xs = (np.linspace(0, w, out_w + 1)).astype(int)
    for bb in range(b):
        for cc in range(c):
            for yi in range(out_h):
                y0, y1 = ys[yi], max(ys[yi] + 1, ys[yi + 1])
                for xi in range(out_w):
                    x0, x1 = xs[xi], max(xs[xi] + 1, xs[xi + 1])
                    out[bb, cc, yi, xi] = arr[bb, cc, y0:y1, x0:x1].mean()
    return out


__all__ = ["BSRGANDegradation"]
