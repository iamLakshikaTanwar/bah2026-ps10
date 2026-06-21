"""irchroma.metrics._common — shared tensor/array plumbing for the metric suite.

Private helper module (Builder-6 / MetricsBuilder). NOT part of the public API.

The evaluation doc (docs/research/06 §A "Library note") warns that the three
metric ecosystems disagree on tensor layout:

  * ``torchmetrics`` expects ``N x C x H x W`` torch tensors,
  * ``sewar`` expects ``H x W x C`` numpy arrays,
  * ``piq`` expects ``N x C x H x W`` in ``[0, 1]``.

Keeping a *single* conversion utility here avoids the silent layout / range bugs
that plague multi-library evaluation code. Every metric in this package routes
its inputs through these helpers.

Tensor convention (the contract, from :mod:`irchroma.interfaces`):
  ``pred`` / ``target`` are RGB ``FloatTensor [B, 3, H, W]`` in ``[0, 1]`` (or a
  single channel ``[B, 1, H, W]`` for SR-domain metrics). We accept torch tensors
  and convert to ``float64`` numpy of shape ``[B, C, H, W]`` internally, where the
  numerics are easiest to get right; library adapters reshape from there.

This module imports with or without numpy / torch installed: numpy is treated as
an optional dependency with a friendly error only raised when a metric actually
needs it (the self-contained PSNR/SSIM/CIEDE2000 paths require numpy, which is a
near-universal base dep, so in practice they always work).
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

from ..interfaces import TORCH_AVAILABLE

# --------------------------------------------------------------------------- #
# Optional numpy (a near-universal base dependency, but guard anyway so the
# module always imports — only metrics that touch numpy fail if it is missing).
# --------------------------------------------------------------------------- #
try:
    import numpy as np  # type: ignore

    NUMPY_AVAILABLE: bool = True
except Exception:  # pragma: no cover - numpy-less environment
    np = None  # type: ignore
    NUMPY_AVAILABLE = False

if TORCH_AVAILABLE:  # pragma: no cover - exercised only with torch installed
    import torch  # type: ignore
else:  # pragma: no cover - torch-less import shim
    torch = None  # type: ignore


def _require_numpy() -> None:
    """Raise a friendly error if numpy is unavailable."""
    if not NUMPY_AVAILABLE:  # pragma: no cover - numpy is virtually always present
        raise RuntimeError(
            "This metric requires numpy. Install it with `pip install numpy`."
        )


def is_torch_tensor(x: Any) -> bool:
    """Return ``True`` if ``x`` is a ``torch.Tensor`` (and torch is importable)."""
    return bool(TORCH_AVAILABLE and torch is not None and isinstance(x, torch.Tensor))


def to_numpy(x: Any) -> "np.ndarray":
    """Convert a torch tensor / array-like to a contiguous numpy ``float64`` array.

    Args:
      x: a ``torch.Tensor``, numpy array, or anything ``np.asarray`` accepts.

    Returns:
      A ``numpy.ndarray`` (``float64``) — detached and moved to CPU if it was a
      CUDA/autograd tensor.
    """
    _require_numpy()
    if is_torch_tensor(x):
        return x.detach().to("cpu").to(torch.float64).numpy()  # type: ignore[union-attr]
    return np.asarray(x, dtype=np.float64)


def as_bchw(x: Any) -> "np.ndarray":
    """Normalize an image tensor/array to a 4-D ``[B, C, H, W]`` numpy array.

    Accepts (and promotes) the following layouts:
      * ``[H, W]``        -> ``[1, 1, H, W]``      (single grayscale image)
      * ``[C, H, W]``     -> ``[1, C, H, W]``      (single CHW image)
      * ``[B, C, H, W]``  -> unchanged             (a batch, the canonical form)

    Note:
      Ambiguity between ``[B, H, W]`` and ``[C, H, W]`` is resolved in favour of
      ``[C, H, W]`` (single image) because the contract passes channels-first
      single-channel SR tensors as ``[B, 1, H, W]`` already; bare 3-D inputs are
      assumed CHW. Pass an explicit 4-D tensor to remove all ambiguity.
    """
    arr = to_numpy(x)
    if arr.ndim == 2:  # H, W
        return arr[None, None, :, :]
    if arr.ndim == 3:  # C, H, W  (single image)
        return arr[None, :, :, :]
    if arr.ndim == 4:  # B, C, H, W
        return arr
    raise ValueError(
        f"Expected an image of rank 2/3/4 ([H,W] / [C,H,W] / [B,C,H,W]); got shape {arr.shape}."
    )


def as_bhwc(x: Any) -> "np.ndarray":
    """Normalize an image to ``[B, H, W, C]`` numpy (the ``sewar`` / OpenCV layout)."""
    bchw = as_bchw(x)
    return np.transpose(bchw, (0, 2, 3, 1))


def check_same_shape(pred: "np.ndarray", target: "np.ndarray") -> None:
    """Raise ``ValueError`` if two arrays differ in shape."""
    if pred.shape != target.shape:
        raise ValueError(
            f"pred and target must share a shape; got {pred.shape} vs {target.shape}."
        )


def shave_border(arr: "np.ndarray", border: int) -> "np.ndarray":
    """Crop ``border`` pixels off every spatial edge of a ``[B, C, H, W]`` array.

    Implements the SR "border shave" convention (docs/research/06 §6): boundary
    artifacts inflate/deflate PSNR/SSIM, so a border (≈ the scale factor, 4–6 px)
    is removed before metrics. A non-positive ``border`` is a no-op.
    """
    if border <= 0:
        return arr
    h, w = arr.shape[-2], arr.shape[-1]
    if 2 * border >= h or 2 * border >= w:
        # Shaving would erase the image; skip rather than crash.
        return arr
    return arr[..., border:h - border, border:w - border]


def rgb_to_y(bchw: "np.ndarray") -> "np.ndarray":
    """Compute the BT.601 luminance (Y) channel from RGB ``[B, 3, H, W]`` in [0,1].

    Returns ``[B, 1, H, W]`` in the BasicSR/MATLAB Y convention (``16..235`` range
    folded back to ``[0, 1]`` by the 16/255 offset + 219 scale), matching
    ``skimage.color.rgb2ycbcr(...)[..., 0]`` used in the SR literature
    (docs/research/06 §6). Single-channel inputs are returned unchanged (already
    luminance-like).
    """
    if bchw.shape[1] == 1:
        return bchw
    r = bchw[:, 0:1, :, :]
    g = bchw[:, 1:2, :, :]
    b = bchw[:, 2:3, :, :]
    # MATLAB/BasicSR rgb2ycbcr (input [0,1]): Y in [16/255, 235/255].
    y = 16.0 / 255.0 + (65.481 * r + 128.553 * g + 24.966 * b) / 255.0
    return y


def rgb_to_ycbcr(bchw: "np.ndarray") -> "np.ndarray":
    """Convert RGB ``[B, 3, H, W]`` in [0,1] to YCbCr ``[B, 3, H, W]`` in [0,1].

    BT.601 full-pipeline (MATLAB ``rgb2ycbcr`` for ``[0,1]`` inputs). Channel order
    of the output is ``[Y, Cb, Cr]``. Used by :class:`ChromaPSNRMetric`.
    """
    if bchw.shape[1] == 1:
        raise ValueError("rgb_to_ycbcr requires a 3-channel RGB image.")
    r = bchw[:, 0:1, :, :]
    g = bchw[:, 1:2, :, :]
    b = bchw[:, 2:3, :, :]
    y = 16.0 / 255.0 + (65.481 * r + 128.553 * g + 24.966 * b) / 255.0
    cb = 128.0 / 255.0 + (-37.797 * r - 74.203 * g + 112.0 * b) / 255.0
    cr = 128.0 / 255.0 + (112.0 * r - 93.786 * g - 18.214 * b) / 255.0
    return np.concatenate([y, cb, cr], axis=1)


def _srgb_to_linear(c: "np.ndarray") -> "np.ndarray":
    """Inverse sRGB companding (gamma expansion); ``c`` in [0,1]."""
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


# D65 reference white (CIE 1931 2 deg observer), used by the Lab conversion.
_D65_XN = 0.95047
_D65_YN = 1.00000
_D65_ZN = 1.08883


def rgb_to_lab(bchw: "np.ndarray") -> "np.ndarray":
    """Convert sRGB ``[B, 3, H, W]`` in [0,1] to CIE L*a*b* ``[B, 3, H, W]``.

    Self-contained D65 sRGB -> XYZ -> Lab pipeline (no scikit-image dependency),
    matching ``skimage.color.rgb2lab`` to numerical tolerance. L in ``[0, 100]``,
    a/b in roughly ``[-128, 127]``. Used by :class:`CIEDE2000Metric` and the
    a*/b* chroma-PSNR path so color metrics NEVER need an optional dep.
    """
    rgb = np.clip(bchw, 0.0, 1.0)
    r = _srgb_to_linear(rgb[:, 0:1, :, :])
    g = _srgb_to_linear(rgb[:, 1:2, :, :])
    b = _srgb_to_linear(rgb[:, 2:3, :, :])
    # Linear sRGB (D65) -> XYZ.
    x = (0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / _D65_XN
    y = (0.2126729 * r + 0.7151522 * g + 0.0721750 * b) / _D65_YN
    z = (0.0193339 * r + 0.1191920 * g + 0.9503041 * b) / _D65_ZN

    eps = 216.0 / 24389.0  # (6/29)^3
    kappa = 24389.0 / 27.0  # (29/3)^3

    def _f(t: "np.ndarray") -> "np.ndarray":
        return np.where(t > eps, np.cbrt(t), (kappa * t + 16.0) / 116.0)

    fx, fy, fz = _f(x), _f(y), _f(z)
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    bb = 200.0 * (fy - fz)
    return np.concatenate([L, a, bb], axis=1)


def sobel_gradient_magnitude(img_hw: "np.ndarray") -> "np.ndarray":
    """Sobel gradient magnitude of a single 2-D ``[H, W]`` image (numpy-only).

    Self-contained 3x3 Sobel convolution with edge replication, returning the
    gradient magnitude ``sqrt(gx^2 + gy^2)``. Used by the hallucination-family
    edge / gradient detectors so they work without OpenCV/skimage.
    """
    _require_numpy()
    a = np.asarray(img_hw, dtype=np.float64)
    p = np.pad(a, 1, mode="edge")
    # 3x3 neighbourhood shifts.
    nw, n_, ne = p[:-2, :-2], p[:-2, 1:-1], p[:-2, 2:]
    w_, _c, e_ = p[1:-1, :-2], p[1:-1, 1:-1], p[1:-1, 2:]
    sw, s_, se = p[2:, :-2], p[2:, 1:-1], p[2:, 2:]
    gx = (ne + 2.0 * e_ + se) - (nw + 2.0 * w_ + sw)
    gy = (sw + 2.0 * s_ + se) - (nw + 2.0 * n_ + ne)
    return np.sqrt(gx * gx + gy * gy)


def pearson_corr(a: "np.ndarray", b: "np.ndarray") -> float:
    """Pearson correlation between two flattened arrays (NaN-safe -> 0.0)."""
    af = np.asarray(a, dtype=np.float64).ravel()
    bf = np.asarray(b, dtype=np.float64).ravel()
    if af.size == 0 or bf.size == 0:
        return 0.0
    af = af - af.mean()
    bf = bf - bf.mean()
    denom = float(np.sqrt(np.sum(af * af) * np.sum(bf * bf)))
    if denom <= 1e-12:
        return 0.0
    return float(np.sum(af * bf) / denom)


__all__ = [
    "NUMPY_AVAILABLE",
    "is_torch_tensor",
    "to_numpy",
    "as_bchw",
    "as_bhwc",
    "check_same_shape",
    "shave_border",
    "rgb_to_y",
    "rgb_to_ycbcr",
    "rgb_to_lab",
    "sobel_gradient_magnitude",
    "pearson_corr",
]
