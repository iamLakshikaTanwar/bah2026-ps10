"""irchroma.metrics.color — Family D: color-specific metrics (colorization heart).

Owner: Builder-6 (MetricsBuilder). Implements docs/research/06 §D.

PSNR/SSIM under-weight chroma (they are dominated by luminance), so they cannot
certify "water -> blue, forest -> green without artifacts" — the heart of PS-10.
These metrics measure *color* correctness directly.

Design priority (per the task contract): **CIEDE2000, Colorfulness, and chroma-PSNR
are implemented self-contained in numpy** (the Lab conversion + the full Sharma
2005 ΔE2000 formula live here / in ``_common``), so they NEVER require an optional
dependency. The ``colour-science`` / scikit-image packages are not needed.

Tensor convention: ``pred`` / ``target`` are RGB ``[B, 3, H, W]`` in ``[0, 1]``.
All metrics return Python ``float``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..interfaces import Metric
from . import _common as _c

try:
    import numpy as np  # type: ignore

    _NUMPY = True
except Exception:  # pragma: no cover
    np = None  # type: ignore
    _NUMPY = False


# --------------------------------------------------------------------------- #
# CIEDE2000 (ΔE00) — self-contained Sharma, Wu & Dalal (2005) implementation.
# Operates on Lab arrays of identical shape; returns the per-pixel ΔE00 array.
# --------------------------------------------------------------------------- #
def _ciede2000(lab1: "np.ndarray", lab2: "np.ndarray") -> "np.ndarray":
    """Compute per-element CIEDE2000 color difference between two Lab arrays.

    Args:
      lab1, lab2: matching-shape arrays whose **last axis is the L*, a*, b*
                  channel** (e.g. ``[..., 3]``). L in [0,100], a/b in ~[-128,127].

    Returns:
      An array of ΔE00 values with the channel axis removed (one per pixel).

    Reference: G. Sharma, W. Wu, E. N. Dalal, "The CIEDE2000 color-difference
    formula", *Color Research & Application* (2005) — the standard reference
    implementation that ``skimage.color.deltaE_ciede2000`` and ``colour-science``
    both follow. Constants ``kL = kC = kH = 1`` (unity parametric factors).
    """
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    kL = kC = kH = 1.0

    C1 = np.sqrt(a1 * a1 + b1 * b1)
    C2 = np.sqrt(a2 * a2 + b2 * b2)
    C_bar = 0.5 * (C1 + C2)

    C_bar7 = C_bar ** 7
    G = 0.5 * (1.0 - np.sqrt(C_bar7 / (C_bar7 + 25.0 ** 7)))

    a1p = (1.0 + G) * a1
    a2p = (1.0 + G) * a2
    C1p = np.sqrt(a1p * a1p + b1 * b1)
    C2p = np.sqrt(a2p * a2p + b2 * b2)

    h1p = np.degrees(np.arctan2(b1, a1p))
    h1p = np.where(h1p < 0.0, h1p + 360.0, h1p)
    h2p = np.degrees(np.arctan2(b2, a2p))
    h2p = np.where(h2p < 0.0, h2p + 360.0, h2p)

    dLp = L2 - L1
    dCp = C2p - C1p

    C1pC2p = C1p * C2p
    dhp = h2p - h1p
    dhp = np.where(dhp > 180.0, dhp - 360.0, dhp)
    dhp = np.where(dhp < -180.0, dhp + 360.0, dhp)
    # Where either chroma is zero, the hue difference is undefined -> 0.
    dhp = np.where(C1pC2p == 0.0, 0.0, dhp)
    dHp = 2.0 * np.sqrt(C1pC2p) * np.sin(np.radians(dhp) / 2.0)

    Lp_bar = 0.5 * (L1 + L2)
    Cp_bar = 0.5 * (C1p + C2p)

    hp_sum = h1p + h2p
    abs_dh = np.abs(h1p - h2p)
    hp_bar = np.where(
        C1pC2p == 0.0,
        hp_sum,
        np.where(
            abs_dh <= 180.0,
            0.5 * hp_sum,
            np.where(hp_sum < 360.0, 0.5 * (hp_sum + 360.0), 0.5 * (hp_sum - 360.0)),
        ),
    )

    T = (
        1.0
        - 0.17 * np.cos(np.radians(hp_bar - 30.0))
        + 0.24 * np.cos(np.radians(2.0 * hp_bar))
        + 0.32 * np.cos(np.radians(3.0 * hp_bar + 6.0))
        - 0.20 * np.cos(np.radians(4.0 * hp_bar - 63.0))
    )

    d_theta = 30.0 * np.exp(-(((hp_bar - 275.0) / 25.0) ** 2))
    Cp_bar7 = Cp_bar ** 7
    Rc = 2.0 * np.sqrt(Cp_bar7 / (Cp_bar7 + 25.0 ** 7))
    Lp_bar_m50_sq = (Lp_bar - 50.0) ** 2
    Sl = 1.0 + (0.015 * Lp_bar_m50_sq) / np.sqrt(20.0 + Lp_bar_m50_sq)
    Sc = 1.0 + 0.045 * Cp_bar
    Sh = 1.0 + 0.015 * Cp_bar * T
    Rt = -np.sin(np.radians(2.0 * d_theta)) * Rc

    term_L = dLp / (kL * Sl)
    term_C = dCp / (kC * Sc)
    term_H = dHp / (kH * Sh)

    dE = np.sqrt(
        term_L * term_L
        + term_C * term_C
        + term_H * term_H
        + Rt * term_C * term_H
    )
    return dE


class CIEDE2000Metric(Metric):
    """CIEDE2000 (ΔE₀₀) — perceptually-uniform color difference in CIE L*a*b*. Lower better.

    **Self-contained** (Sharma 2005 ΔE2000 + a D65 sRGB->Lab conversion, both in
    numpy) so it NEVER needs an optional dependency — the single most important
    *color-accuracy* number for PS-10 (doc 06 §D.25). Rule of thumb: ΔE < 1
    imperceptible, 1–2 perceptible on close inspection, > 5 clearly different.

    Input is RGB ``[B, 3, H, W]`` in ``[0, 1]``; converted to Lab internally and
    the mean per-pixel ΔE00 is returned.
    """

    name: str = "ciede2000"
    higher_is_better: bool = False
    family: str = "color"

    def __init__(self, border: int = 0) -> None:
        self.border = int(border)

    def __call__(
        self,
        pred: Any,
        target: Any = None,
        ctx: Optional[Dict[str, Any]] = None,
    ) -> float:
        if target is None:
            raise ValueError("CIEDE2000Metric requires a target (full-reference).")
        p = _c.as_bchw(pred)
        t = _c.as_bchw(target)
        _c.check_same_shape(p, t)
        if p.shape[1] != 3:
            raise ValueError("CIEDE2000 requires 3-channel RGB inputs.")
        if self.border > 0:
            p = _c.shave_border(p, self.border)
            t = _c.shave_border(t, self.border)
        lab_p = _c.rgb_to_lab(p)  # [B, 3, H, W]
        lab_t = _c.rgb_to_lab(t)
        # Move channel axis last for the per-pixel formula.
        lab_p = np.moveaxis(lab_p, 1, -1)  # [B, H, W, 3]
        lab_t = np.moveaxis(lab_t, 1, -1)
        de = _ciede2000(lab_p, lab_t)
        return float(np.mean(de))


class ColorfulnessMetric(Metric):
    """Colorfulness match (Hasler–Süsstrunk M3). Lower better (reported as a delta).

    **Self-contained** (doc 06 §D.26). The Hasler–Süsstrunk M3 colorfulness of an
    image, from the opponent channels ``rg = R - G`` and ``yb = 0.5(R+G) - B``::

        M3 = sqrt(σ_rg² + σ_yb²) + 0.3 · sqrt(μ_rg² + μ_yb²)

    Correlates > 90% with human ratings. When a ``target`` is given we report the
    **absolute difference** ``|M3_pred - M3_GT|`` (lower better) — a desaturated GAN
    output flags here even when PSNR is fine. With no target, the raw ``M3_pred`` is
    returned (and ``higher_is_better`` is then irrelevant — more colorful = larger).

    RGB values are taken on the standard 0–255 scale internally (the formula's 0.3
    constant assumes 8-bit opponent channels), so inputs in ``[0, 1]`` are scaled.
    """

    name: str = "colorfulness"
    higher_is_better: bool = False  # reported as |Δ| vs GT
    family: str = "color"

    def __init__(self, border: int = 0) -> None:
        self.border = int(border)

    @staticmethod
    def _m3(bchw_255: "np.ndarray") -> float:
        """Hasler–Süsstrunk M3 colorfulness for a ``[B, 3, H, W]`` array in [0,255]."""
        r = bchw_255[:, 0, :, :]
        g = bchw_255[:, 1, :, :]
        b = bchw_255[:, 2, :, :]
        rg = r - g
        yb = 0.5 * (r + g) - b
        std_root = np.sqrt(np.std(rg) ** 2 + np.std(yb) ** 2)
        mean_root = np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2)
        return float(std_root + 0.3 * mean_root)

    def __call__(
        self,
        pred: Any,
        target: Any = None,
        ctx: Optional[Dict[str, Any]] = None,
    ) -> float:
        p = _c.as_bchw(pred)
        if p.shape[1] != 3:
            raise ValueError("ColorfulnessMetric requires 3-channel RGB inputs.")
        if self.border > 0:
            p = _c.shave_border(p, self.border)
        m3_pred = self._m3(p * 255.0)
        if target is None:
            return m3_pred
        t = _c.as_bchw(target)
        if self.border > 0:
            t = _c.shave_border(t, self.border)
        m3_gt = self._m3(t * 255.0)
        return float(abs(m3_pred - m3_gt))


class ChromaPSNRMetric(Metric):
    """Chrominance PSNR — PSNR on the *color* channels only. Higher better.

    Isolates color reconstruction from luminance by computing PSNR on the chroma
    channels (doc 06 §D.27). Two color spaces are supported:

      * ``space="ycbcr"`` (default): PSNR on the Cb and Cr channels (BT.601),
        averaged. Pairs with Y-channel PSNR (Family A) to separate "structure" vs
        "color" errors.
      * ``space="lab"``: PSNR on the a* and b* chroma channels (perceptually
        uniform). The data range is taken as the a*/b* span (default 255) since
        Lab chroma is not in [0,1].

    **Self-contained** (numpy color conversions in ``_common``); no optional dep.
    """

    name: str = "chroma_psnr"
    higher_is_better: bool = True
    family: str = "color"

    def __init__(
        self,
        space: str = "ycbcr",
        border: int = 0,
        lab_range: float = 255.0,
    ) -> None:
        self.space = str(space).lower()
        self.border = int(border)
        self.lab_range = float(lab_range)
        if self.space not in ("ycbcr", "lab"):
            raise ValueError("ChromaPSNRMetric.space must be 'ycbcr' or 'lab'.")
        self.name = "chroma_psnr_ab" if self.space == "lab" else "chroma_psnr"

    def __call__(
        self,
        pred: Any,
        target: Any = None,
        ctx: Optional[Dict[str, Any]] = None,
    ) -> float:
        if target is None:
            raise ValueError("ChromaPSNRMetric requires a target (full-reference).")
        p = _c.as_bchw(pred)
        t = _c.as_bchw(target)
        _c.check_same_shape(p, t)
        if p.shape[1] != 3:
            raise ValueError("ChromaPSNRMetric requires 3-channel RGB inputs.")
        if self.border > 0:
            p = _c.shave_border(p, self.border)
            t = _c.shave_border(t, self.border)
        if self.space == "ycbcr":
            cp = _c.rgb_to_ycbcr(p)[:, 1:3, :, :]  # Cb, Cr in [0,1]
            ct = _c.rgb_to_ycbcr(t)[:, 1:3, :, :]
            data_range = 1.0
        else:  # lab a*, b*
            cp = _c.rgb_to_lab(p)[:, 1:3, :, :]
            ct = _c.rgb_to_lab(t)[:, 1:3, :, :]
            data_range = self.lab_range
        mse = float(np.mean((cp - ct) ** 2))
        if mse <= 1e-12:
            return float("inf")
        return float(10.0 * np.log10((data_range ** 2) / mse))


__all__ = [
    "CIEDE2000Metric",
    "ColorfulnessMetric",
    "ChromaPSNRMetric",
]
