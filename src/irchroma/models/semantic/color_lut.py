"""irchroma.models.semantic.color_lut — O(1) class->CIE-Lab chroma clamp + AdaInt 3D-LUT.

This is the **color-consistency core** (docs/research/04 "Color-LUT design";
ARCHITECTURE.md §7.3). It contains two genuinely *O(1)-per-pixel* color operators:

* :class:`ClassColorLUT` — the class-conditioned chroma clamp implementing
  :class:`irchroma.interfaces.ColorLUTProtocol`. A precomputed ``[num_classes, 3]``
  CIE-Lab palette (seeded from :data:`irchroma.config.DEFAULT_LAB_PALETTE`) plus a
  per-class half-width gives, per class, a clamp box ``[mu - k*sigma, mu + k*sigma]``
  in Lab. At ``apply`` time we **gather** the per-pixel target box by indexing with the
  semantic label (one gather = O(1)/pixel, vectorized over the tile), convert the input
  RGB to Lab, **clamp the chroma channels** ``a*, b*`` into the class box while leaving
  ``L*`` (luminance) free so SR texture/detail survives, then convert back to RGB.
* :class:`AdaIntLUT` — a small learned **image-adaptive 3D LUT** (AdaInt-style,
  docs/research/05 §B2): a tiny CNN predicts blend weights over a few learnable basis
  3D LUTs (and, optionally, non-uniform AdaInt sampling intervals); the per-pixel
  transform is a single **trilinear interpolation** (8-tap) = O(1)/pixel. Optional
  learned color refinement applied *after* the class clamp.

Both operators are fully vectorized and differentiable (the RGB<->Lab conversions
below are pure-torch and back-proppable), so they double as the building blocks for
the ``color_lut_outofclass`` loss term.

Tensor conventions (see :mod:`irchroma.interfaces`):
  * ``rgb``      : ``FloatTensor [B, 3, H, W]`` in ``[0, 1]`` (sRGB, R, G, B order).
  * ``semantic`` : ``LongTensor  [B, H, W]`` — class indices into ``LULC_CLASSES``.

The module imports even without torch installed (guarded import + stubs); the layers
are simply not instantiable on that path.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from irchroma.config import (
    DEFAULT_LAB_PALETTE,
    HARD_PRIOR_CLASSES,
    LULC_CLASSES,
    LULC_NAME_TO_INDEX,
    NUM_LULC_CLASSES,
    SemanticConfig,
)
from irchroma.interfaces import TORCH_AVAILABLE

# --------------------------------------------------------------------------- #
# Guarded torch import (module must import even without torch).
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - exercised only when torch is present
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor
except Exception:  # pragma: no cover - torch-less environments (docs/CI)
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    Tensor = object  # type: ignore


if TORCH_AVAILABLE:  # pragma: no branch
    _Module = nn.Module  # type: ignore[assignment]
else:  # pragma: no cover

    class _Module:  # minimal stand-in base when torch is unavailable
        """Placeholder base when torch is unavailable (never instantiated)."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError(
                "torch is not installed; irchroma.models.semantic.color_lut requires torch."
            )


# Import BaseModel lazily-safe: it is a thin marker over nn.Module.
from irchroma.interfaces import BaseModel  # noqa: E402  (after torch guard by design)

__all__ = ["ClassColorLUT", "AdaIntLUT", "rgb_to_lab", "lab_to_rgb"]


# =========================================================================== #
# Differentiable sRGB <-> CIE-Lab conversion (pure torch, documented matrices).
# =========================================================================== #
# Pipeline (D65 white point, standard 2-degree observer):
#   sRGB' (gamma) --inverse-gamma--> linear sRGB --M_RGB2XYZ--> XYZ
#     --/white--> normalized XYZ --f()--> L*a*b*    (and the exact inverse).
#
# sRGB -> linear-XYZ matrix (IEC 61966-2-1, D65). Rows map linear (R,G,B) -> (X,Y,Z):
_RGB2XYZ = (
    (0.4124564, 0.3575761, 0.1804375),
    (0.2126729, 0.7151522, 0.0721750),
    (0.0193339, 0.1191920, 0.9503041),
)
# Inverse (XYZ -> linear sRGB):
_XYZ2RGB = (
    (3.2404542, -1.5371385, -0.4985314),
    (-0.9692660, 1.8760108, 0.0415560),
    (0.0556434, -0.2040259, 1.0572252),
)
# D65 reference white (Xn, Yn, Zn), Y normalized to 1.0 (2-degree observer):
_D65 = (0.95047, 1.00000, 1.08883)
# CIE-Lab constants:
_LAB_EPS = 216.0 / 24389.0  # (6/29)^3 ~ 0.008856
_LAB_KAPPA = 24389.0 / 27.0  # (29/3)^3 ~ 903.3


def _srgb_to_linear(c: "Tensor") -> "Tensor":
    """Inverse sRGB gamma (companding). ``c`` in [0,1] -> linear in [0,1]."""
    c = c.clamp(0.0, 1.0)
    return torch.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c: "Tensor") -> "Tensor":
    """Forward sRGB gamma. linear -> companded sRGB in [0,1]."""
    c = c.clamp(0.0, 1.0)
    return torch.where(c <= 0.0031308, c * 12.92, 1.055 * (c ** (1.0 / 2.4)) - 0.055)


def _lab_f(t: "Tensor") -> "Tensor":
    """Lab forward nonlinearity ``f(t)`` (cube-root with linear toe)."""
    return torch.where(t > _LAB_EPS, t.clamp_min(0.0) ** (1.0 / 3.0), (_LAB_KAPPA * t + 16.0) / 116.0)


def _lab_f_inv(t: "Tensor") -> "Tensor":
    """Inverse Lab nonlinearity."""
    t3 = t ** 3
    return torch.where(t3 > _LAB_EPS, t3, (116.0 * t - 16.0) / _LAB_KAPPA)


def _matmul_channels(img: "Tensor", mat: Tuple[Tuple[float, ...], ...]) -> "Tensor":
    """Apply a 3x3 color matrix to ``[B, 3, H, W]`` along the channel dim."""
    m = torch.as_tensor(mat, dtype=img.dtype, device=img.device)  # [3, 3]
    b, c, h, w = img.shape
    flat = img.reshape(b, c, h * w)  # [B, 3, HW]
    out = torch.einsum("ij,bjn->bin", m, flat)  # [B, 3, HW]
    return out.reshape(b, 3, h, w)


def rgb_to_lab(rgb: "Tensor") -> "Tensor":
    """Convert sRGB in ``[0,1]`` to CIE-Lab. **Differentiable, vectorized.**

    Args:
      rgb: ``FloatTensor [B, 3, H, W]`` in ``[0, 1]`` (R, G, B; sRGB, D65).

    Returns:
      ``FloatTensor [B, 3, H, W]`` = ``(L*, a*, b*)`` with ``L in [0,100]`` and
      ``a, b`` roughly in ``[-128, 127]``.
    """
    if rgb.dim() != 4 or rgb.shape[1] != 3:
        raise ValueError(f"rgb_to_lab expects [B, 3, H, W]; got {tuple(rgb.shape)}.")
    linear = _srgb_to_linear(rgb)
    xyz = _matmul_channels(linear, _RGB2XYZ)  # [B, 3, H, W]
    wx, wy, wz = _D65
    white = torch.as_tensor([wx, wy, wz], dtype=rgb.dtype, device=rgb.device).view(1, 3, 1, 1)
    xyz_n = xyz / white
    fx = _lab_f(xyz_n[:, 0:1])
    fy = _lab_f(xyz_n[:, 1:2])
    fz = _lab_f(xyz_n[:, 2:3])
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return torch.cat([L, a, b], dim=1)


def lab_to_rgb(lab: "Tensor", clamp: bool = True) -> "Tensor":
    """Convert CIE-Lab back to sRGB in ``[0,1]``. **Differentiable, vectorized.**

    Args:
      lab:   ``FloatTensor [B, 3, H, W]`` = ``(L*, a*, b*)``.
      clamp: if ``True`` clamp the output into ``[0, 1]`` (gamut-clip).

    Returns:
      ``FloatTensor [B, 3, H, W]`` sRGB in ``[0, 1]``.
    """
    if lab.dim() != 4 or lab.shape[1] != 3:
        raise ValueError(f"lab_to_rgb expects [B, 3, H, W]; got {tuple(lab.shape)}.")
    L = lab[:, 0:1]
    a = lab[:, 1:2]
    b = lab[:, 2:3]
    fy = (L + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0
    xr = _lab_f_inv(fx)
    yr = _lab_f_inv(fy)
    zr = _lab_f_inv(fz)
    wx, wy, wz = _D65
    white = torch.as_tensor([wx, wy, wz], dtype=lab.dtype, device=lab.device).view(1, 3, 1, 1)
    xyz = torch.cat([xr, yr, zr], dim=1) * white
    linear = _matmul_channels(xyz, _XYZ2RGB)
    srgb = _linear_to_srgb(linear)
    if clamp:
        srgb = srgb.clamp(0.0, 1.0)
    return srgb


# =========================================================================== #
# ClassColorLUT — the O(1) class -> Lab chroma clamp (ColorLUTProtocol).
# =========================================================================== #
class ClassColorLUT(_Module):  # type: ignore[misc]
    """O(1)-per-pixel class-conditioned CIE-Lab chroma clamp.

    Satisfies :class:`irchroma.interfaces.ColorLUTProtocol`. The table is built once
    from config: per class ``c`` we store a Lab centroid ``mu[c]`` (from
    :data:`irchroma.config.DEFAULT_LAB_PALETTE`) and a per-channel half-width
    ``half[c] = k_c * sigma`` giving the clamp box ``[mu - half, mu + half]``. Hard-prior
    classes (:data:`irchroma.config.HARD_PRIOR_CLASSES`, e.g. water/snow) use a tighter
    ``k`` (``lut_hard_prior_sigma``) since their color is near-deterministic and a
    mislabel is the most dangerous hallucination.

    **Mechanism (genuinely O(1)/pixel, vectorized over the tile):**
      1. ``lo, hi = LUT[semantic]`` — a single ``index_select`` gather of the precomputed
         per-class boxes by the per-pixel label (O(1)/pixel).
      2. ``lab = rgb_to_lab(rgb)`` — differentiable conversion.
      3. Clamp **chroma** ``a*, b*`` into ``[lo, hi]`` (leave ``L*`` free when
         ``chroma_only``); blend toward the clamp by ``strength`` (0 = passthrough,
         1 = fully clamped).
      4. ``out = lab_to_rgb(lab)``.

    Args:
      cfg:           a :class:`irchroma.config.SemanticConfig` (clamp sigmas, flags).
                     Defaults to ``SemanticConfig()``.
      lab_palette:   optional ``{class_name: (L, a, b)}`` override; defaults to
                     :data:`irchroma.config.DEFAULT_LAB_PALETTE`.
      strength:      blend factor in ``[0, 1]`` (default ``1.0`` = hard clamp).
      chroma_margin: extra additive half-width (in Lab units) added to every class box
                     so the clamp is not razor-thin (default ``4.0``).
      base_sigma:    per-channel sigma seed used when no measured covariance is supplied
                     (the runtime LUT-build widens this from real statistics); the clamp
                     half-width is ``k * base_sigma + chroma_margin``.
    """

    def __init__(
        self,
        cfg: Optional[SemanticConfig] = None,
        lab_palette: Optional[Dict[str, Tuple[float, float, float]]] = None,
        strength: float = 1.0,
        chroma_margin: float = 4.0,
        base_sigma: float = 10.0,
    ) -> None:
        super().__init__()  # type: ignore[misc]
        self.cfg = cfg or SemanticConfig()
        self.num_classes = NUM_LULC_CLASSES
        self.strength = float(strength)
        self.chroma_margin = float(chroma_margin)
        self.base_sigma = float(base_sigma)
        self.chroma_only = bool(self.cfg.lut_clamp_chroma_only)

        palette = lab_palette if lab_palette is not None else DEFAULT_LAB_PALETTE
        hard = {LULC_NAME_TO_INDEX[n] for n in HARD_PRIOR_CLASSES if n in LULC_NAME_TO_INDEX}

        # Build [num_classes, 3] centroid and half-width tables from config palettes.
        mu_rows = []
        half_rows = []
        for idx, name in enumerate(LULC_CLASSES):
            lab = palette.get(name, (50.0, 0.0, 0.0))
            mu_rows.append([float(lab[0]), float(lab[1]), float(lab[2])])
            k = float(self.cfg.lut_hard_prior_sigma) if idx in hard else float(self.cfg.lut_clamp_sigma)
            hw = k * self.base_sigma + self.chroma_margin
            # L* half-width is only used if chroma_only is False; keep it generous so
            # luminance stays effectively free even then.
            half_rows.append([hw * 4.0, hw, hw])

        mu = torch.as_tensor(mu_rows, dtype=torch.float32)  # [K, 3]
        half = torch.as_tensor(half_rows, dtype=torch.float32)  # [K, 3]
        lo = mu - half
        hi = mu + half

        # Register as buffers so they move with .to(device) and are saved in state_dict.
        self.register_buffer("mu_lab", mu, persistent=True)  # [K, 3]
        self.register_buffer("lut_lo", lo, persistent=True)  # [K, 3] = mu - half
        self.register_buffer("lut_hi", hi, persistent=True)  # [K, 3] = mu + half
        # Indices of the chroma channels to clamp (a*, b*). L* (0) is left free.
        chroma_idx = [1, 2] if self.chroma_only else [0, 1, 2]
        self.register_buffer(
            "chroma_idx", torch.as_tensor(chroma_idx, dtype=torch.long), persistent=False
        )

    # ------------------------------------------------------------------ #
    def _gather_box(self, semantic: "Tensor") -> Tuple["Tensor", "Tensor"]:
        """Gather per-pixel ``(lo, hi)`` Lab clamp boxes by the semantic label.

        O(1)/pixel: a single ``index_select`` per bound, reshaped to ``[B, 3, H, W]``.

        Args:
          semantic: ``LongTensor [B, H, W]`` of class indices.

        Returns:
          ``(lo, hi)`` each ``FloatTensor [B, 3, H, W]``.
        """
        if semantic.dim() == 4 and semantic.shape[1] == 1:
            semantic = semantic[:, 0]
        if semantic.dim() != 3:
            raise ValueError(
                f"ClassColorLUT expects semantic [B, H, W]; got {tuple(semantic.shape)}."
            )
        b, h, w = semantic.shape
        idx = semantic.long().clamp(min=0, max=self.num_classes - 1).reshape(-1)  # [B*H*W]
        lo = self.lut_lo.index_select(0, idx)  # [B*H*W, 3]
        hi = self.lut_hi.index_select(0, idx)  # [B*H*W, 3]
        lo = lo.reshape(b, h, w, 3).permute(0, 3, 1, 2).contiguous()  # [B, 3, H, W]
        hi = hi.reshape(b, h, w, 3).permute(0, 3, 1, 2).contiguous()
        return lo, hi

    def apply(self, rgb: "Tensor", semantic: "Tensor") -> "Tensor":
        """Return color-constrained RGB (``ColorLUTProtocol.apply``).

        Args:
          rgb:      ``FloatTensor [B, 3, H, W]`` in ``[0, 1]`` — raw predicted color.
          semantic: ``LongTensor  [B, H, W]`` — per-pixel LULC class indices.

        Returns:
          ``FloatTensor [B, 3, H, W]`` in ``[0, 1]`` with chroma clamped toward the
          class palette (luminance left freer so SR texture/detail survives).
        """
        if rgb.dim() != 4 or rgb.shape[1] != 3:
            raise ValueError(f"ClassColorLUT.apply expects rgb [B, 3, H, W]; got {tuple(rgb.shape)}.")
        lo, hi = self._gather_box(semantic)  # [B, 3, H, W] each, O(1)/pixel gather
        lab = rgb_to_lab(rgb)  # [B, 3, H, W]

        # Hard-clamp only the chroma channels; assemble back so L* stays untouched.
        clamped = torch.minimum(torch.maximum(lab, lo), hi)  # full clamp candidate
        if self.chroma_only:
            # Keep L* (channel 0) exactly; replace a*, b* with the clamped values.
            L = lab[:, 0:1]
            ab = clamped[:, 1:3]
            clamped = torch.cat([L, ab], dim=1)

        if self.strength >= 1.0:
            out_lab = clamped
        elif self.strength <= 0.0:
            out_lab = lab
        else:
            out_lab = lab + self.strength * (clamped - lab)  # differentiable blend

        return lab_to_rgb(out_lab, clamp=True)

    def forward(self, rgb: "Tensor", semantic: "Tensor") -> "Tensor":
        """Alias for :meth:`apply` so the LUT is usable as an ``nn.Module``."""
        return self.apply(rgb, semantic)

    def out_of_class_distance(self, rgb: "Tensor", semantic: "Tensor") -> "Tensor":
        """Per-pixel Lab distance of the chroma *outside* the class clamp box.

        Useful for the ``color_lut_outofclass`` loss: ``relu(lab - hi) + relu(lo - lab)``
        on the chroma channels, summed over channels -> ``[B, 1, H, W]`` (zero inside
        the box). Differentiable; O(1)/pixel.
        """
        lo, hi = self._gather_box(semantic)
        lab = rgb_to_lab(rgb)
        over = F.relu(lab - hi)
        under = F.relu(lo - lab)
        dist = over + under  # [B, 3, H, W]
        ch = self.chroma_idx.to(device=dist.device)
        dist = dist.index_select(1, ch)  # only the clamped channels
        return dist.abs().sum(dim=1, keepdim=True)  # [B, 1, H, W]


# =========================================================================== #
# AdaIntLUT — learned image-adaptive 3D LUT (AdaInt-style). O(1)/pixel apply.
# =========================================================================== #
class AdaIntLUT(BaseModel):  # type: ignore[misc]
    """Learned image-adaptive 3D LUT for optional color refinement (AdaInt-style).

    A tiny CNN ("weight predictor") looks at a thumbnail of the image and predicts
    ``n_basis`` blend weights (once per image). The effective LUT is
    ``sum_k weight_k * basis_lut_k`` over ``n_basis`` learnable basis 3D LUTs of lattice
    size ``dim^3 x 3``. The per-pixel color transform is a single **trilinear
    interpolation** (8-tap) into that lattice -> **O(1)/pixel**, vectorized over the
    tile via :func:`torch.nn.functional.grid_sample` (5-D / volumetric sampling).

    When ``adaint=True`` an extra tiny head predicts non-uniform per-axis sampling
    *intervals* (the AdaInt contribution), letting the lattice allocate resolution where
    color density is highest. (docs/research/05 §B2; ARCHITECTURE.md §7.3, §9.2.)

    Args:
      cfg:        :class:`irchroma.config.SemanticConfig` (reads ``lut3d_dim``,
                  ``lut3d_n_basis``, ``lut3d_adaint``). Defaults to ``SemanticConfig()``.
      thumb_size: spatial size of the thumbnail fed to the weight predictor.
      hidden:     width of the weight-predictor conv stack.

    forward: ``forward(rgb) -> rgb`` (``[B, 3, H, W]`` in ``[0, 1]`` -> refined RGB).
    """

    name = "adaint_3dlut"

    def __init__(
        self,
        cfg: Optional[SemanticConfig] = None,
        thumb_size: int = 32,
        hidden: int = 32,
    ) -> None:
        super().__init__()  # type: ignore[misc]
        self.cfg = cfg or SemanticConfig()
        self.dim = int(self.cfg.lut3d_dim)
        self.n_basis = int(self.cfg.lut3d_n_basis)
        self.adaint = bool(self.cfg.lut3d_adaint)
        self.thumb_size = int(thumb_size)

        # ---- Basis 3D LUTs initialized to the IDENTITY mapping. -------------- #
        # Lattice grid in [0,1]^3; shape stored as [n_basis, 3, D, D, D] (out RGB at
        # each (r,g,b) lattice node). Identity => node (i,j,k) maps to (r,g,b) coords.
        identity = self._identity_lut(self.dim)  # [3, D, D, D]
        basis = identity.unsqueeze(0).repeat(self.n_basis, 1, 1, 1, 1).clone()  # [N,3,D,D,D]
        # All basis LUTs start at the identity transform (standard 3D-LUT init); the
        # per-image weight predictor supplies distinct gradients to each basis during
        # training, so they diverge into a useful basis. Initializing at identity makes
        # the *untrained* module a near-no-op refinement (safe to insert into the pipeline).
        self.basis_luts = nn.Parameter(basis)  # learnable

        # ---- Weight predictor: thumbnail -> n_basis blend weights. ----------- #
        self.predictor = nn.Sequential(
            nn.Conv2d(3, hidden, kernel_size=3, stride=2, padding=1),  # /2
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden, hidden * 2, kernel_size=3, stride=2, padding=1),  # /4
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.weight_head = nn.Linear(hidden * 2, self.n_basis)
        # AdaInt: predict (dim-1) positive interval widths per axis (R,G,B).
        if self.adaint:
            self.interval_head: Optional[nn.Linear] = nn.Linear(hidden * 2, 3 * (self.dim - 1))
        else:
            self.interval_head = None

    # ------------------------------------------------------------------ #
    @staticmethod
    def _identity_lut(dim: int) -> "Tensor":
        """Build the identity 3D LUT of shape ``[3, D, D, D]`` over ``[0,1]^3``.

        Node ``(i, j, k)`` (indexing R, G, B axes) maps to the color
        ``(i, j, k) / (D - 1)`` -> identity transform.
        """
        coords = torch.linspace(0.0, 1.0, steps=dim)
        r = coords.view(dim, 1, 1).expand(dim, dim, dim)
        g = coords.view(1, dim, 1).expand(dim, dim, dim)
        b = coords.view(1, 1, dim).expand(dim, dim, dim)
        return torch.stack([r, g, b], dim=0)  # [3, D, D, D]

    def _predict(self, rgb: "Tensor") -> Tuple["Tensor", Optional["Tensor"]]:
        """Predict per-image blend weights (and optional AdaInt intervals).

        Returns:
          ``weights`` ``[B, n_basis]`` (softmax-normalized) and ``intervals``
          ``[B, 3, dim-1]`` or ``None``.
        """
        thumb = F.interpolate(
            rgb, size=(self.thumb_size, self.thumb_size), mode="bilinear", align_corners=False
        )
        feat = self.predictor(thumb).flatten(1)  # [B, hidden*2]
        weights = torch.softmax(self.weight_head(feat), dim=1)  # [B, n_basis]
        intervals: Optional["Tensor"] = None
        if self.interval_head is not None:
            raw = self.interval_head(feat).view(-1, 3, self.dim - 1)
            # Positive, normalized so each axis' intervals sum to 1 (cover [0,1]).
            pos = F.softmax(raw, dim=2)
            intervals = pos  # [B, 3, dim-1]
        return weights, intervals

    @staticmethod
    def _intervals_to_vertices(intervals: "Tensor") -> "Tensor":
        """Cumulative interval widths -> per-axis vertex coordinates in ``[0,1]``.

        Args:
          intervals: ``[B, 3, dim-1]`` positive widths summing to 1 per axis.

        Returns:
          ``[B, 3, dim]`` monotonically increasing vertex positions from 0 to 1.
        """
        b, three, dm1 = intervals.shape
        zeros = intervals.new_zeros(b, three, 1)
        cum = torch.cumsum(intervals, dim=2)  # ends at 1
        return torch.cat([zeros, cum], dim=2)  # [B, 3, dim]

    def forward(self, rgb: "Tensor") -> "Tensor":
        """Apply the image-adaptive 3D LUT (O(1)/pixel via trilinear interpolation).

        Args:
          rgb: ``FloatTensor [B, 3, H, W]`` in ``[0, 1]``.

        Returns:
          ``FloatTensor [B, 3, H, W]`` in ``[0, 1]`` — color-refined RGB.
        """
        if rgb.dim() != 4 or rgb.shape[1] != 3:
            raise ValueError(f"AdaIntLUT.forward expects rgb [B, 3, H, W]; got {tuple(rgb.shape)}.")
        b, _, h, w = rgb.shape
        weights, intervals = self._predict(rgb)

        # Blend basis LUTs per-image: [B, 3, D, D, D].
        # basis_luts: [N, 3, D, D, D]; weights: [B, N].
        blended = torch.einsum("bn,ncdef->bcdef", weights, self.basis_luts)

        # Map input RGB -> sampling coordinates in [-1, 1] for grid_sample (5-D).
        # grid_sample volumetric expects grid[..., (x, y, z)] indexing dims (W, H, D)
        # of the input volume. Our volume axes are (R, G, B) = (dim2, dim3, dim4) i.e.
        # (D_r, D_g, D_b). grid_sample input is [B, C, D_in, H_in, W_in] sampled with
        # grid [B, D_out, H_out, W_out, 3] giving (x->W_in, y->H_in, z->D_in).
        # We treat: x<-blue, y<-green, z<-red so that (z,y,x)=(R,G,B) axis order matches
        # blended[..., R, G, B]. Build per-axis normalized coords in [-1, 1].
        r = rgb[:, 0]
        g = rgb[:, 1]
        bch = rgb[:, 2]

        if intervals is not None:
            # AdaInt: convert each channel value to a fractional lattice position using
            # the predicted non-uniform vertices, then to [-1, 1].
            verts = self._intervals_to_vertices(intervals)  # [B, 3, D]
            r_n = self._to_grid_coord(r, verts[:, 0])
            g_n = self._to_grid_coord(g, verts[:, 1])
            b_n = self._to_grid_coord(bch, verts[:, 2])
        else:
            # Uniform lattice: value v in [0,1] -> [-1, 1].
            r_n = r * 2.0 - 1.0
            g_n = g * 2.0 - 1.0
            b_n = bch * 2.0 - 1.0

        # grid: [B, D_out=1, H_out=H, W_out=W, 3] with last-dim order (x, y, z).
        grid = torch.stack([b_n, g_n, r_n], dim=-1)  # (x=blue, y=green, z=red)
        grid = grid.unsqueeze(1)  # [B, 1, H, W, 3]

        sampled = F.grid_sample(
            blended, grid, mode="bilinear", padding_mode="border", align_corners=True
        )  # [B, 3, 1, H, W]  (trilinear in 3-D == bilinear flag for 5-D input)
        out = sampled.squeeze(2)  # [B, 3, H, W]
        return out.clamp(0.0, 1.0)

    @staticmethod
    def _to_grid_coord(value: "Tensor", vertices: "Tensor") -> "Tensor":
        """Map per-pixel ``value`` in ``[0,1]`` to ``[-1, 1]`` via non-uniform ``vertices``.

        Piecewise-linear inverse of the lattice vertex positions (AdaInt). For each
        pixel value we find its fractional index along the ``dim`` vertices and rescale
        to ``[-1, 1]`` (the grid_sample coordinate convention).

        Args:
          value:    ``[B, H, W]`` channel values in ``[0, 1]``.
          vertices: ``[B, dim]`` monotonically increasing vertex positions in ``[0, 1]``.

        Returns:
          ``[B, H, W]`` coordinates in ``[-1, 1]``.
        """
        b, h, w = value.shape
        dim = vertices.shape[1]
        v = value.reshape(b, -1)  # [B, HW]
        # searchsorted per-batch: index of the upper vertex bounding each value.
        idx = torch.searchsorted(vertices, v.clamp(0.0, 1.0), right=True)  # [B, HW]
        idx = idx.clamp(1, dim - 1)
        lo = torch.gather(vertices, 1, idx - 1)  # [B, HW]
        hi = torch.gather(vertices, 1, idx)
        denom = (hi - lo).clamp_min(1e-6)
        frac = (v - lo) / denom  # in [0, 1] within the cell
        # Fractional lattice index in [0, dim-1]:
        cont = (idx - 1).to(v.dtype) + frac
        norm = cont / (dim - 1)  # [0, 1]
        coord = norm * 2.0 - 1.0  # [-1, 1]
        return coord.reshape(b, h, w)
