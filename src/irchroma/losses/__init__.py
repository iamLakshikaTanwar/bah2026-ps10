"""irchroma.losses — the fidelity-dominant composite loss stack.

Owner: Builder-5 (LossBuilder). Implements every term in the loss design
(ARCHITECTURE §8; docs/research/02 §7, 03 §9, 04 "losses"): Charbonnier/L1, MS-SSIM,
gradient/edge (SGA), FFT/Fourier, VGG/LPIPS perceptual, Pix2PixHD feature-matching,
LSGAN/hinge/vanilla adversarial, Lab chrominance, color-histogram (Wasserstein),
colorfulness, palette out-of-class penalty, spectral-angle (SAM),
segmentation-consistency (frozen segmenter), and total variation.

Each term subclasses :class:`irchroma.interfaces.LossTerm` and returns a **scalar**
``forward(output, target, ctx) -> Tensor``. They are assembled via
:class:`irchroma.interfaces.CompositeLoss` from weights in
:class:`irchroma.config.LossConfig` using :func:`build_composite_loss`.

``LossConfig`` weight field -> loss term mapping (see :func:`build_composite_loss`):

  Stage-1 SR
    sr_charbonnier         -> CharbonnierLoss(sr vs ir)
    sr_gradient            -> GradientLoss(sr vs ir)
    sr_fft                 -> FFTLoss(sr vs ir)
    sr_lpips               -> LPIPSLoss(sr vs <sr-gt>, falls back to VGG)
    sr_adversarial         -> AdversarialLoss(gan_mode)
  Stage-2 colorization
    color_l1               -> CharbonnierLoss(rgb vs rgb)   [pixel anchor]
    color_ms_ssim          -> MSSSIMLoss(rgb vs rgb)
    color_edge             -> GradientLoss(rgb vs rgb)
    color_lpips            -> LPIPSLoss(rgb vs rgb)
    color_feature_matching -> FeatureMatchingLoss
    color_adversarial      -> AdversarialLoss(gan_mode)
    color_lab_chroma       -> ChrominanceLoss
    color_histogram        -> ColorHistogramLoss(wasserstein)
    color_colorfulness     -> ColorfulnessLoss
    color_sam              -> SpectralAngleLoss
    color_tv               -> TVLoss
  Semantic / faithfulness
    seg_consistency        -> SegConsistencyLoss
    color_lut_outofclass   -> PaletteConsistencyLoss
    task_perceptual        -> VGGPerceptualLoss(rgb, resize) [task-feature proxy]

Note: ``uncertainty_weight`` and ``cycle_consistency`` are loop-level / unpaired knobs
(per ARCHITECTURE §8 they modulate other terms or require an IR re-encoder); they are
intentionally NOT wired as standalone terms here. ``gan_mode`` / ``r1_gamma`` /
``charbonnier_eps`` are hyper-parameters consumed by the factories, not weights.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping

from ..interfaces import CompositeLoss, LossTerm

from .reconstruction import (
    CharbonnierLoss,
    FFTLoss,
    GradientLoss,
    MSSSIMLoss,
)
from .perceptual import (
    LPIPSLoss,
    VGGPerceptualLoss,
)
from .adversarial import (
    AdversarialLoss,
    FeatureMatchingLoss,
)
from .color import (
    ChrominanceLoss,
    ColorHistogramLoss,
    ColorfulnessLoss,
    PaletteConsistencyLoss,
    rgb_to_lab,
)
from .semantic import (
    SegConsistencyLoss,
    SpectralAngleLoss,
    TVLoss,
)


def _term_factories(loss_cfg: Any) -> Dict[str, Callable[[], LossTerm]]:
    """Build the ``{LossConfig field name: zero-arg LossTerm factory}`` mapping.

    Keys match ``irchroma.config.LossConfig`` attribute names EXACTLY so
    :meth:`CompositeLoss.from_config` can wire weights by name. Factories capture the
    relevant scalar hyper-parameters (``charbonnier_eps``, ``gan_mode``) from
    ``loss_cfg`` so the constructed terms are self-contained.
    """
    eps = float(getattr(loss_cfg, "charbonnier_eps", 1e-3) or 1e-3)
    gan_mode = str(getattr(loss_cfg, "gan_mode", "hinge") or "hinge")

    factories: Dict[str, Callable[[], LossTerm]] = {
        # ---- Stage-1 SR -------------------------------------------------- #
        "sr_charbonnier": lambda: CharbonnierLoss(eps=eps, pred_key="sr", target_key="ir"),
        "sr_gradient": lambda: GradientLoss(pred_key="sr", target_key="ir"),
        "sr_fft": lambda: FFTLoss(pred_key="sr", target_key="ir"),
        "sr_lpips": lambda: LPIPSLoss(pred_key="sr", target_key="ir"),
        "sr_adversarial": lambda: AdversarialLoss(gan_mode=gan_mode),
        # ---- Stage-2 colorization --------------------------------------- #
        "color_l1": lambda: CharbonnierLoss(eps=eps, pred_key="rgb", target_key="rgb"),
        "color_ms_ssim": lambda: MSSSIMLoss(pred_key="rgb", target_key="rgb"),
        "color_edge": lambda: GradientLoss(pred_key="rgb", target_key="rgb"),
        "color_lpips": lambda: LPIPSLoss(pred_key="rgb", target_key="rgb"),
        "color_feature_matching": lambda: FeatureMatchingLoss(),
        "color_adversarial": lambda: AdversarialLoss(gan_mode=gan_mode),
        "color_lab_chroma": lambda: ChrominanceLoss(),
        "color_histogram": lambda: ColorHistogramLoss(mode="wasserstein"),
        "color_colorfulness": lambda: ColorfulnessLoss(),
        "color_sam": lambda: SpectralAngleLoss(pred_key="rgb", target_key="rgb"),
        "color_tv": lambda: TVLoss(pred_key="rgb"),
        # ---- Semantic / faithfulness ------------------------------------ #
        "seg_consistency": lambda: SegConsistencyLoss(),
        "color_lut_outofclass": lambda: PaletteConsistencyLoss(),
        "task_perceptual": lambda: VGGPerceptualLoss(pred_key="rgb", target_key="rgb", resize=True),
    }
    return factories


def build_composite_loss(config: Any) -> CompositeLoss:
    """Assemble the full :class:`CompositeLoss` from a config.

    Accepts either a top-level :class:`irchroma.config.Config` (uses ``config.loss``)
    or a :class:`irchroma.config.LossConfig` directly. Each enabled term (non-zero
    weight in ``LossConfig``) is instantiated and wired to its weight by name via
    :meth:`CompositeLoss.from_config`; zero-weight terms are skipped (cheap).

    Args:
      config: a ``Config`` (with a ``.loss`` section) or a ``LossConfig``.

    Returns:
      A :class:`CompositeLoss` whose ``forward(output, target, ctx)`` yields
      ``(total_scalar, {term_name: weighted_value})``.
    """
    loss_cfg = getattr(config, "loss", config)
    factories = _term_factories(loss_cfg)
    return CompositeLoss.from_config(loss_cfg, factories)


__all__ = [
    # reconstruction
    "CharbonnierLoss",
    "MSSSIMLoss",
    "GradientLoss",
    "FFTLoss",
    # perceptual
    "VGGPerceptualLoss",
    "LPIPSLoss",
    # adversarial
    "AdversarialLoss",
    "FeatureMatchingLoss",
    # color
    "ChrominanceLoss",
    "ColorHistogramLoss",
    "ColorfulnessLoss",
    "PaletteConsistencyLoss",
    "rgb_to_lab",
    # semantic
    "SegConsistencyLoss",
    "SpectralAngleLoss",
    "TVLoss",
    # builder
    "build_composite_loss",
]
