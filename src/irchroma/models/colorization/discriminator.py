"""irchroma.models.colorization.discriminator — multi-scale PatchGAN discriminator.

Owner: ColorBuilder (Builder-3). Implements the Pix2PixHD-style **multi-scale
PatchGAN** discriminator with **spectral normalization** used to train the Stage-2
colorizer. Operating on several input resolutions (full, 1/2, 1/4, ...) widens the
effective receptive field without a deeper net, which sharpens detail and stabilizes
training; spectral norm bounds the Lipschitz constant for a stable hinge/LSGAN game.

Crucially, the discriminator returns **intermediate features per layer** so the
training stack can compute the **Pix2PixHD feature-matching loss**
(``LossConfig.color_feature_matching``) — matching the discriminator's internal
activations between real and generated images, which is the dominant stabilizing /
sharpening term in the §7 loss stack.

Output contract:
  ``forward(x) -> List[List[Tensor]]`` — a list over **scales** of a list over
  **layers**; for each scale the inner list is the sequence of activations after each
  conv stage, with the **last element being the final 1-channel patch-logit map**
  ``[B, 1, h, w]`` (not sigmoid-activated — apply hinge/LSGAN/BCE in the loss). The
  preceding inner elements are the intermediate feature maps for feature matching.

Tensor conventions: the conditional input ``x`` is typically the channel-wise
concatenation of the condition (IR/semantic) and the image (real or generated RGB),
``FloatTensor [B, C, Hs, Ws]``; this module is agnostic to ``C`` (set ``in_channels``).

torch is assumed present at runtime; this file must still ``py_compile`` without it.
"""

from __future__ import annotations

from typing import List, Optional

from irchroma.config import ColorizationConfig
from irchroma.interfaces import BaseModel, TORCH_AVAILABLE, Tensor

if TORCH_AVAILABLE:  # pragma: no cover - exercised only where torch exists
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.nn.utils import spectral_norm as _spectral_norm
else:  # pragma: no cover - torch-less docs/CI environment
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    _spectral_norm = None  # type: ignore


__all__ = ["NLayerDiscriminator", "MultiScaleDiscriminator"]


def _maybe_spectral_norm(module: "nn.Module", use_sn: bool) -> "nn.Module":
    """Wrap ``module`` in spectral normalization when ``use_sn`` is True."""
    if use_sn and _spectral_norm is not None:
        return _spectral_norm(module)
    return module


class NLayerDiscriminator(BaseModel):
    """A single-scale ``n_layers``-deep PatchGAN discriminator (one of the multi-scale set).

    Architecture (Pix2PixHD / pix2pix 70×70 PatchGAN generalized):
      ``Conv(4,s2) → LReLU`` then ``n_layers - 1`` × ``Conv(4,s2) → norm → LReLU`` with
      growing width (capped at 512), one ``Conv(4,s1) → norm → LReLU`` stride-1 stage,
      then a final ``Conv(4,s1) → 1`` producing the patch-logit map. Each conv may be
      spectral-normalized. ``forward`` returns the list of per-stage activations so the
      caller can read intermediate features (feature matching) and the final logits
      (the last element).

    Args:
      in_channels:   number of input channels (condition ⊕ image), e.g. ``C_ir + 3``.
      ndf:           base filter count.
      n_layers:      number of stride-2 conv stages (PatchGAN depth).
      use_spectral_norm: apply spectral norm to every conv.
      norm:          ``"instance"`` | ``"batch"`` | ``"none"`` for the inner norm layers.
    """

    name: str = "nlayer_patchgan"

    def __init__(
        self,
        in_channels: int = 4,
        ndf: int = 64,
        n_layers: int = 3,
        use_spectral_norm: bool = True,
        norm: str = "instance",
    ) -> None:
        super().__init__()  # type: ignore[misc]
        self.in_channels = int(in_channels)
        self.ndf = int(ndf)
        self.n_layers = int(n_layers)
        self.use_spectral_norm = bool(use_spectral_norm)

        if not TORCH_AVAILABLE:  # pragma: no cover
            self.stages = None
            return

        def _norm(num_features: int) -> "nn.Module":
            if norm == "batch":
                return nn.BatchNorm2d(num_features)
            if norm == "none":
                return nn.Identity()
            return nn.InstanceNorm2d(num_features, affine=False)

        kw, padw = 4, 2
        sn = self.use_spectral_norm
        # Each "stage" is an nn.Sequential; we keep them in a ModuleList so forward can
        # collect the activation after each stage (for feature matching).
        stages: List["nn.Module"] = []

        # Stage 0: no norm.
        stages.append(
            nn.Sequential(
                _maybe_spectral_norm(
                    nn.Conv2d(self.in_channels, ndf, kernel_size=kw, stride=2, padding=padw), sn
                ),
                nn.LeakyReLU(0.2, inplace=True),
            )
        )

        nf = ndf
        for _ in range(1, self.n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            stages.append(
                nn.Sequential(
                    _maybe_spectral_norm(
                        nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw), sn
                    ),
                    _norm(nf),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )

        # Penultimate stride-1 stage.
        nf_prev = nf
        nf = min(nf * 2, 512)
        stages.append(
            nn.Sequential(
                _maybe_spectral_norm(
                    nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw), sn
                ),
                _norm(nf),
                nn.LeakyReLU(0.2, inplace=True),
            )
        )

        # Final 1-channel patch-logit stage (no norm, no activation).
        stages.append(
            nn.Sequential(
                _maybe_spectral_norm(
                    nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw), sn
                )
            )
        )

        self.stages = nn.ModuleList(stages)

    def forward(self, x: Tensor) -> List[Tensor]:
        """Run the discriminator, returning per-stage activations.

        Args:
          x: ``FloatTensor [B, in_channels, H, W]`` (condition ⊕ image).

        Returns:
          ``List[Tensor]`` of length ``len(self.stages)``: the activation after each
          stage. The **last** element is the final patch-logit map ``[B, 1, h, w]``
          (raw logits); earlier elements are intermediate features for feature matching.
        """
        if not TORCH_AVAILABLE:  # pragma: no cover
            raise RuntimeError("torch is required to run NLayerDiscriminator.forward().")
        feats: List[Tensor] = []
        out = x
        for stage in self.stages:  # type: ignore[union-attr]
            out = stage(out)
            feats.append(out)
        return feats


class MultiScaleDiscriminator(BaseModel):
    """Pix2PixHD multi-scale PatchGAN: several :class:`NLayerDiscriminator` s over scales.

    The input is processed at ``num_scales`` resolutions; the full-resolution image is
    fed to discriminator 0, a ``×1/2`` average-pooled copy to discriminator 1, ``×1/4``
    to discriminator 2, and so on. Combining scales gives a large effective receptive
    field (coarse-scale D sees global structure; fine-scale D sees texture) while each
    sub-D stays shallow and fast.

    Args:
      in_channels:        channels of the (conditioned) input — condition ⊕ image.
      num_scales:         number of resolution scales / sub-discriminators.
      n_layers:           PatchGAN depth of each sub-discriminator.
      ndf:                base filter count of each sub-discriminator.
      use_spectral_norm:  apply spectral norm in every sub-discriminator.

    The convenience constructor :meth:`from_config` reads ``num_discriminators``,
    ``disc_n_layers``, ``disc_ndf`` and ``spectral_norm`` from a ``ColorizationConfig``.
    """

    name: str = "multiscale_patchgan"

    def __init__(
        self,
        in_channels: int = 4,
        num_scales: int = 3,
        n_layers: int = 3,
        ndf: int = 64,
        use_spectral_norm: bool = True,
    ) -> None:
        super().__init__()  # type: ignore[misc]
        self.in_channels = int(in_channels)
        self.num_scales = max(1, int(num_scales))
        self.n_layers = int(n_layers)
        self.ndf = int(ndf)
        self.use_spectral_norm = bool(use_spectral_norm)

        if not TORCH_AVAILABLE:  # pragma: no cover
            self.discriminators = None
            return

        self.discriminators = nn.ModuleList(
            [
                NLayerDiscriminator(
                    in_channels=self.in_channels,
                    ndf=self.ndf,
                    n_layers=self.n_layers,
                    use_spectral_norm=self.use_spectral_norm,
                )
                for _ in range(self.num_scales)
            ]
        )
        # Anti-aliased downsampler shared across scales (count-include-pad off).
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)

    @classmethod
    def from_config(
        cls,
        config: ColorizationConfig,
        in_channels: int = 4,
    ) -> "MultiScaleDiscriminator":
        """Build from a :class:`ColorizationConfig` (reads disc_* / spectral_norm)."""
        return cls(
            in_channels=int(in_channels),
            num_scales=int(config.num_discriminators),
            n_layers=int(config.disc_n_layers),
            ndf=int(config.disc_ndf),
            use_spectral_norm=bool(config.spectral_norm),
        )

    def forward(self, x: Tensor) -> List[List[Tensor]]:
        """Run every scale and return nested per-scale, per-layer activations.

        Args:
          x: ``FloatTensor [B, in_channels, H, W]`` — the (conditioned) input.

        Returns:
          ``List[List[Tensor]]`` of length ``num_scales``. Element ``s`` is the list of
          per-stage activations from sub-discriminator ``s`` (as returned by
          :meth:`NLayerDiscriminator.forward`): intermediate features followed by the
          final patch-logit map ``[B, 1, h_s, w_s]``. Use the last element of each inner
          list for the adversarial loss and the earlier elements for feature matching.
        """
        if not TORCH_AVAILABLE:  # pragma: no cover
            raise RuntimeError("torch is required to run MultiScaleDiscriminator.forward().")
        results: List[List[Tensor]] = []
        inp = x
        for i, disc in enumerate(self.discriminators):  # type: ignore[union-attr]
            if i > 0:
                inp = self.downsample(inp)
            results.append(disc(inp))
        return results
