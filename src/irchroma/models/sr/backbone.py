"""irchroma.models.sr.backbone — the shared NAFNet restoration backbone.

Owner: Builder-2 (Stage-1 super-resolution / restoration).

This module implements :class:`NAFNetBackbone`, a clean, real **NAFNet**-style
restoration network (Chen et al., *Simple Baselines for Image Restoration*,
ECCV 2022, https://github.com/megvii-research/NAFNet). It is the shared
restoration backbone referenced in ``ARCHITECTURE.md`` §5 (Stage-1 enhancement)
and recommended by ``docs/research/03-super-resolution.md`` §7/§12 as the
"sweet spot for combined sharpen + denoise at low compute".

Why NAFNet here
---------------
Thermal / IR is low-native-resolution and texture-poor; the dominant real
degradations are **blur + noise**, exactly what a restoration backbone removes.
NAFNet replaces all nonlinear activations (ReLU/GELU/sigmoid) with a
**SimpleGate** (channel-split + elementwise product) and a **Simplified Channel
Attention** (SCA — global-average-pool + 1x1 conv + multiply), which is both
cheaper and stronger than RCAN-style attention. The encoder-decoder uses
**pixel-unshuffle / pixel-shuffle** for down/up sampling (no learned strided
convs), keeping the field-of-view large while staying fast on tiny CPU tiles.

Tensor convention
------------------
``forward(x: Tensor) -> Tensor`` is a **same-resolution** restoration map::

    x : FloatTensor [B, C, H, W]   ->   y : FloatTensor [B, C, H, W]

i.e. it restores (denoises / sharpens) features or an image **without** changing
the spatial size. Upsampling by ``scale`` is the job of :class:`GuidedSR` in
``guided_sr.py`` (which consumes this backbone). When ``in_channels`` differs
from ``out_channels`` the network maps between them; with a single channel it
adds a global residual ``y = x + body(x)`` so it behaves as a true restorer.

The module is config-driven (width / enc / middle / dec block counts) and keeps
its defaults modest so the synthetic CPU demo on tiny tiles stays fast.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from ...config import SRConfig
from ...interfaces import TORCH_AVAILABLE, BaseModel, Tensor

# --------------------------------------------------------------------------- #
# Guarded torch import. Like every irchroma module, this file must IMPORT even
# when torch is unavailable (docs/CI boxes); the classes are only *constructed*
# on a real runtime where torch is present.
# --------------------------------------------------------------------------- #
if TORCH_AVAILABLE:  # pragma: no cover - exercised only with torch installed
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
else:  # pragma: no cover - torch-less import shim
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore


# =========================================================================== #
# Primitive NAFNet building blocks
# =========================================================================== #
if TORCH_AVAILABLE:  # pragma: no cover - definitions require torch.nn

    class LayerNorm2d(nn.Module):
        """Channels-first LayerNorm over the channel dimension of an NCHW tensor.

        Equivalent to ``nn.LayerNorm`` applied per spatial location across the ``C``
        channels, but implemented directly on ``[B, C, H, W]`` so it composes with
        convolutional blocks without permute round-trips. This is the normalization
        NAFNet uses at the head of every block.
        """

        def __init__(self, channels: int, eps: float = 1e-6) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(channels))
            self.bias = nn.Parameter(torch.zeros(channels))
            self.eps = float(eps)

        def forward(self, x: Tensor) -> Tensor:
            # Normalize over the channel axis (dim=1) per pixel.
            mu = x.mean(dim=1, keepdim=True)
            var = x.var(dim=1, keepdim=True, unbiased=False)
            x_hat = (x - mu) / torch.sqrt(var + self.eps)
            return x_hat * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)

    class SimpleGate(nn.Module):
        """NAFNet SimpleGate: split channels in half and multiply the halves.

        ``SimpleGate([a, b]) = a * b`` where ``a, b`` are the two channel halves.
        This replaces every nonlinear activation in the network (no ReLU/GELU),
        halving the channel count of its input.
        """

        def forward(self, x: Tensor) -> Tensor:
            x1, x2 = x.chunk(2, dim=1)
            return x1 * x2

    class SimplifiedChannelAttention(nn.Module):
        """NAFNet Simplified Channel Attention (SCA).

        ``y = x * conv1x1(global_avg_pool(x))`` — a content-adaptive per-channel
        gate with no nonlinearity (cf. RCAN/SE which use ReLU+sigmoid). O(C^2) and
        cheap; the global pool gives a full-image receptive field for free.
        """

        def __init__(self, channels: int) -> None:
            super().__init__()
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.conv = nn.Conv2d(channels, channels, kernel_size=1, stride=1,
                                  padding=0, bias=True)

        def forward(self, x: Tensor) -> Tensor:
            return x * self.conv(self.pool(x))

    class NAFBlock(nn.Module):
        """A single NAFNet block (the core repeated unit of the backbone).

        Two residual sub-blocks, each pre-normalized by :class:`LayerNorm2d`:

          1. **Spatial / gated conv** branch::

                 LN -> 1x1 conv (expand x2) -> 3x3 depthwise conv
                    -> SimpleGate -> SCA -> 1x1 conv (project) -> * beta  (+ residual)

          2. **Channel MLP** branch (FFN)::

                 LN -> 1x1 conv (expand x2) -> SimpleGate -> 1x1 conv (project)
                    -> * gamma  (+ residual)

        ``beta`` / ``gamma`` are learnable per-channel residual scales (NAFNet's
        "LayerScale"-style trainable skip weights), initialized to zero so the block
        starts as an identity and training is stable. ``dw_expand`` / ``ffn_expand``
        are the channel-expansion factors before each SimpleGate (which then halves
        them), matching the reference implementation.
        """

        def __init__(
            self,
            channels: int,
            dw_expand: int = 2,
            ffn_expand: int = 2,
            drop_out_rate: float = 0.0,
        ) -> None:
            super().__init__()
            dw_channels = channels * dw_expand
            # --- Branch 1: gated depthwise conv + SCA ---
            self.norm1 = LayerNorm2d(channels)
            self.conv1 = nn.Conv2d(channels, dw_channels, kernel_size=1, stride=1,
                                   padding=0, bias=True)
            self.conv2 = nn.Conv2d(dw_channels, dw_channels, kernel_size=3, stride=1,
                                   padding=1, groups=dw_channels, bias=True)
            self.gate1 = SimpleGate()  # halves dw_channels -> dw_channels // 2
            self.sca = SimplifiedChannelAttention(dw_channels // 2)
            self.conv3 = nn.Conv2d(dw_channels // 2, channels, kernel_size=1, stride=1,
                                   padding=0, bias=True)

            # --- Branch 2: channel MLP / FFN ---
            ffn_channels = channels * ffn_expand
            self.norm2 = LayerNorm2d(channels)
            self.conv4 = nn.Conv2d(channels, ffn_channels, kernel_size=1, stride=1,
                                   padding=0, bias=True)
            self.gate2 = SimpleGate()  # halves ffn_channels -> ffn_channels // 2
            self.conv5 = nn.Conv2d(ffn_channels // 2, channels, kernel_size=1, stride=1,
                                   padding=0, bias=True)

            # Optional regularization (off by default for the demo).
            self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
            self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()

            # Learnable residual scales, zero-init -> block starts as identity.
            self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
            self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

        def forward(self, x: Tensor) -> Tensor:
            # Branch 1: gated conv + simplified channel attention.
            y = self.norm1(x)
            y = self.conv1(y)
            y = self.conv2(y)
            y = self.gate1(y)
            y = self.sca(y)
            y = self.conv3(y)
            y = self.dropout1(y)
            x = x + y * self.beta

            # Branch 2: channel MLP.
            y = self.norm2(x)
            y = self.conv4(y)
            y = self.gate2(y)
            y = self.conv5(y)
            y = self.dropout2(y)
            x = x + y * self.gamma
            return x


# =========================================================================== #
# NAFNetBackbone — encoder / middle / decoder with pixel-(un)shuffle sampling
# =========================================================================== #
class NAFNetBackbone(BaseModel):
    """NAFNet-style encoder-decoder restoration backbone (same-resolution output).

    This is the shared restoration backbone of Stage-1 (``ARCHITECTURE.md`` §5.2,
    "Restoration backbone"). It restores / sharpens / denoises its input at the
    **same** spatial resolution; the SR upsampler lives in :class:`GuidedSR`.

    Structure (a symmetric U-Net of :class:`NAFBlock` s)::

        in_conv (3x3)               # C_in -> width
        for each encoder stage i:
            enc_blocks[i] x NAFBlock # at width * 2**i
            down  = PixelUnshuffle(2) + 1x1 conv   # halve H,W; double channels
        middle_blocks x NAFBlock     # at the bottleneck width
        for each decoder stage i (reversed):
            up    = 1x1 conv + PixelShuffle(2)     # double H,W; halve channels
            x = x + skip[i]          # additive long skip (NAFNet style)
            dec_blocks[i] x NAFBlock
        out_conv (3x3)              # width -> C_out
        y = out_conv(...) ; return y (+ x_in if C_in == C_out)

    The number of down/up steps equals ``len(enc_blocks)`` (== ``len(dec_blocks)``).
    A global residual is added when ``in_channels == out_channels`` so the network
    predicts a restoration *residual* (standard for SR/denoise and easier to train).

    Args:
        in_channels:   input channel count ``C_in`` (default: IR channels).
        out_channels:  output channel count ``C_out`` (default: same as input).
        width:         base feature width at the top (full-resolution) stage.
        enc_blocks:    NAFBlock count per encoder stage (len == #downsample steps).
        middle_blocks: NAFBlock count at the bottleneck.
        dec_blocks:    NAFBlock count per decoder stage (must match ``enc_blocks`` len).
        dw_expand:     depthwise channel-expansion factor inside each NAFBlock.
        ffn_expand:    FFN channel-expansion factor inside each NAFBlock.

    Shape:
        - input:  ``[B, in_channels, H, W]``
        - output: ``[B, out_channels, H, W]``

    .. note::
        H and W must each be divisible by ``2 ** len(enc_blocks)`` for the
        pixel-unshuffle pyramid to be exact. :meth:`forward` pads the input to the
        nearest valid multiple and crops the result back, so arbitrary tile sizes
        are accepted transparently.
    """

    name: str = "nafnet_backbone"

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: Optional[int] = None,
        width: int = 32,
        enc_blocks: Optional[Sequence[int]] = None,
        middle_blocks: int = 2,
        dec_blocks: Optional[Sequence[int]] = None,
        dw_expand: int = 2,
        ffn_expand: int = 2,
    ) -> None:
        super().__init__()
        if not TORCH_AVAILABLE:  # pragma: no cover - cannot build modules torch-less
            raise RuntimeError(
                "NAFNetBackbone requires torch. Install torch to instantiate the model."
            )

        out_channels = int(in_channels if out_channels is None else out_channels)
        enc_list: List[int] = list(enc_blocks if enc_blocks is not None else [1, 1, 1])
        dec_list: List[int] = list(dec_blocks if dec_blocks is not None else [1, 1, 1])
        if len(enc_list) != len(dec_list):
            raise ValueError(
                f"enc_blocks ({len(enc_list)}) and dec_blocks ({len(dec_list)}) "
                "must have the same length (one entry per up/down sampling step)."
            )

        self.in_channels = int(in_channels)
        self.out_channels = out_channels
        self.width = int(width)
        self.num_stages = len(enc_list)

        # The PixelUnshuffle downsample (conv to chan//2 then unshuffle x2 -> chan*2)
        # is only exactly invertible when ``chan`` stays even at every stage. The
        # channel count doubles each stage, so it suffices that ``width`` is even.
        if self.width % 2 != 0:
            raise ValueError(
                f"width must be even for the pixel-(un)shuffle pyramid; got {self.width}."
            )

        # Head / tail 3x3 convs.
        self.intro = nn.Conv2d(self.in_channels, self.width, kernel_size=3, stride=1,
                               padding=1, bias=True)
        self.ending = nn.Conv2d(self.width, self.out_channels, kernel_size=3, stride=1,
                                padding=1, bias=True)

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()

        chan = self.width
        # ---- Encoder: blocks then downsample (PixelUnshuffle doubles channels) ---
        for n_blocks in enc_list:
            self.encoders.append(
                nn.Sequential(*[
                    NAFBlock(chan, dw_expand=dw_expand, ffn_expand=ffn_expand)
                    for _ in range(int(n_blocks))
                ])
            )
            # 1x1 conv to chan*2 followed by PixelUnshuffle(2): H,W -> H/2,W/2 and
            # channels chan*2 -> chan*2*4. To end at chan*2 we instead conv to
            # chan//2 then unshuffle by 2 (chan//2 * 4 = chan*2). Reference NAFNet
            # uses ``Conv2d(c, 2c, 2, 2)``; we use the unshuffle formulation for an
            # exactly-invertible, parameter-light downsample.
            self.downs.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan // 2, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.PixelUnshuffle(2),
                )
            )
            chan = chan * 2

        # ---- Middle (bottleneck) blocks ----
        self.middle = nn.Sequential(*[
            NAFBlock(chan, dw_expand=dw_expand, ffn_expand=ffn_expand)
            for _ in range(int(middle_blocks))
        ])

        # ---- Decoder: upsample (PixelShuffle halves channels) then blocks ----
        for n_blocks in dec_list:
            # 1x1 conv to chan*2 then PixelShuffle(2): channels chan*2 -> chan*2/4
            # and H,W -> 2H,2W. chan*2/4 = chan/2, so we end one stage shallower.
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.PixelShuffle(2),
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(*[
                    NAFBlock(chan, dw_expand=dw_expand, ffn_expand=ffn_expand)
                    for _ in range(int(n_blocks))
                ])
            )

        #: Spatial multiple the input must be divisible by (auto-padded in forward).
        self.size_divisor = 2 ** self.num_stages

    # ------------------------------------------------------------------ #
    @classmethod
    def from_config(cls, cfg: SRConfig) -> "NAFNetBackbone":
        """Build a :class:`NAFNetBackbone` from an :class:`~irchroma.config.SRConfig`.

        Reads ``width``, ``enc_blocks``, ``middle_blocks``, ``dec_blocks`` and the
        IR channel count (``in_channels``) from the config. The backbone is
        same-channel by default (restoration), so ``out_channels == in_channels``.
        """
        return cls(
            in_channels=int(cfg.in_channels),
            out_channels=int(cfg.in_channels),
            width=int(cfg.width),
            enc_blocks=list(cfg.enc_blocks),
            middle_blocks=int(cfg.middle_blocks),
            dec_blocks=list(cfg.dec_blocks),
        )

    # ------------------------------------------------------------------ #
    def _pad_to_multiple(self, x: Tensor) -> Tensor:
        """Reflect-pad ``x`` so H and W are multiples of ``self.size_divisor``."""
        _, _, h, w = x.shape
        d = self.size_divisor
        pad_h = (d - h % d) % d
        pad_w = (d - w % d) % d
        if pad_h == 0 and pad_w == 0:
            return x
        # Pad on right/bottom; reflect to avoid border artifacts.
        return F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

    def forward(self, x: Tensor) -> Tensor:
        """Restore ``x`` at the same spatial resolution.

        Args:
            x: ``FloatTensor [B, in_channels, H, W]``.

        Returns:
            ``FloatTensor [B, out_channels, H, W]`` — the restored map (with a global
            residual added when ``in_channels == out_channels``).
        """
        _, _, h0, w0 = x.shape
        inp = x
        x = self._pad_to_multiple(x)

        feat = self.intro(x)

        skips: List[Tensor] = []
        for encoder, down in zip(self.encoders, self.downs):
            feat = encoder(feat)
            skips.append(feat)
            feat = down(feat)

        feat = self.middle(feat)

        # Decode with additive long skips (NAFNet uses ``+``, not concat).
        for decoder, up, skip in zip(self.decoders, self.ups, reversed(skips)):
            feat = up(feat)
            feat = feat + skip
            feat = decoder(feat)

        out = self.ending(feat)

        # Global residual when channel counts match (predict a restoration residual).
        if self.out_channels == self.in_channels:
            out = out + x

        # Crop back to the original (pre-pad) resolution.
        return out[:, :, :h0, :w0]


__all__ = ["NAFNetBackbone"]
