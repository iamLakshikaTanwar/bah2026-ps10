"""irchroma.models.sr.guided_sr — the primary Stage-1 guided SR model.

Owner: Builder-2 (Stage-1 super-resolution / restoration).

This module implements :class:`GuidedSR`, the **primary** Stage-1 model of the
irchroma pipeline (``ARCHITECTURE.md`` §5, ``docs/research/03-super-resolution.md``
§0/§6/§12). It performs **guided cross-sensor super-resolution**: a coarse
single-channel IR band is super-resolved by ``scale`` while a *co-registered*
higher-resolution guide band (Landsat 15 m panchromatic / Sentinel-2 10 m) injects
**real** high-frequency structure.

Rationale (the key design decision)
-----------------------------------
Low-native-resolution thermal IR holds very little *real* high-frequency content,
so pure single-image SR (SISR) **must hallucinate** texture — which violates the
project's no-fake-objects rule. Supplying a genuine HR guide turns invention into
fusion: the network borrows structure from a measured band instead of dreaming it
up. This yields higher fidelity *and* lower hallucination (IR-SR survey + PBVS
thermal-SR consensus). When no guide is available the model degrades **gracefully**
to single-image SR (still useful, just without the structural anchor).

Architecture
------------
::

    ir   [B, C_ir, H, W] ─► [IR backbone: NAFNetBackbone] ─┐
                                                            ├─► [fuse] ─► [PixelShuffle xS] ─► sr [B, C_ir, H*S, W*S]
    guide[B, C_g, Hg,Wg] ─► [resize→H,W] ─► [guide encoder]─┘

  * **IR branch**     — a same-resolution :class:`NAFNetBackbone` restores the IR.
  * **guidance branch** — a small conv encoder embeds the HR guide; the guide is
    first resized to the IR grid (``H, W``) so features align spatially.
  * **fusion** (``SRConfig.guide_fusion``) injects guide features into the IR
    features. Two modes are implemented:
      - ``"concat"``           : channel-concatenate + 1x1 conv (cheap, robust).
      - ``"cross_attention"``  : a lightweight 1x1 cross-attention where the IR
                                 features query the guide features (default).
    A learnable residual gate keeps fusion stable and lets the model fall back to
    the IR-only path when the guide is unhelpful / absent.
  * **upsampler** — ESPCN-style **pixel-shuffle** in LR space (``SRConfig.upsampler``):
    a conv expands channels by ``C_ir * scale**2`` and ``PixelShuffle(scale)``
    rearranges them to the SR grid. A bilinearly-upsampled IR skip is added so the
    network predicts an SR *residual* on top of a sane base (anti-hallucination,
    easy to train).

Tensor convention (matches ``irchroma.interfaces``)
---------------------------------------------------
``forward(ir, guide=None) -> {"sr": Tensor, "feat": Tensor}`` (a dict) by default::

    ir    : FloatTensor [B, C_ir, H, W]        (~[0,1])
    guide : FloatTensor [B, C_g, Hg, Wg]  or None
    sr    : FloatTensor [B, C_ir, H*scale, W*scale]
    feat  : FloatTensor [B, width,  H,       W      ]   (LR restoration features)

The dict lets the downstream Stage-2 colorizer reuse the shared restoration
``feat`` (``ModelConfig.feature_feedback`` / ``share_encoder``). For callers that
only want the tensor, pass ``return_dict=False`` to get the ``sr`` tensor directly,
or read ``out["sr"]``.
"""

from __future__ import annotations

from typing import Dict, Optional, Union

from ...config import SRConfig
from ...interfaces import TORCH_AVAILABLE, BaseModel, Tensor
from .backbone import NAFNetBackbone

if TORCH_AVAILABLE:  # pragma: no cover - exercised only with torch installed
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
else:  # pragma: no cover - torch-less import shim
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore


# =========================================================================== #
# Guidance encoder + fusion blocks
# =========================================================================== #
if TORCH_AVAILABLE:  # pragma: no cover - definitions require torch.nn

    class _ConvBlock(nn.Module):
        """A small ``Conv -> GELU -> Conv`` residual block (guide encoder unit)."""

        def __init__(self, channels: int) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
            self.act = nn.GELU()
            self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)

        def forward(self, x: Tensor) -> Tensor:
            return x + self.conv2(self.act(self.conv1(x)))

    class GuidanceEncoder(nn.Module):
        """Encode the (resized) HR guide band into ``width`` feature channels.

        Operates at the IR resolution (the guide is resized to ``H, W`` before this
        runs), so its output aligns pixel-for-pixel with the IR features and can be
        fused directly. Modest depth keeps it cheap on tiny CPU tiles.
        """

        def __init__(self, in_channels: int, width: int, num_blocks: int = 2) -> None:
            super().__init__()
            self.proj = nn.Conv2d(in_channels, width, kernel_size=3, padding=1, bias=True)
            self.body = nn.Sequential(*[_ConvBlock(width) for _ in range(int(num_blocks))])

        def forward(self, x: Tensor) -> Tensor:
            return self.body(self.proj(x))

    class ConcatFusion(nn.Module):
        """Fuse IR + guide features by channel-concat then a 1x1 projection.

        ``fused = ir_feat + gate * conv1x1([ir_feat ; guide_feat])`` — a gated
        residual so the guide can only *add* structure on top of the IR features,
        and the learnable ``gate`` (zero-init) lets training discover how much guide
        to trust (and ignore it entirely when absent).
        """

        def __init__(self, width: int) -> None:
            super().__init__()
            self.fuse = nn.Conv2d(width * 2, width, kernel_size=1, bias=True)
            self.gate = nn.Parameter(torch.zeros(1, width, 1, 1))

        def forward(self, ir_feat: Tensor, guide_feat: Tensor) -> Tensor:
            merged = self.fuse(torch.cat([ir_feat, guide_feat], dim=1))
            return ir_feat + self.gate * merged

    class CrossAttentionFusion(nn.Module):
        """Lightweight 1x1 cross-attention: IR features attend to guide features.

        A convolutional (per-pixel channel-gating) attention rather than full
        spatial self-attention, so it is cheap and tiles well. The IR features form
        the query; the guide features form key/value. Within each head, a softmax
        over the head's channels turns the query-key affinity into a per-channel
        weight that re-weights the guide value channels at that pixel::

            q = Wq(ir_feat) ; k = Wk(guide_feat) ; v = Wv(guide_feat)
            attn = softmax(q ⊙ k / sqrt(d), over head_dim)   # [B, heads, head_dim, H, W]
            out  = Wo(attn ⊙ v)
            fused = ir_feat + gate * out

        This injects guide high-frequency structure where it correlates with the IR
        content, with a zero-init residual gate for stable, fall-back-safe training.
        """

        def __init__(self, width: int, heads: int = 4) -> None:
            super().__init__()
            # Keep heads compatible with the width.
            while width % heads != 0 and heads > 1:
                heads -= 1
            self.heads = heads
            self.head_dim = width // heads
            self.scale = float(self.head_dim) ** -0.5
            self.to_q = nn.Conv2d(width, width, kernel_size=1, bias=True)
            self.to_k = nn.Conv2d(width, width, kernel_size=1, bias=True)
            self.to_v = nn.Conv2d(width, width, kernel_size=1, bias=True)
            self.proj = nn.Conv2d(width, width, kernel_size=1, bias=True)
            self.gate = nn.Parameter(torch.zeros(1, width, 1, 1))

        def forward(self, ir_feat: Tensor, guide_feat: Tensor) -> Tensor:
            b, c, h, w = ir_feat.shape
            q = self.to_q(ir_feat).view(b, self.heads, self.head_dim, h, w)
            k = self.to_k(guide_feat).view(b, self.heads, self.head_dim, h, w)
            v = self.to_v(guide_feat).view(b, self.heads, self.head_dim, h, w)
            # Per-pixel channel attention within each head: softmax over head_dim so
            # each guide value channel gets a query-conditioned weight at that pixel.
            attn = torch.softmax(q * k * self.scale, dim=2)
            out = (attn * v).reshape(b, c, h, w)
            out = self.proj(out)
            return ir_feat + self.gate * out

    class PixelShuffleUpsampler(nn.Module):
        """ESPCN-style pixel-shuffle upsampler from feature space to the SR grid.

        ``conv(width -> out_ch * scale**2) -> PixelShuffle(scale)`` produces the SR
        residual; non-power-of-two scales are handled by a single shuffle with the
        exact ``scale`` factor (PixelShuffle supports any integer upscale).
        """

        def __init__(self, width: int, out_channels: int, scale: int) -> None:
            super().__init__()
            self.scale = int(scale)
            self.out_channels = int(out_channels)
            self.conv = nn.Conv2d(width, out_channels * (scale ** 2), kernel_size=3,
                                  padding=1, bias=True)
            self.shuffle = nn.PixelShuffle(scale)

        def forward(self, x: Tensor) -> Tensor:
            return self.shuffle(self.conv(x))


# =========================================================================== #
# GuidedSR — the primary Stage-1 model
# =========================================================================== #
class GuidedSR(BaseModel):
    """Guided cross-sensor super-resolution of IR (primary Stage-1 model).

    Combines an IR restoration backbone (:class:`NAFNetBackbone`) with an HR-guide
    fusion branch and a pixel-shuffle upsampler to produce SR IR at ``scale`` x the
    input resolution. Degrades gracefully to single-image SR when ``guide is None``.

    Args:
        scale:          super-resolution factor (``SRConfig.scale``).
        in_channels:    IR input channels ``C_ir`` (``SRConfig.in_channels``).
        out_channels:   SR output channels (keeps the IR channel count;
                        ``SRConfig.out_channels``).
        width:          base feature width (``SRConfig.width``).
        guide_channels: HR guide channels ``C_g`` (``SRConfig.guide_channels``).
        use_guide:      whether to build the guidance branch (``SRConfig.use_guide``).
        guide_fusion:   fusion mode, ``{"concat", "cross_attention"}``
                        (``SRConfig.guide_fusion``; unknown values fall back to concat).
        enc_blocks/middle_blocks/dec_blocks:
                        NAFNet backbone depth (``SRConfig.*``).

    Shape:
        - ``ir``    : ``[B, in_channels, H, W]``
        - ``guide`` : ``[B, guide_channels, Hg, Wg]`` or ``None``
        - ``sr``    : ``[B, out_channels, H*scale, W*scale]``
        - ``feat``  : ``[B, width, H, W]`` (shared LR restoration features)

    Returns (from :meth:`forward`):
        ``{"sr": sr, "feat": feat}`` by default (``return_dict=True``), or the ``sr``
        tensor alone when ``return_dict=False``.
    """

    name: str = "guided_sr"

    def __init__(
        self,
        scale: int = 4,
        in_channels: int = 1,
        out_channels: Optional[int] = None,
        width: int = 32,
        guide_channels: int = 1,
        use_guide: bool = True,
        guide_fusion: str = "cross_attention",
        enc_blocks: Optional[list] = None,
        middle_blocks: int = 2,
        dec_blocks: Optional[list] = None,
    ) -> None:
        super().__init__()
        if not TORCH_AVAILABLE:  # pragma: no cover - cannot build modules torch-less
            raise RuntimeError(
                "GuidedSR requires torch. Install torch to instantiate the model."
            )

        out_channels = int(in_channels if out_channels is None else out_channels)
        self.scale = int(scale)
        self.in_channels = int(in_channels)
        self.out_channels = out_channels
        self.width = int(width)
        self.guide_channels = int(guide_channels)
        self.use_guide = bool(use_guide)
        self.guide_fusion_mode = str(guide_fusion)

        # IR restoration backbone (same-resolution, maps C_ir -> width features).
        # We output ``width`` channels so the LR features can be reused by Stage-2.
        self.ir_backbone = NAFNetBackbone(
            in_channels=self.in_channels,
            out_channels=self.width,
            width=self.width,
            enc_blocks=enc_blocks,
            middle_blocks=middle_blocks,
            dec_blocks=dec_blocks,
        )

        # Guidance branch + fusion (only if guidance is enabled).
        if self.use_guide:
            self.guide_encoder = GuidanceEncoder(self.guide_channels, self.width)
            if self.guide_fusion_mode == "cross_attention":
                self.fusion = CrossAttentionFusion(self.width)
            else:
                # ``concat`` and any unrecognized mode (e.g. "deformable") fall back
                # to the robust concat fusion so the model always builds.
                self.fusion = ConcatFusion(self.width)
        else:
            self.guide_encoder = None
            self.fusion = None

        # A couple of refinement blocks after fusion before upsampling.
        self.refine = nn.Sequential(
            nn.Conv2d(self.width, self.width, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(self.width, self.width, kernel_size=3, padding=1, bias=True),
        )

        # ESPCN-style pixel-shuffle upsampler -> SR residual on the HR grid.
        self.upsampler = PixelShuffleUpsampler(self.width, self.out_channels, self.scale)

    # ------------------------------------------------------------------ #
    @classmethod
    def from_config(cls, cfg: SRConfig) -> "GuidedSR":
        """Build a :class:`GuidedSR` from an :class:`~irchroma.config.SRConfig`."""
        return cls(
            scale=int(cfg.scale),
            in_channels=int(cfg.in_channels),
            out_channels=int(cfg.out_channels),
            width=int(cfg.width),
            guide_channels=int(cfg.guide_channels),
            use_guide=bool(cfg.use_guide),
            guide_fusion=str(cfg.guide_fusion),
            enc_blocks=list(cfg.enc_blocks),
            middle_blocks=int(cfg.middle_blocks),
            dec_blocks=list(cfg.dec_blocks),
        )

    # ------------------------------------------------------------------ #
    def _base_upsample(self, ir: Tensor) -> Tensor:
        """Bilinear-upsample the IR input to the SR grid (the residual base).

        Maps ``C_ir`` to ``out_channels`` if they differ (channel-mean / repeat) so
        the SR head only has to predict the high-frequency *residual* over a sane,
        non-hallucinated base image. This is a strong anti-hallucination prior.
        """
        base = F.interpolate(
            ir, scale_factor=self.scale, mode="bilinear", align_corners=False
        )
        if self.out_channels == self.in_channels:
            return base
        if self.in_channels == 1:
            # Broadcast the single IR channel across all output channels.
            return base.repeat(1, self.out_channels, 1, 1)
        # General case: collapse to a mean then broadcast (rare; C_ir>1, C_out!=C_ir).
        return base.mean(dim=1, keepdim=True).repeat(1, self.out_channels, 1, 1)

    def forward(
        self,
        ir: Tensor,
        guide: Optional[Tensor] = None,
        return_dict: bool = True,
    ) -> Union[Dict[str, Tensor], Tensor]:
        """Super-resolve ``ir`` by ``scale``, optionally guided by ``guide``.

        Args:
            ir:          ``FloatTensor [B, C_ir, H, W]`` — the low-res IR (~[0,1]).
            guide:       ``FloatTensor [B, C_g, Hg, Wg]`` co-registered HR guide, or
                         ``None`` to run single-image SR.
            return_dict: if ``True`` (default) return ``{"sr": ..., "feat": ...}``;
                         if ``False`` return only the ``sr`` tensor.

        Returns:
            ``{"sr": [B, C_ir, H*scale, W*scale], "feat": [B, width, H, W]}`` or, when
            ``return_dict=False``, the ``sr`` tensor.
        """
        _, _, h, w = ir.shape

        # 1) IR restoration features at the LR resolution.
        feat = self.ir_backbone(ir)  # [B, width, H, W]

        # 2) Guidance fusion (graceful fallback when no guide / branch disabled).
        if self.use_guide and self.fusion is not None and guide is not None:
            # Align the guide to the IR grid (handles any Hg, Wg).
            if guide.shape[-2:] != (h, w):
                guide = F.interpolate(
                    guide, size=(h, w), mode="bilinear", align_corners=False
                )
            guide_feat = self.guide_encoder(guide)  # [B, width, H, W]
            feat = self.fusion(feat, guide_feat)

        # 3) Refine, then upsample to the SR residual.
        refined = self.refine(feat)
        residual = self.upsampler(refined)  # [B, out_channels, H*scale, W*scale]

        # 4) SR = bilinear base + predicted high-frequency residual.
        sr = self._base_upsample(ir) + residual

        if return_dict:
            return {"sr": sr, "feat": feat}
        return sr


__all__ = ["GuidedSR"]
