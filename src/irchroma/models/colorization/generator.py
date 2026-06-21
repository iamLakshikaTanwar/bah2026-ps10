"""irchroma.models.colorization.generator — Stage-2 IR→RGB primary generator.

Owner: ColorBuilder (Builder-3). Implements the **primary** colorization model:
a **Pix2PixHD-style coarse-to-fine** encoder→residual-blocks→decoder generator that
takes the (super-resolved) IR ``[B, C_ir, Hs, Ws]`` and an optional per-pixel land-
cover ``semantic`` map, conditions the decoder via **SPADEResBlocks** (when
``use_spade`` *and* a semantic map are provided; otherwise plain InstanceNorm/BN
residual blocks), and finishes with a **DDColor-style query-based color decoder head**
that predicts chroma. The output is a full RGB image ``[B, 3, Hs, Ws]`` in ``[0, 1]``.

Tensor conventions (match :mod:`irchroma.interfaces`):
  * input  ``ir``      : ``FloatTensor [B, C_ir, Hs, Ws]`` — already super-resolved /
                         at the target resolution (Stage-2 operates at SR scale, so this
                         module does **not** upsample; H_out == H_in).
  * input  ``semantic``: ``LongTensor  [B, Hs, Ws]`` of LULC indices, or ``None``.
  * output ``rgb``     : ``FloatTensor [B, 3, Hs, Ws]`` in ``[0, 1]`` (sRGB, R,G,B).

------------------------------------------------------------------------------
Color-space choice (Lab vs RGB) — documented per the contract
------------------------------------------------------------------------------
``ColorizationConfig.output_space`` selects how color is parameterized *inside* the
net; the public ``forward`` always returns **RGB in [0, 1]** regardless:

  * ``output_space == "lab"`` (default, recommended): the network predicts a CIE-L*a*b*
    image — **L** (luminance) from the spatial *pixel* decoder, and the **a*, b*** chroma
    planes from the DDColor query color head. We then convert Lab→sRGB in-module. This is
    the design that lets the downstream **O(1) class→Lab color-LUT clamp chroma (a,b)
    while leaving L freer** (applied *later* by the semantic/pipeline stage, NOT here),
    so colors stay class-correct while SR texture/detail survives. We expose the raw Lab
    tensor on the module (``self.last_lab``) so the pipeline can clamp+convert if it
    prefers; the default ``forward`` returns the already-converted RGB for convenience.
  * ``output_space == "rgb"``: the head predicts RGB directly (final sigmoid → [0,1]).

The class-conditioned **ColorLUT** (``ColorLUTProtocol``) and the learned 3D-LUT are
**not** applied in this module — they are a separate, later O(1) refinement owned by the
semantic builder / pipeline. This generator only produces the *raw* colorized RGB.

torch is assumed present at runtime; this file must still ``py_compile`` without it.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

from irchroma.config import ColorizationConfig, NUM_LULC_CLASSES
from irchroma.interfaces import BaseModel, TORCH_AVAILABLE, Tensor

# --------------------------------------------------------------------------- #
# Guarded torch import: the module must import even on a torch-less box.
# --------------------------------------------------------------------------- #
if TORCH_AVAILABLE:  # pragma: no cover - exercised only where torch exists
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
else:  # pragma: no cover - torch-less docs/CI environment
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore


__all__ = ["ColorizationGenerator"]


# =========================================================================== #
# Small building blocks (plain, config-light).
# =========================================================================== #
def _norm_layer(num_features: int, norm: str = "instance") -> "nn.Module":
    """Return a 2D normalization layer by name (``instance`` | ``batch`` | ``none``)."""
    if norm == "batch":
        return nn.BatchNorm2d(num_features)
    if norm == "none":
        return nn.Identity()
    # default: InstanceNorm (Pix2PixHD default; robust on flat satellite masks)
    return nn.InstanceNorm2d(num_features, affine=False)


class ResnetBlock(nn.Module if TORCH_AVAILABLE else object):  # type: ignore[misc]
    """Standard Pix2PixHD residual block (reflection-pad → conv → norm → ReLU ×2).

    Used as the **fallback** decoder/bottleneck block when SPADE is unavailable or no
    semantic map is supplied. Channel count is preserved (identity skip).
    """

    def __init__(self, dim: int, norm: str = "instance") -> None:
        super().__init__()  # type: ignore[misc]
        layers: List["nn.Module"] = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0),
            _norm_layer(dim, norm),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0),
            _norm_layer(dim, norm),
        ]
        self.conv_block = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Return ``x + conv_block(x)`` (shape-preserving residual)."""
        return x + self.conv_block(x)


# =========================================================================== #
# Optional SPADE residual block (lazy/guarded import; SemanticBuilder owns it).
# =========================================================================== #
def _try_import_spade_resblock() -> Optional[Any]:
    """Late, guarded import of ``SPADEResBlock`` from the semantic package.

    The semantic builder owns ``irchroma.models.semantic.spade`` (producing ``SPADE``
    and ``SPADEResBlock``). We import it **lazily** inside the generator so this module
    still ``py_compile`` s and imports even while that file is mid-build (or absent).
    Returns the ``SPADEResBlock`` class, or ``None`` if it cannot be imported.
    """
    try:
        from irchroma.models.semantic.spade import SPADEResBlock  # type: ignore

        return SPADEResBlock
    except Exception:  # pragma: no cover - semantic module mid-build / missing
        return None


# =========================================================================== #
# DDColor-style query-based color decoder head.
# =========================================================================== #
class ColorQueryDecoder(nn.Module if TORCH_AVAILABLE else object):  # type: ignore[misc]
    """Minimal DDColor-style query-based color decoder.

    A set of ``num_queries`` learnable **color queries** attend (cross-attention) to the
    flattened multi-scale image features, are refined by a few transformer decoder
    layers, then projected to a per-query color basis. The decoded queries are correlated
    against a per-pixel feature embedding to produce a spatial chroma map — this is the
    mechanism that enforces "water→blue, veg→green" while suppressing **color bleeding**
    (each query specializes to a semantic-color mode), in a single deterministic pass.

    Produces ``out_chroma`` channels (2 for Lab a*,b*; 3 for direct RGB).
    """

    def __init__(
        self,
        feat_dim: int,
        num_queries: int = 100,
        num_layers: int = 3,
        num_heads: int = 4,
        out_chroma: int = 2,
    ) -> None:
        super().__init__()  # type: ignore[misc]
        self.feat_dim = int(feat_dim)
        self.num_queries = int(num_queries)
        self.out_chroma = int(out_chroma)

        # Learnable color queries [num_queries, feat_dim].
        self.color_queries = nn.Parameter(torch.randn(self.num_queries, self.feat_dim) * 0.02)

        # A stack of transformer decoder layers (queries attend to image tokens).
        self.layers = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    d_model=self.feat_dim,
                    nhead=num_heads,
                    dim_feedforward=self.feat_dim * 2,
                    dropout=0.0,
                    batch_first=True,
                )
                for _ in range(max(1, int(num_layers)))
            ]
        )
        # Project refined queries to a color basis embedding (same dim as pixel embed).
        self.query_proj = nn.Linear(self.feat_dim, self.feat_dim)
        # Per-pixel embedding head: maps image features -> feat_dim for query correlation.
        self.pixel_embed = nn.Conv2d(self.feat_dim, self.feat_dim, kernel_size=1)
        # Map the num_queries correlation maps -> chroma channels.
        self.to_chroma = nn.Conv2d(self.num_queries, self.out_chroma, kernel_size=1)

    def forward(self, feats: Tensor) -> Tensor:
        """Predict a chroma map from spatial features.

        Args:
          feats: ``FloatTensor [B, feat_dim, H, W]`` — decoder feature map.

        Returns:
          ``FloatTensor [B, out_chroma, H, W]`` — raw (pre-activation) chroma logits.
        """
        b, c, h, w = feats.shape
        # Image tokens: [B, H*W, feat_dim].
        tokens = feats.flatten(2).transpose(1, 2)
        # Broadcast learnable queries over the batch: [B, num_queries, feat_dim].
        queries = self.color_queries.unsqueeze(0).expand(b, -1, -1)
        # Cross-attend queries -> image tokens (tgt=queries, memory=tokens).
        for layer in self.layers:
            queries = layer(queries, tokens)
        queries = self.query_proj(queries)  # [B, num_queries, feat_dim]

        # Per-pixel embedding and correlation with each color query.
        pix = self.pixel_embed(feats)  # [B, feat_dim, H, W]
        pix_flat = pix.flatten(2)  # [B, feat_dim, H*W]
        # corr[b, q, p] = <query_q, pixel_p>  ->  [B, num_queries, H*W]
        corr = torch.bmm(queries, pix_flat)
        corr = corr / float(self.feat_dim) ** 0.5
        corr = corr.view(b, self.num_queries, h, w)
        chroma = self.to_chroma(corr)  # [B, out_chroma, H, W]
        return chroma


# =========================================================================== #
# Lab <-> RGB conversion utilities (D65, sRGB). Differentiable, vectorized.
# =========================================================================== #
def _lab_to_rgb(lab: Tensor) -> Tensor:
    """Convert a CIE-L*a*b* image to sRGB in ``[0, 1]`` (D65 white point).

    Args:
      lab: ``FloatTensor [B, 3, H, W]`` with L in ``[0, 100]``, a*,b* in ~``[-128, 127]``.

    Returns:
      ``FloatTensor [B, 3, H, W]`` sRGB clamped to ``[0, 1]``.

    Implementation: Lab → XYZ (D65) → linear-sRGB → gamma-encoded sRGB. All ops are
    elementwise / differentiable so this can sit inside the forward graph.
    """
    L = lab[:, 0:1, :, :]
    a = lab[:, 1:2, :, :]
    b = lab[:, 2:3, :, :]

    fy = (L + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0

    eps = 6.0 / 29.0

    def _f_inv(t: Tensor) -> Tensor:
        # Inverse of the Lab nonlinearity.
        return torch.where(t > eps, t ** 3, 3.0 * (eps ** 2) * (t - 4.0 / 29.0))

    # D65 reference white (Xn, Yn, Zn).
    xn, yn, zn = 0.95047, 1.0, 1.08883
    x = xn * _f_inv(fx)
    y = yn * _f_inv(fy)
    z = zn * _f_inv(fz)

    # Linear sRGB from XYZ (sRGB D65 matrix).
    r = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z
    g = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z
    bl = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z

    rgb = torch.cat([r, g, bl], dim=1)
    rgb = torch.clamp(rgb, 0.0, 1.0)

    # Linear -> gamma-encoded sRGB.
    a_thr = 0.0031308
    rgb = torch.where(
        rgb <= a_thr,
        12.92 * rgb,
        1.055 * torch.clamp(rgb, min=1e-8) ** (1.0 / 2.4) - 0.055,
    )
    return torch.clamp(rgb, 0.0, 1.0)


# =========================================================================== #
# The primary generator.
# =========================================================================== #
class ColorizationGenerator(BaseModel):
    """Pix2PixHD coarse-to-fine generator + DDColor query color head (primary model).

    Pipeline (single deterministic forward pass):
      1. **Encoder** — reflection-pad stem + ``n_downsample`` strided convs build a
         compact feature pyramid from the SR'd IR (optionally with appended palette-prior
         channels via ``condition_on_palette``).
      2. **Bottleneck** — ``n_blocks`` residual blocks. When ``use_spade`` and a semantic
         map is provided, these are **SPADEResBlocks** (lazy-imported from the semantic
         package) that inject spatially-adaptive denormalization from the one-hot label
         map, so class info isn't washed out by normalization (Pix2PixHD's failure mode
         on flat water/field masks). Otherwise plain :class:`ResnetBlock` s are used.
      3. **Decoder** — transposed-conv upsampling back to input resolution producing a
         spatial feature map (the DDColor "pixel decoder").
      4. **Heads** — a small conv head predicts **luminance L** (Lab mode) or nothing
         (RGB mode); the :class:`ColorQueryDecoder` predicts **chroma** (a*,b* in Lab, or
         a 3-channel RGB residual in RGB mode). Lab is converted to sRGB; RGB is squashed
         by a sigmoid. Output is always RGB ``[B, 3, Hs, Ws]`` in ``[0, 1]``.

    The class→Lab **ColorLUT** chroma clamp and learned 3D-LUT are applied *later* by the
    semantic/pipeline stage, not here. The raw predicted Lab tensor (if any) is cached on
    ``self.last_lab`` for that stage to consume.

    Args:
      config:          a :class:`irchroma.config.ColorizationConfig` (channels/ngf/blocks
                       read from here). If ``None``, defaults are used.
      use_spade:       master switch for the optional SPADE conditioning path. Even when
                       ``True``, SPADE is only used if (a) the semantic builder's module
                       is importable and (b) a semantic map is passed to ``forward``.
      label_nc:        number of one-hot label channels fed to SPADE (defaults to the
                       config's ``spade_label_nc`` / the LULC class count).
    """

    name: str = "pix2pixhd_ddcolor"

    def __init__(
        self,
        config: Optional[ColorizationConfig] = None,
        use_spade: bool = True,
        label_nc: Optional[int] = None,
    ) -> None:
        super().__init__()  # type: ignore[misc]
        self.config = config if config is not None else ColorizationConfig()
        cfg = self.config

        self.in_channels: int = int(cfg.in_channels)
        self.out_channels: int = int(cfg.out_channels)
        self.ngf: int = int(cfg.ngf)
        self.n_downsample: int = int(cfg.n_downsample_global)
        self.n_blocks: int = int(cfg.n_blocks_global)
        self.output_space: str = str(cfg.output_space).lower()
        self.condition_on_palette: bool = bool(cfg.condition_on_palette)
        self.label_nc: int = int(label_nc) if label_nc is not None else int(cfg.spade_label_nc)

        # Whether to *attempt* SPADE; resolved against import availability below.
        self.use_spade: bool = bool(use_spade and cfg.use_spade)

        #: Set on each forward (Lab mode): the raw predicted Lab image [B,3,H,W].
        self.last_lab: Optional[Tensor] = None

        if not TORCH_AVAILABLE:  # pragma: no cover - nothing to build without torch
            self._spade_cls = None
            return

        # ---- Palette-prior channels (optional) ----------------------------- #
        # When conditioning on palette, we append `label_nc` per-class prior planes
        # (caller may pass them via `forward(..., palette=...)`; if absent we build a
        # zero/one-hot-derived prior from the semantic map). Reserve the input width.
        self._palette_nc: int = self.label_nc if self.condition_on_palette else 0
        enc_in = self.in_channels + self._palette_nc

        # ---- Encoder (stem + downsampling) --------------------------------- #
        ngf = self.ngf
        encoder: List["nn.Module"] = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(enc_in, ngf, kernel_size=7, padding=0),
            _norm_layer(ngf, "instance"),
            nn.ReLU(inplace=True),
        ]
        ch = ngf
        for _ in range(self.n_downsample):
            encoder += [
                nn.Conv2d(ch, ch * 2, kernel_size=3, stride=2, padding=1),
                _norm_layer(ch * 2, "instance"),
                nn.ReLU(inplace=True),
            ]
            ch *= 2
        self.encoder = nn.Sequential(*encoder)
        self.bottleneck_ch: int = ch  # feature width at the bottleneck

        # ---- Bottleneck residual blocks (SPADE or plain) ------------------- #
        self._spade_cls = _try_import_spade_resblock() if self.use_spade else None
        # If SPADE requested but unavailable, transparently fall back.
        self.spade_active: bool = self._spade_cls is not None

        blocks: List["nn.Module"] = []
        for _ in range(self.n_blocks):
            if self.spade_active:
                try:
                    # SemanticBuilder's SPADEResBlock(fin, fout, label_nc=...): the
                    # bottleneck block is channel-preserving, so fin == fout.
                    blocks.append(
                        self._spade_cls(  # type: ignore[misc]
                            self.bottleneck_ch,
                            self.bottleneck_ch,
                            label_nc=self.label_nc,
                        )
                    )
                except Exception:  # pragma: no cover - signature mismatch -> fallback
                    self.spade_active = False
                    blocks.append(ResnetBlock(self.bottleneck_ch, norm="instance"))
            else:
                blocks.append(ResnetBlock(self.bottleneck_ch, norm="instance"))
        # Use ModuleList (not Sequential): SPADE blocks need the extra `seg` argument.
        self.blocks = nn.ModuleList(blocks)

        # ---- Decoder (upsample back to input resolution) ------------------- #
        decoder: List["nn.Module"] = []
        for _ in range(self.n_downsample):
            decoder += [
                nn.ConvTranspose2d(
                    ch, ch // 2, kernel_size=3, stride=2, padding=1, output_padding=1
                ),
                _norm_layer(ch // 2, "instance"),
                nn.ReLU(inplace=True),
            ]
            ch //= 2
        self.decoder = nn.Sequential(*decoder)
        self.decoder_ch: int = ch  # == ngf

        # ---- Heads --------------------------------------------------------- #
        use_color_decoder = bool(cfg.use_color_decoder)
        if self.output_space == "lab":
            # Luminance L head from the spatial pixel decoder (0..100 via 100*sigmoid).
            self.l_head = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(self.decoder_ch, 1, kernel_size=3, padding=0),
            )
            chroma_out = 2  # a*, b*
        else:
            self.l_head = None  # type: ignore[assignment]
            chroma_out = 3  # direct RGB

        if use_color_decoder:
            self.color_decoder: Optional[ColorQueryDecoder] = ColorQueryDecoder(
                feat_dim=self.decoder_ch,
                num_queries=int(cfg.num_color_queries),
                num_layers=int(cfg.color_decoder_layers),
                num_heads=4,
                out_chroma=chroma_out,
            )
            self.chroma_conv = None  # type: ignore[assignment]
        else:
            # Simpler conv chroma head if the query decoder is disabled.
            self.color_decoder = None
            self.chroma_conv = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(self.decoder_ch, chroma_out, kernel_size=3, padding=0),
            )

        # Chroma scale for Lab a*,b* (tanh -> [-1,1] -> [-AB_RANGE, AB_RANGE]).
        self._ab_range: float = 110.0

    # ------------------------------------------------------------------ #
    # Conditioning helpers.
    # ------------------------------------------------------------------ #
    def _one_hot_semantic(self, semantic: Tensor, height: int, width: int) -> Tensor:
        """One-hot encode a ``[B, H, W]`` Long label map to ``[B, label_nc, H, W]`` float.

        Resizes (nearest) to ``(height, width)`` if it differs from the label map size.
        Negative / ignore indices are mapped to 0 before encoding (kept as a valid class
        channel; the SPADE block / loss masks them out elsewhere).
        """
        sem = semantic
        if sem.dim() == 4 and sem.shape[1] == 1:
            sem = sem[:, 0, :, :]
        sem = sem.long().clamp(min=0, max=self.label_nc - 1)
        one_hot = F.one_hot(sem, num_classes=self.label_nc)  # [B, H, W, C]
        one_hot = one_hot.permute(0, 3, 1, 2).float()  # [B, C, H, W]
        if one_hot.shape[-2] != height or one_hot.shape[-1] != width:
            one_hot = F.interpolate(one_hot, size=(height, width), mode="nearest")
        return one_hot

    def _palette_prior(self, one_hot: Tensor) -> Tensor:
        """Build per-class palette-prior input planes from a one-hot label map.

        For the encoder we simply reuse the one-hot class planes as a cheap, stable
        palette/semantic prior (the explicit Lab-centroid palette is injected by the
        downstream ColorLUT; here we only need a class-identity hint so the encoder can
        specialize). Returns ``[B, label_nc, H, W]`` matching ``self._palette_nc``.
        """
        return one_hot

    # ------------------------------------------------------------------ #
    # Forward.
    # ------------------------------------------------------------------ #
    def forward(self, ir: Tensor, semantic: Optional[Tensor] = None) -> Tensor:
        """Colorize a (super-resolved) IR tile.

        Args:
          ir:       ``FloatTensor [B, C_ir, Hs, Ws]`` — SR'd IR / IR at target resolution.
          semantic: ``LongTensor [B, Hs, Ws]`` of LULC indices, or ``None``. When provided
                    and ``use_spade`` is active, drives SPADE denormalization in the
                    bottleneck; also used (one-hot) as the palette-prior input when
                    ``condition_on_palette``.

        Returns:
          ``FloatTensor [B, 3, Hs, Ws]`` RGB in ``[0, 1]`` (the *raw* colorization; the
          class→Lab LUT clamp / 3D-LUT are applied later by the pipeline). In Lab mode the
          intermediate Lab image is also cached on ``self.last_lab``.
        """
        if not TORCH_AVAILABLE:  # pragma: no cover
            raise RuntimeError("torch is required to run ColorizationGenerator.forward().")

        b, _, h, w = ir.shape

        # Normalize the semantic map to a Long [B, H, W] and build a one-hot copy.
        # The Long map is what SemanticBuilder's SPADEResBlock expects (it one-hots +
        # resizes internally); the one-hot copy feeds the palette-prior input concat.
        sem_idx: Optional[Tensor] = None
        one_hot: Optional[Tensor] = None
        if semantic is not None:
            sem_idx = semantic
            if sem_idx.dim() == 4 and sem_idx.shape[1] == 1:
                sem_idx = sem_idx[:, 0, :, :]
            sem_idx = sem_idx.long()
            one_hot = self._one_hot_semantic(sem_idx, h, w)

        # ---- Encoder input (optionally append palette-prior planes) -------- #
        x = ir
        if self._palette_nc > 0:
            if one_hot is not None:
                prior = self._palette_prior(one_hot)
            else:
                # No semantic map: feed zero priors so channel count still matches.
                prior = ir.new_zeros((b, self._palette_nc, h, w))
            x = torch.cat([x, prior], dim=1)

        feats = self.encoder(x)  # [B, bottleneck_ch, h', w']

        # ---- Bottleneck (SPADE-conditioned or plain residual blocks) ------- #
        # SemanticBuilder's SPADEResBlock.forward(features, semantic) expects the RAW
        # Long label map [B, H, W] (it one-hots + resizes to the feature size itself),
        # NOT our one-hot. Pass `sem_idx`; fall back to plain blocks if absent.
        use_spade_now = self.spade_active and sem_idx is not None
        for block in self.blocks:
            if use_spade_now:
                feats = block(feats, sem_idx)
            else:
                feats = block(feats)

        # ---- Decoder (pixel decoder) --------------------------------------- #
        dec = self.decoder(feats)  # [B, decoder_ch, ~h, ~w]
        # Strided encoders floor odd sizes, so the symmetric decoder can land 1px off
        # for non-power-of-2 tiles. Snap back to the exact input resolution so the RGB
        # output is guaranteed [B, 3, Hs, Ws] (the contract). No-op when sizes match.
        if dec.shape[-2] != h or dec.shape[-1] != w:
            dec = F.interpolate(dec, size=(h, w), mode="bilinear", align_corners=False)

        # ---- Chroma head --------------------------------------------------- #
        if self.color_decoder is not None:
            chroma = self.color_decoder(dec)
        else:
            chroma = self.chroma_conv(dec)

        if self.output_space == "lab":
            # L from the spatial decoder; a*,b* from the (query) chroma head.
            l_logits = self.l_head(dec)  # type: ignore[misc]
            L = torch.sigmoid(l_logits) * 100.0  # [B,1,H,W] in [0,100]
            ab = torch.tanh(chroma) * self._ab_range  # [B,2,H,W] in [-range, range]
            lab = torch.cat([L, ab], dim=1)  # [B,3,H,W]
            self.last_lab = lab
            rgb = _lab_to_rgb(lab)
        else:
            # Direct RGB head (sigmoid -> [0,1]).
            self.last_lab = None
            rgb = torch.sigmoid(chroma)

        return rgb
