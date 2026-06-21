"""irchroma.models.semantic.spade — SPADE spatially-adaptive normalization.

Implements **SPADE** (Park et al., *Semantic Image Synthesis with Spatially-Adaptive
Normalization*, CVPR 2019, arXiv:1903.07291) and a residual block built on it. These
are the semantic-conditioning primitives consumed by the Stage-2 colorization
generator (``irchroma.models.colorization``): the per-pixel LULC label map drives a
learned per-pixel affine ``(gamma, beta)`` injected after a *parameter-free*
normalization, so semantic information is not washed out by normalization on flat /
uniform regions (water, fields) — exactly pix2pixHD's failure mode that SPADE fixes.
(docs/research/04 mechanisms §1; ARCHITECTURE.md §7.2.)

Tensor conventions (see :mod:`irchroma.interfaces`):
  * ``features`` : ``FloatTensor [B, C, Hf, Wf]`` — the decoder feature map.
  * ``semantic`` : ``LongTensor  [B, H, W]`` — per-pixel LULC class indices into
    ``irchroma.config.LULC_CLASSES`` (``label_nc = len(LULC_CLASSES)``). It is one-hot
    encoded then nearest-neighbour resized to ``(Hf, Wf)`` so the conditioning aligns
    with the feature grid at every decoder resolution.

forward contract: ``forward(features, semantic) -> features`` (same ``[B, C, Hf, Wf]``).

If ``torch`` is unavailable this module still imports (light stubs from
:mod:`irchroma.interfaces`); the layers are simply not instantiable on that path.
"""

from __future__ import annotations

from typing import Optional

from irchroma.config import NUM_LULC_CLASSES
from irchroma.interfaces import TORCH_AVAILABLE

# --------------------------------------------------------------------------- #
# Guarded torch import. The module must import even without torch installed so
# that ``import irchroma.models.semantic`` never fails during parallel dev / docs.
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


# Base class for nn.Module subclasses: the real ``nn.Module`` when torch is present,
# else a tiny stub so the ``class X(_Module)`` statements remain importable.
if TORCH_AVAILABLE:  # pragma: no branch
    _Module = nn.Module  # type: ignore[assignment]
else:  # pragma: no cover

    class _Module:  # minimal stand-in
        """Placeholder base when torch is unavailable (never instantiated)."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError(
                "torch is not installed; irchroma.models.semantic.spade requires torch."
            )


__all__ = ["SPADE", "SPADEResBlock"]


class SPADE(_Module):  # type: ignore[misc]
    """Spatially-Adaptive (DE)normalization conditioned on a LULC label map.

    The forward pass:
      1. **Parameter-free normalize** the input features (batch/instance/group norm
         with ``affine=False``) so the learned modulation is the only source of
         per-channel scale/shift — this is what prevents the semantic signal from
         being normalized away.
      2. One-hot encode ``semantic`` to ``label_nc`` channels and nearest-neighbour
         resize it to the feature spatial size ``(Hf, Wf)``.
      3. A shared conv (``mlp_shared``) embeds the one-hot map; two ``1x1`` heads
         (``mlp_gamma``, ``mlp_beta``) regress per-pixel modulation tensors.
      4. Return ``normalized * (1 + gamma) + beta`` (residual ``gamma`` parameterization
         per the SPADE paper, so ``gamma == 0`` is identity scaling).

    Args:
      norm_nc:   number of feature channels ``C`` being normalized.
      label_nc:  number of semantic label channels (one-hot width); defaults to
                 ``len(LULC_CLASSES)`` from config.
      hidden_nc: width of the shared conv embedding (modest, config-driven).
      ks:        kernel size of the modulation convs (SPADE uses ``3``).
      param_free_norm: ``{"batch", "instance", "group", "syncbatch"}`` — the
                 normalization applied before modulation (always ``affine=False``).
      n_groups:  number of groups when ``param_free_norm == "group"``.
    """

    def __init__(
        self,
        norm_nc: int,
        label_nc: int = NUM_LULC_CLASSES,
        hidden_nc: int = 128,
        ks: int = 3,
        param_free_norm: str = "batch",
        n_groups: int = 32,
    ) -> None:
        super().__init__()  # type: ignore[misc]
        self.norm_nc = int(norm_nc)
        self.label_nc = int(label_nc)
        self.hidden_nc = int(hidden_nc)
        self.param_free_norm_type = str(param_free_norm)

        # ---- Parameter-free normalization (no learnable affine: that is SPADE's job). #
        kind = param_free_norm.lower()
        if kind == "instance":
            self.param_free_norm: nn.Module = nn.InstanceNorm2d(norm_nc, affine=False)
        elif kind == "group":
            self.param_free_norm = nn.GroupNorm(
                num_groups=min(int(n_groups), int(norm_nc)), num_channels=norm_nc, affine=False
            )
        elif kind in ("syncbatch", "sync_batch"):
            self.param_free_norm = nn.SyncBatchNorm(norm_nc, affine=False)
        elif kind == "batch":
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError(
                f"Unknown param_free_norm {param_free_norm!r}; expected one of "
                "{'batch', 'instance', 'group', 'syncbatch'}."
            )

        pw = ks // 2
        # Shared embedding of the (one-hot) label map, then two heads -> gamma, beta.
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(self.label_nc, hidden_nc, kernel_size=ks, padding=pw),
            nn.ReLU(inplace=True),
        )
        self.mlp_gamma = nn.Conv2d(hidden_nc, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(hidden_nc, norm_nc, kernel_size=ks, padding=pw)

    # ------------------------------------------------------------------ #
    def _to_one_hot(self, semantic: "Tensor", size: "tuple[int, int]") -> "Tensor":
        """One-hot encode a Long label map and resize (nearest) to ``size``.

        Args:
          semantic: ``LongTensor [B, H, W]`` (or ``[B, 1, H, W]``) of class indices.
          size:     target ``(Hf, Wf)`` spatial size of the feature map.

        Returns:
          ``FloatTensor [B, label_nc, Hf, Wf]`` one-hot label channels.
        """
        if semantic.dim() == 4 and semantic.shape[1] == 1:
            semantic = semantic[:, 0]
        if semantic.dim() != 3:
            raise ValueError(
                f"SPADE expects semantic of shape [B, H, W]; got {tuple(semantic.shape)}."
            )
        # Clamp stray/ignore indices into valid range before one-hot (IGNORE_INDEX=-1
        # or out-of-range values would break one_hot). Negative -> 0 (water) is a safe
        # default; the LUT / loss handle ignore separately.
        labels = semantic.long().clamp(min=0, max=self.label_nc - 1)
        one_hot = F.one_hot(labels, num_classes=self.label_nc)  # [B, H, W, label_nc]
        one_hot = one_hot.permute(0, 3, 1, 2).to(dtype=torch.float32)  # [B, label_nc, H, W]
        if one_hot.shape[-2:] != tuple(size):
            one_hot = F.interpolate(one_hot, size=size, mode="nearest")
        return one_hot

    def forward(self, features: "Tensor", semantic: "Tensor") -> "Tensor":
        """Apply spatially-adaptive denormalization.

        Args:
          features: ``FloatTensor [B, C, Hf, Wf]`` decoder features.
          semantic: ``LongTensor  [B, H, W]`` LULC class indices.

        Returns:
          ``FloatTensor [B, C, Hf, Wf]`` modulated features.
        """
        normalized = self.param_free_norm(features)
        seg = self._to_one_hot(semantic, size=tuple(features.shape[-2:]))
        seg = seg.to(dtype=features.dtype)
        actv = self.mlp_shared(seg)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        # Residual gamma parameterization: identity scale at gamma == 0.
        return normalized * (1.0 + gamma) + beta


class SPADEResBlock(_Module):  # type: ignore[misc]
    """Residual block whose normalizations are SPADE (label-conditioned).

    Mirrors the SPADE generator's ``SPADEResnetBlock``: two SPADE-normalized,
    spectrally-normalizable ``3x3`` convs with a learned ``1x1`` shortcut when the
    channel count changes. LeakyReLU activations (slope ``0.2``) as in the paper.

    Args:
      fin:       input channels.
      fout:      output channels.
      label_nc:  semantic one-hot width (defaults to ``len(LULC_CLASSES)``).
      hidden_nc: SPADE embedding width.
      ks:        conv kernel size (``3``).
      spectral_norm: wrap the convs in spectral norm (recommended for the generator).
      param_free_norm: normalization kind passed through to each :class:`SPADE`.
    """

    def __init__(
        self,
        fin: int,
        fout: int,
        label_nc: int = NUM_LULC_CLASSES,
        hidden_nc: int = 128,
        ks: int = 3,
        spectral_norm: bool = True,
        param_free_norm: str = "batch",
    ) -> None:
        super().__init__()  # type: ignore[misc]
        self.fin = int(fin)
        self.fout = int(fout)
        self.learned_shortcut = fin != fout
        fmiddle = min(int(fin), int(fout))
        pw = ks // 2

        conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=ks, padding=pw)
        conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=ks, padding=pw)
        conv_s: Optional[nn.Module] = (
            nn.Conv2d(fin, fout, kernel_size=1, bias=False) if self.learned_shortcut else None
        )

        if spectral_norm:
            sn = nn.utils.spectral_norm
            conv_0 = sn(conv_0)
            conv_1 = sn(conv_1)
            if conv_s is not None:
                conv_s = sn(conv_s)

        self.conv_0 = conv_0
        self.conv_1 = conv_1
        self.conv_s = conv_s

        self.norm_0 = SPADE(fin, label_nc, hidden_nc, ks, param_free_norm)
        self.norm_1 = SPADE(fmiddle, label_nc, hidden_nc, ks, param_free_norm)
        if self.learned_shortcut:
            self.norm_s: Optional[SPADE] = SPADE(fin, label_nc, hidden_nc, ks, param_free_norm)
        else:
            self.norm_s = None

    # ------------------------------------------------------------------ #
    def _shortcut(self, x: "Tensor", semantic: "Tensor") -> "Tensor":
        if self.learned_shortcut:
            assert self.conv_s is not None and self.norm_s is not None
            return self.conv_s(self.norm_s(x, semantic))
        return x

    @staticmethod
    def _act(x: "Tensor") -> "Tensor":
        return F.leaky_relu(x, negative_slope=0.2)

    def forward(self, features: "Tensor", semantic: "Tensor") -> "Tensor":
        """Residual SPADE block.

        Args:
          features: ``FloatTensor [B, fin, Hf, Wf]``.
          semantic: ``LongTensor  [B, H, W]`` LULC class indices.

        Returns:
          ``FloatTensor [B, fout, Hf, Wf]``.
        """
        x_s = self._shortcut(features, semantic)
        dx = self.conv_0(self._act(self.norm_0(features, semantic)))
        dx = self.conv_1(self._act(self.norm_1(dx, semantic)))
        return x_s + dx
