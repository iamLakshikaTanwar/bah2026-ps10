"""irchroma.models.colorization — Stage-2 IR→RGB colorization.

Owner: ColorBuilder (Builder-3). Implements the paired conditional GAN: restoration
front-end + Pix2PixHD coarse-to-fine generator + SPADE semantic conditioning + a
DDColor-style query color decoder, and a multi-scale spectral-norm PatchGAN
discriminator. The BBDM diffusion backup lives here too.
See docs/research/02-colorization-models.md and ARCHITECTURE.md §6.

Public registry (all subclass :class:`irchroma.interfaces.BaseModel`):
  * :class:`ColorizationGenerator`    — primary generator; ``forward(ir, semantic=None)``
                                        → RGB ``[B, 3, Hs, Ws]`` in ``[0, 1]``.
  * :class:`MultiScaleDiscriminator`  — Pix2PixHD multi-scale spectral-norm PatchGAN;
                                        ``forward(x)`` → ``List[List[Tensor]]``
                                        (per-scale → per-layer features; last = logits).
  * :class:`NLayerDiscriminator`      — a single PatchGAN scale (building block).
  * :class:`BBDMColorizer`            — BACKUP Brownian-Bridge diffusion colorizer.

Imports are **resilient**: a failure inside any one submodule (e.g. another builder's
file mid-edit) does not break ``import irchroma.models.colorization`` — the offending
name is simply omitted from ``__all__``. This keeps parallel development unblocked.
"""

from __future__ import annotations

from typing import List

__all__: List[str] = []

# --- Primary generator ----------------------------------------------------- #
try:
    from .generator import ColorizationGenerator

    __all__.append("ColorizationGenerator")
except Exception:  # pragma: no cover - keep package importable during parallel dev
    ColorizationGenerator = None  # type: ignore

# --- Multi-scale discriminator (+ single-scale building block) ------------- #
try:
    from .discriminator import MultiScaleDiscriminator, NLayerDiscriminator

    __all__.extend(["MultiScaleDiscriminator", "NLayerDiscriminator"])
except Exception:  # pragma: no cover
    MultiScaleDiscriminator = None  # type: ignore
    NLayerDiscriminator = None  # type: ignore

# --- BBDM diffusion backup ------------------------------------------------- #
try:
    from .diffusion import BBDMColorizer

    __all__.append("BBDMColorizer")
except Exception:  # pragma: no cover
    BBDMColorizer = None  # type: ignore
