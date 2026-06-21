"""irchroma.models.sr — Stage-1 enhancement (guided cross-sensor super-resolution).

Owner: Builder-2. Implements the restoration backbone (NAFNet-style encoder-decoder
with LayerNorm + SimpleGate + Simplified Channel Attention and pixel-(un)shuffle
sampling), the primary guided-SR model (IR backbone + HR-guide fusion branch +
pixel-shuffle upsampler), and a blind real-world degradation synthesizer
(Real-ESRGAN/BSRGAN-style + Landsat-TIRS sensor MTF). See
docs/research/03-super-resolution.md and ARCHITECTURE.md §5.

Public registry
---------------
  * :class:`NAFNetBackbone`   — shared restoration backbone (``backbone.py``);
    ``forward(x) -> x`` at the same resolution.
  * :class:`GuidedSR`         — primary Stage-1 model (``guided_sr.py``);
    ``forward(ir, guide=None) -> {"sr": ..., "feat": ...}``.
  * :class:`BSRGANDegradation` — LR<-HR degradation synthesizer (``degradation.py``);
    ``__call__(hr) -> lr`` and ``paired_degrade(hr, scale) -> (hr, lr)``.

Concrete models subclass :class:`irchroma.interfaces.BaseModel` and follow the SR
tensor convention: IR ``[B, C_ir, H, W]`` -> SR ``[B, C_ir, H*scale, W*scale]``.

Imports are **resilient**: a failure to import any one submodule (e.g. an optional
dependency missing in a docs/CI box) does not break ``import irchroma.models.sr`` —
the offending name is simply absent from this namespace and ``__all__``.
"""

from __future__ import annotations

from typing import List

__all__: List[str] = []

# --- NAFNetBackbone -------------------------------------------------------- #
try:
    from .backbone import NAFNetBackbone  # noqa: F401

    __all__.append("NAFNetBackbone")
except Exception:  # pragma: no cover - keep package import safe during dev
    NAFNetBackbone = None  # type: ignore

# --- GuidedSR -------------------------------------------------------------- #
try:
    from .guided_sr import GuidedSR  # noqa: F401

    __all__.append("GuidedSR")
except Exception:  # pragma: no cover - keep package import safe during dev
    GuidedSR = None  # type: ignore

# --- BSRGANDegradation ----------------------------------------------------- #
try:
    from .degradation import BSRGANDegradation  # noqa: F401

    __all__.append("BSRGANDegradation")
except Exception:  # pragma: no cover - keep package import safe during dev
    BSRGANDegradation = None  # type: ignore
