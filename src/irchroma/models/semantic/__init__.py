"""irchroma.models.semantic — semantic guidance, consistency checker, color-LUT.

Owner: Builder-4. Implements the semantic-conditioning + color-consistency +
no-hallucination stack (docs/research/04; ARCHITECTURE.md §7):

  * :class:`SPADE`, :class:`SPADEResBlock` — spatially-adaptive normalization that
    injects the per-pixel LULC label map into the colorization decoder so semantics are
    not washed out on uniform regions (water, fields). (``spade``)
  * :class:`ClassColorLUT` — the O(1)/pixel class -> CIE-Lab **chroma clamp** satisfying
    :class:`irchroma.interfaces.ColorLUTProtocol`; seeded from the config palettes
    (``DEFAULT_LAB_PALETTE``). Plus :class:`AdaIntLUT`, a learned image-adaptive 3D LUT
    (trilinear, O(1)/pixel) for optional color refinement. (``color_lut``)
  * :class:`LandCoverLabeler` — builds the fused LULC label map (GEE majority-vote of
    ESA WorldCover ⊕ Dynamic World ⊕ JRC water; or model-based ``from_segmenter``); and
    :class:`FrozenSegmenter`, the frozen independent consistency checker (transformers
    SegFormer if available, else a pure-torch fallback). (``landcover``)
  * :func:`flag_uncertain`, :func:`desaturate_low_confidence` — Dynamic-World-style
    honest low-confidence flagging + desaturation. (``uncertainty``)

The default palette / taxonomy is defined in :mod:`irchroma.config` (``LULC_CLASSES``,
``DEFAULT_SRGB_PALETTE``, ``DEFAULT_LAB_PALETTE``).

Imports are RESILIENT: individual submodule import failures (e.g. a torch-less docs box,
or a half-written sibling during parallel development) are swallowed so
``import irchroma.models.semantic`` never hard-fails; the affected names are simply
absent from the namespace. ``__all__`` always advertises the full public registry.
"""

from __future__ import annotations

__all__ = [
    # spade
    "SPADE",
    "SPADEResBlock",
    # color_lut
    "ClassColorLUT",
    "AdaIntLUT",
    "rgb_to_lab",
    "lab_to_rgb",
    # landcover
    "LandCoverLabeler",
    "FrozenSegmenter",
    "WORLDCOVER_TO_LULC",
    "DYNAMIC_WORLD_TO_LULC",
    # uncertainty
    "flag_uncertain",
    "desaturate_low_confidence",
    "uncertainty_from_probs",
]

# Resilient submodule imports: never let one missing optional dependency (or a
# torch-less environment) break the whole package import.
try:
    from .spade import SPADE, SPADEResBlock
except Exception:  # pragma: no cover - resilient to torch-less / partial builds
    pass

try:
    from .color_lut import AdaIntLUT, ClassColorLUT, lab_to_rgb, rgb_to_lab
except Exception:  # pragma: no cover
    pass

try:
    from .landcover import (
        DYNAMIC_WORLD_TO_LULC,
        WORLDCOVER_TO_LULC,
        FrozenSegmenter,
        LandCoverLabeler,
    )
except Exception:  # pragma: no cover
    pass

try:
    from .uncertainty import (
        desaturate_low_confidence,
        flag_uncertain,
        uncertainty_from_probs,
    )
except Exception:  # pragma: no cover
    pass
