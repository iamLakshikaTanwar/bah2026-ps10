"""irchroma.models — neural network building blocks and the end-to-end pipeline.

Sub-packages:
  - ``sr``            : Stage-1 guided super-resolution / restoration (Builder-2).
  - ``colorization``  : Stage-2 IR→RGB colorization generator + discriminator (Builder-3).
  - ``semantic``      : LULC segmentation guidance, frozen checker, O(1) color-LUT (Builder-4).

The composed two-stage pipeline (:class:`IRChromaPipeline`) that satisfies
:class:`irchroma.interfaces.PipelineProtocol` lives in ``pipeline.py``; build it with
:func:`build_pipeline`.

Imports are **resilient**: a failure inside any one submodule (e.g. a torch-less docs
box, or a sibling mid-edit during parallel development) does not break
``import irchroma.models`` — the offending name is simply omitted and set to ``None``.
"""

from __future__ import annotations

from typing import List

__all__: List[str] = []

# --- End-to-end pipeline + factory ----------------------------------------- #
try:
    from .pipeline import IRChromaPipeline, build_pipeline

    __all__.extend(["IRChromaPipeline", "build_pipeline"])
except Exception:  # pragma: no cover - keep package importable during parallel dev
    IRChromaPipeline = None  # type: ignore
    build_pipeline = None  # type: ignore
