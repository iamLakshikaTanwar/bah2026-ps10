"""irchroma — Infrared satellite image super-resolution + IR→RGB colorization.

BAH 2026 Problem Statement 10. An end-to-end, fidelity-dominant, no-hallucination
framework that (1) super-resolves/sharpens monochrome IR satellite imagery and
(2) colorizes it IR→RGB realistically while preserving semantic integrity.

This top-level ``__init__`` is intentionally minimal. It imports ONLY the two
contract modules (:mod:`irchroma.config` and :mod:`irchroma.interfaces`), both of
which are designed to import with stdlib (+ optional torch/omegaconf guarded).
Sub-packages (``models``, ``losses``, ``metrics``, ...) are NOT eagerly imported
here so that parallel builders can add modules without breaking ``import irchroma``.

See ``ARCHITECTURE.md`` (repo root) and ``docs/research/*.md`` for the full design.
"""

from __future__ import annotations

__version__ = "0.1.0"

# Only the stable contract surface is re-exported at the top level. These two
# modules are guaranteed import-safe (torch/omegaconf are guarded inside them).
from . import config, interfaces  # noqa: F401

__all__ = ["config", "interfaces", "__version__"]
