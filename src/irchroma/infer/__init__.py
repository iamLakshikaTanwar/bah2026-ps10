"""irchroma.infer — fast O(1)-per-tile inference.

Owner: Builder-7 (InferServeBuilder). Implements the fast-platform inference layer
of docs/research/05 / ARCHITECTURE.md §9:

  * :class:`~irchroma.infer.engine.InferenceEngine` — loads the two-stage IRChroma
    pipeline (lazy ``build_pipeline``), runs ``predict(ir, guide, semantic) -> rgb``,
    and offers the optimized-runtime exits: FP16 via :meth:`~InferenceEngine.half`,
    ONNX export (:meth:`~InferenceEngine.export_onnx`) + ONNX Runtime
    (:meth:`~InferenceEngine.load_onnx` / :meth:`~InferenceEngine.predict_onnx`), and
    a guarded TensorRT FP16/INT8 engine build (:meth:`~InferenceEngine.build_tensorrt`).
  * :func:`~irchroma.infer.tiling.tile_inference` /
    :func:`~irchroma.infer.tiling.raised_cosine_window` — patch-tiled inference with
    raised-cosine overlap-blend (the seamless large-scene path; O(1)/tile,
    O(#tiles)/scene).
  * :func:`~irchroma.infer.color_refine.apply_color_lut` — the closing O(1)/pixel
    color refinement (class→Lab chroma clamp + optional learned 3D-LUT).

Import resilience: every submodule is imported defensively so ``import
irchroma.infer`` never fails because torch or an optional optimized-runtime backend
is missing — public names are exported only when their submodule imports, and a
dynamic ``__all__`` reflects what is actually available.
"""

from __future__ import annotations

from typing import List

__all__: List[str] = []

# --- Tiling (pure torch/numpy; import-safe) -------------------------------- #
try:
    from .tiling import raised_cosine_window, tile_inference  # noqa: F401

    __all__ += ["raised_cosine_window", "tile_inference"]
except Exception:  # pragma: no cover - never expected (no hard deps at import)
    raised_cosine_window = None  # type: ignore
    tile_inference = None  # type: ignore

# --- Color refinement (lazy ClassColorLUT/AdaIntLUT; import-safe) ---------- #
try:
    from .color_refine import apply_color_lut, build_adaint_lut, build_class_lut  # noqa: F401

    __all__ += ["apply_color_lut", "build_class_lut", "build_adaint_lut"]
except Exception:  # pragma: no cover
    apply_color_lut = None  # type: ignore
    build_class_lut = None  # type: ignore
    build_adaint_lut = None  # type: ignore

# --- Inference engine (guarded torch/onnx/tensorrt; import-safe) ----------- #
try:
    from .engine import InferenceEngine  # noqa: F401

    __all__ += ["InferenceEngine"]
except Exception:  # pragma: no cover
    InferenceEngine = None  # type: ignore
