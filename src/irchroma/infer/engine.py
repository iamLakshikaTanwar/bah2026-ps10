"""irchroma.infer.engine — the fast-platform inference engine.

:class:`InferenceEngine` is the single entry point for **running** the two-stage
IRChroma pipeline at serving time (docs/research/05; ARCHITECTURE.md §9). It loads
a trained pipeline from a checkpoint, runs ``predict(ir, guide, semantic) -> rgb``,
and offers the optimized-runtime exits of the fast platform:

  * **PyTorch** (default) — eager / FP16 via :meth:`half`.
  * **ONNX export** (:meth:`export_onnx`) and **ONNX Runtime** inference
    (:meth:`load_onnx` / :meth:`predict_onnx`) — portable, CUDA/TensorRT/OpenVINO
    execution providers (docs/research/05 §C1).
  * **TensorRT** (:meth:`build_tensorrt`) — best GPU latency via FP16/INT8 engines;
    documented below and guarded (the engine build only runs where TensorRT exists).

The full O(1)-per-tile serving path this engine anchors (docs/research/05 §C5,
ARCHITECTURE.md §9.2–9.3):

    COG range read (1 tile)  ─►  preprocess  ─►  **TensorRT/ONNX FP16/INT8 forward**
        ─►  class-LUT + learned 3D-LUT color refine (O(1)/pixel)  ─►  encode tile.

Cache lookup (quadkey/H3) short-circuits the whole path on a hit (see
:mod:`irchroma.serve`). Large scenes are stitched by
:func:`irchroma.infer.tiling.tile_inference` (raised-cosine overlap-blend), keeping
per-tile cost O(1) and whole-scene cost O(#tiles), embarrassingly parallel.

Import posture (the contract): ``torch`` and **every** optimized-runtime backend
(``onnx``, ``onnxruntime``, ``tensorrt``) are import-guarded so this module imports
on a bare box; each guarded feature raises a clear, friendly :class:`RuntimeError`
only when actually called. The pipeline factory (``irchroma.models.pipeline``) is
imported **lazily** inside methods so this module compiles/imports even while that
file is being written concurrently.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

from ..config import Config, InferConfig

# --------------------------------------------------------------------------- #
# Guarded torch import — module must import without torch.
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - exercised on the real (torch) runtime
    import torch
    from torch import Tensor

    _HAS_TORCH = True
except Exception:  # pragma: no cover - torch-less environments (docs/CI)
    torch = None  # type: ignore
    Tensor = object  # type: ignore
    _HAS_TORCH = False

# Optional optimized-runtime backends (all guarded; only needed when their feature
# is called). We import lazily inside the methods too, but probe availability here.
try:  # pragma: no cover
    import onnx  # type: ignore

    _HAS_ONNX = True
except Exception:  # pragma: no cover
    onnx = None  # type: ignore
    _HAS_ONNX = False

try:  # pragma: no cover
    import onnxruntime  # type: ignore

    _HAS_ORT = True
except Exception:  # pragma: no cover
    onnxruntime = None  # type: ignore
    _HAS_ORT = False

try:  # pragma: no cover
    import tensorrt  # type: ignore

    _HAS_TRT = True
except Exception:  # pragma: no cover
    tensorrt = None  # type: ignore
    _HAS_TRT = False


__all__ = ["InferenceEngine"]


def _require_torch() -> None:
    """Raise a clear error if torch is unavailable on the call path."""
    if not _HAS_TORCH or torch is None:  # pragma: no cover
        raise RuntimeError(
            "irchroma.infer.engine.InferenceEngine requires PyTorch. Install torch "
            "(`pip install torch`) to load and run the pipeline."
        )


def _as_infer_config(config: Any) -> InferConfig:
    """Coerce a ``Config`` / ``InferConfig`` / ``None`` into an :class:`InferConfig`."""
    if config is None:
        return InferConfig()
    if isinstance(config, InferConfig):
        return config
    if isinstance(config, Config):
        return config.infer
    inner = getattr(config, "infer", None)
    if isinstance(inner, InferConfig):
        return inner
    if hasattr(config, "runtime") and hasattr(config, "tile_size"):
        return config  # type: ignore[return-value]
    return InferConfig()


class InferenceEngine:
    """Run the IRChroma pipeline for serving — torch / ONNX / TensorRT, FP16/INT8.

    Construct directly with a (already built) pipeline, or via :meth:`from_checkpoint`.
    The engine is deliberately thin: it owns a ``PipelineProtocol`` object plus the
    serving config, and exposes the optimized-runtime exits documented at module level.

    Args:
      pipeline: a :class:`irchroma.interfaces.PipelineProtocol` (the canonical
                ``IRChromaPipeline``), or ``None`` to attach one later (e.g. when only
                an ONNX/TensorRT engine is used).
      config:   an :class:`~irchroma.config.InferConfig` / :class:`~irchroma.config.Config`
                / ``None`` (serving knobs: ``tile_size``, ``device``, ``precision``,
                ``runtime``, cache settings, 3D-LUT toggles).
      device:   explicit device override (``"cuda"``/``"cpu"``); defaults to
                ``config.device``.

    Attributes:
      pipeline:  the wrapped pipeline (or ``None``).
      cfg:       the resolved :class:`InferConfig`.
      device:    the active ``torch.device``.
      ort_session: the ONNX Runtime session once :meth:`load_onnx` has been called.
    """

    def __init__(
        self,
        pipeline: Optional[Any] = None,
        config: Any = None,
        device: Optional[str] = None,
    ) -> None:
        self.cfg: InferConfig = _as_infer_config(config)
        self.pipeline = pipeline
        self._fp16 = str(getattr(self.cfg, "precision", "fp16")).lower() == "fp16"
        self.ort_session: Optional[Any] = None
        self._trt_engine: Optional[Any] = None
        # Resolve the device lazily so the module imports without torch/CUDA.
        self._device_str = device if device is not None else getattr(self.cfg, "device", "cpu")
        self._device: Optional[Any] = None
        if _HAS_TORCH:
            self._device = self._resolve_device(self._device_str)
            if self.pipeline is not None:
                self._prepare_pipeline()

    # ------------------------------------------------------------------ #
    # Construction.
    # ------------------------------------------------------------------ #
    @property
    def device(self) -> Any:
        """The active ``torch.device`` (resolving CPU fallback if CUDA is absent)."""
        if self._device is None:
            _require_torch()
            self._device = self._resolve_device(self._device_str)
        return self._device

    @staticmethod
    def _resolve_device(device_str: str) -> Any:
        """Resolve a device string to a ``torch.device``, falling back to CPU if needed."""
        _require_torch()
        want = str(device_str or "cpu")
        if want.startswith("cuda") and not torch.cuda.is_available():
            return torch.device("cpu")
        if want == "mps" and not getattr(torch.backends, "mps", None):  # pragma: no cover
            return torch.device("cpu")
        try:
            return torch.device(want)
        except Exception:  # pragma: no cover - defensive
            return torch.device("cpu")

    def _prepare_pipeline(self) -> None:
        """Move the pipeline to the device, set eval mode, and apply FP16 if configured."""
        if self.pipeline is None:
            return
        if hasattr(self.pipeline, "to"):
            try:
                self.pipeline = self.pipeline.to(self.device)
            except Exception:  # pragma: no cover - non-module pipelines are fine
                pass
        if hasattr(self.pipeline, "eval"):
            try:
                self.pipeline.eval()
            except Exception:  # pragma: no cover
                pass
        if self._fp16 and self.device.type == "cuda":
            self.half()

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        config: Any = None,
        device: Optional[str] = None,
        strict: bool = False,
    ) -> "InferenceEngine":
        """Build an engine by loading a trained pipeline from a checkpoint file.

        The pipeline is constructed via ``irchroma.models.pipeline.build_pipeline``
        (imported **lazily** so this module loads even before that file exists), then
        its weights are loaded from ``path``. Accepts either a raw ``state_dict`` or a
        training checkpoint dict carrying one under ``"model"``/``"state_dict"``/
        ``"pipeline"`` (and an optional embedded ``"config"`` used when ``config`` is
        ``None``).

        Args:
          path:   filesystem path to the checkpoint (``.pt``/``.pth``).
          config: an :class:`~irchroma.config.Config`/:class:`~irchroma.config.InferConfig`
                  (or ``None`` to use the checkpoint's embedded config / defaults).
          device: device override (``"cuda"``/``"cpu"``).
          strict: passed to ``load_state_dict`` (default ``False`` so partially-trained
                  or differently-keyed checkpoints still load the overlap).

        Returns:
          A ready :class:`InferenceEngine` with the pipeline on ``device`` in eval mode.
        """
        _require_torch()
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path!r}")

        ckpt = torch.load(path, map_location="cpu")
        # Recover an embedded config if the caller did not pass one.
        full_cfg = config
        if full_cfg is None and isinstance(ckpt, dict) and "config" in ckpt:
            try:
                full_cfg = Config.from_dict(ckpt["config"])  # type: ignore[arg-type]
            except Exception:  # pragma: no cover - tolerate odd embedded configs
                full_cfg = None

        pipeline = cls._build_pipeline(full_cfg)

        # Extract the state_dict from a variety of checkpoint layouts.
        state = ckpt
        if isinstance(ckpt, dict):
            for key in ("pipeline", "model", "state_dict", "model_state_dict"):
                if key in ckpt and isinstance(ckpt[key], dict):
                    state = ckpt[key]
                    break
        if hasattr(pipeline, "load_state_dict") and isinstance(state, dict):
            try:
                pipeline.load_state_dict(state, strict=strict)
            except Exception as exc:  # pragma: no cover - report but keep going on non-strict
                if strict:
                    raise
                # Non-strict: surface a warning-style message but proceed (random init).
                import warnings

                warnings.warn(
                    f"Partial/failed state_dict load from {path!r}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )

        return cls(pipeline=pipeline, config=full_cfg if full_cfg is not None else config, device=device)

    @staticmethod
    def _build_pipeline(config: Any) -> Any:
        """Lazily import and call ``irchroma.models.pipeline.build_pipeline`` (guarded).

        Kept out-of-line and lazy so :mod:`irchroma.infer.engine` imports/compiles even
        while ``irchroma.models.pipeline`` is still being written (Wave-B concurrent
        development). Raises a clear error only when a pipeline is actually needed.
        """
        try:
            from irchroma.models.pipeline import build_pipeline  # local lazy import
        except Exception as exc:
            raise RuntimeError(
                "Could not import irchroma.models.pipeline.build_pipeline. Ensure the "
                "pipeline module is available and torch is installed. Original error: "
                f"{exc}"
            ) from exc
        cfg = config if config is not None else Config()
        return build_pipeline(cfg)

    def attach_pipeline(self, pipeline: Any) -> "InferenceEngine":
        """Attach (or replace) the wrapped pipeline and prepare it on the device."""
        self.pipeline = pipeline
        if _HAS_TORCH:
            self._prepare_pipeline()
        return self

    # ------------------------------------------------------------------ #
    # Precision.
    # ------------------------------------------------------------------ #
    def half(self) -> "InferenceEngine":
        """Cast the wrapped pipeline to FP16 (``torch.float16``) for ~2× throughput.

        FP16 is near-lossless for SR/colorization and the default serving precision
        (docs/research/05 §C2). On CPU this is a no-op-ish cast (CPU FP16 kernels are
        limited); FP16 is intended for the CUDA serving path. Inputs passed to
        :meth:`predict` are cast to match automatically.
        """
        _require_torch()
        self._fp16 = True
        if self.pipeline is not None and hasattr(self.pipeline, "half"):
            try:
                self.pipeline = self.pipeline.half()
            except Exception:  # pragma: no cover - defensive
                pass
        return self

    def float(self) -> "InferenceEngine":
        """Cast the wrapped pipeline back to FP32 (``torch.float32``)."""
        _require_torch()
        self._fp16 = False
        if self.pipeline is not None and hasattr(self.pipeline, "float"):
            try:
                self.pipeline = self.pipeline.float()
            except Exception:  # pragma: no cover
                pass
        return self

    @property
    def dtype(self) -> Any:
        """The active compute dtype (``torch.float16`` if FP16 else ``torch.float32``)."""
        _require_torch()
        return torch.float16 if self._fp16 else torch.float32

    # ------------------------------------------------------------------ #
    # Core prediction (PyTorch).
    # ------------------------------------------------------------------ #
    def _to_input(self, x: Optional["Tensor"], *, is_label: bool = False) -> Optional["Tensor"]:
        """Move/cast an input tensor to the engine device & dtype (labels stay Long)."""
        if x is None:
            return None
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        x = x.to(self.device)
        if is_label:
            return x.long()
        return x.to(self.dtype)

    def predict(
        self,
        ir: "Tensor",
        guide: Optional["Tensor"] = None,
        semantic: Optional["Tensor"] = None,
        return_output: bool = False,
    ) -> "Tensor":
        """Run the pipeline: ``ir (+guide,+semantic) -> rgb`` (the colorized product).

        Builds a :class:`irchroma.interfaces.Sample` batch from the inputs, runs the
        wrapped pipeline's ``forward``, and returns the ``rgb`` field
        (``FloatTensor [B, 3, H*scale, W*scale]`` in ``[0, 1]``). Runs under
        ``torch.no_grad`` and moves/casts inputs to the engine device & dtype.

        Args:
          ir:       ``FloatTensor [B, C_ir, H, W]`` (or ``[C_ir, H, W]``) — IR input.
          guide:    optional HR guide band ``[B, C_g, Hg, Wg]`` for guided SR.
          semantic: optional LULC labels ``[B, H, W]`` (used by semantic conditioning
                    and the class-LUT color clamp).
          return_output: if ``True`` return the **full** ``PipelineOutput`` dict
                    (``rgb``/``sr``/``semantic_pred``/``uncertainty``/``aux``) instead
                    of just ``rgb``.

        Returns:
          ``rgb`` tensor by default, or the full ``PipelineOutput`` mapping if
          ``return_output=True``.
        """
        _require_torch()
        if self.pipeline is None:
            raise RuntimeError(
                "InferenceEngine has no pipeline attached. Use from_checkpoint(...) or "
                "attach_pipeline(...) (or use predict_onnx for the ONNX runtime path)."
            )
        squeeze = False
        if torch.is_tensor(ir) and ir.dim() == 3:
            ir = ir.unsqueeze(0)
            squeeze = True
        batch: Dict[str, Any] = {
            "ir": self._to_input(ir),
            "guide": self._to_input(guide),
            "semantic": self._to_input(semantic, is_label=True),
            "meta": {},
        }
        with torch.no_grad():
            output = self.pipeline.forward(batch)  # type: ignore[union-attr]
        if return_output:
            return output  # type: ignore[return-value]
        rgb = output["rgb"] if isinstance(output, dict) else output
        if squeeze and torch.is_tensor(rgb):
            rgb = rgb[0]
        return rgb

    __call__ = predict

    # ------------------------------------------------------------------ #
    # ONNX export.
    # ------------------------------------------------------------------ #
    def export_onnx(
        self,
        path: Optional[str] = None,
        sample_input: Optional["Tensor"] = None,
        opset: int = 17,
        dynamic_batch: bool = True,
        dynamic_hw: bool = True,
    ) -> str:
        """Export the wrapped pipeline to ONNX (``torch.onnx.export``), guarded.

        Produces a portable graph that ONNX Runtime / TensorRT / OpenVINO can execute
        (docs/research/05 §C1). A wrapper ``nn.Module`` adapts the pipeline's
        ``forward(Sample) -> PipelineOutput`` into a plain ``forward(ir) -> rgb`` graph
        (IR-only is the most portable signature; guide/semantic conditioning, if any,
        is exercised through the full torch path). Marks the batch and spatial dims as
        dynamic so the engine handles variable tile counts/sizes.

        Requires ``torch.onnx`` (always present with torch); the optional ``onnx``
        package is used only to *verify* the graph if installed. Raises a friendly
        error if no pipeline is attached.

        Args:
          path:         output ``.onnx`` path (defaults to ``config.export_path``).
          sample_input: example IR tensor for tracing (a small zero tensor is created
                        from ``config.tile_size`` if ``None``).
          opset:        ONNX opset version.
          dynamic_batch / dynamic_hw: mark batch / H,W as dynamic axes.

        Returns:
          The path the ONNX model was written to.
        """
        _require_torch()
        if self.pipeline is None:
            raise RuntimeError("export_onnx requires an attached pipeline.")
        try:
            import torch.onnx  # noqa: F401  (present with torch; explicit for clarity)
        except Exception as exc:  # pragma: no cover - extremely unusual
            raise RuntimeError(f"torch.onnx is unavailable: {exc}") from exc

        out_path = path or getattr(self.cfg, "export_path", "checkpoints/irchroma.onnx")
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)

        tile = int(getattr(self.cfg, "tile_size", 512))
        if sample_input is None:
            sample_input = torch.zeros((1, 1, tile, tile), dtype=torch.float32)
        sample_input = sample_input.to(self.device).to(self.dtype)

        wrapper = _PipelineONNXWrapper(self.pipeline).to(self.device)
        if self._fp16:
            try:
                wrapper = wrapper.half()
            except Exception:  # pragma: no cover
                pass
        wrapper.eval()

        dynamic_axes: Dict[str, Dict[int, str]] = {}
        if dynamic_batch:
            dynamic_axes.setdefault("ir", {})[0] = "batch"
            dynamic_axes.setdefault("rgb", {})[0] = "batch"
        if dynamic_hw:
            dynamic_axes.setdefault("ir", {}).update({2: "height", 3: "width"})
            dynamic_axes.setdefault("rgb", {}).update({2: "out_height", 3: "out_width"})

        torch.onnx.export(
            wrapper,
            sample_input,
            out_path,
            input_names=["ir"],
            output_names=["rgb"],
            opset_version=int(opset),
            dynamic_axes=dynamic_axes or None,
            do_constant_folding=True,
        )

        # Optional structural check if the `onnx` package is available.
        if _HAS_ONNX:  # pragma: no cover - only when onnx installed
            try:
                model = onnx.load(out_path)
                onnx.checker.check_model(model)
            except Exception as exc:
                import warnings

                warnings.warn(f"ONNX graph check warning: {exc}", RuntimeWarning, stacklevel=2)
        return out_path

    # ------------------------------------------------------------------ #
    # ONNX Runtime inference.
    # ------------------------------------------------------------------ #
    def load_onnx(
        self,
        path: Optional[str] = None,
        providers: Optional[list] = None,
    ) -> "InferenceEngine":
        """Load an ONNX model into an ONNX Runtime session, guarded by ``onnxruntime``.

        Selects execution providers automatically when ``providers`` is ``None``,
        preferring TensorRT → CUDA → CPU among those available
        (``onnxruntime.get_available_providers``), matching the docs/research/05 §C1
        portable-runtime story. Raises a friendly error if ``onnxruntime`` is absent.

        Args:
          path:      path to the ``.onnx`` file (defaults to ``config.export_path``).
          providers: explicit ONNX Runtime execution-provider list (overrides auto).

        Returns:
          ``self`` (the session is stored on ``self.ort_session``).
        """
        if not _HAS_ORT or onnxruntime is None:
            raise RuntimeError(
                "ONNX Runtime is not installed. Install it to run the ONNX inference "
                "path (`pip install onnxruntime` or `onnxruntime-gpu`)."
            )
        model_path = path or getattr(self.cfg, "export_path", "checkpoints/irchroma.onnx")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model not found: {model_path!r}")
        if providers is None:
            available = set(onnxruntime.get_available_providers())
            preference = [
                "TensorrtExecutionProvider",
                "CUDAExecutionProvider",
                "OpenVINOExecutionProvider",
                "CPUExecutionProvider",
            ]
            providers = [p for p in preference if p in available] or None
        self.ort_session = onnxruntime.InferenceSession(model_path, providers=providers)
        return self

    def predict_onnx(self, ir: Any, input_name: Optional[str] = None) -> Any:
        """Run IR→RGB through the loaded ONNX Runtime session (guarded).

        :meth:`load_onnx` must have been called first. Accepts a torch tensor or a
        numpy array; returns a numpy ``ndarray`` ``[B, 3, H*scale, W*scale]`` in
        ``[0, 1]`` (the ONNX runtime's native output). This is the portable serving
        exit (CUDA/TensorRT/OpenVINO EP) from docs/research/05 §C1.

        Args:
          ir:         IR input ``[B, C_ir, H, W]`` (torch tensor or numpy array; a
                      ``[C_ir, H, W]`` input gets a batch dim).
          input_name: ONNX graph input name (defaults to the session's first input).

        Returns:
          ``numpy.ndarray`` of the RGB output.
        """
        if self.ort_session is None:
            raise RuntimeError("predict_onnx requires load_onnx(...) to be called first.")
        try:
            import numpy as np  # local import; numpy is the ORT I/O type
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("predict_onnx requires numpy.") from exc

        if _HAS_TORCH and torch is not None and torch.is_tensor(ir):
            arr = ir.detach().cpu().numpy()
        else:
            arr = np.asarray(ir)
        if arr.ndim == 3:
            arr = arr[None, ...]
        arr = arr.astype(np.float32)

        name = input_name or self.ort_session.get_inputs()[0].name
        outputs = self.ort_session.run(None, {name: arr})
        return outputs[0]

    # ------------------------------------------------------------------ #
    # TensorRT (optional, best GPU latency).
    # ------------------------------------------------------------------ #
    def build_tensorrt(
        self,
        onnx_path: Optional[str] = None,
        engine_path: Optional[str] = None,
        precision: Optional[str] = None,
        workspace_gb: int = 4,
        int8_calibrator: Optional[Any] = None,
        min_shape: Optional[Tuple[int, int, int, int]] = None,
        opt_shape: Optional[Tuple[int, int, int, int]] = None,
        max_shape: Optional[Tuple[int, int, int, int]] = None,
    ) -> str:
        """Build a TensorRT engine from an ONNX model — best GPU latency (guarded).

        TensorRT (NVIDIA) gives the lowest GPU latency via **layer/tensor fusion** and
        reduced precision (docs/research/05 §C1–C2; ARCHITECTURE.md §9.2 #4):

          * **FP16** — ~2× throughput, near-lossless for SR/colorization; the default
            serving precision. Enabled with the builder's FP16 flag.
          * **INT8** — further latency drop; **use QAT (quantization-aware training)**
            or a representative calibration dataset (pass ``int8_calibrator``) to keep
            accuracy near FP32 (NVIDIA QAT shows ≈FP32 accuracy). Caveat: INT8 can be
            *slower* than FP16 on some layers/GPUs — **benchmark, don't assume**.

        A dynamic optimization profile (``min``/``opt``/``max`` shapes) lets one engine
        serve variable tile counts/sizes; defaults derive from ``config.tile_size``.

        This method only runs where the ``tensorrt`` Python package and an NVIDIA GPU
        are present; otherwise it raises a clear, friendly error describing how to get
        the FP16/INT8 engine path. The serialized engine is written to ``engine_path``.

        Args:
          onnx_path:    source ONNX model (defaults to ``config.export_path``; call
                        :meth:`export_onnx` first if you have not).
          engine_path:  output ``.engine``/``.plan`` path (defaults next to the ONNX).
          precision:    ``"fp16"`` (default from config) / ``"int8"`` / ``"fp32"``.
          workspace_gb: builder workspace memory pool (GiB).
          int8_calibrator: a ``tensorrt.IInt8Calibrator`` for INT8 PTQ (else QAT is
                        assumed if INT8 is requested without one).
          min_shape/opt_shape/max_shape: dynamic ``(N, C, H, W)`` profile bounds.

        Returns:
          Path to the serialized TensorRT engine.
        """
        if not _HAS_TRT or tensorrt is None:
            raise RuntimeError(
                "TensorRT is not installed. The FP16/INT8 TensorRT engine path needs "
                "the `tensorrt` Python package and an NVIDIA GPU. Use the ONNX Runtime "
                "path (load_onnx/predict_onnx) for a portable optimized runtime instead."
            )
        trt = tensorrt
        src = onnx_path or getattr(self.cfg, "export_path", "checkpoints/irchroma.onnx")
        if not os.path.exists(src):
            raise FileNotFoundError(
                f"ONNX model not found: {src!r}. Call export_onnx(...) first."
            )
        prec = str(precision or getattr(self.cfg, "precision", "fp16")).lower()
        out_path = engine_path or (os.path.splitext(src)[0] + ".engine")

        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(network_flags)
        parser = trt.OnnxParser(network, logger)
        with open(src, "rb") as fh:
            if not parser.parse(fh.read()):
                errs = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
                raise RuntimeError(f"Failed to parse ONNX for TensorRT: {errs}")

        bcfg = builder.create_builder_config()
        # Workspace memory pool (new API) with a fallback to the deprecated setter.
        try:
            bcfg.set_memory_pool_limit(
                trt.MemoryPoolType.WORKSPACE, int(workspace_gb) * (1 << 30)
            )
        except Exception:  # pragma: no cover - older TensorRT
            bcfg.max_workspace_size = int(workspace_gb) * (1 << 30)  # type: ignore[attr-defined]

        if prec == "fp16" and builder.platform_has_fast_fp16:
            bcfg.set_flag(trt.BuilderFlag.FP16)
        elif prec == "int8" and builder.platform_has_fast_int8:
            bcfg.set_flag(trt.BuilderFlag.INT8)
            # PTQ via an explicit calibrator; otherwise assume the network was QAT-trained
            # (carrying Q/DQ nodes) which keeps INT8 accuracy near FP32.
            if int8_calibrator is not None:
                bcfg.int8_calibrator = int8_calibrator

        # Dynamic shape optimization profile so one engine serves variable tiles.
        tile = int(getattr(self.cfg, "tile_size", 512))
        min_s = min_shape or (1, 1, tile, tile)
        opt_s = opt_shape or (int(getattr(self.cfg, "batch_tiles", 1) or 1), 1, tile, tile)
        max_s = max_shape or (max(opt_s[0], 1), 1, tile, tile)
        profile = builder.create_optimization_profile()
        in_name = network.get_input(0).name
        profile.set_shape(in_name, min_s, opt_s, max_s)
        bcfg.add_optimization_profile(profile)

        # Build + serialize (API differs across TensorRT versions).
        engine_bytes = None
        if hasattr(builder, "build_serialized_network"):
            engine_bytes = builder.build_serialized_network(network, bcfg)
        else:  # pragma: no cover - older TensorRT
            engine = builder.build_engine(network, bcfg)
            engine_bytes = engine.serialize() if engine is not None else None
        if engine_bytes is None:
            raise RuntimeError("TensorRT engine build failed (see the TRT logger output).")
        with open(out_path, "wb") as fh:
            fh.write(engine_bytes)
        self._trt_engine = out_path
        return out_path

    # ------------------------------------------------------------------ #
    # Introspection.
    # ------------------------------------------------------------------ #
    def info(self) -> Dict[str, Any]:
        """Return a small dict describing the engine's runtime configuration."""
        return {
            "runtime": getattr(self.cfg, "runtime", "torch"),
            "precision": "fp16" if self._fp16 else "fp32",
            "device": str(self._device) if self._device is not None else self._device_str,
            "tile_size": int(getattr(self.cfg, "tile_size", 512)),
            "has_pipeline": self.pipeline is not None,
            "has_onnx_session": self.ort_session is not None,
            "has_tensorrt_engine": self._trt_engine is not None,
            "backends_available": {
                "torch": _HAS_TORCH,
                "onnx": _HAS_ONNX,
                "onnxruntime": _HAS_ORT,
                "tensorrt": _HAS_TRT,
            },
        }


# =========================================================================== #
# ONNX export wrapper — adapts Sample->PipelineOutput to a plain ir->rgb graph.
# =========================================================================== #
if _HAS_TORCH:  # pragma: no branch - only define the nn.Module when torch is present
    import torch.nn as nn  # noqa: E402

    class _PipelineONNXWrapper(nn.Module):  # type: ignore[misc]
        """Wrap a ``forward(Sample)->PipelineOutput`` pipeline as ``forward(ir)->rgb``.

        ONNX traces a flat tensor-in/tensor-out signature most reliably; this adapter
        packs the single IR tensor into a minimal :class:`Sample` dict, runs the
        pipeline, and returns just the ``rgb`` output tensor.
        """

        def __init__(self, pipeline: Any) -> None:
            super().__init__()
            self.pipeline = pipeline

        def forward(self, ir: "Tensor") -> "Tensor":  # type: ignore[override]
            out = self.pipeline.forward({"ir": ir, "meta": {}})
            return out["rgb"] if isinstance(out, dict) else out
else:  # pragma: no cover - torch-less stand-in so references resolve

    class _PipelineONNXWrapper:  # type: ignore[no-redef]
        """Placeholder used only on torch-less boxes (never instantiated)."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("torch is required to build the ONNX export wrapper.")
