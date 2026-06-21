"""irchroma.models.pipeline — the end-to-end two-stage IR→RGB pipeline.

Owner: Builder-7 (training / serving). This module wires the (Wave-A) building
blocks into the single object that satisfies
:class:`irchroma.interfaces.PipelineProtocol`:

    ``IRChromaPipeline.forward(batch: Sample) -> PipelineOutput``

Flow (``ARCHITECTURE.md`` §3.1 "End-to-end tensor flow"):

  1. **Stage-1 — guided SR** : :class:`~irchroma.models.sr.GuidedSR`
     ``ir (+guide) -> sr [B, C_ir, Hs, Ws]`` (also exposes shared LR ``feat``).
  2. **Stage-2 — colorize**  : :class:`~irchroma.models.colorization.ColorizationGenerator`
     ``sr (+semantic) -> rgb_raw [B, 3, Hs, Ws]`` in ``[0, 1]``.
  3. **Refine — class LUT**  : :class:`~irchroma.models.semantic.ClassColorLUT`
     ``rgb_raw (+semantic) -> rgb`` chroma-clamped toward the per-class palette
     (skipped gracefully when no semantic map is available).
  4. **Refine — 3D-LUT**     : optional :class:`~irchroma.models.semantic.AdaIntLUT`
     image-adaptive trilinear refinement (config flag ``use_learned_3dlut``).
  5. **Audit — checker**     : optional frozen :class:`~irchroma.models.semantic.FrozenSegmenter`
     produces ``semantic_pred`` logits + a Dynamic-World-style ``uncertainty`` map;
     high-uncertainty pixels are honestly desaturated (``desaturate_uncertain``).

Conventions (see :mod:`irchroma.interfaces`):
  * IR        ``FloatTensor [B, C_ir, H, W]``
  * SR        ``FloatTensor [B, C_ir, H*scale, W*scale]``
  * RGB       ``FloatTensor [B, 3, H*scale, W*scale]`` in ``[0, 1]``
  * semantic  ``LongTensor  [B, H, W]`` (LULC indices)

Robustness contract: the pipeline must run with **only** ``ir`` present (inference,
no ``guide``/``semantic``/``rgb``). Missing ``guide`` → blind SISR; missing
``semantic`` → either derive a coarse label map from the frozen segmenter (if
``derive_semantic`` and a segmenter is built) or skip the class-LUT clamp entirely.

The :class:`MultiScaleDiscriminator` is **built and exposed** as ``.discriminator`` so
the :class:`~irchroma.train.Trainer` can run the two-player game; it is **not** used in
``forward`` (inference is generator-only).

``torch`` is assumed present at runtime; this file must still ``py_compile`` without it.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..config import Config, ModelConfig
from ..interfaces import BaseModel, PipelineOutput, Sample, TORCH_AVAILABLE, Tensor

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


__all__ = ["IRChromaPipeline", "build_pipeline"]


def _resolve_model_cfg(config: Any) -> ModelConfig:
    """Coerce a :class:`Config` / :class:`ModelConfig` / ``None`` into a ``ModelConfig``."""
    if config is None:
        return ModelConfig()
    if isinstance(config, ModelConfig):
        return config
    inner = getattr(config, "model", None)
    if isinstance(inner, ModelConfig):
        return inner
    # Duck-typed object that already looks like a ModelConfig (has .sr/.colorization).
    if hasattr(config, "sr") and hasattr(config, "colorization"):
        return config  # type: ignore[return-value]
    return ModelConfig()


class IRChromaPipeline(BaseModel):
    """End-to-end IR→(SR, colorized RGB, semantics, uncertainty) pipeline.

    Satisfies :class:`irchroma.interfaces.PipelineProtocol`. Construct via
    :func:`build_pipeline` (the canonical factory that wires every submodule from a
    :class:`~irchroma.config.Config`), or pass the already-built submodules directly.

    Args:
      sr_model:        Stage-1 :class:`~irchroma.models.sr.GuidedSR` (required).
      colorizer:       Stage-2 :class:`~irchroma.models.colorization.ColorizationGenerator`
                       (required).
      class_lut:       :class:`~irchroma.models.semantic.ClassColorLUT` chroma clamp, or
                       ``None`` to disable the class-conditioned clamp.
      adaint_lut:      optional :class:`~irchroma.models.semantic.AdaIntLUT` 3D-LUT refine.
      segmenter:       optional frozen :class:`~irchroma.models.semantic.FrozenSegmenter`
                       consistency checker (drives ``semantic_pred`` + ``uncertainty``).
      discriminator:   optional :class:`~irchroma.models.colorization.MultiScaleDiscriminator`
                       for training (exposed as ``.discriminator``; unused in ``forward``).
      scale:           end-to-end SR factor (``ModelConfig.scale``; bookkeeping only).
      apply_class_lut: master switch for step 3 (default ``True``).
      apply_adaint:    master switch for step 4 (default: built iff ``adaint_lut`` given).
      emit_uncertainty: run the segmenter audit + uncertainty in ``forward`` (default
                       ``True`` when a segmenter is present).
      desaturate_uncertain: honestly desaturate high-uncertainty pixels (``SemanticConfig``).
      uncertainty_threshold: confidence/uncertainty switch for desaturation.
      derive_semantic: when no ``semantic`` is supplied, derive a coarse label map from
                       the frozen segmenter (run on the raw colorization) so the class-LUT
                       clamp can still apply (default ``True`` iff a segmenter exists).
    """

    name: str = "irchroma_pipeline"

    def __init__(
        self,
        sr_model: BaseModel,
        colorizer: BaseModel,
        class_lut: Optional[Any] = None,
        adaint_lut: Optional[BaseModel] = None,
        segmenter: Optional[BaseModel] = None,
        discriminator: Optional[BaseModel] = None,
        scale: int = 4,
        apply_class_lut: bool = True,
        apply_adaint: Optional[bool] = None,
        emit_uncertainty: Optional[bool] = None,
        desaturate_uncertain: bool = True,
        uncertainty_threshold: float = 0.5,
        derive_semantic: Optional[bool] = None,
    ) -> None:
        super().__init__()  # type: ignore[misc]

        self.sr_model = sr_model
        self.colorizer = colorizer
        # ClassColorLUT is an nn.Module (registers buffers); keep it a child module so
        # ``.to(device)`` moves its lookup tables. AdaIntLUT / segmenter likewise.
        self.class_lut = class_lut
        self.adaint_lut = adaint_lut
        self.segmenter = segmenter
        # The discriminator is a *training-only* sibling. We deliberately do NOT register
        # it as a forward-used child via assignment-to-`discriminator` confusion; storing
        # it on the module is fine (it is an nn.Module so `.to()` still moves it), but it
        # is never touched by `forward`.
        self.discriminator = discriminator

        self.scale = int(scale)
        self.apply_class_lut = bool(apply_class_lut) and class_lut is not None
        self.apply_adaint = (
            bool(apply_adaint) if apply_adaint is not None else (adaint_lut is not None)
        ) and adaint_lut is not None
        self.emit_uncertainty = (
            bool(emit_uncertainty)
            if emit_uncertainty is not None
            else (segmenter is not None)
        ) and segmenter is not None
        self.desaturate_uncertain = bool(desaturate_uncertain)
        self.uncertainty_threshold = float(uncertainty_threshold)
        self.derive_semantic = (
            bool(derive_semantic) if derive_semantic is not None else (segmenter is not None)
        ) and segmenter is not None

    # ------------------------------------------------------------------ #
    @classmethod
    def from_config(cls, config: Any) -> "IRChromaPipeline":
        """Build a fully-wired pipeline from a :class:`Config` (see :func:`build_pipeline`)."""
        return build_pipeline(config)

    # ------------------------------------------------------------------ #
    # Helpers.
    # ------------------------------------------------------------------ #
    @staticmethod
    def _to_long_bhw(semantic: Tensor) -> Tensor:
        """Normalize a semantic tensor to a Long ``[B, H, W]`` label map."""
        sem = semantic
        if sem.dim() == 4 and sem.shape[1] == 1:
            sem = sem[:, 0, :, :]
        return sem.long()

    def _segment_logits(self, rgb: Tensor) -> Optional[Tensor]:
        """Run the frozen checker on ``rgb`` → ``[B, K, Hs, Ws]`` logits (or ``None``)."""
        if self.segmenter is None:
            return None
        return self.segmenter(rgb)

    # ------------------------------------------------------------------ #
    # Forward.
    # ------------------------------------------------------------------ #
    def forward(self, batch: Sample) -> PipelineOutput:
        """Run IR → (SR, colorized RGB, optional semantics/uncertainty).

        Args:
          batch: a :class:`~irchroma.interfaces.Sample`. ``ir`` is required; ``guide``
                 and ``semantic`` are optional (missing → graceful degradation).

        Returns:
          A :class:`~irchroma.interfaces.PipelineOutput` with ``rgb`` + ``sr`` (always),
          plus ``semantic_pred`` / ``uncertainty`` (when the segmenter audit runs) and an
          ``aux`` dict carrying intermediates the trainer consumes (``sr``, ``rgb_raw``,
          ``feat``, ``semantic`` used for the LUT/SPADE, ``disc_cond`` discriminator
          conditioning channels).
        """
        if not TORCH_AVAILABLE:  # pragma: no cover
            raise RuntimeError("torch is required to run IRChromaPipeline.forward().")

        ir = batch.get("ir") if hasattr(batch, "get") else batch["ir"]  # type: ignore[index]
        if ir is None:
            raise ValueError("IRChromaPipeline.forward: batch['ir'] is required.")
        guide = batch.get("guide") if hasattr(batch, "get") else None
        semantic = batch.get("semantic") if hasattr(batch, "get") else None

        aux: Dict[str, Any] = {}

        # ---- Stage 1: guided super-resolution ------------------------------ #
        sr_out = self.sr_model(ir, guide=guide, return_dict=True)
        if isinstance(sr_out, dict):
            sr = sr_out["sr"]
            feat = sr_out.get("feat")
        else:  # tolerate a bare-tensor SR model
            sr = sr_out
            feat = None
        aux["feat"] = feat

        # Normalize a provided semantic map to Long [B, H, W].
        sem_idx: Optional[Tensor] = None
        if semantic is not None:
            sem_idx = self._to_long_bhw(semantic)

        # ---- Stage 2: colorization (SPADE-conditioned when semantic present) #
        rgb_raw = self.colorizer(sr, semantic=sem_idx)
        aux["rgb_raw"] = rgb_raw

        # ---- Optional: derive a coarse semantic map for the LUT if absent --- #
        # At inference an IR-only sample has no label map; if a frozen segmenter is
        # available we run it once on the raw colorization to get a class map so the
        # O(1) class-LUT clamp can still steer color toward the per-class palette.
        sem_for_lut = sem_idx
        if sem_for_lut is None and self.derive_semantic and self.segmenter is not None:
            with torch.no_grad():
                seg_logits = self.segmenter(rgb_raw)  # [B, K, Hs, Ws]
            sem_for_lut = seg_logits.argmax(dim=1)  # [B, Hs, Ws]
            aux["semantic_derived"] = sem_for_lut

        # ---- Refine 3: class→Lab chroma clamp (O(1)/pixel) ----------------- #
        rgb = rgb_raw
        if self.apply_class_lut and self.class_lut is not None and sem_for_lut is not None:
            # The LUT clamp operates at the RGB (Hs, Ws) grid; the class map must match
            # that resolution. Resize (nearest) if it is at the LR grid.
            sem_hr = sem_for_lut
            if sem_hr.shape[-2:] != rgb.shape[-2:]:
                sem_hr = (
                    F.interpolate(
                        sem_hr.unsqueeze(1).float(), size=rgb.shape[-2:], mode="nearest"
                    )
                    .long()
                    .squeeze(1)
                )
            rgb = self.class_lut.apply(rgb, sem_hr)
            aux["semantic"] = sem_hr
        elif sem_for_lut is not None:
            aux["semantic"] = sem_for_lut

        # ---- Refine 4: learned image-adaptive 3D-LUT (optional) ------------ #
        if self.apply_adaint and self.adaint_lut is not None:
            rgb = self.adaint_lut(rgb)

        # The fully-colorized product BEFORE the (inference-only) honesty desaturation —
        # exposed so callers (the demo panel) can show the raw colorization regardless of
        # the audit's desaturation of low-confidence pixels.
        aux["rgb_colorized"] = rgb

        # ---- Audit 5: frozen checker → semantic_pred + uncertainty --------- #
        semantic_pred: Optional[Tensor] = None
        uncertainty: Optional[Tensor] = None
        if self.emit_uncertainty and self.segmenter is not None:
            # Late-import the honesty helpers (guarded; they require torch but the
            # import itself is cheap and resilient).
            try:
                from .semantic.uncertainty import (
                    desaturate_low_confidence,
                    uncertainty_from_probs,
                )
            except Exception:  # pragma: no cover - keep forward robust
                desaturate_low_confidence = None  # type: ignore
                uncertainty_from_probs = None  # type: ignore

            with torch.no_grad():
                logits = self.segmenter(rgb)  # [B, K, Hs, Ws]
            semantic_pred = logits
            if uncertainty_from_probs is not None:
                uncertainty = uncertainty_from_probs(logits, is_logits=True)  # [B,1,Hs,Ws]
                # Honest desaturation is an INFERENCE-time product step only: during
                # training we must optimize the true colorization (desaturating with a
                # randomly-initialized fallback checker would gray everything out and
                # corrupt the color gradients). The uncertainty map is still emitted for
                # logging/QA in both modes.
                if (
                    self.desaturate_uncertain
                    and desaturate_low_confidence is not None
                    and not self.training
                ):
                    rgb = desaturate_low_confidence(
                        rgb,
                        uncertainty,
                        threshold=self.uncertainty_threshold,
                        confidence_is_uncertainty=True,
                    )

        aux["sr"] = sr
        # Discriminator conditioning the trainer concatenates with the RGB image: the
        # super-resolved IR (the condition Stage-2 saw). Stored so the trainer doesn't
        # have to re-derive it. Shape [B, C_ir, Hs, Ws].
        aux["disc_cond"] = sr

        out: PipelineOutput = {
            "rgb": rgb,
            "sr": sr,
            "semantic_pred": semantic_pred,
            "uncertainty": uncertainty,
            "aux": aux,
        }
        return out


# =========================================================================== #
# Factory: wire every submodule from a Config.
# =========================================================================== #
def build_pipeline(config: Any) -> IRChromaPipeline:
    """Construct a fully-wired :class:`IRChromaPipeline` from a :class:`Config`.

    Reads the model sub-configs and instantiates each stage via its own
    ``from_config`` (so the pipeline stays a thin composition layer):

      * Stage-1  : ``GuidedSR.from_config(model.sr)``
      * Stage-2  : ``ColorizationGenerator(config=model.colorization, use_spade=..., label_nc=...)``
      * class LUT: ``ClassColorLUT(model.semantic)``           (iff ``color_space_clamp``)
      * 3D-LUT   : ``AdaIntLUT(model.semantic)``               (iff ``use_learned_3dlut``)
      * segmenter: ``FrozenSegmenter(model.semantic, force_fallback=True)`` (CPU-safe checker)
      * disc     : ``MultiScaleDiscriminator.from_config(model.colorization, in_channels=C_ir+3)``

    Submodules whose Wave-A file failed to import (torch-less docs box, or a missing
    optional dep) are skipped where optional; the two mandatory stages (SR + colorizer)
    raise a clear error if unavailable.

    Args:
      config: a :class:`~irchroma.config.Config` (uses ``config.model``) or a bare
              :class:`~irchroma.config.ModelConfig`.

    Returns:
      A ready-to-run :class:`IRChromaPipeline` (call ``.to(device)`` / ``.eval()`` /
      ``.train()`` as usual).
    """
    model_cfg = _resolve_model_cfg(config)

    # ---- Stage-1: guided SR (mandatory) ----------------------------------- #
    from .sr.guided_sr import GuidedSR

    sr_model = GuidedSR.from_config(model_cfg.sr)

    # ---- Stage-2: colorization generator (mandatory) ---------------------- #
    from .colorization.generator import ColorizationGenerator

    color_cfg = model_cfg.colorization
    colorizer = ColorizationGenerator(
        config=color_cfg,
        use_spade=bool(getattr(color_cfg, "use_spade", True)),
        label_nc=int(getattr(color_cfg, "spade_label_nc", model_cfg.num_classes)),
    )

    # ---- Refine: class→Lab chroma clamp (optional) ------------------------ #
    class_lut: Optional[Any] = None
    apply_class_lut = bool(getattr(color_cfg, "color_space_clamp", True))
    if apply_class_lut:
        try:
            from .semantic.color_lut import ClassColorLUT

            class_lut = ClassColorLUT(model_cfg.semantic)
        except Exception:  # pragma: no cover - keep build resilient
            class_lut = None
            apply_class_lut = False

    # ---- Refine: learned 3D-LUT (optional, config-gated) ------------------ #
    adaint_lut: Optional[BaseModel] = None
    use_3dlut = bool(getattr(model_cfg.semantic, "use_learned_3dlut", False))
    if use_3dlut:
        try:
            from .semantic.color_lut import AdaIntLUT

            adaint_lut = AdaIntLUT(model_cfg.semantic)
        except Exception:  # pragma: no cover
            adaint_lut = None

    # ---- Audit: frozen consistency checker (optional) --------------------- #
    # Force the pure-torch fallback so the pipeline is CPU-friendly and never tries to
    # download SegFormer weights at build time (the integration worker runs on CPU).
    segmenter: Optional[BaseModel] = None
    try:
        from .semantic.landcover import FrozenSegmenter

        segmenter = FrozenSegmenter(model_cfg.semantic, force_fallback=True)
    except Exception:  # pragma: no cover
        segmenter = None

    # ---- Discriminator (training only) ------------------------------------ #
    # Conditional input is the (super-resolved IR) condition ⊕ the RGB image, so
    # in_channels = C_ir + 3 (Pix2PixHD conditional PatchGAN).
    discriminator: Optional[BaseModel] = None
    try:
        from .colorization.discriminator import MultiScaleDiscriminator

        disc_in = int(model_cfg.c_ir) + 3
        discriminator = MultiScaleDiscriminator.from_config(color_cfg, in_channels=disc_in)
    except Exception:  # pragma: no cover
        discriminator = None

    sem_cfg = model_cfg.semantic
    pipeline = IRChromaPipeline(
        sr_model=sr_model,
        colorizer=colorizer,
        class_lut=class_lut,
        adaint_lut=adaint_lut,
        segmenter=segmenter,
        discriminator=discriminator,
        scale=int(model_cfg.scale),
        apply_class_lut=apply_class_lut,
        apply_adaint=adaint_lut is not None,
        emit_uncertainty=segmenter is not None,
        desaturate_uncertain=bool(getattr(sem_cfg, "desaturate_uncertain", True)),
        uncertainty_threshold=float(getattr(sem_cfg, "uncertainty_threshold", 0.5)),
        derive_semantic=segmenter is not None,
    )
    return pipeline
