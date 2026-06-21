"""irchroma.interfaces — the CODE CONTRACT shared by all seven builders.

This module is the single source of truth for the framework's runtime types,
tensor conventions, and abstract base classes. Every model, loss, metric, and
pipeline conforms to the protocols defined here. Keep it STABLE.

==============================================================================
TENSOR CONVENTIONS  (the contract — all builders MUST follow exactly)
==============================================================================
All image tensors are PyTorch ``FloatTensor`` in **N C H W** (channels-first)
layout unless stated otherwise.

  * IR input            : ``FloatTensor [B, C_ir, H, W]``, ``C_ir >= 1`` (default 1),
                          radiometrically standardized to approximately ``[0, 1]``.
  * HR guide (optional) : ``FloatTensor [B, C_g, Hg, Wg]`` — a co-registered higher-
                          resolution band (e.g. Landsat 15 m pan / Sentinel-2 10 m)
                          used by the guided-SR branch. ``None`` if no guide.
  * Semantic labels     : ``LongTensor  [B, H, W]`` — integer indices into the LULC
                          taxonomy ``irchroma.config.LULC_CLASSES`` (0..NUM_LULC_CLASSES-1);
                          ``irchroma.config.IGNORE_INDEX`` marks ignore/no-data.
  * SR output           : ``FloatTensor [B, C_ir, H*scale, W*scale]`` — super-resolved
                          IR (same channel count as the IR input).
  * RGB output          : ``FloatTensor [B, 3, H*scale, W*scale]`` in ``[0, 1]`` (sRGB,
                          R, G, B order). This is the final colorized product.
  * Uncertainty (opt.)  : ``FloatTensor [B, 1, H*scale, W*scale]`` in ``[0, 1]`` — per-
                          pixel confidence-complement (1 = least reliable). High values
                          are desaturated / flagged (anti-hallucination honesty).

Coordinate / metadata travels in ``Sample["meta"]`` (a plain dict): CRS, transform,
bounds, tile z/x/y or H3/quadkey, acquisition date, source id, scale factor, etc.

``scale`` is the end-to-end super-resolution factor (``irchroma.config.ModelConfig.scale``).
At inference, ``rgb`` and ``sr`` are produced at ``scale``x the input IR resolution.

If ``torch`` is not importable (e.g. a docs/CI box without it), this module still
imports: light stand-ins are defined for ``torch.nn.Module`` / ``Tensor`` so type
references resolve. **Assume torch is normally present at runtime.**
"""

from __future__ import annotations

import abc
import typing
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Protocol,
    Tuple,
    runtime_checkable,
)

try:  # TypedDict lives in typing on 3.8+, but be defensive.
    from typing import TypedDict
except Exception:  # pragma: no cover
    from typing_extensions import TypedDict  # type: ignore

# --------------------------------------------------------------------------- #
# Guarded torch import. interfaces.py must import even without torch installed.
# --------------------------------------------------------------------------- #
try:
    import torch  # type: ignore
    from torch import Tensor  # type: ignore
    import torch.nn as nn  # type: ignore

    TORCH_AVAILABLE: bool = True
    _ModuleBase = nn.Module  # real base class
except Exception:  # pragma: no cover - torch-less environments (docs/CI)
    torch = None  # type: ignore
    TORCH_AVAILABLE = False

    class _TensorStub:  # minimal stand-in so annotations like ``-> Tensor`` resolve
        """Placeholder for ``torch.Tensor`` when torch is unavailable.

        Exists purely so this contract module imports in torch-less environments;
        it is never instantiated on the real runtime path.
        """

        def __init__(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
            raise RuntimeError(
                "torch is not installed; irchroma.interfaces.Tensor is a stub. "
                "Install torch to use the runtime tensor types."
            )

    Tensor = _TensorStub  # type: ignore

    class _ModuleStub:  # stand-in base so ``class X(BaseModel)`` is definable
        """Placeholder for ``torch.nn.Module`` when torch is unavailable."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
            pass

        def __call__(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
            return self.forward(*args, **kwargs)

        def forward(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
            raise NotImplementedError

    class _NNNamespace:  # expose ``nn.Module`` name for downstream code
        Module = _ModuleStub

    nn = _NNNamespace()  # type: ignore
    _ModuleBase = _ModuleStub  # type: ignore


# =========================================================================== #
# 1. CORE DATA STRUCTURES (TypedDicts)
# =========================================================================== #
class Sample(TypedDict, total=False):
    """One training/inference example flowing through the pipeline.

    Keys (all optional via ``total=False`` so an inference-time IR-only sample is
    valid, but ``ir`` and ``meta`` are effectively REQUIRED in practice):

      * ``ir``        : ``FloatTensor [B, C_ir, H, W]`` — the IR input (required).
      * ``rgb``       : ``FloatTensor [B, 3, H*scale, W*scale]`` in [0,1] — ground-truth
                        RGB target. ``None`` / absent at inference.
      * ``guide``     : ``FloatTensor [B, C_g, Hg, Wg]`` — optional HR guide band.
      * ``semantic``  : ``LongTensor  [B, H, W]`` — optional per-pixel LULC labels
                        (indices into ``irchroma.config.LULC_CLASSES``).
      * ``meta``      : ``dict`` — geospatial + provenance metadata (CRS, transform,
                        bounds, tile key, date, source, scale, uncertainty sources...).
    """

    ir: Tensor
    rgb: Optional[Tensor]
    guide: Optional[Tensor]
    semantic: Optional[Tensor]
    meta: Dict[str, Any]


class PipelineOutput(TypedDict, total=False):
    """What :class:`PipelineProtocol.forward` returns.

      * ``rgb``           : ``FloatTensor [B, 3, H*scale, W*scale]`` in [0,1] — final
                            colorized super-resolved product (required).
      * ``sr``            : ``FloatTensor [B, C_ir, H*scale, W*scale]`` — the Stage-1
                            super-resolved IR (required; enables SR-only metrics).
      * ``semantic_pred`` : ``LongTensor [B, H*scale, W*scale]`` or class-logits
                            ``FloatTensor [B, K, H*scale, W*scale]`` — predicted LULC
                            (optional; from the guidance/checker head).
      * ``uncertainty``   : ``FloatTensor [B, 1, H*scale, W*scale]`` in [0,1] — per-pixel
                            uncertainty map (optional; high => desaturate + flag).
      * ``aux``           : ``dict`` — anything else (discriminator features, intermediate
                            Lab tensors, attention maps, timings, LUT indices, ...).
    """

    rgb: Tensor
    sr: Tensor
    semantic_pred: Optional[Tensor]
    uncertainty: Optional[Tensor]
    aux: Dict[str, Any]


# =========================================================================== #
# 2. MODEL BASE + PIPELINE PROTOCOL
# =========================================================================== #
class BaseModel(_ModuleBase):  # type: ignore[misc]
    """Common base for every neural module in irchroma (subclasses ``torch.nn.Module``).

    Subclasses MUST implement ``forward`` following the tensor conventions in this
    module's docstring. Conventions by stage:

      * SR model           : ``forward(ir, guide=None) -> sr``
                             ``ir``  : ``[B, C_ir, H, W]`` ;
                             ``guide``: ``[B, C_g, Hg, Wg]`` or ``None`` ;
                             ``sr``  : ``[B, C_ir, H*scale, W*scale]``.
      * Colorization model : ``forward(sr_or_ir, semantic=None, guide=None) -> rgb``
                             returns ``[B, 3, H*scale, W*scale]`` in [0,1].
      * Segmenter/checker  : ``forward(rgb) -> logits [B, K, H, W]`` (K = num classes).

    This base intentionally adds no required state so it stays a thin, stable marker
    that tooling (and the ``CompositeLoss``/pipeline) can rely on. Builders may add
    ``@property name`` / ``num_parameters`` helpers as needed.
    """

    #: Short identifier set by subclasses (e.g. "nafnet_sr", "pix2pixhd_ddcolor").
    name: str = "base_model"

    def forward(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - abstract
        raise NotImplementedError("BaseModel subclasses must implement forward().")

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Return the parameter count (helper for the efficiency metrics)."""
        if not TORCH_AVAILABLE:  # pragma: no cover
            return 0
        params = self.parameters()  # type: ignore[attr-defined]
        return int(sum(p.numel() for p in params if (p.requires_grad or not trainable_only)))


@runtime_checkable
class PipelineProtocol(Protocol):
    """The end-to-end pipeline interface (structural typing).

    Any object implementing ``forward(batch: Sample) -> PipelineOutput`` satisfies
    this protocol. The canonical implementation is ``IRChromaPipeline`` in
    ``irchroma.models`` (two-stage: guided SR -> semantic-conditioned colorization
    -> O(1) 3D-LUT refinement -> no-hallucination audit).
    """

    def forward(self, batch: Sample) -> PipelineOutput:
        """Run IR -> (SR, colorized RGB, optional semantics/uncertainty)."""
        ...


# =========================================================================== #
# 3. LOSS CONTRACT
# =========================================================================== #
class LossTerm(_ModuleBase):  # type: ignore[misc]
    """A single named loss term (subclasses ``torch.nn.Module``).

    Contract:
      ``forward(output: PipelineOutput, target: Sample, ctx: dict) -> Tensor``
    returning a **scalar** tensor (0-dim). ``ctx`` carries shared, possibly heavy
    state so terms don't recompute it: e.g. the frozen segmenter/checker, the
    discriminator and its features, the color-LUT, a precomputed Lab conversion,
    the current training step, per-pixel weights, and the active ``LossConfig``.

    Terms must be robust to missing optional inputs: if a term needs ``target['rgb']``
    (paired GT) and it is ``None`` (pure inference / unpaired tile), return a zero
    scalar rather than raising, so the same stack works paired and unpaired.
    """

    #: Stable key used by ``CompositeLoss`` and ``LossConfig`` (e.g. "color_l1").
    name: str = "loss_term"

    def forward(
        self,
        output: PipelineOutput,
        target: Sample,
        ctx: Dict[str, Any],
    ) -> Tensor:  # pragma: no cover - abstract
        raise NotImplementedError("LossTerm subclasses must implement forward().")


class CompositeLoss(_ModuleBase):  # type: ignore[misc]
    """Weighted sum of named :class:`LossTerm` s.

    Holds ``{name: (LossTerm, weight)}`` and, on ``forward``, evaluates each term,
    multiplies by its weight, and returns::

        (total: Tensor, components: Dict[str, float])

    where ``components`` maps each term name to its *weighted* scalar value (float,
    detached) for logging. Terms with weight ``0.0`` are skipped entirely (cheap).

    Construct directly, or via :meth:`from_config` which reads weights by attribute
    name from an ``irchroma.config.LossConfig`` (each registered term's ``name`` must
    match a ``LossConfig`` field).
    """

    def __init__(self, terms: Optional[Mapping[str, Tuple[LossTerm, float]]] = None) -> None:
        super().__init__()  # type: ignore[misc]
        # Keep modules registered (so .to(device)/.parameters() work) while also
        # tracking weights in a parallel plain dict.
        self._weights: Dict[str, float] = {}
        if TORCH_AVAILABLE:
            self._terms = nn.ModuleDict()  # type: ignore[attr-defined]
        else:  # pragma: no cover
            self._terms = {}  # type: ignore[assignment]
        for name, (term, weight) in (terms or {}).items():
            self.add_term(name, term, weight)

    def add_term(self, name: str, term: LossTerm, weight: float) -> None:
        """Register (or replace) a named term with its scalar weight."""
        self._terms[name] = term  # type: ignore[index]
        self._weights[name] = float(weight)

    def set_weight(self, name: str, weight: float) -> None:
        """Update a term's weight in place (e.g. for loss curricula / annealing)."""
        if name not in self._weights:
            raise KeyError(f"Unknown loss term {name!r}.")
        self._weights[name] = float(weight)

    @property
    def weights(self) -> Dict[str, float]:
        """Read-only copy of the current ``{name: weight}`` mapping."""
        return dict(self._weights)

    def forward(
        self,
        output: PipelineOutput,
        target: Sample,
        ctx: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Return ``(total_scalar, {name: weighted_value})``."""
        ctx = ctx if ctx is not None else {}
        components: Dict[str, float] = {}
        total: Any = None
        for name, term in self._terms.items():  # type: ignore[union-attr]
            weight = self._weights.get(name, 0.0)
            if weight == 0.0:
                continue
            value = term(output, target, ctx)  # scalar Tensor
            weighted = value * weight
            total = weighted if total is None else (total + weighted)
            # Detach for logging without holding the graph.
            try:
                components[name] = float(weighted.detach().item())  # type: ignore[attr-defined]
            except Exception:  # pragma: no cover - non-tensor fallback
                components[name] = float(weighted)
        if total is None:
            # No active terms -> zero scalar (keep dtype/device neutral on torch).
            if TORCH_AVAILABLE:
                total = torch.zeros((), dtype=torch.float32)  # type: ignore[union-attr]
            else:  # pragma: no cover
                total = 0.0  # type: ignore[assignment]
        return total, components  # type: ignore[return-value]

    @classmethod
    def from_config(
        cls,
        loss_cfg: Any,
        term_factories: Mapping[str, Callable[[], LossTerm]],
    ) -> "CompositeLoss":
        """Build a :class:`CompositeLoss` from a ``LossConfig`` + term factories.

        ``term_factories`` maps a term name (matching a ``LossConfig`` attribute) to a
        zero-arg callable that constructs the corresponding :class:`LossTerm`. Only
        terms whose config weight is non-zero are instantiated (lazy / cheap).
        """
        composite = cls()
        for name, factory in term_factories.items():
            weight = float(getattr(loss_cfg, name, 0.0) or 0.0)
            if weight == 0.0:
                continue
            composite.add_term(name, factory(), weight)
        return composite


# =========================================================================== #
# 4. METRIC CONTRACT
# =========================================================================== #
class Metric(abc.ABC):
    """A single evaluation metric (callable; NOT an ``nn.Module`` by default).

    Contract:
      ``__call__(pred, target=None, ctx=None) -> float``
    where ``pred`` is the model output (tensor / array / image-set path, metric-
    dependent) and ``target`` is the reference (``None`` for no-reference IQA and
    most efficiency metrics). ``ctx`` passes shared helpers (frozen detector/
    segmenter, device, normalization spec, label maps, ...).

    Each metric declares:
      * ``name``            : stable identifier (e.g. "psnr", "clean_fid", "ciede2000").
      * ``higher_is_better``: ``True`` if larger is better (PSNR/SSIM/mAP), else ``False``
                              (FID/LPIPS/CIEDE2000/latency).
      * ``family``          : one of {"fidelity","perceptual","no_reference","color",
                              "faithfulness","downstream","efficiency"} (docs/research/06).
    """

    name: str = "metric"
    higher_is_better: bool = True
    family: str = "fidelity"

    @abc.abstractmethod
    def __call__(
        self,
        pred: Any,
        target: Any = None,
        ctx: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Compute the metric and return a Python ``float``."""
        raise NotImplementedError


class MetricSuite:
    """Runs a named collection of :class:`Metric` s and aggregates results.

    Holds ``{name: Metric}``; :meth:`evaluate` runs each over a (pred, target, ctx)
    triple and returns ``{name: float}``. Metrics that raise are recorded as ``NaN``
    (so one missing optional dependency never aborts a whole evaluation run) unless
    ``strict=True``.
    """

    def __init__(self, metrics: Optional[Mapping[str, Metric]] = None) -> None:
        self._metrics: Dict[str, Metric] = dict(metrics or {})

    def add(self, metric: Metric, name: Optional[str] = None) -> None:
        """Register a metric (keyed by ``name`` or ``metric.name``)."""
        self._metrics[name or metric.name] = metric

    @property
    def names(self) -> List[str]:
        """The registered metric names."""
        return list(self._metrics.keys())

    def directions(self) -> Dict[str, bool]:
        """Map ``{name: higher_is_better}`` for ranking / Pareto checks."""
        return {n: m.higher_is_better for n, m in self._metrics.items()}

    def evaluate(
        self,
        pred: Any,
        target: Any = None,
        ctx: Optional[Dict[str, Any]] = None,
        strict: bool = False,
    ) -> Dict[str, float]:
        """Run all metrics; return ``{name: value}`` (NaN on error unless ``strict``)."""
        results: Dict[str, float] = {}
        for name, metric in self._metrics.items():
            try:
                results[name] = float(metric(pred, target, ctx))
            except Exception:
                if strict:
                    raise
                results[name] = float("nan")
        return results


# =========================================================================== #
# 5. COLOR-LUT PROTOCOL (O(1) class -> CIE-Lab chroma clamp)
# =========================================================================== #
@runtime_checkable
class ColorLUTProtocol(Protocol):
    """O(1)-per-pixel class-conditioned color constraint (docs/research/04, 05 §B).

    Implementations clamp/steer the predicted color toward the per-class plausible
    palette (water->blue, veg->green, built->gray) via a precomputed
    ``class_id -> (mu_Lab, sigma_Lab, [lo, hi])`` table, indexed by the per-pixel
    semantic label. The op is a single gather + elementwise clip -> genuinely O(1)
    per pixel and fully vectorized over a tile.
    """

    def apply(self, rgb: Tensor, semantic: Tensor) -> Tensor:
        """Return color-constrained RGB.

        Args:
          rgb     : ``FloatTensor [B, 3, H, W]`` in [0,1] — the raw predicted color.
          semantic: ``LongTensor  [B, H, W]`` — per-pixel LULC class indices.

        Returns:
          ``FloatTensor [B, 3, H, W]`` in [0,1] with chroma clamped toward the class
          palette (luminance left freer so SR texture/detail survives).
        """
        ...


# =========================================================================== #
# 6. CONVENIENCE TYPE ALIASES (shared vocabulary for builders)
# =========================================================================== #
#: A factory that builds a fresh loss term (used by ``CompositeLoss.from_config``).
LossTermFactory = Callable[[], LossTerm]
#: A shared context dict threaded through losses/metrics/pipeline.
Context = MutableMapping[str, Any]
#: Z/X/Y web-mercator tile coordinate (cache-key source; docs/research/05 §A5).
TileXYZ = Tuple[int, int, int]


__all__ = [
    # torch availability
    "TORCH_AVAILABLE",
    "Tensor",
    # data structures
    "Sample",
    "PipelineOutput",
    # model + pipeline
    "BaseModel",
    "PipelineProtocol",
    # losses
    "LossTerm",
    "CompositeLoss",
    "LossTermFactory",
    # metrics
    "Metric",
    "MetricSuite",
    # color LUT
    "ColorLUTProtocol",
    # aliases
    "Context",
    "TileXYZ",
]
