"""irchroma.metrics — the 6-family, 44-metric evaluation suite.

Owner: Builder-6 (MetricsBuilder). Implements reconstruction-fidelity
(PSNR/SSIM/MS-SSIM/SAM/ERGAS/RMSE/MAE), perceptual (clean-FID/CLIP-FID/KID/LPIPS),
no-reference IQA (NIQE/BRISQUE/MUSIQ), color-specific
(CIEDE2000/colorfulness/chroma-PSNR), hallucination / semantic-faithfulness
(seg-consistency mIoU, edge IoU, gradient correlation), and efficiency
(latency/throughput/FLOPs/params/VRAM). See docs/research/06-evaluation-metrics.md
and ARCHITECTURE.md §10.

Each metric subclasses :class:`irchroma.interfaces.Metric` (callable
``__call__(pred, target=None, ctx=None) -> float`` with ``name`` /
``higher_is_better`` / ``family``); the collection is assembled via
:class:`irchroma.interfaces.MetricSuite`.

Evaluation protocol — the **geographic / scene / WRS-2-path-row holdout** (doc 06
§7.1): NEVER a random tile split (adjacent tiles are highly correlated; random
splitting leaks near-duplicates into test and inflates every metric). Train on one
set of path-rows, validate on disjoint ones, test on a *third* disjoint set spanning
different biomes (forest/water/urban/desert/snow) and seasons (≈ 70/15/15 by scene
area; ``DataConfig.split_*``). A separate **real-IR-only** set (no RGB GT) feeds the
no-reference family + qualitative hallucination panels. Acceptance is **Pareto-good
across families** (Blau–Michaeli): a win in one family that regresses another (great
FID, failing seg-consistency) is flagged as *probable hallucination, not success*.

Dependency posture (the contract): self-contained PSNR / SSIM / MS-SSIM / RMSE /
MAE / SAM / ERGAS / CIEDE2000 / colorfulness / chroma-PSNR / edge-IoU /
gradient-correlation NEVER need an optional dependency (pure numpy). LPIPS / FID /
KID / NIQE / BRISQUE / MUSIQ / seg-consistency / FLOPs are guarded — the modules
always import and the metric raises a friendly error only when *called* without its
backend; :func:`build_metric_suite` and :class:`MetricSuite` skip / NaN unavailable
metrics gracefully so one missing dep never aborts a whole evaluation run.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, List, Optional

from ..config import Config, EvalConfig
from ..interfaces import Metric, MetricSuite

# --- Family A: fidelity (self-contained numpy) ----------------------------- #
from .fidelity import (
    ERGASMetric,
    MAEMetric,
    MSSSIMMetric,
    PSNRMetric,
    RMSEMetric,
    SAMMetric,
    SSIMMetric,
)

# --- Family B: perceptual (guarded) ---------------------------------------- #
from .perceptual import FIDMetric, KIDMetric, LPIPSMetric

# --- Family C: no-reference IQA (guarded: pyiqa) --------------------------- #
from .noref import BRISQUEMetric, MUSIQMetric, NIQEMetric

# --- Family D: color (self-contained numpy) -------------------------------- #
from .color import CIEDE2000Metric, ChromaPSNRMetric, ColorfulnessMetric

# --- Family E: hallucination / faithfulness -------------------------------- #
from .hallucination import (
    EdgeIoUMetric,
    GradientCorrelationMetric,
    SegConsistencyMetric,
)

# --- Family G: efficiency -------------------------------------------------- #
from .efficiency import (
    LatencyMetric,
    LatencyMetricP95,
    ThroughputMetric,
    count_flops,
    count_params,
    peak_vram,
)


def _as_eval_config(config: Any) -> EvalConfig:
    """Coerce a ``Config`` / ``EvalConfig`` / ``None`` into an :class:`EvalConfig`."""
    if config is None:
        return EvalConfig()
    if isinstance(config, EvalConfig):
        return config
    if isinstance(config, Config):
        return config.eval
    # Duck-type: anything exposing a `.eval` EvalConfig, or already eval-like.
    inner = getattr(config, "eval", None)
    if isinstance(inner, EvalConfig):
        return inner
    if hasattr(config, "families"):
        return config  # type: ignore[return-value]
    return EvalConfig()


def _try_add(
    suite: MetricSuite,
    factory: Callable[[], Metric],
    *,
    skip_unavailable: bool,
) -> None:
    """Construct a metric via ``factory`` and add it to ``suite``, skipping on error.

    Some guarded metrics (e.g. pyiqa NR-IQA) may attempt to construct a backend at
    ``__init__`` time; if the optional dependency is missing we *warn and skip*
    rather than crash the suite assembly (matching the doc-06 / MetricSuite robustness
    contract). Pure-numpy metrics never fail here.
    """
    try:
        metric = factory()
    except Exception as exc:  # pragma: no cover - only when an optional dep is missing
        if not skip_unavailable:
            raise
        warnings.warn(
            f"Skipping a metric whose optional dependency is unavailable: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    suite.add(metric)


def build_metric_suite(
    config: Any = None,
    *,
    skip_unavailable: bool = True,
) -> MetricSuite:
    """Assemble the enabled metrics into a :class:`MetricSuite` from an ``EvalConfig``.

    Reads ``EvalConfig`` (or a full ``Config``) and registers, per enabled family in
    ``eval.families``, the metrics this package owns. SR conventions are wired from
    the config: when ``eval.y_channel_metrics`` is set, **both** RGB-domain and
    **Y-channel** PSNR/SSIM are registered (BasicSR/RCAN standard), and
    ``eval.border_shave`` is applied as the metric border (doc 06 §6). FID is
    configured for clean-fid (``eval.fid_mode``), with CLIP-FID
    (``eval.use_clip_fid``), KID (``eval.use_kid``), and precomputed reference stats
    (``eval.fid_reference_stats``). Efficiency timing uses ``eval.warmup_iters`` /
    ``eval.timing_iters`` and registers an fp16 variant when ``eval.report_fp16``.

    Any metric whose optional dependency is unavailable is **skipped with a warning**
    (``skip_unavailable=True``, the default) so assembly never crashes; set
    ``skip_unavailable=False`` to surface construction errors. Note that even when a
    guarded metric *is* registered, :meth:`MetricSuite.evaluate` still records ``NaN``
    if it raises at call time (e.g. a backend that only fails on first use).

    Args:
      config:           a :class:`~irchroma.config.Config`, an
                        :class:`~irchroma.config.EvalConfig`, or ``None`` (defaults).
      skip_unavailable: warn-and-skip metrics that fail to construct (default True).

    Returns:
      A populated :class:`~irchroma.interfaces.MetricSuite`.
    """
    cfg = _as_eval_config(config)
    families = set(getattr(cfg, "families", []) or [])
    border = int(getattr(cfg, "border_shave", 0) or 0)
    y_chan = bool(getattr(cfg, "y_channel_metrics", False))

    suite = MetricSuite()

    # ---- Family A: reconstruction fidelity (always available) ------------- #
    if "fidelity" in families:
        _try_add(suite, lambda: PSNRMetric(border=border), skip_unavailable=skip_unavailable)
        _try_add(suite, lambda: SSIMMetric(border=border), skip_unavailable=skip_unavailable)
        _try_add(suite, lambda: MSSSIMMetric(border=border), skip_unavailable=skip_unavailable)
        _try_add(suite, lambda: RMSEMetric(border=border), skip_unavailable=skip_unavailable)
        _try_add(suite, lambda: MAEMetric(border=border), skip_unavailable=skip_unavailable)
        _try_add(suite, lambda: SAMMetric(border=border), skip_unavailable=skip_unavailable)
        _try_add(suite, lambda: ERGASMetric(border=border), skip_unavailable=skip_unavailable)
        if y_chan:
            # Y-channel PSNR/SSIM (BasicSR/RCAN SR convention) — distinct names.
            _try_add(
                suite,
                lambda: PSNRMetric(border=border, y_channel=True),
                skip_unavailable=skip_unavailable,
            )
            _try_add(
                suite,
                lambda: SSIMMetric(border=border, y_channel=True),
                skip_unavailable=skip_unavailable,
            )

    # ---- Family B: perceptual / realism (guarded) ------------------------- #
    if "perceptual" in families:
        fid_mode = str(getattr(cfg, "fid_mode", "clean") or "clean")
        ref_stats = getattr(cfg, "fid_reference_stats", None)
        _try_add(suite, lambda: LPIPSMetric(net="alex"), skip_unavailable=skip_unavailable)
        _try_add(suite, lambda: LPIPSMetric(net="vgg"), skip_unavailable=skip_unavailable)
        _try_add(
            suite,
            lambda: FIDMetric(mode=fid_mode, dataset_name=ref_stats),
            skip_unavailable=skip_unavailable,
        )
        if bool(getattr(cfg, "use_clip_fid", False)):
            _try_add(
                suite,
                lambda: FIDMetric(mode=fid_mode, model_name="clip_vit_b_32"),
                skip_unavailable=skip_unavailable,
            )
        if bool(getattr(cfg, "use_kid", False)):
            _try_add(suite, lambda: KIDMetric(), skip_unavailable=skip_unavailable)

    # ---- Family C: no-reference IQA (guarded: pyiqa) ---------------------- #
    if "no_reference" in families:
        _try_add(suite, lambda: NIQEMetric(), skip_unavailable=skip_unavailable)
        _try_add(suite, lambda: BRISQUEMetric(), skip_unavailable=skip_unavailable)
        _try_add(suite, lambda: MUSIQMetric(), skip_unavailable=skip_unavailable)

    # ---- Family D: color-specific (self-contained) ------------------------ #
    if "color" in families:
        _try_add(suite, lambda: CIEDE2000Metric(border=border), skip_unavailable=skip_unavailable)
        _try_add(suite, lambda: ColorfulnessMetric(border=border), skip_unavailable=skip_unavailable)
        _try_add(
            suite,
            lambda: ChromaPSNRMetric(space="ycbcr", border=border),
            skip_unavailable=skip_unavailable,
        )

    # ---- Family E: hallucination / faithfulness --------------------------- #
    if "faithfulness" in families:
        num_classes = None
        # seg-consistency may want the LULC class count if the caller knows it.
        _try_add(
            suite,
            lambda: SegConsistencyMetric(num_classes=num_classes),
            skip_unavailable=skip_unavailable,
        )
        _try_add(suite, lambda: EdgeIoUMetric(border=border), skip_unavailable=skip_unavailable)
        _try_add(
            suite,
            lambda: GradientCorrelationMetric(border=border),
            skip_unavailable=skip_unavailable,
        )

    # ---- Family G: efficiency --------------------------------------------- #
    if "efficiency" in families:
        warmup = int(getattr(cfg, "warmup_iters", 50) or 50)
        timing = int(getattr(cfg, "timing_iters", 100) or 100)
        _try_add(
            suite,
            lambda: LatencyMetric(warmup_iters=warmup, timing_iters=timing),
            skip_unavailable=skip_unavailable,
        )
        _try_add(
            suite,
            lambda: LatencyMetricP95(warmup_iters=warmup, timing_iters=timing),
            skip_unavailable=skip_unavailable,
        )
        _try_add(
            suite,
            lambda: ThroughputMetric(warmup_iters=warmup, timing_iters=timing),
            skip_unavailable=skip_unavailable,
        )
        if bool(getattr(cfg, "report_fp16", False)):
            _try_add(
                suite,
                lambda: LatencyMetric(
                    warmup_iters=warmup, timing_iters=timing, fp16=True
                ),
                skip_unavailable=skip_unavailable,
            )
            _try_add(
                suite,
                lambda: ThroughputMetric(
                    warmup_iters=warmup, timing_iters=timing, fp16=True
                ),
                skip_unavailable=skip_unavailable,
            )

    return suite


__all__ = [
    # Family A — fidelity
    "PSNRMetric",
    "SSIMMetric",
    "MSSSIMMetric",
    "RMSEMetric",
    "MAEMetric",
    "SAMMetric",
    "ERGASMetric",
    # Family B — perceptual
    "LPIPSMetric",
    "FIDMetric",
    "KIDMetric",
    # Family C — no-reference IQA
    "NIQEMetric",
    "BRISQUEMetric",
    "MUSIQMetric",
    # Family D — color
    "CIEDE2000Metric",
    "ColorfulnessMetric",
    "ChromaPSNRMetric",
    # Family E — hallucination / faithfulness
    "SegConsistencyMetric",
    "EdgeIoUMetric",
    "GradientCorrelationMetric",
    # Family G — efficiency
    "LatencyMetric",
    "LatencyMetricP95",
    "ThroughputMetric",
    "count_params",
    "count_flops",
    "peak_vram",
    # suite assembly
    "build_metric_suite",
]
