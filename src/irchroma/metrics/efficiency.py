"""irchroma.metrics.efficiency — Family G: efficiency / scalability (per-tile).

Owner: Builder-6 (MetricsBuilder). Implements docs/research/06 §G.

PS-10 requires inference time per tile. Naive ``time.time()`` around a CUDA call is
*wrong* because GPU ops are asynchronous (doc 06 §G.39). These metrics measure it
**rigorously**:

  * :class:`LatencyMetric` — ms/tile with **warmup** (GPU power-state ramp),
    **``torch.cuda.synchronize()``** (or CUDA events), averaging over ≥100 runs,
    reporting mean / std / **p50 / p95**, with an **fp16** (AMP) option (the
    deployment-relevant number). Lower better.
  * :class:`ThroughputMetric` — tiles/s at a given batch (saturates parallelism).
    Higher better.
  * :func:`count_params` — total / trainable parameter count.
  * :func:`count_flops` — forward MACs/FLOPs via ``thop`` or ``fvcore`` (guarded).
  * :func:`peak_vram` — peak CUDA memory (MB) after a forward, via
    ``torch.cuda.max_memory_allocated`` (guarded; returns NaN without CUDA).

These operate on a **model + input** (not pred/target). The :class:`Metric` contract
``__call__(pred, target=None, ctx=None)`` is honoured by treating ``pred`` as the
**model** and reading the input tensor/shape from ``ctx`` (e.g.
``ctx['input']`` or ``ctx['input_shape']``) or from sensible defaults.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Sequence, Tuple

from ..interfaces import TORCH_AVAILABLE, Metric

try:
    import numpy as np  # type: ignore

    _NUMPY = True
except Exception:  # pragma: no cover
    np = None  # type: ignore
    _NUMPY = False

if TORCH_AVAILABLE:  # pragma: no cover - exercised only with torch installed
    import torch  # type: ignore
else:  # pragma: no cover
    torch = None  # type: ignore

# Guarded FLOP counters.
try:
    from thop import profile as _thop_profile  # type: ignore

    _HAS_THOP = True
except Exception:  # pragma: no cover
    _thop_profile = None  # type: ignore
    _HAS_THOP = False

try:
    from fvcore.nn import FlopCountAnalysis as _FvFlops  # type: ignore

    _HAS_FVCORE = True
except Exception:  # pragma: no cover
    _FvFlops = None  # type: ignore
    _HAS_FVCORE = False

# Guarded NVML for power/energy (optional; doc 06 §G.44).
try:
    import pynvml as _pynvml  # type: ignore

    _HAS_PYNVML = True
except Exception:  # pragma: no cover
    _pynvml = None  # type: ignore
    _HAS_PYNVML = False


# Default input shape for an efficiency probe: one 512x512 single-channel IR tile.
_DEFAULT_INPUT_SHAPE: Tuple[int, int, int, int] = (1, 1, 512, 512)


def _percentile(values: Sequence[float], q: float) -> float:
    """Return the ``q``-th percentile (0–100) of ``values`` without numpy if needed."""
    if _NUMPY:
        return float(np.percentile(np.asarray(values, dtype=np.float64), q))
    if not values:  # pragma: no cover
        return float("nan")
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    pos = (q / 100.0) * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return float(s[lo] * (1.0 - frac) + s[hi] * frac)


def _resolve_input(model: Any, ctx: Optional[Dict[str, Any]]) -> "Any":
    """Build the input tensor for a timing/FLOP probe from ctx or defaults.

    Resolution order:
      1. ``ctx['input']`` — an explicit input tensor (used as-is).
      2. ``ctx['input_shape']`` — a shape tuple; a random tensor is allocated.
      3. ``_DEFAULT_INPUT_SHAPE`` (one 512x512 single-channel tile).
    The tensor is placed on ``ctx['device']`` (or the model's device if discoverable,
    else CPU).
    """
    if not TORCH_AVAILABLE:  # pragma: no cover
        raise RuntimeError("Efficiency metrics require torch.")
    ctx = ctx or {}
    if ctx.get("input") is not None:
        return ctx["input"]
    shape = tuple(ctx.get("input_shape", _DEFAULT_INPUT_SHAPE))
    device = ctx.get("device")
    if device is None:
        try:
            device = next(model.parameters()).device  # type: ignore[union-attr]
        except Exception:
            device = "cpu"
    return torch.randn(*shape, device=device)  # type: ignore[union-attr]


def _is_cuda(x: Any) -> bool:
    """True if a tensor/model lives on CUDA."""
    try:
        if hasattr(x, "is_cuda"):
            return bool(x.is_cuda)
        dev = next(x.parameters()).device  # type: ignore[union-attr]
        return dev.type == "cuda"
    except Exception:
        return False


class LatencyMetric(Metric):
    """Inference latency per tile (ms). Lower better. Rigorous GPU timing.

    Protocol (doc 06 §G.39): warmup (power-state ramp), ``cuda.synchronize()`` (or
    CUDA events) around each timed forward, average over ``timing_iters`` (≥100),
    report **mean** (default) with std and **p50/p95** available, and an **fp16**
    (AMP autocast) option (the deployment-relevant number). Fixed tile size; batch
    size is part of the input shape.

    Usage via the :class:`Metric` contract: ``pred`` is the **model**; the input is
    taken from ``ctx['input']`` / ``ctx['input_shape']`` (default one 512x512x1 tile).
    ``__call__`` returns the chosen ``report`` statistic in **milliseconds**; richer
    stats are available via :meth:`measure`.

    Args:
      warmup_iters: dummy forwards before timing (default 50; ``EvalConfig.warmup_iters``).
      timing_iters: timed forwards to average (default 100; ``EvalConfig.timing_iters``).
      fp16:         run under ``torch.autocast`` fp16 (CUDA) — the deployment number.
      report:       which statistic ``__call__`` returns: ``mean`` | ``p50`` | ``p95`` | ``std``.
    """

    name: str = "latency_ms"
    higher_is_better: bool = False
    family: str = "efficiency"

    def __init__(
        self,
        warmup_iters: int = 50,
        timing_iters: int = 100,
        fp16: bool = False,
        report: str = "mean",
        device: Optional[str] = None,
    ) -> None:
        self.warmup_iters = int(warmup_iters)
        self.timing_iters = int(timing_iters)
        self.fp16 = bool(fp16)
        self.report = str(report)
        self.device = device
        if fp16:
            self.name = "latency_ms_fp16"
        if report != "mean":
            self.name = f"{self.name}_{report}"

    def measure(self, model: Any, ctx: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Time the model and return ``{mean, std, p50, p95, min, max}`` in ms."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("LatencyMetric requires torch.")
        inp = _resolve_input(model, ctx)
        if self.device is not None and hasattr(inp, "to"):
            inp = inp.to(self.device)
            if hasattr(model, "to"):
                model = model.to(self.device)
        on_cuda = _is_cuda(inp) or _is_cuda(model)
        if hasattr(model, "eval"):
            model.eval()

        use_amp = self.fp16 and on_cuda

        def _forward() -> None:
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):  # type: ignore[union-attr]
                    model(inp)
            else:
                model(inp)

        with torch.no_grad():  # type: ignore[union-attr]
            # Warmup (GPU clocks/caches ramp; also triggers lazy init).
            for _ in range(max(0, self.warmup_iters)):
                _forward()
            if on_cuda:
                torch.cuda.synchronize()  # type: ignore[union-attr]

            times_ms = []
            for _ in range(max(1, self.timing_iters)):
                if on_cuda:
                    start = torch.cuda.Event(enable_timing=True)  # type: ignore[union-attr]
                    end = torch.cuda.Event(enable_timing=True)  # type: ignore[union-attr]
                    start.record()
                    _forward()
                    end.record()
                    torch.cuda.synchronize()  # type: ignore[union-attr]
                    times_ms.append(float(start.elapsed_time(end)))  # ms
                else:
                    t0 = time.perf_counter()
                    _forward()
                    times_ms.append((time.perf_counter() - t0) * 1000.0)

        mean = float(sum(times_ms) / len(times_ms))
        if _NUMPY:
            std = float(np.std(np.asarray(times_ms, dtype=np.float64)))
        else:  # pragma: no cover
            std = float((sum((x - mean) ** 2 for x in times_ms) / len(times_ms)) ** 0.5)
        return {
            "mean": mean,
            "std": std,
            "p50": _percentile(times_ms, 50.0),
            "p95": _percentile(times_ms, 95.0),
            "min": float(min(times_ms)),
            "max": float(max(times_ms)),
        }

    def __call__(
        self,
        pred: Any,
        target: Any = None,
        ctx: Optional[Dict[str, Any]] = None,
    ) -> float:
        stats = self.measure(pred, ctx)
        return float(stats.get(self.report, stats["mean"]))


class ThroughputMetric(Metric):
    """Throughput (tiles/s). Higher better.

    ``throughput = (timing_iters * batch_size) / total_time`` at the input batch
    (saturates parallelism; doc 06 §G.40). Reuses :class:`LatencyMetric`'s rigorous
    timing (warmup + sync) on the per-batch latency, then converts. ``pred`` is the
    model; the input/batch is taken from ``ctx`` (default one 512x512x1 tile).
    """

    name: str = "throughput_tiles_s"
    higher_is_better: bool = True
    family: str = "efficiency"

    def __init__(
        self,
        warmup_iters: int = 50,
        timing_iters: int = 100,
        fp16: bool = False,
        device: Optional[str] = None,
    ) -> None:
        self._latency = LatencyMetric(
            warmup_iters=warmup_iters,
            timing_iters=timing_iters,
            fp16=fp16,
            report="mean",
            device=device,
        )
        if fp16:
            self.name = "throughput_tiles_s_fp16"

    def __call__(
        self,
        pred: Any,
        target: Any = None,
        ctx: Optional[Dict[str, Any]] = None,
    ) -> float:
        if not TORCH_AVAILABLE:
            raise RuntimeError("ThroughputMetric requires torch.")
        inp = _resolve_input(pred, ctx)
        try:
            batch = int(inp.shape[0])
        except Exception:
            batch = 1
        stats = self._latency.measure(pred, ctx)
        ms_per_batch = stats["mean"]
        if ms_per_batch <= 0:
            return float("inf")
        return float(batch * 1000.0 / ms_per_batch)


def count_params(model: Any, trainable_only: bool = False) -> int:
    """Return a model's parameter count (doc 06 §G.42). Lower better.

    Args:
      model:          a ``torch.nn.Module``.
      trainable_only: count only parameters with ``requires_grad``.

    Returns:
      Integer parameter count (0 if torch is unavailable).
    """
    if not TORCH_AVAILABLE:  # pragma: no cover
        return 0
    try:
        return int(
            sum(
                p.numel()
                for p in model.parameters()  # type: ignore[union-attr]
                if (p.requires_grad or not trainable_only)
            )
        )
    except Exception:
        return 0


def count_flops(
    model: Any,
    input_shape: Sequence[int] = _DEFAULT_INPUT_SHAPE,
    return_macs: bool = False,
    device: Optional[str] = None,
) -> float:
    """Count forward FLOPs (or MACs) for ``model`` on a probe input. Lower better.

    Prefers ``thop`` (``thop.profile`` -> MACs), falls back to ``fvcore``
    (``FlopCountAnalysis`` -> MACs). **1 MAC = 2 FLOPs** (doc 06 §G.41); by default
    this returns **FLOPs** (``return_macs=False`` -> ``2 * MACs``).

    Args:
      model:        a ``torch.nn.Module``.
      input_shape:  ``[B, C, H, W]`` probe shape (default one 512x512x1 tile).
      return_macs:  return MACs instead of FLOPs.
      device:       device to allocate the probe input on.

    Raises:
      RuntimeError: if neither ``thop`` nor ``fvcore`` is installed (friendly error).
    """
    if not TORCH_AVAILABLE:  # pragma: no cover
        raise RuntimeError("count_flops requires torch.")
    if device is None:
        try:
            device = next(model.parameters()).device  # type: ignore[union-attr]
        except Exception:
            device = "cpu"
    inp = torch.randn(*tuple(input_shape), device=device)  # type: ignore[union-attr]
    if hasattr(model, "eval"):
        model.eval()
    macs: Optional[float] = None
    if _HAS_THOP:
        with torch.no_grad():  # type: ignore[union-attr]
            macs_val, _params = _thop_profile(model, inputs=(inp,), verbose=False)  # type: ignore[misc]
        macs = float(macs_val)
    elif _HAS_FVCORE:
        with torch.no_grad():  # type: ignore[union-attr]
            macs = float(_FvFlops(model, inp).total())  # type: ignore[misc]
    if macs is None:
        raise RuntimeError(
            "count_flops requires the optional dependency 'thop' or 'fvcore'. "
            "Install one with `pip install thop` (or `pip install fvcore`)."
        )
    return macs if return_macs else macs * 2.0


def peak_vram(
    model: Any = None,
    input_shape: Sequence[int] = _DEFAULT_INPUT_SHAPE,
    ctx: Optional[Dict[str, Any]] = None,
) -> float:
    """Peak CUDA memory (MB) for a forward pass. Lower better (doc 06 §G.43).

    Resets the CUDA peak-memory stat, runs one forward at the target tile/batch, and
    returns ``torch.cuda.max_memory_allocated`` in MB. If CUDA is unavailable returns
    ``float('nan')`` (a graceful no-op so the suite never crashes on CPU).

    If ``model`` is ``None`` the *current* peak (already accumulated) is reported.
    """
    if not TORCH_AVAILABLE or not torch.cuda.is_available():  # type: ignore[union-attr]
        return float("nan")
    device = (ctx or {}).get("device")
    torch.cuda.reset_peak_memory_stats(device)  # type: ignore[union-attr]
    if model is not None:
        if device is None:
            try:
                device = next(model.parameters()).device  # type: ignore[union-attr]
            except Exception:
                device = "cuda"
        inp = (ctx or {}).get("input")
        if inp is None:
            inp = torch.randn(*tuple(input_shape), device=device)  # type: ignore[union-attr]
        if hasattr(model, "eval"):
            model.eval()
        with torch.no_grad():  # type: ignore[union-attr]
            model(inp)
        torch.cuda.synchronize(device)  # type: ignore[union-attr]
    peak_bytes = int(torch.cuda.max_memory_allocated(device))  # type: ignore[union-attr]
    return float(peak_bytes) / (1024.0 ** 2)


class LatencyMetricP95(LatencyMetric):
    """Convenience subclass reporting the **p95** latency (ms). Lower better.

    Equivalent to ``LatencyMetric(report='p95')``; provided as a named class so the
    suite can register both mean and tail latency cleanly.
    """

    def __init__(
        self,
        warmup_iters: int = 50,
        timing_iters: int = 100,
        fp16: bool = False,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(
            warmup_iters=warmup_iters,
            timing_iters=timing_iters,
            fp16=fp16,
            report="p95",
            device=device,
        )


__all__ = [
    "LatencyMetric",
    "LatencyMetricP95",
    "ThroughputMetric",
    "count_params",
    "count_flops",
    "peak_vram",
]
