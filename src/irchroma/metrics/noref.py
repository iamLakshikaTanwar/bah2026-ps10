"""irchroma.metrics.noref — Family C: no-reference IQA (real IR, no GT).

Owner: Builder-6 (MetricsBuilder). Implements docs/research/06 §C.

Real operational IR scenes have *no* paired RGB ground truth, so Families A/B
(full-reference) are impossible there. **No-reference (NR) IQA** scores the
*output's* quality blindly, which is exactly what we need to demonstrate
quality on real IR -> RGB products (and as an extra cross-check on the paired
test set, where NR scores should *improve* vs the raw IR input; doc 06 §C).

Backend: the **``pyiqa`` (IQA-PyTorch)** toolbox — uniform API
``m = pyiqa.create_metric(name); score = m(img_tensor)``; each metric exposes
``m.lower_better``. We mirror that flag onto ``higher_is_better`` at construction.

All metrics here are **guarded**: the module always imports; a friendly error is
raised only when a metric is *called* without ``pyiqa`` installed, and
:class:`MetricSuite` records NaN / skips gracefully. Per doc 06 these NR models are
themselves imperfect — use the *panel* (NIQE + BRISQUE + MUSIQ + …), not any one.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..interfaces import TORCH_AVAILABLE, Metric
from . import _common as _c

if TORCH_AVAILABLE:  # pragma: no cover - exercised only with torch installed
    import torch  # type: ignore
else:  # pragma: no cover
    torch = None  # type: ignore

try:  # pyiqa (IQA-PyTorch) — the NR-IQA toolbox.
    import pyiqa as _pyiqa  # type: ignore

    _HAS_PYIQA = True
except Exception:  # pragma: no cover
    _pyiqa = None  # type: ignore
    _HAS_PYIQA = False


def _pyiqa_missing(metric: str, key: str) -> RuntimeError:
    """Build a friendly ``RuntimeError`` when pyiqa is unavailable."""
    return RuntimeError(
        f"{metric} (pyiqa metric '{key}') requires the optional dependency "
        "'pyiqa', which is not installed. Install it with `pip install pyiqa` "
        "(IQA-PyTorch). It provides NIQE/BRISQUE/PIQE/MUSIQ/MANIQA/CLIP-IQA."
    )


def _to_bchw_torch(x: Any) -> "Any":
    """Coerce an image to a torch ``[B, C, H, W]`` float tensor in [0,1] for pyiqa."""
    if not TORCH_AVAILABLE:  # pragma: no cover
        raise RuntimeError("No-reference IQA requires torch (for the pyiqa backend).")
    if _c.is_torch_tensor(x):
        t = x.detach().float()
        if t.dim() == 2:
            t = t[None, None]
        elif t.dim() == 3:
            t = t[None]
        return t
    arr = _c.as_bchw(x)
    return torch.as_tensor(arr, dtype=torch.float32)


class _PyIQAMetric(Metric):
    """Base class for a pyiqa-backed no-reference metric (lazy model construction).

    Subclasses set ``_pyiqa_key`` (the ``pyiqa.create_metric`` name), ``name`` and a
    sensible default ``higher_is_better``. When pyiqa is available the actual
    direction is read from ``model.lower_better`` at construction time (authoritative).
    """

    family: str = "no_reference"
    _pyiqa_key: str = ""

    def __init__(self, device: Optional[str] = None) -> None:
        self.device = device
        self._model = None
        # If pyiqa is importable, fix the direction from the model metadata.
        if _HAS_PYIQA:
            try:
                self._ensure_model()
                self.higher_is_better = not bool(
                    getattr(self._model, "lower_better", not self.higher_is_better)
                )
            except Exception:
                # Construction may need network weights; keep the class default.
                self._model = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        if not _HAS_PYIQA:
            raise _pyiqa_missing(type(self).__name__, self._pyiqa_key)
        device = self.device
        if device is None and TORCH_AVAILABLE:
            device = "cuda" if torch.cuda.is_available() else "cpu"  # type: ignore[union-attr]
        self._model = _pyiqa.create_metric(self._pyiqa_key, device=device)  # type: ignore[union-attr]

    def __call__(
        self,
        pred: Any,
        target: Any = None,
        ctx: Optional[Dict[str, Any]] = None,
    ) -> float:
        # No-reference: ``target`` is ignored (may be None). ``pred`` is the image
        # whose blind quality we score (the IR->RGB output).
        self._ensure_model()
        img = _to_bchw_torch(pred)
        if self.device is not None:
            img = img.to(self.device)
        with torch.no_grad():  # type: ignore[union-attr]
            score = self._model(img)  # type: ignore[misc]
        # pyiqa returns a tensor (possibly per-image); reduce to a scalar mean.
        try:
            return float(score.mean().item())
        except Exception:  # pragma: no cover - non-tensor return
            return float(score)


class NIQEMetric(_PyIQAMetric):
    """NIQE — Natural Image Quality Evaluator. Lower better (opinion-unaware).

    Distance of the output's NSS features from a pristine-image model; needs no
    training labels -> a good default NR metric (doc 06 §C.19). Backed by
    ``pyiqa.create_metric('niqe')``. Guarded (friendly error if pyiqa missing).
    """

    name: str = "niqe"
    higher_is_better: bool = False
    _pyiqa_key: str = "niqe"


class BRISQUEMetric(_PyIQAMetric):
    """BRISQUE — spatial NSS-based blind quality. Lower better (classic, fast).

    Doc 06 §C.20. Backed by ``pyiqa.create_metric('brisque')``. Guarded.
    """

    name: str = "brisque"
    higher_is_better: bool = False
    _pyiqa_key: str = "brisque"


class MUSIQMetric(_PyIQAMetric):
    """MUSIQ — Multi-scale Image Quality Transformer. Higher better.

    Deep multi-scale NR-IQA with strong human correlation (default KonIQ weights;
    doc 06 §C.22). Backed by ``pyiqa.create_metric('musiq')``. Guarded (friendly
    error if pyiqa missing).
    """

    name: str = "musiq"
    higher_is_better: bool = True
    _pyiqa_key: str = "musiq"


__all__ = [
    "NIQEMetric",
    "BRISQUEMetric",
    "MUSIQMetric",
]
