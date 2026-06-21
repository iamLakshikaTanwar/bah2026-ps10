"""irchroma.metrics.perceptual — Family B: perceptual / realism (deep-feature).

Owner: Builder-6 (MetricsBuilder). Implements docs/research/06 §B.

These quantify "does the output look like a real RGB satellite image" using deep
features and distribution-level distances. Report **alongside** Family A
(fidelity) — by the Perception–Distortion tradeoff (Blau & Michaeli, CVPR'18) a
GAN can score excellent FID while *hallucinating*, so these are never sufficient
on their own (see Family E for the faithfulness audit).

Dependencies (all guarded — modules import even if absent; the metric raises a
friendly error only when *called* without its backend, and :class:`MetricSuite`
records NaN / skips gracefully):
  * LPIPS  -> ``lpips`` (canonical) or ``torchmetrics`` fallback.
  * FID    -> ``clean-fid`` (PREFERRED, the headline number) or ``torchmetrics``.
  * CLIP-FID -> ``clean-fid`` with ``model_name="clip_vit_b_32"``.
  * KID    -> ``clean-fid`` or ``torchmetrics``.

**clean-FID vs naive FID (the key pitfall, doc 06 §B / §10):** PyTorch/TF FID use a
fixed-width bilinear resize that *aliases* when downsampling, and JPEG quantization
can swing the score even when images look identical. ``clean-fid`` (Parmar et al.,
CVPR 2022) uses an antialiased PIL-bicubic-matched resize; keep both image sets as
lossless PNG and equal/large N. We therefore prefer ``clean-fid`` for the headline
number and fall back to torchmetrics' in-loop estimate only when it is unavailable.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..interfaces import TORCH_AVAILABLE, Metric
from . import _common as _c

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

# --------------------------------------------------------------------------- #
# Guarded optional backends. Probe import availability only (lazy construct).
# --------------------------------------------------------------------------- #
try:  # canonical LPIPS package
    import lpips as _lpips_pkg  # type: ignore

    _HAS_LPIPS = True
except Exception:  # pragma: no cover
    _lpips_pkg = None  # type: ignore
    _HAS_LPIPS = False

try:  # clean-fid (preferred for FID/CLIP-FID/KID)
    from cleanfid import fid as _cleanfid  # type: ignore

    _HAS_CLEANFID = True
except Exception:  # pragma: no cover
    _cleanfid = None  # type: ignore
    _HAS_CLEANFID = False

try:  # torchmetrics fallbacks
    import torchmetrics as _tm  # type: ignore
    import torchmetrics.image as _tmi  # type: ignore

    _HAS_TORCHMETRICS = True
except Exception:  # pragma: no cover
    _tm = None  # type: ignore
    _tmi = None  # type: ignore
    _HAS_TORCHMETRICS = False


def _missing(dep: str, metric: str, hint: str = "") -> RuntimeError:
    """Build a friendly ``RuntimeError`` for an unavailable optional backend."""
    extra = f" {hint}" if hint else ""
    return RuntimeError(
        f"{metric} requires the optional dependency '{dep}', which is not "
        f"installed. Install it with `pip install {dep}`.{extra}"
    )


def _to_bchw_torch(x: Any) -> "Any":
    """Coerce an image to a torch ``[B, C, H, W]`` float tensor in [0,1].

    Accepts torch tensors (passed through with shape promotion) or numpy arrays.
    """
    if not TORCH_AVAILABLE:  # pragma: no cover
        raise _missing("torch", "Perceptual metric")
    if _c.is_torch_tensor(x):
        t = x.detach().float()
    else:
        arr = _c.as_bchw(x)
        t = torch.as_tensor(arr, dtype=torch.float32)
        return t
    if t.dim() == 2:
        t = t[None, None]
    elif t.dim() == 3:
        t = t[None]
    return t


# =========================================================================== #
# LPIPS
# =========================================================================== #
class LPIPSMetric(Metric):
    """Learned Perceptual Image Patch Similarity — deep-feature distance. Lower better.

    Correlates with human perception of similarity, ≈ [0, 1] (doc 06 §B.12). Report
    **both backbones**: AlexNet (original, fastest) and VGG (often used as a loss).

    **Normalization pitfall (doc 06 §10):** the ``lpips`` package expects inputs in
    ``[-1, 1]``. Our contract delivers ``[0, 1]``, so we map ``x -> 2x - 1`` before
    calling ``lpips`` (its ``normalize=True`` flag would do this internally, but we
    set ``normalize=True`` and pass ``[0,1]`` to be explicit and bug-proof). The
    torchmetrics fallback is configured with ``normalize=True`` (expects ``[0,1]``).

    Guarded: requires ``lpips`` (preferred) or ``torchmetrics[image]``. Raises a
    friendly error if neither is installed.
    """

    name: str = "lpips"
    higher_is_better: bool = False
    family: str = "perceptual"

    def __init__(self, net: str = "alex", device: Optional[str] = None) -> None:
        self.net = str(net)
        self.device = device
        self.name = f"lpips_{self.net}"
        self._model = None  # lazily constructed
        self._backend: Optional[str] = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        if not TORCH_AVAILABLE:
            raise _missing("torch", "LPIPSMetric")
        if _HAS_LPIPS:
            model = _lpips_pkg.LPIPS(net=self.net)  # type: ignore[union-attr]
            if self.device is not None:
                model = model.to(self.device)
            model.eval()
            self._model = model
            self._backend = "lpips"
        elif _HAS_TORCHMETRICS:
            net_type = "alex" if self.net not in ("alex", "vgg", "squeeze") else self.net
            model = _tmi.LearnedPerceptualImagePatchSimilarity(  # type: ignore[union-attr]
                net_type=net_type, normalize=True
            )
            if self.device is not None:
                model = model.to(self.device)
            model.eval()
            self._model = model
            self._backend = "torchmetrics"
        else:
            raise _missing(
                "lpips", "LPIPSMetric", hint="(or `pip install torchmetrics[image]`)."
            )

    def __call__(
        self,
        pred: Any,
        target: Any = None,
        ctx: Optional[Dict[str, Any]] = None,
    ) -> float:
        if target is None:
            raise ValueError("LPIPSMetric requires a target (full-reference).")
        self._ensure_model()
        p = _to_bchw_torch(pred)
        t = _to_bchw_torch(target)
        if self.device is not None:
            p, t = p.to(self.device), t.to(self.device)
        with torch.no_grad():  # type: ignore[union-attr]
            if self._backend == "lpips":
                # Pass [0,1] with normalize=True (the package rescales to [-1,1]).
                val = self._model(p, t, normalize=True)  # type: ignore[misc]
                return float(val.mean().item())
            # torchmetrics: configured with normalize=True -> expects [0,1].
            self._model.reset()  # type: ignore[union-attr]
            self._model.update(p, t)  # type: ignore[union-attr]
            return float(self._model.compute().item())  # type: ignore[union-attr]


# =========================================================================== #
# FID
# =========================================================================== #
class FIDMetric(Metric):
    """Fréchet Inception Distance — distance between feature distributions. Lower better.

    **Prefers ``clean-fid``** (the headline number; antialiased resize, doc 06 §B.13)
    and falls back to ``torchmetrics`` for an in-loop estimate. Two usage modes:

    1. **Directory / paired call** — ``__call__(pred, target)`` where ``pred`` and
       ``target`` are either (a) directory paths of PNGs (clean-fid's native form,
       strongly preferred — keep them lossless PNG, equal & large N), or (b) image
       batches, in which case features are accumulated then compared in one shot.

    2. **Streaming** — ``update(images, real=False)`` accumulates generated/real
       feature batches across many calls (torchmetrics backend), then
       ``compute()`` returns the FID. This matches how FID is computed over a whole
       test set rather than per-tile.

    Set ``mode="clean"`` (default), ``"legacy_pytorch"``, or ``"legacy_tensorflow"``
    to reproduce other implementations' numbers; ``model_name="clip_vit_b_32"``
    gives CLIP-FID (robust for domain-shifted satellite data, doc 06 §B.14).

    Guarded: needs ``clean-fid`` or ``torchmetrics[image]``; raises friendly error
    otherwise.
    """

    name: str = "clean_fid"
    higher_is_better: bool = False
    family: str = "perceptual"

    def __init__(
        self,
        mode: str = "clean",
        model_name: str = "inception_v3",
        feature: int = 2048,
        device: Optional[str] = None,
        dataset_name: Optional[str] = None,
    ) -> None:
        self.mode = str(mode)
        self.model_name = str(model_name)
        self.feature = int(feature)
        self.device = device
        self.dataset_name = dataset_name  # precomputed reference stats (clean-fid)
        if model_name == "clip_vit_b_32":
            self.name = "clip_fid"
        elif mode != "clean":
            self.name = f"fid_{mode}"
        self._tm_metric = None  # lazily constructed torchmetrics FID (streaming)

    # ---- streaming (torchmetrics) ----------------------------------------- #
    def _ensure_tm(self) -> None:
        if self._tm_metric is not None:
            return
        if not (_HAS_TORCHMETRICS and TORCH_AVAILABLE):
            raise _missing(
                "torchmetrics[image]",
                "FIDMetric.update/compute",
                hint="(streaming FID; or use clean-fid with directory paths).",
            )
        m = _tmi.FrechetInceptionDistance(feature=self.feature, normalize=True)  # type: ignore[union-attr]
        if self.device is not None:
            m = m.to(self.device)
        self._tm_metric = m

    def update(self, images: Any, real: bool = False) -> None:
        """Accumulate a batch of images into the running FID (torchmetrics backend).

        Args:
          images: ``[B, 3, H, W]`` in ``[0, 1]`` (torch tensor / numpy array).
          real:   ``True`` for ground-truth/real images, ``False`` for generated.
        """
        self._ensure_tm()
        t = _to_bchw_torch(images)
        if self.device is not None:
            t = t.to(self.device)
        self._tm_metric.update(t, real=bool(real))  # type: ignore[union-attr]

    def compute(self) -> float:
        """Return the accumulated FID (torchmetrics backend) as a float."""
        if self._tm_metric is None:
            raise RuntimeError("FIDMetric.compute() called before any update().")
        return float(self._tm_metric.compute().item())  # type: ignore[union-attr]

    def reset(self) -> None:
        """Clear the streaming accumulator."""
        if self._tm_metric is not None:
            self._tm_metric.reset()  # type: ignore[union-attr]

    # ---- one-shot paired call --------------------------------------------- #
    def _fid_dirs(self, fake_dir: str, real_dir: Optional[str]) -> float:
        if not _HAS_CLEANFID:
            raise _missing("clean-fid", "FIDMetric (directory mode)")
        kwargs: Dict[str, Any] = {"mode": self.mode}
        if self.model_name == "clip_vit_b_32":
            kwargs["model_name"] = "clip_vit_b_32"
        if self.device is not None:
            kwargs["device"] = self.device
        if self.dataset_name is not None and real_dir is None:
            # Compare a fake directory against precomputed custom reference stats.
            return float(
                _cleanfid.compute_fid(  # type: ignore[union-attr]
                    fake_dir,
                    dataset_name=self.dataset_name,
                    dataset_split="custom",
                    **kwargs,
                )
            )
        if real_dir is None:
            raise ValueError(
                "FIDMetric: provide a real directory/target or a precomputed "
                "dataset_name for clean-fid."
            )
        return float(_cleanfid.compute_fid(real_dir, fake_dir, **kwargs))  # type: ignore[union-attr]

    def __call__(
        self,
        pred: Any,
        target: Any = None,
        ctx: Optional[Dict[str, Any]] = None,
    ) -> float:
        # Directory paths -> clean-fid (the preferred, headline path).
        if isinstance(pred, str):
            return self._fid_dirs(pred, target if isinstance(target, str) else None)
        # Otherwise treat pred/target as image batches -> streaming torchmetrics.
        if target is None:
            raise ValueError("FIDMetric requires a target set (real images / dir).")
        self._ensure_tm()
        self.reset()
        self.update(target, real=True)
        self.update(pred, real=False)
        return self.compute()


# =========================================================================== #
# KID
# =========================================================================== #
class KIDMetric(Metric):
    """Kernel Inception Distance — squared MMD of Inception features. Lower better.

    **Unbiased for small samples** (unlike FID) and ships with a variance estimate
    — preferred when the test set is small (doc 06 §B.15). Prefers ``clean-fid``
    (``compute_kid``); falls back to ``torchmetrics.KernelInceptionDistance``
    (reports mean ± std; mind ``subset_size`` for small sets). Returns the mean KID.

    Guarded: needs ``clean-fid`` or ``torchmetrics[image]``; friendly error otherwise.
    """

    name: str = "kid"
    higher_is_better: bool = False
    family: str = "perceptual"

    def __init__(
        self,
        subset_size: int = 50,
        device: Optional[str] = None,
    ) -> None:
        self.subset_size = int(subset_size)
        self.device = device
        self._tm_metric = None

    def _ensure_tm(self) -> None:
        if self._tm_metric is not None:
            return
        if not (_HAS_TORCHMETRICS and TORCH_AVAILABLE):
            raise _missing(
                "torchmetrics[image]",
                "KIDMetric (batch mode)",
                hint="(or use clean-fid with directory paths).",
            )
        m = _tmi.KernelInceptionDistance(  # type: ignore[union-attr]
            subset_size=self.subset_size, normalize=True
        )
        if self.device is not None:
            m = m.to(self.device)
        self._tm_metric = m

    def update(self, images: Any, real: bool = False) -> None:
        """Accumulate a batch (torchmetrics backend); see :meth:`FIDMetric.update`."""
        self._ensure_tm()
        t = _to_bchw_torch(images)
        if self.device is not None:
            t = t.to(self.device)
        self._tm_metric.update(t, real=bool(real))  # type: ignore[union-attr]

    def compute(self) -> float:
        """Return the accumulated KID mean (torchmetrics backend)."""
        if self._tm_metric is None:
            raise RuntimeError("KIDMetric.compute() called before any update().")
        mean, _std = self._tm_metric.compute()  # type: ignore[union-attr]
        return float(mean.item())

    def reset(self) -> None:
        """Clear the streaming accumulator."""
        if self._tm_metric is not None:
            self._tm_metric.reset()  # type: ignore[union-attr]

    def __call__(
        self,
        pred: Any,
        target: Any = None,
        ctx: Optional[Dict[str, Any]] = None,
    ) -> float:
        if isinstance(pred, str):
            if not _HAS_CLEANFID:
                raise _missing("clean-fid", "KIDMetric (directory mode)")
            if not isinstance(target, str):
                raise ValueError("KIDMetric directory mode needs a real directory target.")
            kwargs: Dict[str, Any] = {}
            if self.device is not None:
                kwargs["device"] = self.device
            return float(_cleanfid.compute_kid(target, pred, **kwargs))  # type: ignore[union-attr]
        if target is None:
            raise ValueError("KIDMetric requires a target set (real images / dir).")
        self._ensure_tm()
        self.reset()
        self.update(target, real=True)
        self.update(pred, real=False)
        return self.compute()


__all__ = [
    "LPIPSMetric",
    "FIDMetric",
    "KIDMetric",
]
