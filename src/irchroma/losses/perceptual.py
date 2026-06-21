"""irchroma.losses.perceptual — VGG / LPIPS feature-space realism terms.

Perceptual losses compare prediction and target in a deep feature space (rather than
pixel space) to add realism and semantic fidelity without raw-pixel overfit
(ARCHITECTURE §8; docs/research/02 §7 — "LPIPS-VGG best in studies"). They power:

  * ``sr_lpips``      — light perceptual term on the SR output.
  * ``color_lpips``   — perceptual realism + semantics on the colorized RGB.
  * ``task_perceptual`` — task-driven feature matching (here served by a deeper VGG
                          stage as a dependency-free stand-in for a frozen detector).

Optional dependencies are guarded so the module ALWAYS imports:

  * :class:`VGGPerceptualLoss` uses ``torchvision`` VGG16 features when available; if
    ``torchvision`` is missing it falls back to a small fixed (learning-free) Gaussian
    multi-scale feature extractor, and ultimately to plain L1 — always returning a
    valid scalar with a one-time note.
  * :class:`LPIPSLoss` uses the ``lpips`` package when available; otherwise it falls
    back to :class:`VGGPerceptualLoss`.

Single-channel inputs (SR IR) are repeated to 3 channels before being fed to the
3-channel feature backbones.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

from ..interfaces import LossTerm, PipelineOutput, Sample, TORCH_AVAILABLE, Tensor

try:  # Real torch at runtime; guarded so the module always imports.
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
    import torch.nn.functional as F  # type: ignore
except Exception:  # pragma: no cover - torch-less docs/CI box
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore


def _zero_scalar(ref: Optional[Tensor] = None) -> Tensor:
    """Return a 0-dim zero tensor matching ``ref``'s device/dtype when possible."""
    if not TORCH_AVAILABLE:  # pragma: no cover
        return 0.0  # type: ignore[return-value]
    if ref is not None and isinstance(ref, torch.Tensor):  # type: ignore[attr-defined]
        return torch.zeros((), dtype=ref.dtype, device=ref.device)  # type: ignore[union-attr]
    return torch.zeros((), dtype=torch.float32)  # type: ignore[union-attr]


def _to_3ch(x: Tensor) -> Tensor:
    """Repeat a single-channel tensor to 3 channels; pass 3-channel through."""
    if x.shape[1] == 1:
        return x.repeat(1, 3, 1, 1)
    if x.shape[1] == 3:
        return x
    # Multi-band (>3) IR: collapse to the mean and repeat to 3ch so backbones run.
    mean = x.mean(dim=1, keepdim=True)
    return mean.repeat(1, 3, 1, 1)


# Try to import torchvision VGG once at module load (guarded).
try:  # pragma: no cover - exercised only when torchvision is installed
    import torchvision  # type: ignore

    _HAS_TORCHVISION = True
except Exception:
    torchvision = None  # type: ignore
    _HAS_TORCHVISION = False


class _LearningFreeFeatures(nn.Module if TORCH_AVAILABLE else object):  # type: ignore[misc]
    """Fixed multi-scale Gaussian-pyramid features used when torchvision is absent.

    Produces a small list of progressively blurred/downsampled maps. This is NOT a
    learned perceptual space, but it gives a stable, dependency-free structural
    feature stack so :class:`VGGPerceptualLoss` still returns a meaningful scalar.
    """

    def __init__(self, levels: int = 4) -> None:
        super().__init__()  # type: ignore[misc]
        self.levels = int(levels)
        if TORCH_AVAILABLE:
            # 5x5 separable Gaussian kernel.
            k1 = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0])  # type: ignore[union-attr]
            k1 = k1 / k1.sum()
            k2 = (k1[:, None] * k1[None, :]).view(1, 1, 5, 5)
            self.register_buffer("blur", k2, persistent=False)  # type: ignore[attr-defined]

    def forward(self, x: Tensor) -> List[Tensor]:
        feats: List[Tensor] = [x]
        cur = x
        c = cur.shape[1]
        blur = self.blur.to(dtype=cur.dtype).repeat(c, 1, 1, 1)  # type: ignore[attr-defined]
        for _ in range(self.levels):
            cur_p = F.pad(cur, (2, 2, 2, 2), mode="replicate")  # type: ignore[union-attr]
            cur = F.conv2d(cur_p, blur, groups=c)  # type: ignore[union-attr]
            cur = F.avg_pool2d(cur, kernel_size=2)  # type: ignore[union-attr]
            feats.append(cur)
        return feats


class VGGPerceptualLoss(LossTerm):
    """VGG16 feature-space L1 perceptual loss (torchvision-guarded).

    Extracts features at several VGG16 relu stages for prediction and target (both
    ImageNet-normalized) and L1-matches them. Serves ``sr_lpips`` / ``color_lpips``
    (as a VGG perceptual proxy) and ``task_perceptual``. If ``torchvision`` is
    unavailable, falls back to a fixed learning-free feature stack, and finally to
    plain L1 — always returning a scalar.

    Args:
      layers:     VGG16 feature indices to tap (after which relu activations).
      resize:     if ``True``, resize inputs to ``224`` before the backbone (matches
                  the ImageNet training distribution; recommended for VGG).
      pred_key/target_key: which tensors to compare (defaults to RGB pair).
    """

    name = "vgg_perceptual"
    #: ImageNet mean/std for sRGB in [0,1].
    _MEAN = (0.485, 0.456, 0.406)
    _STD = (0.229, 0.224, 0.225)
    #: Default relu tap points in torchvision VGG16.features (relu1_2..relu4_3).
    _DEFAULT_LAYERS = (3, 8, 15, 22)

    def __init__(
        self,
        layers: Optional[List[int]] = None,
        resize: bool = False,
        pred_key: str = "rgb",
        target_key: str = "rgb",
    ) -> None:
        super().__init__()  # type: ignore[misc]
        self.layers = list(layers) if layers is not None else list(self._DEFAULT_LAYERS)
        self.resize = bool(resize)
        self.pred_key = pred_key
        self.target_key = target_key
        self._fallback_warned = False
        self.backbone: Any = None
        self._fallback: Any = None

        if TORCH_AVAILABLE:
            self.register_buffer(  # type: ignore[attr-defined]
                "_mean", torch.tensor(self._MEAN).view(1, 3, 1, 1), persistent=False  # type: ignore[union-attr]
            )
            self.register_buffer(  # type: ignore[attr-defined]
                "_std", torch.tensor(self._STD).view(1, 3, 1, 1), persistent=False  # type: ignore[union-attr]
            )
            self._build_backbone()

    def _build_backbone(self) -> None:
        """Construct the (frozen) VGG16 feature backbone, or the fallback stack."""
        if _HAS_TORCHVISION:
            try:  # pragma: no cover - needs torchvision weights
                try:
                    weights = torchvision.models.VGG16_Weights.IMAGENET1K_V1  # type: ignore[attr-defined]
                    vgg = torchvision.models.vgg16(weights=weights).features  # type: ignore[attr-defined]
                except Exception:
                    vgg = torchvision.models.vgg16(pretrained=True).features  # type: ignore[attr-defined]
                for p in vgg.parameters():
                    p.requires_grad_(False)
                vgg.eval()
                self.backbone = vgg
                return
            except Exception:
                self.backbone = None
        # torchvision unavailable or failed -> learning-free fallback.
        self._fallback = _LearningFreeFeatures(levels=4)

    def _normalize(self, x: Tensor) -> Tensor:
        """ImageNet-normalize a 3-channel tensor in ``[0, 1]``."""
        x = x.clamp(0.0, 1.0)
        return (x - self._mean.to(x.dtype)) / self._std.to(x.dtype)  # type: ignore[attr-defined]

    def _vgg_features(self, x: Tensor) -> List[Tensor]:
        """Run VGG16 and collect activations at the configured tap indices."""
        feats: List[Tensor] = []
        max_idx = max(self.layers)
        out = x
        for idx, layer in enumerate(self.backbone):  # type: ignore[union-attr]
            out = layer(out)
            if idx in self.layers:
                feats.append(out)
            if idx >= max_idx:
                break
        return feats

    def forward(
        self,
        output: PipelineOutput,
        target: Sample,
        ctx: Dict[str, Any],
    ) -> Tensor:
        pred = output.get(self.pred_key, None) if hasattr(output, "get") else None
        gt = target.get(self.target_key, None) if hasattr(target, "get") else None
        if pred is None or gt is None:
            return _zero_scalar(pred if pred is not None else gt)

        pred3 = _to_3ch(pred)
        gt3 = _to_3ch(gt)
        if pred3.shape[-2:] != gt3.shape[-2:]:
            gt3 = F.interpolate(  # type: ignore[union-attr]
                gt3, size=pred3.shape[-2:], mode="bilinear", align_corners=False
            )

        if self.backbone is not None:
            x = self._normalize(pred3)
            y = self._normalize(gt3)
            if self.resize:
                x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)  # type: ignore[union-attr]
                y = F.interpolate(y, size=(224, 224), mode="bilinear", align_corners=False)  # type: ignore[union-attr]
            fx = self._vgg_features(x)
            fy = self._vgg_features(y)
            loss = pred3.new_zeros(())  # type: ignore[attr-defined]
            for a, b in zip(fx, fy):
                loss = loss + (a - b).abs().mean()
            return loss / max(1, len(fx))

        if self._fallback is not None:
            if not self._fallback_warned:
                warnings.warn(
                    "VGGPerceptualLoss: torchvision unavailable; using a learning-free "
                    "Gaussian-pyramid feature fallback (structural, not learned VGG).",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._fallback_warned = True
            fx = self._fallback(pred3)
            fy = self._fallback(gt3)
            loss = pred3.new_zeros(())  # type: ignore[attr-defined]
            for a, b in zip(fx, fy):
                loss = loss + (a - b).abs().mean()
            return loss / max(1, len(fx))

        # Ultimate fallback: plain L1 (should be unreachable on the torch path).
        return (pred3 - gt3).abs().mean()


class LPIPSLoss(LossTerm):
    """LPIPS perceptual loss (``lpips`` package guarded; VGG fallback).

    Uses the ``lpips`` package (AlexNet/VGG LPIPS) when installed, normalizing inputs
    to ``[-1, 1]`` as LPIPS expects. When ``lpips`` is unavailable it transparently
    delegates to :class:`VGGPerceptualLoss`, so ``sr_lpips`` / ``color_lpips`` always
    produce a sensible scalar.

    Args:
      net:        LPIPS backbone name (``"alex"`` | ``"vgg"`` | ``"squeeze"``).
      pred_key/target_key: which tensors to compare (defaults to RGB pair).
    """

    name = "lpips"

    def __init__(
        self,
        net: str = "vgg",
        pred_key: str = "rgb",
        target_key: str = "rgb",
    ) -> None:
        super().__init__()  # type: ignore[misc]
        self.net = str(net)
        self.pred_key = pred_key
        self.target_key = target_key
        self._lpips: Any = None
        self._vgg_fallback: Any = None
        self._warned = False
        if TORCH_AVAILABLE:
            self._build()

    def _build(self) -> None:
        """Try to construct the LPIPS module; else prepare the VGG fallback."""
        try:  # pragma: no cover - needs the lpips package + weights
            import lpips  # type: ignore

            net = lpips.LPIPS(net=self.net)  # type: ignore[attr-defined]
            for p in net.parameters():
                p.requires_grad_(False)
            net.eval()
            self._lpips = net
        except Exception:
            self._vgg_fallback = VGGPerceptualLoss(
                pred_key=self.pred_key, target_key=self.target_key
            )

    def forward(
        self,
        output: PipelineOutput,
        target: Sample,
        ctx: Dict[str, Any],
    ) -> Tensor:
        if self._lpips is None:
            if not self._warned:
                warnings.warn(
                    "LPIPSLoss: 'lpips' package unavailable; falling back to "
                    "VGGPerceptualLoss.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._warned = True
            return self._vgg_fallback(output, target, ctx)  # type: ignore[misc]

        pred = output.get(self.pred_key, None) if hasattr(output, "get") else None
        gt = target.get(self.target_key, None) if hasattr(target, "get") else None
        if pred is None or gt is None:  # pragma: no cover - needs lpips installed
            return _zero_scalar(pred if pred is not None else gt)

        pred3 = _to_3ch(pred)
        gt3 = _to_3ch(gt)
        if pred3.shape[-2:] != gt3.shape[-2:]:  # pragma: no cover
            gt3 = F.interpolate(  # type: ignore[union-attr]
                gt3, size=pred3.shape[-2:], mode="bilinear", align_corners=False
            )
        # lpips expects inputs in [-1, 1].
        x = pred3.clamp(0.0, 1.0) * 2.0 - 1.0
        y = gt3.clamp(0.0, 1.0) * 2.0 - 1.0
        return self._lpips(x, y).mean()  # type: ignore[misc]


__all__ = [
    "VGGPerceptualLoss",
    "LPIPSLoss",
]
