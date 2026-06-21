"""irchroma.losses.adversarial — GAN objective + Pix2PixHD feature matching.

These terms drive *bounded* generative realism for the colorization stage
(ARCHITECTURE §8, docs/research/02 §7):

  * :class:`AdversarialLoss`    — generator GAN loss (``color_adversarial`` /
                                  optional ``sr_adversarial``) supporting
                                  ``'lsgan' | 'hinge' | 'vanilla'``.
  * :class:`FeatureMatchingLoss`— Pix2PixHD feature matching (``color_feature_matching``):
                                  L1 between the discriminator's intermediate features
                                  on real vs fake, across scales and layers. Stabilizes
                                  the GAN and sharpens output (heavily weighted).

==============================================================================
DISCRIMINATOR OUTPUT CONTRACT  (coordinated with ColorBuilder's MultiScaleDiscriminator)
==============================================================================
The Pix2PixHD-style multi-scale discriminator ``D`` is expected to return, for a
single input image batch, a **list over scales** of **list over layers** of
``Tensor``::

    D(x) -> List[List[Tensor]]
            outer list : one entry per discriminator scale (e.g. 3 resolutions)
            inner list : the per-layer activations of that scale's PatchGAN, in
                         forward order. The **LAST** inner element is the final
                         patch *score* map (logits); the preceding elements are the
                         intermediate features used for feature matching.

So for ``D(x) = outs``:
  * ``outs[s][-1]``     is the patch-score logits at scale ``s`` (NCHW, any spatial).
  * ``outs[s][:-1]``    are intermediate feature maps at scale ``s``.

This module is defensive about the exact shape: it also accepts a flat
``List[Tensor]`` of per-scale scores (treating each as a single-layer scale), and a
bare ``Tensor`` (one scale, one layer). ``FeatureMatchingLoss`` needs the full nested
features; if only scores are provided it degrades to matching the score maps.

How the discriminator outputs reach the losses
----------------------------------------------
The training loop runs ``D`` on the **fake** and **real** images and threads the
results through ``ctx`` so loss terms don't re-run the (heavy) discriminator:

  * ``ctx['disc_out_fake']`` : ``D(rgb_fake)``  — required for both terms.
  * ``ctx['disc_out_real']`` : ``D(rgb_real)``  — required for feature matching.

If those are absent, the terms try ``ctx['discriminator']`` (a callable ``D``) and run
it on ``output['rgb']`` / ``target['rgb']`` themselves. If neither a precomputed
output nor a discriminator is available, they return a zero scalar (so a
discriminator-free configuration still runs).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

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


def _normalize_disc_out(out: Any) -> List[List[Tensor]]:
    """Coerce a discriminator output into nested ``List[List[Tensor]]`` form.

    Accepts:
      * ``List[List[Tensor]]`` — returned as-is (the canonical format).
      * ``List[Tensor]``       — treated as one layer per scale -> ``[[t] for t]``.
      * ``Tensor``             — single scale, single layer -> ``[[t]]``.
    Anything else yields an empty list (callers then return a zero scalar).
    """
    if out is None:
        return []
    if TORCH_AVAILABLE and isinstance(out, torch.Tensor):  # type: ignore[attr-defined]
        return [[out]]
    if isinstance(out, (list, tuple)):
        if len(out) == 0:
            return []
        first = out[0]
        if TORCH_AVAILABLE and isinstance(first, torch.Tensor):  # type: ignore[attr-defined]
            # Flat list of per-scale score tensors.
            return [[t] for t in out]  # type: ignore[list-item]
        if isinstance(first, (list, tuple)):
            return [list(scale) for scale in out]  # type: ignore[arg-type]
    return []


def _get_disc_outputs(
    output: PipelineOutput,
    target: Sample,
    ctx: Dict[str, Any],
    need_real: bool,
) -> Any:
    """Resolve ``(fake_out, real_out)`` discriminator results from ctx / output.

    Returns nested-list discriminator outputs (possibly empty). ``real_out`` is only
    populated when ``need_real`` is True (feature matching).
    """
    fake_out = _normalize_disc_out(ctx.get("disc_out_fake")) if ctx else []
    real_out = _normalize_disc_out(ctx.get("disc_out_real")) if (ctx and need_real) else []

    disc: Optional[Callable[..., Any]] = ctx.get("discriminator") if ctx else None
    if not fake_out and disc is not None:
        rgb = output.get("rgb", None) if hasattr(output, "get") else None
        if rgb is not None:
            fake_out = _normalize_disc_out(disc(rgb))
    if need_real and not real_out and disc is not None:
        rgb_gt = target.get("rgb", None) if hasattr(target, "get") else None
        if rgb_gt is not None:
            # Real features should not backprop into anything; detach the GT path.
            with _no_grad():
                real_out = _normalize_disc_out(disc(rgb_gt))
    return fake_out, real_out


class _NoGrad:
    """Context manager: ``torch.no_grad()`` when torch is present, else a no-op."""

    def __enter__(self) -> "Any":
        if TORCH_AVAILABLE:
            self._cm = torch.no_grad()  # type: ignore[union-attr]
            return self._cm.__enter__()
        return None

    def __exit__(self, *exc: Any) -> None:
        if TORCH_AVAILABLE:
            self._cm.__exit__(*exc)


def _no_grad() -> "_NoGrad":
    return _NoGrad()


class AdversarialLoss(LossTerm):
    """Generator-side GAN loss over a multi-scale discriminator's score maps.

    Supports three standard objectives (``gan_mode``):
      * ``'lsgan'``   : ``mean((D(fake) - 1)^2)`` (least-squares).
      * ``'hinge'``   : ``-mean(D(fake))`` (generator hinge).
      * ``'vanilla'`` : ``mean(softplus(-D(fake)))`` (BCE-with-logits, stable form).

    The score per scale is the **last** element of that scale's layer list
    (see the discriminator contract at module top); losses are averaged over scales.
    This is the bounded ``color_adversarial`` term (and the optional ``sr_adversarial``).

    The discriminator output is taken from ``ctx['disc_out_fake']`` if present, else by
    running ``ctx['discriminator']`` on ``output['rgb']``. With neither available the
    loss is a zero scalar (so a D-free run still works).

    Args:
      gan_mode: one of ``{'lsgan', 'hinge', 'vanilla'}`` (mirrors
                ``LossConfig.gan_mode`` / ``ColorizationConfig.gan_mode``).
    """

    name = "adversarial"

    def __init__(self, gan_mode: str = "hinge") -> None:
        super().__init__()  # type: ignore[misc]
        mode = str(gan_mode).lower()
        if mode not in ("lsgan", "hinge", "vanilla"):
            raise ValueError(
                f"AdversarialLoss: unsupported gan_mode {gan_mode!r}; "
                "expected 'lsgan' | 'hinge' | 'vanilla'."
            )
        self.gan_mode = mode

    def _g_loss_on_score(self, score: Tensor) -> Tensor:
        """Generator loss for one score map (wants D to call the fake 'real')."""
        if self.gan_mode == "lsgan":
            return (score - 1.0).pow(2).mean()
        if self.gan_mode == "hinge":
            return -score.mean()
        # vanilla: BCEWithLogits toward the 'real' (=1) target == softplus(-score).
        return F.softplus(-score).mean()  # type: ignore[union-attr]

    def forward(
        self,
        output: PipelineOutput,
        target: Sample,
        ctx: Dict[str, Any],
    ) -> Tensor:
        ctx = ctx or {}
        fake_out, _ = _get_disc_outputs(output, target, ctx, need_real=False)
        if not fake_out:
            ref = output.get("rgb", None) if hasattr(output, "get") else None
            return _zero_scalar(ref)
        losses: List[Tensor] = []
        for scale in fake_out:
            if not scale:
                continue
            score = scale[-1]  # last element per scale = patch-score logits
            losses.append(self._g_loss_on_score(score))
        if not losses:
            ref = output.get("rgb", None) if hasattr(output, "get") else None
            return _zero_scalar(ref)
        return torch.stack(losses).mean()  # type: ignore[union-attr]


class FeatureMatchingLoss(LossTerm):
    """Pix2PixHD feature-matching loss (L1 over discriminator intermediate features).

    For each scale, L1-matches the discriminator's intermediate activations on the
    **fake** image to those on the **real** image, averaged over layers and scales.
    Real features are detached (they are a moving target, not a gradient source). This
    is the heavily-weighted ``color_feature_matching`` stabilizer/sharpener
    (docs/research/02 §7; Pix2PixHD).

    Requires both ``ctx['disc_out_fake']`` and ``ctx['disc_out_real']`` (or a callable
    ``ctx['discriminator']`` to compute them from ``output['rgb']`` /
    ``target['rgb']``). If only final score maps are available (no intermediate
    features), it degrades to matching the score maps. Returns a zero scalar when no
    discriminator information or no paired real image is available.

    Args:
      include_final: if ``True`` also match the final score map (default ``False``;
                     the GAN term already supervises the score).
    """

    name = "feature_matching"

    def __init__(self, include_final: bool = False) -> None:
        super().__init__()  # type: ignore[misc]
        self.include_final = bool(include_final)

    def forward(
        self,
        output: PipelineOutput,
        target: Sample,
        ctx: Dict[str, Any],
    ) -> Tensor:
        ctx = ctx or {}
        fake_out, real_out = _get_disc_outputs(output, target, ctx, need_real=True)
        ref = output.get("rgb", None) if hasattr(output, "get") else None
        if not fake_out or not real_out:
            return _zero_scalar(ref)

        total: Any = None
        count = 0
        num_scales = min(len(fake_out), len(real_out))
        for s in range(num_scales):
            fake_layers = fake_out[s]
            real_layers = real_out[s]
            if not fake_layers or not real_layers:
                continue
            n_layers = min(len(fake_layers), len(real_layers))
            # Match all-but-last (intermediate features) unless include_final; but if
            # the scale has only one map (score-only), match that single map.
            upper = n_layers if (self.include_final or n_layers == 1) else n_layers - 1
            for layer in range(upper):
                a = fake_layers[layer]
                b = real_layers[layer]
                if not (TORCH_AVAILABLE and isinstance(a, torch.Tensor)):  # type: ignore[attr-defined]
                    continue
                b = b.detach()
                if a.shape[-2:] != b.shape[-2:]:  # pragma: no cover - shape guard
                    b = F.interpolate(  # type: ignore[union-attr]
                        b, size=a.shape[-2:], mode="bilinear", align_corners=False
                    )
                term = (a - b).abs().mean()
                total = term if total is None else (total + term)
                count += 1
        if total is None or count == 0:
            return _zero_scalar(ref)
        return total / float(count)


__all__ = [
    "AdversarialLoss",
    "FeatureMatchingLoss",
]
