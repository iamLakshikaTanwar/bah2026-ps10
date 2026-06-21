"""Tests for the fidelity-dominant composite loss — :mod:`irchroma.losses`.

Requires torch. We verify that:

* ``build_composite_loss(cfg)`` assembles a non-empty CompositeLoss whose enabled
  terms match the non-zero ``LossConfig`` weights.
* Each enabled term returns a **finite scalar** on a tiny ``(output, target, ctx)``.
* The composite ``total`` is finite and the ``components`` dict is per-term weighted.
* The **missing-semantic / missing-GT** paths return 0 (a zero scalar) without
  crashing — the same stack must work paired and unpaired.

The tiny ``(output, target)`` is built *self-consistently* (``sr`` and ``ir`` share a
spatial size; ``output['rgb']`` and ``target['rgb']`` share a size) so every term is
actually computable on CPU. ``ctx`` is left minimal: adversarial / feature-matching /
seg-consistency terms degrade to a zero scalar when their heavy ctx (discriminator /
frozen segmenter) is absent — which is exactly the robustness we assert.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")  # skip the whole module without torch

from irchroma.config import LossConfig
from irchroma.losses import build_composite_loss


def _tiny_io(b: int = 2, hw: int = 8, seed: int = 0):
    """A self-consistent tiny (output, target) pair on CPU.

    ``sr``/``ir`` share ``[B,1,hw,hw]`` and ``rgb`` pairs share ``[B,3,hw,hw]`` so
    even the SR terms (which compare ``output['sr']`` vs ``target['ir']``) line up
    spatially. ``semantic`` is a valid Long map for the palette / seg terms.
    """
    g = torch.Generator().manual_seed(seed)
    ir = torch.rand((b, 1, hw, hw), generator=g)
    sr = ir + 0.01 * torch.randn((b, 1, hw, hw), generator=g)
    rgb_pred = torch.rand((b, 3, hw, hw), generator=g)
    rgb_gt = torch.rand((b, 3, hw, hw), generator=g)
    sem = torch.randint(0, 10, (b, hw, hw), generator=g)
    output = {"rgb": rgb_pred.clamp(0, 1), "sr": sr, "aux": {}}
    target = {"ir": ir, "rgb": rgb_gt, "semantic": sem, "meta": {}}
    return output, target


def _is_finite_scalar(t) -> bool:
    """True if ``t`` is a 0-dim (or single-element) finite torch scalar."""
    if not torch.is_tensor(t):
        return False
    if t.numel() != 1:
        return False
    return bool(torch.isfinite(t).all())


# --------------------------------------------------------------------------- #
# Assembly.
# --------------------------------------------------------------------------- #
def test_build_composite_loss_enables_nonzero_terms():
    """Only the non-zero-weight LossConfig terms are wired into the composite."""
    cfg = LossConfig()
    comp = build_composite_loss(cfg)
    weights = comp.weights
    assert len(weights) > 0
    # Every wired weight is the (non-zero) LossConfig value.
    for name, w in weights.items():
        assert w != 0.0
        assert float(getattr(cfg, name)) == pytest.approx(w)
    # A representative pixel-anchor term (color_l1, weight 10.0) is present.
    assert "color_l1" in weights
    assert weights["color_l1"] == pytest.approx(10.0)
    # A zero-weighted term (sr_adversarial = 0.0) is NOT wired.
    assert "sr_adversarial" not in weights


def test_build_composite_loss_accepts_full_config(cfg):
    """``build_composite_loss`` accepts a top-level Config (uses ``.loss``)."""
    comp = build_composite_loss(cfg)
    assert len(comp.weights) > 0


# --------------------------------------------------------------------------- #
# Per-term + composite finiteness.
# --------------------------------------------------------------------------- #
def test_each_enabled_term_returns_finite_scalar():
    """Every enabled term yields a finite 0-dim scalar on the tiny paired sample."""
    comp = build_composite_loss(LossConfig())
    output, target = _tiny_io()
    ctx = {}
    # Access the registered terms via the parallel weights dict + ModuleDict.
    for name in comp.weights:
        term = comp._terms[name]  # internal ModuleDict (test-only access)
        value = term(output, target, ctx)
        assert _is_finite_scalar(value), f"term {name!r} did not return a finite scalar"


def test_composite_total_is_finite_and_components_match():
    """The composite total is finite; components are the per-term weighted values."""
    comp = build_composite_loss(LossConfig())
    output, target = _tiny_io()
    total, components = comp(output, target, {})
    assert _is_finite_scalar(total)
    assert isinstance(components, dict) and len(components) > 0
    # Components are floats and finite.
    for name, val in components.items():
        assert isinstance(val, float)
        assert math.isfinite(val)
    # Total approximately equals the sum of weighted components (some terms may be
    # exactly zero, e.g. adversarial without a discriminator in ctx).
    assert float(total) == pytest.approx(sum(components.values()), abs=1e-4)


# --------------------------------------------------------------------------- #
# Missing-input robustness (paired AND unpaired must both work).
# --------------------------------------------------------------------------- #
def test_missing_semantic_path_returns_zero_without_crashing():
    """With no ``semantic`` map, semantic terms return 0 (no crash); total stays finite."""
    comp = build_composite_loss(LossConfig())
    output, target = _tiny_io()
    target.pop("semantic")  # unpaired-ish: drop the label map
    total, components = comp(output, target, {})
    assert _is_finite_scalar(total)
    # The palette out-of-class term must be exactly zero with no semantic map.
    if "color_lut_outofclass" in components:
        assert components["color_lut_outofclass"] == pytest.approx(0.0)


def test_missing_rgb_gt_path_returns_finite():
    """With no GT RGB, color terms that need it return 0; total stays finite."""
    comp = build_composite_loss(LossConfig())
    output, target = _tiny_io()
    target["rgb"] = None  # inference-time: no paired RGB target
    total, components = comp(output, target, {})
    assert _is_finite_scalar(total)
    # color_l1 (pixel anchor) needs the GT; it must be zero when GT is absent.
    if "color_l1" in components:
        assert components["color_l1"] == pytest.approx(0.0)


def test_adversarial_terms_zero_without_discriminator():
    """Adversarial / feature-matching terms return 0 when no discriminator is in ctx."""
    comp = build_composite_loss(LossConfig())
    output, target = _tiny_io()
    _, components = comp(output, target, {})  # empty ctx => no discriminator
    for name in ("color_adversarial", "color_feature_matching"):
        if name in components:
            assert components[name] == pytest.approx(0.0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
