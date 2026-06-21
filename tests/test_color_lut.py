"""Tests for the O(1) class->Lab chroma clamp — :class:`ClassColorLUT`.

Requires torch. This is the **semantic color-consistency core** (water->blue,
veg->green). We feed a *uniform* RGB tile tagged entirely with one class and assert
the LUT steers chroma toward that class's palette direction in CIE-Lab while leaving
luminance essentially untouched (the contract: clamp a*,b*, leave L* free so SR
texture survives). Concretely:

  * ``water``  -> b* becomes negative (blue) AND the blue channel rises *relative to*
                  red/green.
  * ``trees``  -> a* becomes negative (green) AND the green channel rises *relative to*
                  red/blue.
  * luminance (CIE L*) is preserved within a small tolerance.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")  # skip the whole module without torch

from irchroma.config import LULC_NAME_TO_INDEX, SemanticConfig
from irchroma.models.semantic.color_lut import ClassColorLUT, rgb_to_lab


def _uniform_rgb(value: float = 0.5, b: int = 1, hw: int = 8):
    """A flat ``[B, 3, hw, hw]`` gray tile in [0,1] (neutral a*=b*=0 start point)."""
    return torch.full((b, 3, hw, hw), float(value), dtype=torch.float32)


def _class_map(class_name: str, b: int = 1, hw: int = 8):
    """A ``[B, hw, hw]`` Long semantic map filled with one class index."""
    idx = LULC_NAME_TO_INDEX[class_name]
    return torch.full((b, hw, hw), idx, dtype=torch.long)


def test_lut_apply_preserves_shape_and_range():
    """``apply`` returns the same shape, dtype, and a valid [0,1] RGB."""
    lut = ClassColorLUT(SemanticConfig())
    rgb = _uniform_rgb()
    sem = _class_map("water")
    out = lut.apply(rgb, sem)
    assert out.shape == rgb.shape
    assert out.dtype == rgb.dtype
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_water_shifts_chroma_toward_blue():
    """A neutral tile labeled 'water' gets bluer: b* < 0 and blue channel relatively up."""
    lut = ClassColorLUT(SemanticConfig())
    rgb = _uniform_rgb(0.5)
    sem = _class_map("water")
    out = lut.apply(rgb, sem)

    out_lab = rgb_to_lab(out)
    # b* (channel 2) is the blue<->yellow axis; water should drive it negative (blue).
    assert float(out_lab[:, 2].mean()) < -1.0

    # In RGB the blue channel should increase *relative to* red & green.
    r, g, b = out[:, 0].mean(), out[:, 1].mean(), out[:, 2].mean()
    assert float(b) > float(r)
    assert float(b) > float(g)
    # And bluer than the neutral input it started from.
    assert float(b) > float(rgb[:, 2].mean())


def test_trees_shifts_chroma_toward_green():
    """A neutral tile labeled 'trees' gets greener: a* < 0 and green channel relatively up."""
    lut = ClassColorLUT(SemanticConfig())
    rgb = _uniform_rgb(0.5)
    sem = _class_map("trees")
    out = lut.apply(rgb, sem)

    out_lab = rgb_to_lab(out)
    # a* (channel 1) is the green<->red axis; trees should drive it negative (green).
    assert float(out_lab[:, 1].mean()) < -1.0

    # In RGB the green channel should exceed red & blue.
    r, g, b = out[:, 0].mean(), out[:, 1].mean(), out[:, 2].mean()
    assert float(g) > float(r)
    assert float(g) > float(b)


def test_luminance_is_preserved():
    """The clamp leaves L* (luminance) essentially unchanged (chroma-only clamp)."""
    lut = ClassColorLUT(SemanticConfig())
    assert lut.chroma_only is True  # default SemanticConfig clamps a,b only
    rgb = _uniform_rgb(0.5)
    for cls in ("water", "trees"):
        sem = _class_map(cls)
        out = lut.apply(rgb, sem)
        L_in = rgb_to_lab(rgb)[:, 0].mean()
        L_out = rgb_to_lab(out)[:, 0].mean()
        # L* is held exactly in Lab; only the sRGB gamut round-trip perturbs it.
        assert abs(float(L_out) - float(L_in)) < 5.0


def test_strength_zero_is_passthrough():
    """``strength=0`` is a near no-op (only the Lab round-trip touches the pixels)."""
    lut = ClassColorLUT(SemanticConfig(), strength=0.0)
    rgb = _uniform_rgb(0.5)
    sem = _class_map("water")
    out = lut.apply(rgb, sem)
    assert torch.allclose(out, rgb, atol=1e-2)


def test_out_of_class_distance_zero_inside_box():
    """``out_of_class_distance`` is ~0 once the color is already inside the class box."""
    lut = ClassColorLUT(SemanticConfig())
    rgb = _uniform_rgb(0.5)
    sem = _class_map("water")
    clamped = lut.apply(rgb, sem)
    dist = lut.out_of_class_distance(clamped, sem)
    assert dist.shape == (rgb.shape[0], 1, rgb.shape[2], rgb.shape[3])
    # After clamping into the box, the residual out-of-box chroma distance is ~0.
    assert float(dist.max()) < 1e-2


def test_forward_aliases_apply():
    """``forward`` (nn.Module call) is identical to ``apply``."""
    lut = ClassColorLUT(SemanticConfig())
    rgb = _uniform_rgb(0.5)
    sem = _class_map("trees")
    assert torch.equal(lut.forward(rgb, sem), lut.apply(rgb, sem))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
