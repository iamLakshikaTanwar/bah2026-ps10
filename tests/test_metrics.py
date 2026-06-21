"""Tests for the evaluation suite — :mod:`irchroma.metrics`.

The **self-contained** family (PSNR / SSIM / MS-SSIM / CIEDE2000 / colorfulness /
chroma-PSNR / RMSE / MAE / SAM) is pure numpy and needs **no torch**, so this module
runs with only numpy installed. We verify sane values on numpy arrays:

  * identical images        -> PSNR = +inf, SSIM ≈ 1.0, MS-SSIM ≈ 1.0, ΔE2000 ≈ 0,
                               colorfulness-delta ≈ 0, RMSE/MAE ≈ 0.
  * a degraded prediction    -> finite PSNR, SSIM < 1, ΔE2000 > 0.

We also assert ``build_metric_suite(cfg).evaluate(...)`` never crashes and returns a
dict (guarded perceptual / no-reference / efficiency metrics that need optional deps
are skipped or NaN'd gracefully, never raising).
"""

from __future__ import annotations

import math

import pytest

np = pytest.importorskip("numpy")  # the self-contained metrics are numpy-only

from irchroma.config import EvalConfig
from irchroma.metrics import (
    CIEDE2000Metric,
    ColorfulnessMetric,
    MAEMetric,
    MSSSIMMetric,
    PSNRMetric,
    RMSEMetric,
    SSIMMetric,
    build_metric_suite,
)


def _structured_rgb(b: int = 1, hw: int = 32, seed: int = 0):
    """A structured (non-constant) RGB array ``[B, 3, hw, hw]`` in [0,1].

    Smooth gradients + a little noise so SSIM/MS-SSIM variance terms are well-defined
    (a flat image makes SSIM degenerate). Deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    ys = np.linspace(0.0, 1.0, hw)[None, :]
    xs = np.linspace(0.0, 1.0, hw)[:, None]
    base = 0.5 + 0.4 * np.sin(6.0 * xs) * np.cos(6.0 * ys)  # [hw, hw]
    img = np.stack(
        [base, np.roll(base, 3, axis=0), np.roll(base, 5, axis=1)], axis=0
    )  # [3, hw, hw]
    img = img[None].repeat(b, axis=0)  # [B, 3, hw, hw]
    img = img + 0.02 * rng.standard_normal(img.shape)
    return np.clip(img, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# Identical-image sanity (the canonical fixed points).
# --------------------------------------------------------------------------- #
def test_psnr_identical_is_inf_and_degraded_finite():
    """PSNR(x, x) = +inf; PSNR(x, x+noise) is a finite positive dB value."""
    x = _structured_rgb()
    assert math.isinf(PSNRMetric()(x, x))
    noisy = np.clip(x + 0.1 * np.random.default_rng(1).standard_normal(x.shape), 0, 1)
    val = PSNRMetric()(noisy, x)
    assert math.isfinite(val) and val > 0.0


def test_ssim_identical_is_one_and_degraded_less():
    """SSIM(x, x) ≈ 1.0; a degraded prediction scores strictly below 1."""
    x = _structured_rgb()
    s_same = SSIMMetric()(x, x)
    assert s_same == pytest.approx(1.0, abs=1e-5)
    noisy = np.clip(x + 0.15 * np.random.default_rng(2).standard_normal(x.shape), 0, 1)
    s_deg = SSIMMetric()(noisy, x)
    assert s_deg < s_same
    assert -1.0 <= s_deg <= 1.0


def test_ms_ssim_identical_is_one():
    """MS-SSIM(x, x) ≈ 1.0 on an image large enough to pyramid."""
    x = _structured_rgb(hw=32)
    val = MSSSIMMetric()(x, x)
    assert val == pytest.approx(1.0, abs=1e-3)
    assert 0.0 <= val <= 1.0 + 1e-6


def test_ciede2000_identical_is_zero_and_positive_otherwise():
    """ΔE2000(x, x) ≈ 0; a different color yields a positive ΔE2000."""
    x = _structured_rgb()
    de_same = CIEDE2000Metric()(x, x)
    assert de_same == pytest.approx(0.0, abs=1e-4)
    # Shift toward blue: a clearly different color must register a positive ΔE.
    shifted = x.copy()
    shifted[:, 2] = np.clip(shifted[:, 2] + 0.3, 0, 1)
    de_diff = CIEDE2000Metric()(shifted, x)
    assert de_diff > 1.0


def test_colorfulness_delta_zero_for_identical():
    """Colorfulness delta is ~0 for identical images; raw M3 >= 0 with no target."""
    x = _structured_rgb()
    assert ColorfulnessMetric()(x, x) == pytest.approx(0.0, abs=1e-6)
    raw = ColorfulnessMetric()(x, None)  # no-reference -> raw M3
    assert raw >= 0.0


def test_rmse_mae_zero_for_identical():
    """RMSE/MAE are ~0 for identical images and positive for a degraded prediction."""
    x = _structured_rgb()
    assert RMSEMetric()(x, x) == pytest.approx(0.0, abs=1e-6)
    assert MAEMetric()(x, x) == pytest.approx(0.0, abs=1e-6)
    noisy = np.clip(x + 0.1 * np.random.default_rng(3).standard_normal(x.shape), 0, 1)
    assert RMSEMetric()(noisy, x) > 0.0
    assert MAEMetric()(noisy, x) > 0.0


def test_metric_directions_declared():
    """Metrics declare a sensible ranking direction (higher/lower is better)."""
    assert PSNRMetric().higher_is_better is True
    assert SSIMMetric().higher_is_better is True
    assert CIEDE2000Metric().higher_is_better is False


def test_single_image_chw_and_hw_accepted():
    """Self-contained metrics accept rank-3 (CHW) and rank-2 (HW) arrays too."""
    chw = _structured_rgb(b=1)[0]  # [3, H, W]
    assert math.isinf(PSNRMetric()(chw, chw))
    gray = chw[0]  # [H, W]
    assert RMSEMetric()(gray, gray) == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# Suite assembly + robustness.
# --------------------------------------------------------------------------- #
def test_build_metric_suite_evaluate_returns_dict():
    """The assembled suite evaluates without crashing and returns a {name: float} dict."""
    # Restrict to the always-available families so the test is deterministic and
    # never depends on optional perceptual/no-reference/efficiency backends.
    cfg = EvalConfig(families=["fidelity", "color"], y_channel_metrics=False, border_shave=0)
    suite = build_metric_suite(cfg)
    assert len(suite.names) > 0

    x = _structured_rgb()
    noisy = np.clip(x + 0.05 * np.random.default_rng(4).standard_normal(x.shape), 0, 1)
    results = suite.evaluate(noisy, x)
    assert isinstance(results, dict) and len(results) == len(suite.names)
    # Every value is a float (NaN allowed for any guarded metric that opts out).
    for name, val in results.items():
        assert isinstance(val, float)
    # PSNR / SSIM / CIEDE2000 are self-contained -> must be finite here.
    assert math.isfinite(results["psnr"])
    assert math.isfinite(results["ssim"])
    assert math.isfinite(results["ciede2000"])


def test_build_metric_suite_with_guarded_families_never_crashes():
    """Requesting guarded families (perceptual/no_reference/efficiency) still works.

    Their optional deps (lpips / clean-fid / pyiqa) are absent in CI; assembly must
    skip them with a warning and ``evaluate`` must NaN any that slip through — never
    raise. We only assert the self-contained metrics stay finite.
    """
    cfg = EvalConfig(
        families=["fidelity", "color", "perceptual", "no_reference"],
        y_channel_metrics=True,
        border_shave=2,
    )
    suite = build_metric_suite(cfg, skip_unavailable=True)
    x = _structured_rgb()
    results = suite.evaluate(x, x)  # identical inputs
    assert isinstance(results, dict)
    # Self-contained fixed points survive regardless of which guarded metrics loaded.
    assert math.isinf(results["psnr"])
    assert results["ssim"] == pytest.approx(1.0, abs=1e-4)
    assert results["ciede2000"] == pytest.approx(0.0, abs=1e-3)


def test_full_config_default_families_assemble():
    """A full default EvalConfig (all six families) assembles + evaluates safely."""
    suite = build_metric_suite(EvalConfig())
    x = _structured_rgb()
    results = suite.evaluate(x, x)
    assert isinstance(results, dict)
    # No exception is the contract; at least the fidelity metrics are present.
    assert "psnr" in results


# --------------------------------------------------------------------------- #
# Optional: torch-tensor inputs are accepted (only if torch is installed).
# --------------------------------------------------------------------------- #
def test_metrics_accept_torch_tensors_when_available():
    """When torch is present, the numpy metrics also accept torch tensors."""
    torch = pytest.importorskip("torch")
    x_np = _structured_rgb()
    x_t = torch.from_numpy(x_np)
    assert math.isinf(PSNRMetric()(x_t, x_t))
    assert SSIMMetric()(x_t, x_t) == pytest.approx(1.0, abs=1e-4)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
