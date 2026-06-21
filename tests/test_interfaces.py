"""Tests for the code contract — :mod:`irchroma.interfaces`.

``interfaces.py`` is designed to import even without torch (it installs stubs for
``nn.Module`` / ``Tensor``), so most of these run torch-free. We verify:

* ``Sample`` / ``PipelineOutput`` TypedDict key sets (the tensor-flow contract).
* ``LossTerm`` / ``CompositeLoss`` exist with the right API (add/set weights, the
  ``forward -> (total, components)`` shape, ``from_config`` term-factory wiring).
  These are exercised with pure-Python stand-in terms so no torch is required.
* ``Metric`` / ``MetricSuite`` API (NaN-on-error robustness, ``directions``).
* ``ColorLUTProtocol`` is a runtime-checkable structural Protocol.
"""

from __future__ import annotations

import pytest

from irchroma import interfaces
from irchroma.interfaces import (
    BaseModel,
    ColorLUTProtocol,
    CompositeLoss,
    LossTerm,
    Metric,
    MetricSuite,
    PipelineOutput,
    PipelineProtocol,
    Sample,
)


# --------------------------------------------------------------------------- #
# TypedDicts: Sample / PipelineOutput key sets.
# --------------------------------------------------------------------------- #
def test_sample_typed_dict_keys():
    """``Sample`` declares exactly the documented optional keys."""
    keys = set(Sample.__annotations__.keys())
    assert keys == {"ir", "rgb", "guide", "semantic", "meta"}
    # total=False => all keys optional (an IR-only inference sample is valid).
    assert Sample.__total__ is False


def test_pipeline_output_typed_dict_keys():
    """``PipelineOutput`` declares the documented keys (rgb + sr required-by-convention)."""
    keys = set(PipelineOutput.__annotations__.keys())
    assert keys == {"rgb", "sr", "semantic_pred", "uncertainty", "aux"}
    assert PipelineOutput.__total__ is False


def test_module_exports_and_torch_flag():
    """The contract advertises a stable ``__all__`` and a TORCH_AVAILABLE flag."""
    assert isinstance(interfaces.TORCH_AVAILABLE, bool)
    for name in (
        "Sample",
        "PipelineOutput",
        "BaseModel",
        "PipelineProtocol",
        "LossTerm",
        "CompositeLoss",
        "Metric",
        "MetricSuite",
        "ColorLUTProtocol",
    ):
        assert name in interfaces.__all__


# --------------------------------------------------------------------------- #
# BaseModel / PipelineProtocol.
# --------------------------------------------------------------------------- #
def test_base_model_has_name_and_num_parameters():
    """``BaseModel`` exposes the stable ``name`` marker + ``num_parameters`` helper."""
    assert hasattr(BaseModel, "name")
    assert hasattr(BaseModel, "num_parameters")


def test_pipeline_protocol_is_runtime_checkable():
    """Any object with ``forward(batch)`` structurally satisfies the pipeline protocol."""

    class _DummyPipeline:
        def forward(self, batch):  # noqa: ANN001
            return {"rgb": None, "sr": None}

    assert isinstance(_DummyPipeline(), PipelineProtocol)

    class _NotAPipeline:
        pass

    assert not isinstance(_NotAPipeline(), PipelineProtocol)


# --------------------------------------------------------------------------- #
# LossTerm / CompositeLoss — torch-free with pure-Python stand-in terms.
# --------------------------------------------------------------------------- #
class _ConstTerm(LossTerm):
    """A LossTerm stand-in returning a fixed Python float (no torch needed).

    ``CompositeLoss.forward`` multiplies by the weight and sums; the float fallback
    path (``float(weighted)``) handles non-tensor values, so a plain float works for
    contract-level API testing without instantiating any nn.Module state.
    """

    def __init__(self, value: float) -> None:
        # Avoid nn.Module.__init__ (it's a stub without torch); set attrs directly.
        self.value = float(value)
        self.name = "const"

    def forward(self, output, target, ctx):  # noqa: ANN001
        return self.value


def test_loss_term_contract_signature():
    """``LossTerm`` advertises a ``name`` and a 3-arg ``forward(output, target, ctx)``."""
    assert hasattr(LossTerm, "name")
    term = _ConstTerm(2.0)
    assert term.forward({}, {}, {}) == 2.0


def test_composite_loss_add_weights_and_forward():
    """CompositeLoss sums weighted terms and reports per-term weighted components."""
    comp = CompositeLoss()
    comp.add_term("a", _ConstTerm(1.0), weight=2.0)  # contributes 2.0
    comp.add_term("b", _ConstTerm(3.0), weight=0.5)  # contributes 1.5
    comp.add_term("zero", _ConstTerm(99.0), weight=0.0)  # skipped entirely
    assert comp.weights == {"a": 2.0, "b": 0.5, "zero": 0.0}

    total, components = comp.forward({}, {}, {})
    assert float(total) == pytest.approx(3.5)
    assert components == {"a": pytest.approx(2.0), "b": pytest.approx(1.5)}
    # Zero-weight term is skipped, so it never appears in components.
    assert "zero" not in components


def test_composite_loss_set_weight_and_unknown_key():
    """``set_weight`` updates in place; unknown keys raise ``KeyError``."""
    comp = CompositeLoss()
    comp.add_term("a", _ConstTerm(1.0), weight=1.0)
    comp.set_weight("a", 5.0)
    total, components = comp.forward({}, {}, {})
    assert float(total) == pytest.approx(5.0)
    assert components["a"] == pytest.approx(5.0)
    with pytest.raises(KeyError):
        comp.set_weight("does_not_exist", 1.0)


def test_composite_loss_empty_returns_zero():
    """An empty / all-zero CompositeLoss returns a 0 total without crashing."""
    comp = CompositeLoss()
    total, components = comp.forward({}, {}, {})
    assert float(total) == pytest.approx(0.0)
    assert components == {}


def test_composite_loss_from_config_wires_by_name():
    """``from_config`` instantiates only non-zero-weight terms, keyed by config attr."""

    class _LossCfgStub:
        a = 2.0
        b = 0.0  # disabled -> factory must NOT be called
        c = 3.0

    called = {"a": 0, "b": 0, "c": 0}

    def _make(name: str):
        def factory():
            called[name] += 1
            return _ConstTerm(1.0)

        return factory

    comp = CompositeLoss.from_config(
        _LossCfgStub(), {"a": _make("a"), "b": _make("b"), "c": _make("c")}
    )
    # Only non-zero-weight factories were invoked (lazy / cheap).
    assert called == {"a": 1, "b": 0, "c": 1}
    assert comp.weights == {"a": 2.0, "c": 3.0}
    total, _ = comp.forward({}, {}, {})
    assert float(total) == pytest.approx(5.0)  # 1*2 + 1*3


# --------------------------------------------------------------------------- #
# Metric / MetricSuite.
# --------------------------------------------------------------------------- #
class _FixedMetric(Metric):
    """A Metric returning a fixed value (or raising) — for suite-robustness tests."""

    def __init__(self, name: str, value: float, higher: bool = True, raises: bool = False):
        self.name = name
        self.value = value
        self.higher_is_better = higher
        self._raises = raises

    def __call__(self, pred, target=None, ctx=None):  # noqa: ANN001
        if self._raises:
            raise RuntimeError("boom (simulated missing backend)")
        return self.value


def test_metric_is_abstract():
    """``Metric`` cannot be instantiated directly (abstract ``__call__``)."""
    with pytest.raises(TypeError):
        Metric()  # type: ignore[abstract]


def test_metric_suite_evaluate_and_directions():
    """``MetricSuite`` evaluates all metrics and reports ranking directions."""
    suite = MetricSuite()
    suite.add(_FixedMetric("psnr", 30.0, higher=True))
    suite.add(_FixedMetric("ciede2000", 2.0, higher=False))
    assert set(suite.names) == {"psnr", "ciede2000"}
    assert suite.directions() == {"psnr": True, "ciede2000": False}
    results = suite.evaluate(pred=None)
    assert results == {"psnr": 30.0, "ciede2000": 2.0}


def test_metric_suite_records_nan_on_error_not_strict():
    """A metric that raises is recorded as NaN (one bad dep never aborts the run)."""
    suite = MetricSuite()
    suite.add(_FixedMetric("ok", 1.0))
    suite.add(_FixedMetric("broken", 0.0, raises=True))
    results = suite.evaluate(pred=None)
    assert results["ok"] == 1.0
    assert results["broken"] != results["broken"]  # NaN != NaN
    # strict=True surfaces the error instead.
    with pytest.raises(RuntimeError):
        suite.evaluate(pred=None, strict=True)


# --------------------------------------------------------------------------- #
# ColorLUTProtocol.
# --------------------------------------------------------------------------- #
def test_color_lut_protocol_structural():
    """Any object exposing ``apply(rgb, semantic)`` satisfies ``ColorLUTProtocol``."""

    class _LUT:
        def apply(self, rgb, semantic):  # noqa: ANN001
            return rgb

    assert isinstance(_LUT(), ColorLUTProtocol)

    class _NotLUT:
        pass

    assert not isinstance(_NotLUT(), ColorLUTProtocol)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
