"""Tests for the configuration contract — :mod:`irchroma.config`.

These tests need **no torch**: ``config.py`` is pure-stdlib (with guarded
omegaconf/pyyaml). They lock down the data contract that all seven builders share:
default values, YAML round-trip, the LULC taxonomy + palettes, and the
synthetic-demo config sanity.
"""

from __future__ import annotations

import json
import os

import pytest

from irchroma import config as cfgmod
from irchroma.config import (
    DEFAULT_LAB_PALETTE,
    DEFAULT_SRGB_PALETTE,
    LULC_CLASSES,
    LULC_NAME_TO_INDEX,
    NUM_LULC_CLASSES,
    Config,
    DataConfig,
    ModelConfig,
    default_config,
    synthetic_demo_config,
)

# Repo root (two levels up from this file: tests/ -> repo root).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_YAML = os.path.join(_REPO_ROOT, "configs", "default.yaml")


def _normalized(d):
    """Round-trip a config dict through JSON to normalize tuple<->list.

    YAML has no tuple type, so a tuple default (e.g. ``train.betas``) round-trips as
    a list. JSON serializes both tuples and lists to arrays, giving a layout-neutral
    structure for equality comparison of *values* (not container subtypes).
    """
    return json.loads(json.dumps(d))


# --------------------------------------------------------------------------- #
# Defaults.
# --------------------------------------------------------------------------- #
def test_config_defaults_are_runnable():
    """``Config()`` yields a fully-populated default with all nested sections."""
    c = Config()
    assert c.name == "irchroma_default"
    assert c.seed == 1337
    # Nested sections exist and are the right dataclass types.
    assert isinstance(c.data, DataConfig)
    assert isinstance(c.model, ModelConfig)
    for section in ("data", "model", "loss", "train", "infer", "eval"):
        assert hasattr(c, section)
    # default_config() is the same as Config().
    assert _normalized(default_config().to_dict()) == _normalized(Config().to_dict())


def test_model_and_data_scale_consistency():
    """The SR scale mirrors across model / sr / data per the contract."""
    c = Config()
    assert c.model.scale == c.model.sr.scale
    assert c.data.scale_factor == c.model.scale
    assert c.model.c_ir == c.data.c_ir
    assert c.model.num_classes == NUM_LULC_CLASSES
    assert c.model.colorization.spade_label_nc == NUM_LULC_CLASSES


# --------------------------------------------------------------------------- #
# LULC taxonomy + palettes.
# --------------------------------------------------------------------------- #
def test_lulc_taxonomy_length_and_indices():
    """The taxonomy has the documented length and a consistent inverse map."""
    assert NUM_LULC_CLASSES == len(LULC_CLASSES)
    assert NUM_LULC_CLASSES == 10
    # name<->index inverse consistency.
    assert LULC_NAME_TO_INDEX == {n: i for i, n in enumerate(LULC_CLASSES)}
    for i, name in enumerate(LULC_CLASSES):
        assert cfgmod.class_index(name) == i
        assert cfgmod.class_name(i) == name
    # Canonical anchors (the contract pins index 0 = water).
    assert LULC_CLASSES[0] == "water"
    assert "trees" in LULC_NAME_TO_INDEX
    assert "built" in LULC_NAME_TO_INDEX


def test_palettes_present_for_every_class():
    """Both the sRGB and Lab palettes cover every LULC class with valid ranges."""
    for name in LULC_CLASSES:
        assert name in DEFAULT_SRGB_PALETTE, f"sRGB palette missing {name}"
        assert name in DEFAULT_LAB_PALETTE, f"Lab palette missing {name}"
        r, g, b = DEFAULT_SRGB_PALETTE[name]
        for ch in (r, g, b):
            assert 0 <= ch <= 255
        L, a, bb = DEFAULT_LAB_PALETTE[name]
        assert 0.0 <= L <= 100.0
        assert -128.0 <= a <= 128.0
        assert -128.0 <= bb <= 128.0


def test_palette_semantics_water_blue_trees_green():
    """Sanity-check the *direction* of the natural-color palette (anti-hallucination)."""
    # Water should be bluish: in sRGB blue dominates; in Lab b* is negative.
    wr, wg, wb = DEFAULT_SRGB_PALETTE["water"]
    assert wb > wr and wb > wg
    assert DEFAULT_LAB_PALETTE["water"][2] < 0.0  # b* negative => blue
    # Trees should be greenish: in Lab a* is negative (green axis).
    assert DEFAULT_LAB_PALETTE["trees"][1] < 0.0  # a* negative => green
    # Built should be ~neutral gray (a*≈b*≈0).
    bL, ba, bb = DEFAULT_LAB_PALETTE["built"]
    assert abs(ba) <= 5.0 and abs(bb) <= 5.0


# --------------------------------------------------------------------------- #
# YAML round-trip.
# --------------------------------------------------------------------------- #
def test_from_yaml_default_round_trip():
    """``Config.from_yaml('configs/default.yaml')`` reconstructs the defaults exactly."""
    assert os.path.exists(_DEFAULT_YAML), f"missing {_DEFAULT_YAML}"
    loaded = Config.from_yaml(_DEFAULT_YAML)
    # The shipped YAML is generated from Config() defaults, so values must match
    # (normalize tuple<->list: YAML has no tuple type, e.g. train.betas).
    assert _normalized(loaded.to_dict()) == _normalized(Config().to_dict())


def test_to_yaml_then_from_dict_round_trips(tmp_path):
    """``to_yaml`` output parses back into an identical Config."""
    c = Config()
    out = tmp_path / "dumped.yaml"
    text = c.to_yaml(str(out))
    assert isinstance(text, str) and len(text) > 0
    assert out.exists()
    reloaded = Config.from_yaml(str(out))
    assert _normalized(reloaded.to_dict()) == _normalized(c.to_dict())


def test_from_dict_ignores_unknown_and_fills_defaults():
    """Partial / forward-compat dicts load: unknown keys dropped, missing defaulted."""
    partial = {
        "name": "custom",
        "model": {"scale": 8, "this_key_does_not_exist": 123},
        "totally_unknown_section": {"x": 1},
    }
    c = Config.from_dict(partial)
    assert c.name == "custom"
    assert c.model.scale == 8
    # A field not overridden keeps its default.
    assert c.model.c_rgb == 3
    # Unknown top-level + nested keys are ignored, not stored.
    assert not hasattr(c, "totally_unknown_section")
    assert not hasattr(c.model, "this_key_does_not_exist")


def test_merge_overrides_deeply():
    """``Config.merge`` deep-merges nested overrides without dropping siblings."""
    c = Config()
    merged = c.merge({"train": {"epochs": 7}})
    assert merged.train.epochs == 7
    # A sibling field in the same section is preserved.
    assert merged.train.batch_size == c.train.batch_size
    # Original config is unchanged (merge returns a new object).
    assert c.train.epochs != 7


# --------------------------------------------------------------------------- #
# Synthetic-demo config.
# --------------------------------------------------------------------------- #
def test_synthetic_demo_config_is_sane_and_cpu():
    """The demo config is tiny, CPU-only, and internally consistent."""
    c = synthetic_demo_config()
    assert c.data.use_synthetic is True
    assert c.train.device == "cpu"
    assert c.infer.device == "cpu"
    assert c.train.precision == "fp32"
    assert c.data.num_workers == 0
    assert c.infer.enable_cache is False
    # Small scale + small nets keep the demo cheap.
    assert c.model.scale == 2
    assert c.model.sr.scale == c.model.scale  # mirror preserved
    assert c.data.ir_patch_size <= 64
    assert c.train.epochs <= 5
    # num_classes consistency survives the demo tweaks.
    assert c.model.num_classes == NUM_LULC_CLASSES


def test_ignore_index_is_negative_one():
    """``IGNORE_INDEX`` is the documented -1 sentinel."""
    assert cfgmod.IGNORE_INDEX == -1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
