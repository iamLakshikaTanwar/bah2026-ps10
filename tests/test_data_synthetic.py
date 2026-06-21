"""Tests for the procedural data layer — :mod:`irchroma.data.synthetic`.

Requires torch (the synthetic factories build real tensors). The module-level
``importorskip`` skips the whole file gracefully on a torch-less box. We pin the
tensor contract from ``synthetic.py``'s docstring:

  * ``ir``       : ``FloatTensor [B, C_ir, H, W]`` in ~[0,1] (low contrast).
  * ``rgb``      : ``FloatTensor [B, 3, H*scale, W*scale]`` in [0,1].
  * ``semantic`` : ``LongTensor  [B, H, W]`` in ``[0, NUM_LULC_CLASSES-1]``.
  * ``guide``    : ``FloatTensor [B, C_g, H*scale, W*scale]`` in [0,1] (or None).

plus determinism by seed and that ``build_synthetic_loader`` yields batched samples.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")  # skip the whole module without torch

from irchroma.config import NUM_LULC_CLASSES
from irchroma.data import (
    SyntheticIRRGBDataset,
    build_synthetic_loader,
    make_synthetic_sample,
)


def _dims(cfg):
    """Resolve the expected (B-agnostic) IR/HR dims from a config."""
    h = int(cfg.data.ir_patch_size)
    scale = int(cfg.model.scale)
    return h, h * scale, scale


# --------------------------------------------------------------------------- #
# Shapes / dtypes / ranges.
# --------------------------------------------------------------------------- #
def test_sample_keys_present(cfg):
    """A synthetic sample carries the full Sample key set."""
    s = make_synthetic_sample(cfg, batch_size=2, seed=0)
    assert set(s.keys()) == {"ir", "rgb", "guide", "semantic", "meta"}


def test_ir_shape_dtype_range(cfg):
    """IR is ``[B, C_ir, H, W]`` float in ~[0,1]."""
    h, _, _ = _dims(cfg)
    s = make_synthetic_sample(cfg, batch_size=2, seed=0)
    ir = s["ir"]
    assert ir.shape == (2, cfg.model.c_ir, h, h)
    assert ir.dtype == torch.float32
    assert torch.is_floating_point(ir)
    assert float(ir.min()) >= 0.0 and float(ir.max()) <= 1.0


def test_rgb_shape_dtype_range(cfg):
    """RGB is ``[B, 3, H*scale, W*scale]`` float strictly in [0,1]."""
    _, hs, _ = _dims(cfg)
    s = make_synthetic_sample(cfg, batch_size=2, seed=0)
    rgb = s["rgb"]
    assert rgb.shape == (2, 3, hs, hs)
    assert rgb.dtype == torch.float32
    assert float(rgb.min()) >= 0.0 and float(rgb.max()) <= 1.0


def test_semantic_shape_dtype_range(cfg):
    """Semantic is a Long ``[B, H, W]`` map with valid class indices."""
    h, _, _ = _dims(cfg)
    s = make_synthetic_sample(cfg, batch_size=2, seed=0)
    sem = s["semantic"]
    assert sem.shape == (2, h, h)
    assert sem.dtype == torch.long
    assert int(sem.min()) >= 0
    assert int(sem.max()) <= NUM_LULC_CLASSES - 1


def test_rgb_is_scale_times_ir(cfg):
    """The RGB grid is exactly ``scale``x the IR grid (the SR target contract)."""
    s = make_synthetic_sample(cfg, batch_size=1, seed=1)
    _, _, scale = _dims(cfg)
    ir_h, ir_w = s["ir"].shape[-2:]
    rgb_h, rgb_w = s["rgb"].shape[-2:]
    assert (rgb_h, rgb_w) == (ir_h * scale, ir_w * scale)


def test_guide_shape_and_range_when_present(cfg):
    """The HR guide (if c_guide>0) is ``[B, C_g, Hs, Ws]`` in [0,1] at RGB resolution."""
    _, hs, _ = _dims(cfg)
    s = make_synthetic_sample(cfg, batch_size=2, seed=0)
    guide = s["guide"]
    if cfg.model.c_guide > 0:
        assert guide is not None
        assert guide.shape == (2, cfg.model.c_guide, hs, hs)
        assert float(guide.min()) >= 0.0 and float(guide.max()) <= 1.0
    else:  # pragma: no cover - default demo cfg has a guide
        assert guide is None


def test_meta_records_synthesis_params(cfg):
    """``meta`` carries provenance + the synthesis parameters for reproducibility."""
    s = make_synthetic_sample(cfg, batch_size=2, seed=7)
    meta = s["meta"]
    assert meta["synthetic"] is True
    assert meta["seed"] == 7
    assert meta["scale"] == cfg.model.scale
    assert meta["num_classes"] == NUM_LULC_CLASSES


def test_batch_size_must_be_positive(cfg):
    """A non-positive batch size is rejected with a clear error."""
    with pytest.raises(ValueError):
        make_synthetic_sample(cfg, batch_size=0, seed=0)


# --------------------------------------------------------------------------- #
# Determinism.
# --------------------------------------------------------------------------- #
def test_determinism_same_seed(cfg):
    """Identical ``(cfg, batch_size, seed)`` -> bit-identical tensors."""
    a = make_synthetic_sample(cfg, batch_size=2, seed=123)
    b = make_synthetic_sample(cfg, batch_size=2, seed=123)
    assert torch.equal(a["ir"], b["ir"])
    assert torch.equal(a["rgb"], b["rgb"])
    assert torch.equal(a["semantic"], b["semantic"])
    assert torch.equal(a["guide"], b["guide"])


def test_different_seed_differs(cfg):
    """Different seeds produce different data (RNG actually varies)."""
    a = make_synthetic_sample(cfg, batch_size=2, seed=0)
    b = make_synthetic_sample(cfg, batch_size=2, seed=1)
    # At least one modality must differ (extremely likely the semantic map does).
    assert not (
        torch.equal(a["ir"], b["ir"])
        and torch.equal(a["rgb"], b["rgb"])
        and torch.equal(a["semantic"], b["semantic"])
    )


# --------------------------------------------------------------------------- #
# Dataset + DataLoader.
# --------------------------------------------------------------------------- #
def test_dataset_item_is_unbatched(cfg):
    """Dataset items have no leading batch dim (DataLoader re-collates them)."""
    ds = SyntheticIRRGBDataset(cfg, n=3, seed=0)
    assert len(ds) == 3
    item = ds[0]
    h, _, _ = _dims(cfg)
    assert item["ir"].shape == (cfg.model.c_ir, h, h)  # [C, H, W], no batch
    assert item["semantic"].shape == (h, h)  # [H, W]


def test_build_synthetic_loader_yields_batches(cfg):
    """The loader yields batched Samples obeying the [B, ...] tensor contract."""
    loader = build_synthetic_loader(
        cfg, batch_size=2, n=4, seed=0, shuffle=False, num_workers=0
    )
    batch = next(iter(loader))
    h, hs, _ = _dims(cfg)
    assert batch["ir"].shape == (2, cfg.model.c_ir, h, h)
    assert batch["rgb"].shape == (2, 3, hs, hs)
    assert batch["semantic"].shape == (2, h, h)
    # meta is collated as a per-item list (length == batch size).
    assert isinstance(batch["meta"], list) and len(batch["meta"]) == 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
