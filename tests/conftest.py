"""Shared pytest fixtures for the irchroma test suite.

The fixtures here keep tile sizes deliberately **tiny** so the torch-dependent
tests (tiny nets, the full pipeline forward) run in well under a second on a CPU
CI runner. Two fixtures are provided:

* ``cfg``             — a tiny, CPU-friendly :class:`irchroma.config.Config`
                        (``synthetic_demo_config()`` further shrunk) usable by
                        *every* test, including the torch-free config/interface
                        tests (building a ``Config`` needs no torch).
* ``synthetic_batch`` — a ``batch_size=2`` :class:`irchroma.interfaces.Sample`
                        built deterministically with ``seed=0``. This fixture
                        REQUIRES torch and is therefore guarded with
                        ``pytest.importorskip("torch")`` *inside the fixture body*
                        so collection never fails on a torch-less box — only the
                        tests that actually request it are skipped.

Torch-dependent test *modules* additionally use a module-level
``torch = pytest.importorskip("torch")`` so the whole file is skipped (not
errored) when torch is absent.
"""

from __future__ import annotations

import pytest

from irchroma.config import synthetic_demo_config


@pytest.fixture()
def cfg():
    """A tiny CPU-only :class:`Config` for the whole suite (no torch needed to build).

    Starts from :func:`irchroma.config.synthetic_demo_config` (scale=2, width=16,
    ngf=16, cpu, fp32) and shrinks the IR patch size further so model forwards on
    the synthetic batch are extremely cheap. All sizes stay multiples that keep the
    SR / strided-encoder math clean (ir_patch_size=16 -> RGB 32 at scale 2).
    """
    c = synthetic_demo_config()
    # Shrink for speed: 16x16 IR -> 32x32 RGB at scale 2.
    c.data.ir_patch_size = 16
    c.data.tile_size = 32
    c.data.synthetic_num_samples = 4
    c.data.num_workers = 0
    c.train.batch_size = 2
    # Keep the colorization generator tiny but with a valid downsample depth for
    # a 32px tile (4 downsamples -> 2px bottleneck, still valid).
    c.model.colorization.ngf = 8
    c.model.colorization.num_color_queries = 8
    c.model.colorization.color_decoder_layers = 1
    return c


@pytest.fixture()
def synthetic_batch(cfg):
    """A deterministic ``batch_size=2`` :class:`Sample` (requires torch).

    Guarded so only tests that request this fixture are skipped when torch is
    unavailable; torch-free tests never touch it.
    """
    pytest.importorskip("torch")
    from irchroma.data import make_synthetic_sample

    return make_synthetic_sample(cfg, batch_size=2, seed=0)
