"""Tests for the Stage-1/Stage-2 models and the end-to-end pipeline.

Requires torch. The component models (``GuidedSR``, ``ColorizationGenerator``,
``MultiScaleDiscriminator``) are Wave-A and complete, so those tests always run.
The composed ``IRChromaPipeline`` / ``build_pipeline`` are written concurrently by
another builder, so the pipeline tests import them **at runtime** and
``importorskip`` the module — they skip (not error) until that file lands, then go
green. Nets are kept tiny via the shrunk ``cfg`` fixture for CPU speed.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")  # skip the whole module without torch

from irchroma.config import SRConfig
from irchroma.models.colorization.discriminator import MultiScaleDiscriminator
from irchroma.models.colorization.generator import ColorizationGenerator
from irchroma.models.sr.guided_sr import GuidedSR


def _sr_cfg(cfg):
    """A tiny SRConfig derived from the demo cfg (scale 2, narrow width)."""
    return SRConfig(
        scale=int(cfg.model.scale),
        in_channels=int(cfg.model.c_ir),
        out_channels=int(cfg.model.c_ir),
        width=16,
        guide_channels=int(cfg.model.c_guide),
        use_guide=True,
        guide_fusion="cross_attention",
        enc_blocks=[1, 1],
        middle_blocks=1,
        dec_blocks=[1, 1],
    )


# --------------------------------------------------------------------------- #
# Stage-1: GuidedSR.
# --------------------------------------------------------------------------- #
def test_guided_sr_forward_shapes_with_guide(cfg, synthetic_batch):
    """GuidedSR upscales IR by ``scale`` and returns a dict with sr + LR feat."""
    model = GuidedSR.from_config(_sr_cfg(cfg)).eval()
    ir = synthetic_batch["ir"]
    guide = synthetic_batch["guide"]
    b, c, h, w = ir.shape
    scale = int(cfg.model.scale)
    with torch.no_grad():
        out = model(ir, guide)
    assert isinstance(out, dict) and set(out.keys()) == {"sr", "feat"}
    assert out["sr"].shape == (b, c, h * scale, w * scale)
    assert out["feat"].shape[-2:] == (h, w)  # LR-resolution restoration features
    assert torch.isfinite(out["sr"]).all()


def test_guided_sr_graceful_without_guide(cfg, synthetic_batch):
    """GuidedSR degrades to single-image SR when no guide is supplied."""
    model = GuidedSR.from_config(_sr_cfg(cfg)).eval()
    ir = synthetic_batch["ir"]
    b, c, h, w = ir.shape
    scale = int(cfg.model.scale)
    with torch.no_grad():
        sr = model(ir, None, return_dict=False)
    assert sr.shape == (b, c, h * scale, w * scale)
    assert torch.isfinite(sr).all()


# --------------------------------------------------------------------------- #
# Stage-2: ColorizationGenerator.
# --------------------------------------------------------------------------- #
def test_colorization_generator_forward_shapes(cfg, synthetic_batch):
    """The generator maps SR'd IR (+ semantic) -> RGB ``[B,3,Hs,Ws]`` in [0,1]."""
    gen = ColorizationGenerator(cfg.model.colorization).eval()
    rgb_gt = synthetic_batch["rgb"]  # [B, 3, Hs, Ws] — gives us the SR resolution
    b, _, hs, ws = rgb_gt.shape
    # Build a 1-channel SR'd-IR-like input + an HR semantic map at SR resolution.
    sr_ir = rgb_gt.mean(dim=1, keepdim=True)  # [B, 1, Hs, Ws] in [0,1]
    sem_lr = synthetic_batch["semantic"]  # [B, H, W]
    sem_hr = (
        torch.nn.functional.interpolate(
            sem_lr.unsqueeze(1).float(), size=(hs, ws), mode="nearest"
        )
        .long()
        .squeeze(1)
    )
    with torch.no_grad():
        rgb = gen(sr_ir, sem_hr)
    assert rgb.shape == (b, 3, hs, ws)
    assert float(rgb.min()) >= 0.0 and float(rgb.max()) <= 1.0
    assert torch.isfinite(rgb).all()


def test_colorization_generator_without_semantic(cfg, synthetic_batch):
    """The generator runs with ``semantic=None`` (plain-residual fallback path)."""
    gen = ColorizationGenerator(cfg.model.colorization).eval()
    rgb_gt = synthetic_batch["rgb"]
    b, _, hs, ws = rgb_gt.shape
    sr_ir = rgb_gt.mean(dim=1, keepdim=True)
    with torch.no_grad():
        rgb = gen(sr_ir, None)
    assert rgb.shape == (b, 3, hs, ws)
    assert float(rgb.min()) >= 0.0 and float(rgb.max()) <= 1.0


# --------------------------------------------------------------------------- #
# Discriminator: MultiScaleDiscriminator output structure.
# --------------------------------------------------------------------------- #
def test_multiscale_discriminator_output_structure(cfg, synthetic_batch):
    """MultiScaleD returns a list (scales) of lists (layers); each ends in a logit map."""
    num_scales = 2
    disc = MultiScaleDiscriminator(
        in_channels=3, num_scales=num_scales, n_layers=2, ndf=8, use_spectral_norm=True
    ).eval()
    rgb = synthetic_batch["rgb"]  # [B, 3, Hs, Ws]
    with torch.no_grad():
        out = disc(rgb)
    # Outer list = scales; each inner = per-stage activations.
    assert isinstance(out, list) and len(out) == num_scales
    for scale_feats in out:
        assert isinstance(scale_feats, list) and len(scale_feats) >= 2
        logits = scale_feats[-1]  # final stage = 1-channel patch logits
        assert logits.shape[0] == rgb.shape[0]
        assert logits.shape[1] == 1
        assert torch.isfinite(logits).all()


def test_discriminator_from_config(cfg):
    """``from_config`` wires disc_* / spectral_norm from the ColorizationConfig."""
    col = cfg.model.colorization
    disc = MultiScaleDiscriminator.from_config(col, in_channels=4)
    assert disc.num_scales == int(col.num_discriminators)
    assert disc.n_layers == int(col.disc_n_layers)


# --------------------------------------------------------------------------- #
# End-to-end pipeline (built concurrently — import at runtime, skip if absent).
# --------------------------------------------------------------------------- #
def test_build_pipeline_forward_is_pipeline_output(cfg, synthetic_batch):
    """build_pipeline(cfg).forward(batch) -> a PipelineOutput with rgb [0,1] + SR'd shapes."""
    pipeline_mod = pytest.importorskip(
        "irchroma.models.pipeline",
        reason="IRChromaPipeline is built concurrently by another worker.",
    )
    build_pipeline = getattr(pipeline_mod, "build_pipeline", None)
    if build_pipeline is None:  # pragma: no cover - constructor not yet exported
        pytest.skip("build_pipeline not yet available in irchroma.models.pipeline")

    pipeline = build_pipeline(cfg)
    # Run in eval / no-grad if it is an nn.Module-like object.
    if hasattr(pipeline, "eval"):
        pipeline.eval()
    with torch.no_grad():
        out = pipeline.forward(synthetic_batch)

    assert "rgb" in out and "sr" in out
    ir = synthetic_batch["ir"]
    b, _, h, w = ir.shape
    scale = int(cfg.model.scale)
    rgb = out["rgb"]
    assert rgb.shape[0] == b and rgb.shape[1] == 3
    # RGB is produced at scale x the input IR resolution.
    assert rgb.shape[-2:] == (h * scale, w * scale)
    assert float(rgb.min()) >= 0.0 and float(rgb.max()) <= 1.0
    assert torch.isfinite(rgb).all()
    # SR keeps the IR channel count and is upscaled by scale.
    sr = out["sr"]
    assert sr.shape[-2:] == (h * scale, w * scale)
    assert torch.isfinite(sr).all()


def test_pipeline_satisfies_protocol(cfg):
    """The built pipeline structurally satisfies ``PipelineProtocol``."""
    from irchroma.interfaces import PipelineProtocol

    pipeline_mod = pytest.importorskip(
        "irchroma.models.pipeline",
        reason="IRChromaPipeline is built concurrently by another worker.",
    )
    build_pipeline = getattr(pipeline_mod, "build_pipeline", None)
    if build_pipeline is None:  # pragma: no cover
        pytest.skip("build_pipeline not yet available")
    pipeline = build_pipeline(cfg)
    assert isinstance(pipeline, PipelineProtocol)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
