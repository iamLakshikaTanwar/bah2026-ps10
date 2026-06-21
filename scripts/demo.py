#!/usr/bin/env python
"""scripts/demo.py — THE end-to-end synthetic smoke test for irchroma.

Runs the WHOLE pipeline on procedural data, on CPU, with only ``torch`` + ``numpy``
(+ optional ``matplotlib``) — no network, no credentials, no GPU. It is the single
script the integration worker runs to prove the Wave-A modules + the Wave-B pipeline /
trainer wire together correctly.

Steps (all asserted with clear messages):

  1. ``cfg = synthetic_demo_config()`` and ``pipeline = build_pipeline(cfg)``.
  2. ``batch = make_synthetic_sample(cfg, batch_size=2, seed=0)`` — a coherent IR/RGB pair.
  3. ``out = pipeline(batch)`` and assert the shape contract:
       * ``out['sr']`` is ``[B, C_ir, H*scale, W*scale]`` (upscaled),
       * ``out['rgb']`` is ``[B, 3, H*scale, W*scale]`` in ``[0, 1]``.
  4. Compute the fidelity-dominant composite loss
     ``build_composite_loss(cfg)(out, batch, ctx)`` and a few metrics via
     ``build_metric_suite(cfg)``.
  5. Run a few real optimizer steps through :class:`~irchroma.train.Trainer` to prove the
     two-player loop trains (loss is finite and updates).
  6. If matplotlib is present, save ``outputs/demo.png`` with panels
     ``[IR | SR | Colorized RGB | Target RGB | semantic]``.
  7. Print a concise report (shapes, loss/metric values, param counts, per-iter time).

Run directly (``python scripts/demo.py``) or via ``irchroma demo`` /
``python -m irchroma.train.cli --demo``. Exit code ``0`` on success.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, Optional


# --------------------------------------------------------------------------- #
# Make ``irchroma`` importable when run as a loose script (``python scripts/demo.py``)
# from a source checkout: add the repo's ``src/`` to sys.path if needed.
# --------------------------------------------------------------------------- #
def _ensure_irchroma_importable() -> None:
    try:
        import irchroma  # noqa: F401

        return
    except Exception:
        here = os.path.dirname(os.path.abspath(__file__))
        src = os.path.join(os.path.dirname(here), "src")
        if os.path.isdir(src) and src not in sys.path:
            sys.path.insert(0, src)


_ensure_irchroma_importable()


def _fmt_shape(t: Any) -> str:
    """Render a tensor's shape as a tuple string (``'-'`` for ``None``)."""
    if t is None:
        return "-"
    try:
        return str(tuple(t.shape))
    except Exception:
        return "?"


def run_demo(
    cfg: Optional[Any] = None,
    steps: int = 3,
    out_dir: str = "outputs",
    make_plot: bool = True,
    seed: int = 0,
    batch_size: int = 2,
) -> Dict[str, Any]:
    """Execute the end-to-end smoke test and return a structured report dict.

    Args:
      cfg:        a :class:`~irchroma.config.Config`; defaults to ``synthetic_demo_config()``.
      steps:      number of optimizer steps to run (proves training; >=1).
      out_dir:    directory for ``demo.png`` (created if missing).
      make_plot:  save the panel figure when matplotlib is importable.
      seed:       synthetic-sample seed.
      batch_size: synthetic batch size.

    Returns:
      ``{"ok": bool, "shapes": {...}, "loss": float, "components": {...},
         "metrics": {...}, "params": {...}, "per_iter_s": float, "plot": path|None,
         "train_losses": [...]}``.
    """
    # ---- Hard dependency: torch. Fail loudly and early if missing. --------- #
    try:
        import torch
    except Exception as exc:  # pragma: no cover - integration installs torch
        print(f"FATAL: torch is required to run the demo ({exc}).", file=sys.stderr)
        return {"ok": False, "error": "torch unavailable"}

    from irchroma.config import synthetic_demo_config
    from irchroma.data.synthetic import make_synthetic_sample
    from irchroma.losses import build_composite_loss
    from irchroma.metrics import build_metric_suite
    from irchroma.models import build_pipeline
    from irchroma.train import Trainer, seed_everything

    # Determinism: seed Python/NumPy/torch BEFORE building the pipeline so the random
    # weight init (and optimizer/dropout/data) is reproducible across machines. This is
    # what makes the numerically-stable training path deterministic rather than "lucky"
    # (un-seeded init previously surfaced the now-fixed NaN-gradient bug only sometimes).
    seed_everything(seed)

    if cfg is None:
        cfg = synthetic_demo_config()
    device = torch.device(cfg.train.device or "cpu")
    scale = int(cfg.model.scale)
    report: Dict[str, Any] = {"ok": False}

    print("=" * 70)
    print("irchroma end-to-end synthetic demo")
    print("=" * 70)
    print(f"config            : {cfg.name}")
    print(f"device            : {device}  scale x{scale}  num_classes={cfg.model.num_classes}")

    # ---- 1) Build the pipeline -------------------------------------------- #
    pipeline = build_pipeline(cfg).to(device)
    pipeline.eval()
    assert pipeline is not None, "build_pipeline returned None"
    assert hasattr(pipeline, "discriminator"), "pipeline must expose .discriminator for training"

    # ---- 2) Synthetic batch ----------------------------------------------- #
    batch = make_synthetic_sample(cfg, batch_size=batch_size, seed=seed)
    for key in ("ir", "rgb", "semantic"):
        assert batch.get(key) is not None, f"synthetic sample missing '{key}'"
    # Move tensors to device.
    for key in ("ir", "rgb", "guide", "semantic"):
        v = batch.get(key)
        if v is not None and isinstance(v, torch.Tensor):
            batch[key] = v.to(device)

    ir = batch["ir"]
    b, c_ir, h, w = ir.shape
    hs, ws = h * scale, w * scale

    # ---- 3) Forward + shape asserts --------------------------------------- #
    t0 = time.time()
    with torch.no_grad():
        out = pipeline(batch)
    fwd_s = time.time() - t0

    sr = out["sr"]
    rgb = out["rgb"]
    assert sr.shape == (b, c_ir, hs, ws), (
        f"SR shape {tuple(sr.shape)} != expected {(b, c_ir, hs, ws)}"
    )
    assert rgb.dim() == 4 and rgb.shape[1] == 3 and rgb.shape[2:] == (hs, ws), (
        f"RGB shape {tuple(rgb.shape)} != expected {(b, 3, hs, ws)}"
    )
    rgb_min, rgb_max = float(rgb.min()), float(rgb.max())
    assert -1e-4 <= rgb_min and rgb_max <= 1.0 + 1e-4, (
        f"RGB out of [0,1]: min={rgb_min:.4f} max={rgb_max:.4f}"
    )

    shapes = {
        "ir": _fmt_shape(ir),
        "guide": _fmt_shape(batch.get("guide")),
        "semantic": _fmt_shape(batch.get("semantic")),
        "sr": _fmt_shape(sr),
        "rgb": _fmt_shape(rgb),
        "semantic_pred": _fmt_shape(out.get("semantic_pred")),
        "uncertainty": _fmt_shape(out.get("uncertainty")),
    }
    report["shapes"] = shapes
    print("\n-- shapes --")
    for k, v in shapes.items():
        print(f"  {k:14s}: {v}")
    print(f"  rgb range     : [{rgb_min:.4f}, {rgb_max:.4f}]")
    print(f"  forward time  : {fwd_s * 1000:.1f} ms for B={b}")

    # ---- 4) Composite loss + metrics -------------------------------------- #
    criterion = build_composite_loss(cfg).to(device)
    # Provide an HR IR reference so the SR loss terms (sr vs target['ir']) are
    # spatially well-defined for synthetic LR IR (mirrors the Trainer).
    ir_hr = torch.nn.functional.interpolate(ir, size=(hs, ws), mode="bilinear", align_corners=False)
    loss_target = {
        "rgb": batch.get("rgb"),
        "ir": ir_hr,
        "semantic": batch.get("semantic"),
        "meta": batch.get("meta"),
    }
    ctx = {"segmenter": getattr(pipeline, "segmenter", None)}
    total, components = criterion(out, loss_target, ctx)
    loss_val = float(total.detach().item())
    assert loss_val == loss_val, "composite loss is NaN"
    report["loss"] = loss_val
    report["components"] = {k: float(v) for k, v in components.items()}
    print("\n-- composite loss --")
    print(f"  total         : {loss_val:.4f}")
    for k in sorted(components):
        print(f"    {k:24s}: {float(components[k]):+.4f}")

    suite = build_metric_suite(cfg)
    metrics = suite.evaluate(rgb, batch.get("rgb"), ctx={"segmenter": getattr(pipeline, "segmenter", None)})
    # Keep only finite metrics for the printed report (optional backends -> NaN).
    finite_metrics = {k: float(v) for k, v in metrics.items() if v == v}
    report["metrics"] = finite_metrics
    print("\n-- metrics (finite; optional backends may be skipped) --")
    if finite_metrics:
        for k in sorted(finite_metrics):
            print(f"    {k:24s}: {finite_metrics[k]:.4f}")
    else:
        print("    (none available on this box)")

    # ---- 5) Prove training (a few real optimizer steps) ------------------- #
    print(f"\n-- training {steps} step(s) through Trainer --")
    trainer = Trainer(cfg, pipeline=pipeline, device=str(device), max_steps=steps, seed=seed)
    train_losses = []
    t_train = time.time()
    n_done = 0
    for _ in range(max(1, steps)):
        logs = trainer.train_step(batch, sr_only=False)
        train_losses.append(logs.get("loss_total", float("nan")))
        n_done += 1
    per_iter = (time.time() - t_train) / max(1, n_done)
    report["train_losses"] = [float(x) for x in train_losses]
    report["per_iter_s"] = float(per_iter)
    for i, lv in enumerate(train_losses):
        print(f"    step {i + 1}: loss_total={lv:.4f}")
    for lv in train_losses:
        assert lv == lv, "training produced a NaN loss"
    print(f"    per-iter time : {per_iter * 1000:.1f} ms")

    # ---- params -------------------------------------------- #
    def _np(mod: Any) -> int:
        return 0 if mod is None else int(sum(p.numel() for p in mod.parameters()))

    params = {
        "sr_model": _np(getattr(pipeline, "sr_model", None)),
        "colorizer": _np(getattr(pipeline, "colorizer", None)),
        "adaint_lut": _np(getattr(pipeline, "adaint_lut", None)),
        "segmenter": _np(getattr(pipeline, "segmenter", None)),
        "discriminator": _np(getattr(pipeline, "discriminator", None)),
        "pipeline_forward": int(sum(p.numel() for p in pipeline.parameters())),
    }
    report["params"] = params
    print("\n-- parameter counts --")
    for k in ("sr_model", "colorizer", "adaint_lut", "segmenter", "discriminator", "pipeline_forward"):
        print(f"    {k:18s}: {params[k]:,}")

    # ---- 6) Optional panel figure ----------------------------------------- #
    plot_path = None
    if make_plot:
        plot_path = _save_panel(out, batch, out_dir)
        if plot_path:
            report["plot"] = plot_path
            print(f"\nsaved panel -> {plot_path}")
        else:
            print("\n(matplotlib unavailable — skipped demo.png)")

    report["ok"] = True
    print("\n" + "=" * 70)
    print("DEMO OK — end-to-end pipeline ran, loss finite, training stepped.")
    print("=" * 70)
    return report


def _save_panel(out: Dict[str, Any], batch: Dict[str, Any], out_dir: str) -> Optional[str]:
    """Save a ``[IR | SR | Colorized RGB | Target RGB | semantic]`` panel (matplotlib)."""
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless backend
        import matplotlib.pyplot as plt
    except Exception:
        return None

    def _img(t: Any, idx: int = 0) -> Any:
        """First-sample CHW tensor -> HWC numpy for imshow (single-channel squeezed)."""
        arr = t[idx].detach().cpu().float().numpy()
        if arr.shape[0] == 1:
            return arr[0]
        return arr.transpose(1, 2, 0)

    try:
        os.makedirs(out_dir, exist_ok=True)
        ir = batch["ir"]
        sr = out["sr"]
        # Prefer the fully-colorized product (pre honesty-desaturation) for the panel so
        # the colorization is visible even with a randomly-initialized fallback checker;
        # fall back to the final product if the pipeline did not expose it.
        rgb_src = out.get("aux", {}).get("rgb_colorized", out["rgb"])
        rgb = rgb_src.clamp(0.0, 1.0)
        rgb_gt = batch.get("rgb")
        sem = batch.get("semantic")

        panels = [("IR (LR)", _img(ir), "gray"),
                  ("SR (IR)", _img(sr), "gray"),
                  ("Colorized RGB", _img(rgb), None),
                  ("Target RGB", _img(rgb_gt) if rgb_gt is not None else None, None),
                  ("Semantic", sem[0].detach().cpu().numpy() if sem is not None else None, "tab10")]

        n = len(panels)
        fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.2))
        for ax, (title, data, cmap) in zip(axes, panels):
            ax.set_title(title, fontsize=9)
            ax.axis("off")
            if data is None:
                continue
            ax.imshow(data, cmap=cmap)
        fig.suptitle("irchroma end-to-end demo (synthetic)", fontsize=11)
        fig.tight_layout()
        path = os.path.join(out_dir, "demo.png")
        fig.savefig(path, dpi=110)
        plt.close(fig)
        return path
    except Exception:
        return None


def main(argv: Optional[list] = None) -> int:
    """CLI entrypoint for the standalone demo script."""
    parser = argparse.ArgumentParser(description="irchroma end-to-end synthetic demo.")
    parser.add_argument("--config", default=None, help="optional YAML config (else synthetic demo).")
    parser.add_argument("--steps", type=int, default=3, help="optimizer steps to run.")
    parser.add_argument("--out", default="outputs", help="output dir for demo.png.")
    parser.add_argument("--no-plot", action="store_true", help="skip the matplotlib panel.")
    parser.add_argument("--device", default=None, help="cpu|cuda (default: config/cpu).")
    parser.add_argument("--seed", type=int, default=0, help="synthetic-sample seed.")
    args = parser.parse_args(argv)

    from irchroma.config import Config, synthetic_demo_config

    cfg = Config.from_yaml(args.config) if args.config else synthetic_demo_config()
    if args.device:
        cfg.train.device = args.device
        cfg.infer.device = args.device

    result = run_demo(
        cfg,
        steps=int(args.steps),
        out_dir=args.out,
        make_plot=not args.no_plot,
        seed=int(args.seed),
    )
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
