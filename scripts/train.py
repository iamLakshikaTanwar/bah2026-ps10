#!/usr/bin/env python
"""scripts/train.py — thin training wrapper for irchroma.

Loads a :class:`~irchroma.config.Config` (default, ``--config`` YAML, or the synthetic
demo config), builds a :class:`~irchroma.train.Trainer`, and trains on synthetic data
(default) or a real-data root (``--data``). Mirrors ``irchroma train`` /
``python -m irchroma.train.cli``; provided as a loose script for source checkouts.

Examples::

    python scripts/train.py --synthetic --steps 20 --device cpu
    python scripts/train.py --config configs/landsat.yaml --out checkpoints/run.pt
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional


def _ensure_irchroma_importable() -> None:
    """Add the repo's ``src/`` to sys.path when run as a loose script."""
    try:
        import irchroma  # noqa: F401

        return
    except Exception:
        here = os.path.dirname(os.path.abspath(__file__))
        src = os.path.join(os.path.dirname(here), "src")
        if os.path.isdir(src) and src not in sys.path:
            sys.path.insert(0, src)


_ensure_irchroma_importable()


def main(argv: Optional[list] = None) -> int:
    """Parse args, build the Trainer, and run training."""
    parser = argparse.ArgumentParser(description="Train the irchroma pipeline.")
    parser.add_argument("--config", default=None, help="YAML config (else synthetic/default).")
    parser.add_argument("--synthetic", action="store_true",
                        help="use synthetic data (default when no --data).")
    parser.add_argument("--data", default=None, help="real-data root (when not synthetic).")
    parser.add_argument("--epochs", type=int, default=None, help="override config.train.epochs.")
    parser.add_argument("--steps", type=int, default=None, help="hard cap on optimizer steps.")
    parser.add_argument("--device", default=None, help="cpu|cuda|mps.")
    parser.add_argument("--out", default=None, help="checkpoint path to save.")
    parser.add_argument("--quiet", action="store_true", help="disable the progress bar.")
    args = parser.parse_args(argv)

    try:
        import torch  # noqa: F401
    except Exception as exc:  # pragma: no cover - integration installs torch
        print(f"FATAL: torch is required to train ({exc}).", file=sys.stderr)
        return 1

    from irchroma.config import Config, synthetic_demo_config

    synthetic = args.synthetic or (args.data is None)
    if args.config:
        cfg = Config.from_yaml(args.config)
    elif synthetic:
        cfg = synthetic_demo_config()
    else:
        cfg = Config()
    if args.device:
        cfg.train.device = args.device
    if synthetic:
        cfg.data.use_synthetic = True

    from irchroma.train import Trainer

    if Trainer is None:  # pragma: no cover
        print("error: irchroma.train.Trainer unavailable (torch import failed).", file=sys.stderr)
        return 1

    trainer = Trainer(cfg, device=args.device, max_steps=args.steps)

    loader = None
    if not synthetic and args.data:
        try:
            from irchroma.data import build_loader  # type: ignore

            loader = build_loader(cfg, root=args.data)
        except Exception as exc:
            print(f"warning: real-data loader unavailable ({exc}); using synthetic.",
                  file=sys.stderr)
            loader = None

    summary = trainer.train(loader=loader, epochs=args.epochs, progress=not args.quiet)
    final = summary.get("final", {})
    print(
        f"trained {summary.get('steps', 0)} steps over {summary.get('epochs', 0)} epoch(s); "
        f"final loss_total={final.get('loss_total', float('nan')):.4f}"
    )
    if args.out:
        path = trainer.save_checkpoint(args.out)
        print(f"saved checkpoint -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
