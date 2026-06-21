"""irchroma.train.cli — the ``irchroma-train`` console entrypoint (+ ``make demo``).

This is the stable entrypoint wired in ``pyproject.toml`` (``irchroma-train``) and
invoked by the Makefile (``python -m irchroma.train.cli --demo|--config ...``). It is a
thin compatibility shim over the unified :mod:`irchroma.cli`:

  * ``--demo``  → run the end-to-end synthetic smoke test (``irchroma demo``).
  * otherwise   → train the pipeline (``irchroma train``) on synthetic (default) or, with
                  ``--data``, real data.

Flags mirror :mod:`irchroma.cli` (``--config``, ``--device``, ``--steps``, ``--epochs``,
``--out``) so existing Makefile / docs invocations keep working unchanged.
"""

from __future__ import annotations

import argparse
from typing import List, Optional


def main(argv: Optional[List[str]] = None) -> int:
    """Entrypoint: parse the train/demo flags and dispatch into :mod:`irchroma.cli`."""
    parser = argparse.ArgumentParser(
        prog="irchroma-train",
        description="Train the irchroma pipeline (or run the synthetic demo with --demo).",
    )
    parser.add_argument("--demo", action="store_true", help="run the end-to-end synthetic demo.")
    parser.add_argument("--config", default=None, help="YAML config (else synthetic/default).")
    parser.add_argument("--synthetic", action="store_true",
                        help="train on synthetic data (default when no --data).")
    parser.add_argument("--data", default=None, help="real-data root (when not synthetic).")
    parser.add_argument("--epochs", type=int, default=None, help="override config.train.epochs.")
    parser.add_argument("--steps", type=int, default=None, help="hard cap on optimizer steps.")
    parser.add_argument("--device", default=None, help="cpu|cuda|mps.")
    parser.add_argument("--out", default=None, help="output dir / checkpoint path.")
    parser.add_argument("--quiet", action="store_true", help="disable the progress bar.")
    parser.add_argument("--no-plot", action="store_true", help="(demo) skip the matplotlib panel.")
    parser.add_argument("--seed", type=int, default=0, help="(demo) synthetic-sample seed.")
    args = parser.parse_args(argv)

    from ..cli import cmd_demo, cmd_train

    if args.demo:
        demo_ns = argparse.Namespace(
            config=args.config,
            steps=3 if args.steps is None else int(args.steps),
            out=args.out or "outputs",
            no_plot=args.no_plot,
            device=args.device,
            seed=args.seed,
        )
        return int(cmd_demo(demo_ns) or 0)

    # Default to synthetic training unless a real --data root is given.
    synthetic = args.synthetic or (args.data is None)
    train_ns = argparse.Namespace(
        config=args.config,
        synthetic=synthetic,
        data=args.data,
        epochs=args.epochs,
        steps=args.steps,
        device=args.device,
        out=args.out,
        quiet=args.quiet,
    )
    return int(cmd_train(train_ns) or 0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
