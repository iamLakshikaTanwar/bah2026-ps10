"""irchroma.cli — the unified ``irchroma`` command-line entrypoint.

Subcommands (``irchroma <cmd> ...`` / ``python -m irchroma.cli <cmd> ...``):

  * ``demo``  — run the end-to-end synthetic smoke test (build pipeline, forward a
                synthetic batch, compute loss + a few metrics, run a few optimizer
                steps, optionally save ``outputs/demo.png``). CPU-only, no network.
  * ``train`` — train the pipeline via :class:`~irchroma.train.Trainer` on synthetic
                (default) or real data; ``--steps`` / ``--device`` / ``--out``.
  * ``eval``  — run the 6-family :func:`~irchroma.metrics.build_metric_suite` over a few
                synthetic batches (a thin standalone path; the full results-dir evaluator
                lives in ``scripts.evaluate`` if present, imported late).
  * ``serve`` — launch the FastAPI tile server (``irchroma.serve.api.create_app``);
                friendly error if FastAPI / the serve extra is missing.
  * ``info``  — print the resolved config + per-model parameter counts.

Heavy work (torch, the models) is imported **lazily inside each handler** so
``python -m irchroma.cli --help`` and ``irchroma info`` stay fast and never hard-fail on
a torch-less box. ``main()`` is the console-script entrypoint.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, List, Optional

from .config import Config, synthetic_demo_config


# =========================================================================== #
# Config loading.
# =========================================================================== #
def load_config(path: Optional[str] = None, demo: bool = False) -> Config:
    """Resolve a :class:`Config` from ``--config`` / ``--demo`` / defaults.

    Precedence: an explicit YAML ``path`` wins; else the tiny CPU-friendly
    :func:`~irchroma.config.synthetic_demo_config` when ``demo``; else the canonical
    :class:`~irchroma.config.Config` defaults.
    """
    if path:
        return Config.from_yaml(path)
    if demo:
        return synthetic_demo_config()
    return Config()


# =========================================================================== #
# Subcommand handlers.
# =========================================================================== #
def cmd_demo(args: argparse.Namespace) -> int:
    """Run the end-to-end synthetic smoke test (delegates to ``scripts/demo.py``).

    The demo is fundamentally a *synthetic* smoke test, so it starts from the tiny
    CPU-friendly :func:`~irchroma.config.synthetic_demo_config`. A ``--config`` is honored
    only when it is itself a synthetic config (``data.use_synthetic`` truthy); a full
    training config (e.g. the Makefile's ``configs/default.yaml``) is intentionally
    downgraded to the synthetic demo so ``make demo`` stays fast and self-contained.
    """
    cfg = synthetic_demo_config()
    if args.config:
        loaded = Config.from_yaml(args.config)
        if bool(getattr(loaded.data, "use_synthetic", False)):
            cfg = loaded  # user explicitly opted into a synthetic config for the demo
    # Allow CLI overrides of the demo dims for a quicker / larger smoke test.
    if getattr(args, "device", None):
        cfg.train.device = args.device
        cfg.infer.device = args.device
    run_demo = _load_run_demo()
    if run_demo is None:  # pragma: no cover - scripts/ unavailable in a bare wheel
        print("error: could not locate the demo runner (scripts/demo.py).", file=sys.stderr)
        return 1
    result = run_demo(
        cfg,
        steps=int(args.steps),
        out_dir=args.out,
        make_plot=not args.no_plot,
        seed=int(args.seed),
    )
    return 0 if result.get("ok") else 1


def cmd_train(args: argparse.Namespace) -> int:
    """Train the pipeline via :class:`~irchroma.train.Trainer`."""
    cfg = load_config(args.config, demo=args.synthetic)
    if args.device:
        cfg.train.device = args.device
    if args.synthetic:
        cfg.data.use_synthetic = True

    from .train import Trainer

    if Trainer is None:  # pragma: no cover - torch-less
        print("error: torch is required to train (irchroma.train.Trainer unavailable).", file=sys.stderr)
        return 1

    try:
        trainer = Trainer(cfg, device=args.device, max_steps=args.steps)
    except RuntimeError as exc:  # pragma: no cover - torch-less box
        print(f"error: {exc}", file=sys.stderr)
        return 1

    loader = None
    if not args.synthetic and args.data:
        loader = _build_real_loader(cfg, args.data)

    summary = trainer.train(loader=loader, epochs=args.epochs, progress=not args.quiet)
    final = summary.get("final", {})
    print(
        f"trained {summary.get('steps', 0)} steps over {summary.get('epochs', 0)} epoch(s); "
        f"final loss_total={final.get('loss_total', float('nan')):.4f} "
        f"loss_d={final.get('loss_d', float('nan')):.4f}"
    )
    if args.out:
        path = trainer.save_checkpoint(args.out)
        print(f"saved checkpoint -> {path}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    """Run the metric suite (thin synthetic path; defers to scripts.evaluate if present)."""
    # Prefer a richer standalone evaluator if a Wave-B sibling provides one.
    try:  # pragma: no cover - optional richer evaluator
        from scripts import evaluate as _evaluate  # type: ignore

        if hasattr(_evaluate, "main"):
            return int(_evaluate.main(args) or 0)
    except Exception:
        pass

    cfg = load_config(args.config, demo=args.synthetic or not args.config)
    if args.device:
        cfg.train.device = args.device

    from .train import Trainer

    if Trainer is None:  # pragma: no cover
        print("error: torch is required to evaluate.", file=sys.stderr)
        return 1
    try:
        trainer = Trainer(cfg, device=args.device)
    except RuntimeError as exc:  # pragma: no cover - torch-less box
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
    metrics = trainer.validate(max_batches=int(args.batches))
    if not metrics:
        print("no metrics produced (all optional backends unavailable on this box).")
    for name in sorted(metrics):
        print(f"  {name:24s} {metrics[name]:.4f}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Launch the FastAPI tile server (late import; friendly error if missing)."""
    cfg = load_config(args.config)
    try:
        from .serve.api import create_app  # type: ignore
    except Exception as exc:
        print(
            "error: could not import irchroma.serve.api.create_app "
            f"({exc}). Install the serve extra: pip install -e '.[serve]'.",
            file=sys.stderr,
        )
        return 1
    try:
        import uvicorn  # type: ignore
    except Exception:
        print("error: uvicorn is required to serve (pip install -e '.[serve]').", file=sys.stderr)
        return 1
    app = create_app(cfg)
    uvicorn.run(app, host=args.host, port=int(args.port))
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    """Print the resolved config + per-model parameter counts."""
    cfg = load_config(args.config, demo=args.demo)
    print(f"config: name={cfg.name} scale={cfg.model.scale} "
          f"num_classes={cfg.model.num_classes} device={cfg.train.device}")
    print(f"  sr.backbone={cfg.model.sr.backbone} width={cfg.model.sr.width} "
          f"colorization.generator={cfg.model.colorization.generator} ngf={cfg.model.colorization.ngf}")

    # Parameter counts require torch + the models; guard so `info` still runs torch-less.
    try:
        from .interfaces import TORCH_AVAILABLE

        if not TORCH_AVAILABLE:
            print("  (torch unavailable — skipping parameter counts)")
            return 0
        from .models import build_pipeline

        pipeline = build_pipeline(cfg)
        parts = {
            "sr_model": getattr(pipeline, "sr_model", None),
            "colorizer": getattr(pipeline, "colorizer", None),
            "adaint_lut": getattr(pipeline, "adaint_lut", None),
            "segmenter": getattr(pipeline, "segmenter", None),
            "discriminator": getattr(pipeline, "discriminator", None),
        }
        total = 0
        for label, mod in parts.items():
            if mod is None:
                continue
            n = sum(p.numel() for p in mod.parameters())
            total += n
            print(f"  params[{label:13s}] = {n:,}")
        # The pipeline total (deduplicated by the framework's own param iterator).
        pipe_total = sum(p.numel() for p in pipeline.parameters())
        print(f"  params[pipeline (forward, dedup)] = {pipe_total:,}")
        print(f"  params[all submodules summed]     = {total:,}")
    except Exception as exc:  # pragma: no cover - keep info resilient
        print(f"  (could not build pipeline for param counts: {exc})")
    return 0


# =========================================================================== #
# Helpers.
# =========================================================================== #
def _load_run_demo() -> Optional[Any]:
    """Locate and import ``run_demo`` from ``scripts/demo.py`` (the authoritative demo).

    Tries, in order: a ``scripts.demo`` package import (if ``scripts`` is on the path);
    then a file-path import resolved relative to this package's repo root
    (``<repo>/scripts/demo.py`` in the ``src/`` layout). Returns the ``run_demo`` callable
    or ``None`` if it cannot be located.
    """
    # 1) Already-importable ``scripts.demo``.
    try:  # pragma: no cover - depends on sys.path layout
        from scripts.demo import run_demo  # type: ignore

        return run_demo
    except Exception:
        pass
    # 2) File-path import relative to the repo root (dev / src layout).
    import importlib.util
    import os

    here = os.path.dirname(os.path.abspath(__file__))          # .../src/irchroma
    repo_root = os.path.dirname(os.path.dirname(here))          # .../<repo>
    demo_path = os.path.join(repo_root, "scripts", "demo.py")
    if not os.path.isfile(demo_path):
        return None
    try:
        spec = importlib.util.spec_from_file_location("_irchroma_demo", demo_path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, "run_demo", None)
    except Exception:  # pragma: no cover
        return None


def _build_real_loader(cfg: Config, data_path: str) -> Optional[Any]:
    """Best-effort build a real-data loader from ``irchroma.data`` (late import)."""
    try:  # pragma: no cover - real-data path not exercised in the synthetic smoke test
        from .data import build_loader  # type: ignore

        return build_loader(cfg, root=data_path)
    except Exception as exc:
        print(f"warning: could not build real-data loader ({exc}); falling back to synthetic.",
              file=sys.stderr)
        return None


# =========================================================================== #
# Argument parser.
# =========================================================================== #
def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level ``irchroma`` argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="irchroma",
        description="IR satellite super-resolution + semantically-faithful IR→RGB colorization.",
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    # ---- demo ---- #
    p_demo = sub.add_parser("demo", help="end-to-end synthetic smoke test (CPU, no network).")
    p_demo.add_argument("--config", default=None, help="optional YAML config (else synthetic demo).")
    p_demo.add_argument("--steps", type=int, default=3, help="optimizer steps to prove training.")
    p_demo.add_argument("--out", default="outputs", help="output dir for demo.png.")
    p_demo.add_argument("--no-plot", action="store_true", help="skip the matplotlib panel.")
    p_demo.add_argument("--device", default=None, help="cpu|cuda (default: config/cpu).")
    p_demo.add_argument("--seed", type=int, default=0, help="synthetic-sample seed.")
    p_demo.set_defaults(func=cmd_demo)

    # ---- train ---- #
    p_train = sub.add_parser("train", help="train the pipeline (synthetic by default).")
    p_train.add_argument("--config", default=None, help="YAML config (else defaults).")
    p_train.add_argument("--synthetic", action="store_true", help="train on synthetic data.")
    p_train.add_argument("--data", default=None, help="real-data root (when not --synthetic).")
    p_train.add_argument("--epochs", type=int, default=None, help="override config.train.epochs.")
    p_train.add_argument("--steps", type=int, default=None, help="hard cap on total optimizer steps.")
    p_train.add_argument("--device", default=None, help="cpu|cuda|mps.")
    p_train.add_argument("--out", default=None, help="checkpoint path to save.")
    p_train.add_argument("--quiet", action="store_true", help="disable the progress bar.")
    p_train.set_defaults(func=cmd_train)

    # ---- eval ---- #
    p_eval = sub.add_parser("eval", help="run the 6-family metric suite.")
    p_eval.add_argument("--config", default=None, help="YAML config (else synthetic demo).")
    p_eval.add_argument("--synthetic", action="store_true", help="evaluate on synthetic data.")
    p_eval.add_argument("--checkpoint", default=None, help="checkpoint to load before eval.")
    p_eval.add_argument("--batches", type=int, default=4, help="number of batches to average.")
    p_eval.add_argument("--device", default=None, help="cpu|cuda|mps.")
    p_eval.set_defaults(func=cmd_eval)

    # ---- serve ---- #
    p_serve = sub.add_parser("serve", help="launch the FastAPI/TiTiler tile server.")
    p_serve.add_argument("--config", default=None, help="YAML config (else defaults).")
    p_serve.add_argument("--host", default="0.0.0.0", help="bind host.")
    p_serve.add_argument("--port", type=int, default=8000, help="bind port.")
    p_serve.set_defaults(func=cmd_serve)

    # ---- info ---- #
    p_info = sub.add_parser("info", help="print config + model parameter counts.")
    p_info.add_argument("--config", default=None, help="YAML config (else defaults).")
    p_info.add_argument("--demo", action="store_true", help="use the synthetic demo config.")
    p_info.set_defaults(func=cmd_info)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Console-script entrypoint. Parse args and dispatch to the chosen subcommand."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
