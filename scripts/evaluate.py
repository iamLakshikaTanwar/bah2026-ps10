#!/usr/bin/env python3
"""Evaluate the IRChroma pipeline with the 6-family metric suite (docs/research/06).

CLI front-end that assembles the :class:`~irchroma.interfaces.MetricSuite` via
:func:`irchroma.metrics.build_metric_suite` and evaluates a pipeline (or precomputed
RGB outputs) over a dataset, then writes a metrics **report to JSON** and a
**markdown table**.

Evaluation protocol (docs/research/06 §7.1; ARCHITECTURE.md §10.3): the test set is a
**geographic / scene / WRS-2-path-row holdout**, NEVER a random tile split (adjacent
tiles are near-duplicates → leakage inflates every metric). For the on-disk GeoTIFF
path (``--data``) this script runs :func:`irchroma.data.geographic_split` and
evaluates the held-out **test** split. The synthetic path (default) generates a fresh,
deterministic, disjoint sample set (no leakage by construction) and runs on **CPU**
with the core (numpy) metric subset — so this script works with zero optional deps.

The 3-condition downstream uplift protocol (raw-IR vs naive-color vs ours;
docs/research/06 §F) is documented and wired as a ``--compare`` **stub**: it explains
the protocol and, given multiple prediction sources, runs the suite per condition so
the report carries side-by-side numbers. The full bootstrap-CI + Wilcoxon significance
machinery lives in the downstream/efficiency metrics and is invoked by the suite.

Examples
--------
Synthetic, CPU, core metrics (always works)::

    python scripts/evaluate.py --synthetic --num 16 --families fidelity color \
        --out results/eval

Real GeoTIFF tiles via a manifest, geographic-holdout test split::

    python scripts/evaluate.py --data data/manifest.json --split-by path_row \
        --families fidelity perceptual color faithfulness --out results/eval

Precomputed outputs (skip the pipeline), compare two conditions::

    python scripts/evaluate.py --synthetic --pred ours=preds/ours --pred naive=preds/naive \
        --compare --out results/eval
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple


# --------------------------------------------------------------------------- #
# Argparse.
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    """Construct the argparse CLI parser."""
    p = argparse.ArgumentParser(
        prog="evaluate.py",
        description=(
            "Evaluate the IRChroma pipeline with the 6-family metric suite over a "
            "geographic-holdout test split (synthetic by default; GeoTIFF via --data)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Data source ------------------------------------------------------- #
    src = p.add_argument_group("data source")
    src.add_argument(
        "--synthetic",
        action="store_true",
        help="Evaluate on procedural synthetic IR/RGB data (CPU, no deps). Default if "
        "neither --data nor --pred is given.",
    )
    src.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to a GeoTIFF tile manifest (JSON list of items with ir_path/rgb_path/"
        "guide_path/semantic_path/path_row) to evaluate via the geographic-holdout split.",
    )
    src.add_argument(
        "--num",
        type=int,
        default=16,
        help="Number of synthetic samples to evaluate (synthetic path only).",
    )
    src.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for pipeline inference.",
    )

    # ---- Split ------------------------------------------------------------- #
    spl = p.add_argument_group("geographic-holdout split (GeoTIFF path)")
    spl.add_argument(
        "--split-by",
        type=str,
        default="path_row",
        help="Geographic grouping key for the holdout split (e.g. path_row, region, scene).",
    )
    spl.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test", "all"],
        help="Which split to evaluate (the held-out 'test' set by default).",
    )
    spl.add_argument(
        "--ratios",
        type=float,
        nargs=3,
        metavar=("TRAIN", "VAL", "TEST"),
        default=(0.7, 0.15, 0.15),
        help="Train/val/test fractions for the geographic split.",
    )

    # ---- Pipeline / checkpoint -------------------------------------------- #
    mdl = p.add_argument_group("pipeline / model")
    mdl.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a trained pipeline checkpoint. If omitted, a freshly-built "
        "(untrained) pipeline is used for a smoke run.",
    )
    mdl.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Inference device (cpu/cuda).",
    )
    mdl.add_argument(
        "--no-pipeline",
        action="store_true",
        help="Do not build/run a pipeline; evaluate only --pred precomputed outputs.",
    )

    # ---- Precomputed predictions (skip pipeline) -------------------------- #
    pre = p.add_argument_group("precomputed predictions")
    pre.add_argument(
        "--pred",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Precomputed predictions as NAME=PATH (a .npy/.npz/.pt of stacked RGB "
        "[N,3,H,W], or a directory of per-tile images). Repeatable for multiple "
        "conditions (used with --compare).",
    )

    # ---- Metric families --------------------------------------------------- #
    met = p.add_argument_group("metrics")
    met.add_argument(
        "--families",
        type=str,
        nargs="+",
        default=None,
        help="Metric families to run (subset of: fidelity perceptual no_reference color "
        "faithfulness downstream efficiency). Defaults to the EvalConfig families; on "
        "the synthetic/CPU path this auto-narrows to the always-available core "
        "(fidelity, color) unless you pass families explicitly.",
    )
    met.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML config to source EvalConfig (families, border_shave, etc.).",
    )

    # ---- Downstream uplift protocol --------------------------------------- #
    cmp = p.add_argument_group("downstream uplift (3-condition protocol)")
    cmp.add_argument(
        "--compare",
        action="store_true",
        help="Run the 3-condition downstream protocol stub (raw-IR vs naive-color vs "
        "ours). Documents the protocol and, when multiple --pred sources are given, "
        "evaluates the suite per condition for side-by-side numbers.",
    )

    # ---- Output ------------------------------------------------------------ #
    out = p.add_argument_group("output")
    out.add_argument(
        "--out",
        type=str,
        default="results/eval",
        help="Output path stem; writes <out>.json and <out>.md.",
    )
    out.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="RNG seed (synthetic data + split shuffles).",
    )
    out.add_argument(
        "--strict",
        action="store_true",
        help="Fail on the first metric error instead of recording NaN.",
    )
    return p


# --------------------------------------------------------------------------- #
# Helpers.
# --------------------------------------------------------------------------- #
def _eprint(*args: Any) -> None:
    """Print to stderr (status / guidance messages)."""
    print(*args, file=sys.stderr)


def _load_config(path: Optional[str]) -> Any:
    """Load a :class:`irchroma.config.Config` from YAML, or the synthetic-demo default."""
    from irchroma.config import Config, synthetic_demo_config

    if path:
        return Config.from_yaml(path)
    return synthetic_demo_config()


def _resolve_families(
    requested: Optional[List[str]],
    cfg: Any,
    *,
    synthetic: bool,
    device: str,
) -> List[str]:
    """Decide which metric families to run.

    If the user passed ``--families`` use them verbatim. Otherwise read the
    ``EvalConfig.families``; on the synthetic/CPU smoke path auto-narrow to the
    always-available **core** (``fidelity``, ``color``) so the script runs end-to-end
    with zero optional dependencies (docs/research/06 dependency posture).
    """
    if requested:
        return list(requested)
    eval_cfg = getattr(cfg, "eval", cfg)
    families = list(getattr(eval_cfg, "families", ["fidelity", "color"]))
    if synthetic or device == "cpu":
        core = [f for f in families if f in ("fidelity", "color")]
        return core or ["fidelity", "color"]
    return families


def _parse_pred_args(pred_args: List[str]) -> "Dict[str, str]":
    """Parse repeated ``NAME=PATH`` ``--pred`` args into an ordered dict."""
    out: Dict[str, str] = {}
    for entry in pred_args or []:
        if "=" not in entry:
            raise ValueError(
                f"--pred expects NAME=PATH (got {entry!r}); e.g. --pred ours=preds/ours"
            )
        name, path = entry.split("=", 1)
        out[name.strip()] = path.strip()
    return out


def _load_pred_stack(path: str) -> Any:
    """Load a precomputed prediction stack ``[N, 3, H, W]`` (torch tensor).

    Supports ``.pt``/``.pth`` (torch.load), ``.npy``/``.npz`` (numpy), or a directory
    of per-tile images (loaded via Pillow into a stacked tensor). Requires torch.
    """
    import torch  # local import (precomputed path needs torch)

    if os.path.isdir(path):
        return _load_pred_dir(path)
    ext = os.path.splitext(path)[1].lower()
    if ext in (".pt", ".pth"):
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict):
            # take the first tensor value if it's a dict
            obj = next(v for v in obj.values() if torch.is_tensor(v))
        return obj.float()
    if ext in (".npy", ".npz"):
        import numpy as np

        arr = np.load(path)
        if hasattr(arr, "files"):  # npz
            arr = arr[arr.files[0]]
        return torch.as_tensor(arr).float()
    raise ValueError(f"Unsupported prediction file type: {path!r}")


def _load_pred_dir(path: str) -> Any:
    """Load a directory of per-tile images into a stacked ``[N, 3, H, W]`` tensor."""
    import torch

    try:
        import numpy as np
        from PIL import Image
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Loading a directory of predicted tiles needs numpy + Pillow."
        ) from exc
    files = sorted(
        f for f in os.listdir(path) if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
    )
    if not files:
        raise ValueError(f"No image files found in prediction dir {path!r}.")
    stack = []
    for f in files:
        im = np.asarray(Image.open(os.path.join(path, f)).convert("RGB"), dtype=np.float32) / 255.0
        stack.append(torch.as_tensor(im).permute(2, 0, 1))  # [3,H,W]
    return torch.stack(stack, dim=0)  # [N,3,H,W]


def _build_engine(checkpoint: Optional[str], cfg: Any, device: str) -> Any:
    """Build an :class:`InferenceEngine`: from a checkpoint, or a fresh untrained pipeline."""
    from irchroma.infer.engine import InferenceEngine

    if checkpoint:
        return InferenceEngine.from_checkpoint(checkpoint, config=cfg, device=device)
    # Fresh build (untrained) for a smoke evaluation; build_pipeline is lazy-imported.
    engine = InferenceEngine(pipeline=None, config=cfg, device=device)
    pipeline = InferenceEngine._build_pipeline(cfg)  # lazy build_pipeline(cfg)
    return engine.attach_pipeline(pipeline)


# --------------------------------------------------------------------------- #
# Dataset assembly (synthetic or GeoTIFF holdout).
# --------------------------------------------------------------------------- #
def _iter_synthetic_batches(cfg: Any, num: int, batch_size: int, seed: int):
    """Yield ``(pred_input_batch)`` synthetic Sample batches for evaluation.

    Generates deterministic, disjoint synthetic samples (no leakage by construction).
    Yields batched :class:`Sample` dicts (``ir``/``rgb``/``guide``/``semantic``/``meta``).
    """
    from irchroma.data.synthetic import make_synthetic_sample

    produced = 0
    chunk_seed = seed
    while produced < num:
        bs = min(batch_size, num - produced)
        sample = make_synthetic_sample(cfg, batch_size=bs, seed=chunk_seed)
        yield sample
        produced += bs
        chunk_seed += 1


def _load_geotiff_test_items(
    manifest: str, split_by: str, ratios: Tuple[float, float, float], which: str, seed: int
) -> List[Any]:
    """Load a GeoTIFF manifest and return the requested geographic-holdout split items."""
    from irchroma.data.dataset import TileItem, geographic_split

    with open(manifest, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if isinstance(raw, dict) and "items" in raw:
        raw = raw["items"]
    items = [TileItem(**it) if isinstance(it, dict) else it for it in raw]
    if which == "all":
        return items
    splits = geographic_split(items, by=split_by, ratios=ratios, seed=seed)
    return splits.get(which, [])


# --------------------------------------------------------------------------- #
# Evaluation core.
# --------------------------------------------------------------------------- #
def _evaluate_pairs(suite: Any, pairs: List[Tuple[Any, Any]], ctx: Dict[str, Any], strict: bool):
    """Run the metric suite over ``(pred, target)`` batch pairs; average per metric.

    Returns ``(mean_results, per_batch_results)`` where ``mean_results`` maps each
    metric name to its mean over batches (NaN-robust). ``MetricSuite.evaluate`` already
    records NaN for any metric that raises (unless ``strict``).
    """
    import math

    per_batch: List[Dict[str, float]] = []
    for pred, target in pairs:
        res = suite.evaluate(pred, target, ctx, strict=strict)
        per_batch.append(res)

    # Aggregate (mean over batches, ignoring NaNs per metric).
    agg: Dict[str, float] = {}
    if per_batch:
        for name in per_batch[0].keys():
            vals = [b[name] for b in per_batch if name in b and not math.isnan(b[name])]
            agg[name] = (sum(vals) / len(vals)) if vals else float("nan")
    return agg, per_batch


def _run_pipeline_eval(
    engine: Any,
    sample_batches: List[Any],
    suite: Any,
    strict: bool,
) -> Tuple[Dict[str, float], int]:
    """Run the pipeline over Sample batches and evaluate predicted vs GT RGB."""
    import torch

    pairs: List[Tuple[Any, Any]] = []
    n_tiles = 0
    for sample in sample_batches:
        ir = sample["ir"]
        guide = sample.get("guide")
        semantic = sample.get("semantic")
        target = sample.get("rgb")
        with torch.no_grad():
            pred = engine.predict(ir, guide=guide, semantic=semantic)
        n_tiles += int(pred.shape[0]) if torch.is_tensor(pred) else 0
        pairs.append((pred, target))
    ctx: Dict[str, Any] = {"device": str(getattr(engine, "device", "cpu"))}
    agg, _ = _evaluate_pairs(suite, pairs, ctx, strict)
    return agg, n_tiles


def _markdown_table(report: Dict[str, Any]) -> str:
    """Render the metrics report as a GitHub-flavored markdown document."""
    lines: List[str] = []
    lines.append("# IRChroma Evaluation Report")
    lines.append("")
    meta = report.get("meta", {})
    lines.append("## Run")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|---|---|")
    for k in ("data_source", "split", "split_by", "num_tiles", "families", "device", "checkpoint", "seed"):
        if k in meta:
            v = meta[k]
            if isinstance(v, (list, tuple)):
                v = ", ".join(str(x) for x in v)
            lines.append(f"| {k} | {v} |")
    lines.append("")

    conditions = report.get("conditions")
    if conditions:
        # Side-by-side table: rows = metrics, columns = conditions.
        names = sorted({m for cond in conditions.values() for m in cond.keys()})
        cols = list(conditions.keys())
        lines.append("## Metrics by condition (downstream uplift protocol)")
        lines.append("")
        lines.append("| Metric | " + " | ".join(cols) + " |")
        lines.append("|---" * (len(cols) + 1) + "|")
        for m in names:
            row = [m]
            for c in cols:
                val = conditions[c].get(m, float("nan"))
                row.append(f"{val:.4f}" if isinstance(val, (int, float)) else str(val))
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    metrics = report.get("metrics")
    if metrics:
        lines.append("## Metrics")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        for name in sorted(metrics.keys()):
            val = metrics[name]
            cell = f"{val:.4f}" if isinstance(val, (int, float)) else str(val)
            lines.append(f"| {name} | {cell} |")
        lines.append("")

    note = report.get("downstream_protocol_note")
    if note:
        lines.append("## Downstream uplift protocol (docs/research/06 §F)")
        lines.append("")
        lines.append(note)
        lines.append("")
    return "\n".join(lines)


_DOWNSTREAM_NOTE = (
    "The **3-condition downstream protocol** measures whether colorization *helps a "
    "downstream task* rather than just looking nicer (the honest, anti-hallucination "
    "yardstick). Run an off-the-shelf detector/segmenter (e.g. YOLOv11-OBB / SegFormer) "
    "on three inputs and compare task scores (mAP / mIoU):\n\n"
    "1. **raw-IR** — the (upsampled) infrared input, no color;\n"
    "2. **naive-color** — a baseline colorization (e.g. a fixed colormap / bicubic+palette);\n"
    "3. **ours** — the IRChroma super-resolved, semantically-conditioned colorization.\n\n"
    "Significance is established with a **paired Wilcoxon signed-rank test** over tiles and "
    "**bootstrap confidence intervals** (`EvalConfig.bootstrap_resamples`, "
    "`EvalConfig.significance_test`). 'Ours' must *significantly* beat both baselines on the "
    "downstream metric to claim uplift; a perceptual win (better FID) that does NOT improve "
    "the downstream task — or that regresses seg-consistency — is flagged as probable "
    "hallucination, not success (Blau-Michaeli Pareto view). This `--compare` flag wires the "
    "side-by-side suite evaluation; full detector wiring is provided by the downstream "
    "metric family."
)


# --------------------------------------------------------------------------- #
# Main.
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    """Entry point: assemble the suite, evaluate, write JSON + markdown report."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Default to synthetic if no explicit source was chosen.
    synthetic = args.synthetic or (args.data is None and not args.pred)

    try:
        cfg = _load_config(args.config)
    except Exception as exc:
        _eprint(f"[evaluate] Failed to load config: {exc}")
        return 2

    # Keep synthetic dims small/CPU-friendly when on the synthetic path.
    if synthetic and args.config is None:
        cfg.data.use_synthetic = True
        cfg.train.device = args.device

    families = _resolve_families(args.families, cfg, synthetic=synthetic, device=args.device)
    _eprint(f"[evaluate] metric families: {families}")

    # Build the metric suite (NaN-robust; skips unavailable optional metrics).
    try:
        from irchroma.metrics import build_metric_suite

        eval_cfg = getattr(cfg, "eval", cfg)
        # Reflect the chosen families into the EvalConfig the suite reads.
        try:
            eval_cfg.families = list(families)
        except Exception:  # pragma: no cover - frozen-ish config edge
            pass
        suite = build_metric_suite(eval_cfg, skip_unavailable=True)
    except Exception as exc:
        _eprint(f"[evaluate] Failed to build metric suite: {exc}")
        return 2
    _eprint(f"[evaluate] metrics registered: {suite.names}")

    report: Dict[str, Any] = {
        "meta": {
            "data_source": "synthetic" if synthetic else ("geotiff" if args.data else "precomputed"),
            "families": families,
            "device": args.device,
            "checkpoint": args.checkpoint,
            "seed": args.seed,
            "split": args.split if args.data else ("n/a" if synthetic else "precomputed"),
            "split_by": args.split_by if args.data else None,
        },
        "metric_directions": suite.directions(),
    }

    pred_sources = _parse_pred_args(args.pred)

    # --------------------------------------------------------------------- #
    # Path A: precomputed predictions (one or more conditions).
    # --------------------------------------------------------------------- #
    if pred_sources:
        _eprint(f"[evaluate] evaluating precomputed predictions: {list(pred_sources)}")
        # Targets: synthetic GT (regenerated deterministically) or GeoTIFF GT.
        try:
            target_stack = _gather_targets_for_precomputed(args, cfg, synthetic)
        except Exception as exc:
            _eprint(f"[evaluate] Could not assemble targets for precomputed eval: {exc}")
            return 2
        conditions: Dict[str, Dict[str, float]] = {}
        for name, path in pred_sources.items():
            try:
                preds = _load_pred_stack(path)
            except Exception as exc:
                _eprint(f"[evaluate]  - '{name}' load failed ({exc}); skipping.")
                continue
            ctx = {"device": args.device}
            agg, _ = _evaluate_pairs(suite, [(preds, target_stack)], ctx, args.strict)
            conditions[name] = agg
        report["conditions"] = conditions
        if len(conditions) == 1:
            # Single source -> also expose as flat metrics for convenience.
            report["metrics"] = next(iter(conditions.values()))
        if args.compare:
            report["downstream_protocol_note"] = _DOWNSTREAM_NOTE

    # --------------------------------------------------------------------- #
    # Path B: run the pipeline over a dataset.
    # --------------------------------------------------------------------- #
    elif not args.no_pipeline:
        try:
            engine = _build_engine(args.checkpoint, cfg, args.device)
        except Exception as exc:
            _eprint(
                f"[evaluate] Could not build the inference engine/pipeline: {exc}\n"
                "            (Is torch installed and the pipeline module available?)"
            )
            return 2

        # Assemble evaluation batches.
        if synthetic:
            batches = list(_iter_synthetic_batches(cfg, args.num, args.batch_size, args.seed))
            report["meta"]["num_tiles"] = args.num
        else:
            try:
                items = _load_geotiff_test_items(
                    args.data, args.split_by, tuple(args.ratios), args.split, args.seed
                )
            except Exception as exc:
                _eprint(f"[evaluate] Failed to load GeoTIFF manifest/split: {exc}")
                return 2
            if not items:
                _eprint(f"[evaluate] No items in the '{args.split}' split; nothing to evaluate.")
                return 1
            batches = list(_iter_geotiff_batches(items, cfg, args.batch_size))
            report["meta"]["num_tiles"] = len(items)

        try:
            metrics, n_tiles = _run_pipeline_eval(engine, batches, suite, args.strict)
        except Exception as exc:
            _eprint(f"[evaluate] Pipeline evaluation failed: {exc}")
            return 2
        report["metrics"] = metrics
        report["meta"]["num_tiles"] = report["meta"].get("num_tiles", n_tiles)
        if args.compare:
            report["downstream_protocol_note"] = _DOWNSTREAM_NOTE
    else:
        _eprint("[evaluate] --no-pipeline set but no --pred given; nothing to do.")
        return 1

    # --------------------------------------------------------------------- #
    # Write outputs.
    # --------------------------------------------------------------------- #
    out_stem = args.out
    out_dir = os.path.dirname(os.path.abspath(out_stem)) or "."
    os.makedirs(out_dir, exist_ok=True)
    json_path = out_stem + ".json"
    md_path = out_stem + ".md"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(_markdown_table(report))

    _eprint(f"[evaluate] wrote {json_path} and {md_path}")
    # Echo a compact summary to stdout.
    summary = report.get("metrics") or report.get("conditions") or {}
    print(json.dumps(summary, indent=2))
    return 0


def _iter_geotiff_batches(items: List[Any], cfg: Any, batch_size: int):
    """Yield batched :class:`Sample` dicts from GeoTIFF tile items (rasterio path)."""
    from irchroma.data.dataset import GeoTIFFTileDataset
    from irchroma.data.synthetic import _collate_samples  # reuse the collate logic

    ds = GeoTIFFTileDataset(items, cfg=getattr(cfg, "data", None), to_tensor=True)
    buf: List[Any] = []
    for i in range(len(ds)):
        buf.append(ds[i])
        if len(buf) >= batch_size:
            yield _collate_samples(buf)
            buf = []
    if buf:
        yield _collate_samples(buf)


def _gather_targets_for_precomputed(args: Any, cfg: Any, synthetic: bool) -> Any:
    """Assemble the ground-truth RGB stack matching precomputed predictions."""
    import torch

    if synthetic:
        from irchroma.data.synthetic import make_synthetic_sample

        sample = make_synthetic_sample(cfg, batch_size=args.num, seed=args.seed)
        return sample["rgb"]
    # GeoTIFF: stack the test split's RGB targets.
    items = _load_geotiff_test_items(
        args.data, args.split_by, tuple(args.ratios), args.split, args.seed
    )
    from irchroma.data.dataset import GeoTIFFTileDataset

    ds = GeoTIFFTileDataset(items, cfg=getattr(cfg, "data", None), to_tensor=True)
    rgbs = [ds[i]["rgb"] for i in range(len(ds)) if ds[i].get("rgb") is not None]
    if not rgbs:
        raise ValueError("No RGB ground-truth tiles found for the precomputed comparison.")
    return torch.stack(rgbs, dim=0)


if __name__ == "__main__":
    raise SystemExit(main())
