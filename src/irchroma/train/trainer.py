"""irchroma.train.trainer — the two-player (G + D) training loop.

Owner: Builder-7. Drives the end-to-end :class:`~irchroma.models.IRChromaPipeline`
through a real adversarial training step:

  * **D update** — score the (conditioned) real and *detached* fake RGB with the
    multi-scale PatchGAN, minimize the discriminator objective (hinge / LSGAN /
    vanilla), clip, step the D optimizer.
  * **G update** — re-score the fake (gradients flow into G), assemble the shared
    ``ctx`` (discriminator + precomputed fake/real scores + frozen segmenter), evaluate
    the fidelity-dominant :class:`~irchroma.interfaces.CompositeLoss`, clip, step the G
    optimizer.

Design notes (ARCHITECTURE §6, §8; ``docs/research/03`` §10 two-stage recipe):

  * **Conditioning** — the Pix2PixHD conditional PatchGAN sees the *condition* ⊕ the
    *image*: ``cat([sr, rgb], dim=1)`` where ``sr`` is the super-resolved IR the
    colorizer consumed (``C_ir + 3`` channels; matches ``MultiScaleDiscriminator``'s
    ``in_channels`` built by :func:`~irchroma.models.build_pipeline`).
  * **SR target** — the composite loss's SR terms compare ``output['sr']`` (HR) against
    ``target['ir']``. Synthetic data only carries the *LR* IR, so this loop supplies a
    high-res IR reference (bilinear upsample of the LR IR to the SR grid) under
    ``target['ir']`` for the loss only; this makes ``sr_charbonnier`` / ``sr_gradient``
    well-defined and supervises the SR *residual* over a sane non-hallucinated base.
  * **Two-stage curriculum** (``two_stage`` / ``TrainConfig.sr_pretrain_epochs``) — an
    optional warm-up phase trains Stage-1 SR only (SR loss terms, no GAN), then the full
    composite loss + adversarial game switches on.

Everything runs on **CPU** for the tiny synthetic config (no AMP, ``device='cpu'``);
AMP/fp16 + grad-scaler are used only when the device is CUDA and the config asks.

``torch`` is assumed present at runtime; this file must still ``py_compile`` without it.
"""

from __future__ import annotations

import os
import random
import time
import warnings
from typing import Any, Dict, List, Optional

from ..config import Config, TrainConfig
from ..interfaces import Sample, TORCH_AVAILABLE

# --------------------------------------------------------------------------- #
# Guarded torch import.
# --------------------------------------------------------------------------- #
if TORCH_AVAILABLE:  # pragma: no cover - exercised only where torch exists
    import torch
    import torch.nn.functional as F
else:  # pragma: no cover - torch-less docs/CI environment
    torch = None  # type: ignore
    F = None  # type: ignore

# Optional progress bar — guarded so importing the trainer never hard-fails.
try:  # pragma: no cover - tqdm is optional
    from tqdm import tqdm as _tqdm
except Exception:  # pragma: no cover
    _tqdm = None  # type: ignore


__all__ = ["Trainer", "seed_everything"]


def _maybe_tqdm(iterable: Any, total: Optional[int], desc: str, enabled: bool) -> Any:
    """Wrap ``iterable`` in a tqdm bar when tqdm is present and ``enabled``."""
    if enabled and _tqdm is not None:
        return _tqdm(iterable, total=total, desc=desc, leave=False)
    return iterable


def seed_everything(seed: int = 0) -> int:
    """Seed Python, NumPy and torch RNGs for reproducible training/inference.

    Seeds ``random``, ``numpy`` (best-effort) and ``torch`` (CPU + CUDA), so behavior is
    reproducible across machines — important for the numerically-stable demo/CI path
    where un-seeded random *weight init* otherwise made the (now-fixed) NaN-gradient bug
    appear only intermittently. Returns the seed used.
    """
    seed = int(seed)
    random.seed(seed)
    try:  # numpy is a hard dep of the project, but stay defensive.
        import numpy as _np

        _np.random.seed(seed % (2**32 - 1))
    except Exception:  # pragma: no cover - numpy always present in practice
        pass
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover - CPU-only CI
            torch.cuda.manual_seed_all(seed)
    return seed


class Trainer:
    """Two-player trainer for the IRChroma pipeline.

    Args:
      config:       a :class:`~irchroma.config.Config` (full) — reads ``train`` for the
                    optimizer/schedule/precision and ``loss`` for the composite weights.
      pipeline:     a prebuilt :class:`~irchroma.models.IRChromaPipeline`; if ``None`` it
                    is built from ``config`` via :func:`~irchroma.models.build_pipeline`.
      device:       override the device string (else ``config.train.device``).
      two_stage:    enable the SR-pretrain → joint curriculum (else
                    ``config.model.two_stage``).
      max_steps:    optional hard cap on total optimizer steps (across epochs); ``None``
                    runs the full ``epochs`` schedule.

    Public API:
      * :meth:`train_step` — one G+D step on a single batch (returns a log dict).
      * :meth:`train`      — full loop over a ``DataLoader`` (synthetic by default).
      * :meth:`validate`   — run the metric suite over a few batches.
      * :meth:`save_checkpoint` / :meth:`load_checkpoint`.
    """

    def __init__(
        self,
        config: Config,
        pipeline: Optional[Any] = None,
        device: Optional[str] = None,
        two_stage: Optional[bool] = None,
        max_steps: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        if not TORCH_AVAILABLE:  # pragma: no cover
            raise RuntimeError("Trainer requires torch. Install torch to train.")

        self.config = config
        self.train_cfg: TrainConfig = config.train
        # Seed RNGs up-front for reproducible training (weight init when we build the
        # pipeline below, optimizer noise, dropout, data shuffling). Falls back to the
        # config's train.seed. This makes the (now-fixed) numerics reproducible too.
        self.seed = seed_everything(
            seed if seed is not None else int(getattr(self.train_cfg, "seed", 0))
        )
        self.device = torch.device(device or self.train_cfg.device or "cpu")

        # ---- Build / adopt the pipeline + discriminator -------------------- #
        if pipeline is None:
            from ..models import build_pipeline

            pipeline = build_pipeline(config)
        self.pipeline = pipeline.to(self.device)
        self.discriminator = getattr(pipeline, "discriminator", None)
        if self.discriminator is not None:
            self.discriminator = self.discriminator.to(self.device)

        # ---- Composite loss (fidelity-dominant) ---------------------------- #
        from ..losses import build_composite_loss

        self.criterion = build_composite_loss(config).to(self.device)
        self.gan_mode = str(getattr(config.loss, "gan_mode", "hinge") or "hinge").lower()

        # ---- Frozen segmenter for the seg-consistency term (shared via ctx) -- #
        self.segmenter = getattr(pipeline, "segmenter", None)

        # ---- Optimizers (separate G and D, Adam) --------------------------- #
        betas = tuple(self.train_cfg.betas)
        g_params = self._generator_parameters()
        self.opt_g = torch.optim.Adam(
            g_params, lr=float(self.train_cfg.lr_generator), betas=betas
        )
        self.opt_d: Optional[Any] = None
        if self.discriminator is not None:
            d_params = [p for p in self.discriminator.parameters() if p.requires_grad]
            if d_params:
                self.opt_d = torch.optim.Adam(
                    d_params, lr=float(self.train_cfg.lr_discriminator), betas=betas
                )

        # ---- LR schedule (cosine over the planned step budget) ------------- #
        self.two_stage = (
            bool(two_stage) if two_stage is not None else bool(config.model.two_stage)
        )
        self.max_steps = max_steps
        self.grad_clip = float(self.train_cfg.grad_clip or 0.0)

        # ---- AMP / fp16 (CUDA only) ---------------------------------------- #
        prec = str(self.train_cfg.precision or "fp32").lower()
        self.use_amp = (self.device.type == "cuda") and prec.startswith("amp")
        self.amp_dtype = torch.bfloat16 if "bf16" in prec else torch.float16
        try:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp and self.amp_dtype == torch.float16)
        except Exception:  # pragma: no cover - older/newer torch API drift
            self.scaler = None

        self.global_step = 0
        self.sched_g: Optional[Any] = None
        self.sched_d: Optional[Any] = None

    # ------------------------------------------------------------------ #
    # Parameter selection.
    # ------------------------------------------------------------------ #
    def _generator_parameters(self) -> List[Any]:
        """All trainable pipeline parameters EXCEPT the (training-only) discriminator.

        Includes the SR model, colorizer, and any learned 3D-LUT; excludes the frozen
        segmenter (its params have ``requires_grad=False`` already) and the
        discriminator (optimized separately).
        """
        disc_param_ids = set()
        if self.discriminator is not None:
            disc_param_ids = {id(p) for p in self.discriminator.parameters()}
        params: List[Any] = []
        for p in self.pipeline.parameters():
            if not p.requires_grad:
                continue
            if id(p) in disc_param_ids:
                continue
            params.append(p)
        return params

    @staticmethod
    def _grads_finite(params: Any) -> bool:
        """Return ``True`` iff every parameter gradient is finite (no NaN/inf).

        Used as a last-line guard before ``optimizer.step()`` so a non-finite gradient
        can never be written into the weights. ``params`` may be any iterable of tensors
        (it is consumed once).
        """
        for p in params:
            g = getattr(p, "grad", None)
            if g is not None and not bool(torch.isfinite(g).all()):
                return False
        return True

    # ------------------------------------------------------------------ #
    # Schedules.
    # ------------------------------------------------------------------ #
    def _build_schedulers(self, total_steps: int) -> None:
        """Build cosine LR schedulers for G (and D) over ``total_steps``."""
        sched = str(self.train_cfg.scheduler or "none").lower()
        total = max(1, int(total_steps))
        if sched == "cosine":
            self.sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.opt_g, T_max=total, eta_min=float(self.train_cfg.min_lr)
            )
            if self.opt_d is not None:
                self.sched_d = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.opt_d, T_max=total, eta_min=float(self.train_cfg.min_lr)
                )
        else:
            self.sched_g = None
            self.sched_d = None

    def _step_schedulers(self) -> None:
        if self.sched_g is not None:
            self.sched_g.step()
        if self.sched_d is not None:
            self.sched_d.step()

    # ------------------------------------------------------------------ #
    # Batch / device helpers.
    # ------------------------------------------------------------------ #
    def _to_device(self, batch: Sample) -> Sample:
        """Move a :class:`Sample`'s tensor fields onto the trainer device."""
        out: Dict[str, Any] = {}
        for key in ("ir", "rgb", "guide", "semantic"):
            v = batch.get(key) if hasattr(batch, "get") else None
            if v is not None and isinstance(v, torch.Tensor):
                out[key] = v.to(self.device, non_blocking=True)
            else:
                out[key] = v
        out["meta"] = batch.get("meta") if hasattr(batch, "get") else None
        return out  # type: ignore[return-value]

    def _loss_target(self, batch: Sample, output: Dict[str, Any]) -> Sample:
        """Build the loss-target dict, supplying an HR IR reference for SR terms.

        The SR loss terms (``sr_charbonnier`` / ``sr_gradient``) compare ``output['sr']``
        (HR) against ``target['ir']``; synthetic data only has the LR IR, so we upsample
        it to the SR grid here (loss-only) so those terms are spatially well-defined.
        The colorization terms keep the real GT ``rgb`` / ``semantic`` untouched.
        """
        sr = output.get("sr")
        ir = batch.get("ir") if hasattr(batch, "get") else None
        tgt: Dict[str, Any] = {
            "rgb": batch.get("rgb") if hasattr(batch, "get") else None,
            "semantic": batch.get("semantic") if hasattr(batch, "get") else None,
            "meta": batch.get("meta") if hasattr(batch, "get") else None,
        }
        if ir is not None and sr is not None and isinstance(ir, torch.Tensor):
            if ir.shape[-2:] != sr.shape[-2:]:
                ir_hr = F.interpolate(
                    ir, size=sr.shape[-2:], mode="bilinear", align_corners=False
                )
            else:
                ir_hr = ir
            tgt["ir"] = ir_hr
        else:
            tgt["ir"] = ir
        return tgt  # type: ignore[return-value]

    @staticmethod
    def _disc_input(cond: Any, image: Any) -> Any:
        """Concatenate the (super-resolved IR) condition with an RGB image for D."""
        if cond is None:
            return image
        if cond.shape[-2:] != image.shape[-2:]:
            cond = F.interpolate(cond, size=image.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([cond, image], dim=1)

    # ------------------------------------------------------------------ #
    # Discriminator objective (D-side; the inline counterpart of AdversarialLoss).
    # ------------------------------------------------------------------ #
    def _d_loss(self, real_out: Any, fake_out: Any) -> Any:
        """Discriminator loss over multi-scale score maps (last layer per scale).

        Mirrors the ``gan_mode`` of the generator-side :class:`AdversarialLoss`:
          * hinge   : ``mean(relu(1 - D(real))) + mean(relu(1 + D(fake)))``
          * lsgan   : ``mean((D(real) - 1)^2) + mean(D(fake)^2)``
          * vanilla : ``softplus(-D(real)) + softplus(D(fake))`` (BCE-with-logits).
        Averaged over scales.
        """
        losses: List[Any] = []
        n = min(len(real_out), len(fake_out))
        for s in range(n):
            real_score = real_out[s][-1]
            fake_score = fake_out[s][-1]
            if self.gan_mode == "hinge":
                loss = F.relu(1.0 - real_score).mean() + F.relu(1.0 + fake_score).mean()
            elif self.gan_mode == "lsgan":
                loss = (real_score - 1.0).pow(2).mean() + fake_score.pow(2).mean()
            else:  # vanilla / BCE-with-logits
                loss = F.softplus(-real_score).mean() + F.softplus(fake_score).mean()
            losses.append(loss)
        if not losses:
            return torch.zeros((), device=self.device)
        return torch.stack(losses).mean()

    # ------------------------------------------------------------------ #
    # One training step (G + D).
    # ------------------------------------------------------------------ #
    def train_step(self, batch: Sample, sr_only: bool = False) -> Dict[str, float]:
        """Run a single optimizer step (D then G) on ``batch``.

        Args:
          batch:   a (device-resident or host) :class:`Sample`.
          sr_only: if ``True`` (SR-pretrain phase) skip the GAN game and supervise only
                   the SR loss terms (still a full forward; the colorizer trains via its
                   own pixel/structure terms being zero-weighted out by the SR-only ctx).

        Returns:
          A log dict ``{"loss_total", "loss_d", ...weighted component values...}``.
        """
        batch = self._to_device(batch)
        self.pipeline.train()
        if self.discriminator is not None:
            self.discriminator.train()

        logs: Dict[str, float] = {}

        # ---- Forward the pipeline (G) -------------------------------------- #
        output = self.pipeline(batch)
        rgb_fake = output["rgb"]
        cond = output.get("aux", {}).get("disc_cond", output.get("sr"))
        rgb_real = batch.get("rgb")

        run_gan = (
            (not sr_only)
            and self.discriminator is not None
            and self.opt_d is not None
            and rgb_real is not None
        )

        # ---- D update ------------------------------------------------------ #
        loss_d_val = 0.0
        if run_gan:
            fake_in = self._disc_input(cond.detach(), rgb_fake.detach())
            real_in = self._disc_input(cond.detach(), rgb_real)
            self.opt_d.zero_grad(set_to_none=True)
            fake_out_d = self.discriminator(fake_in)
            real_out_d = self.discriminator(real_in)
            loss_d = self._d_loss(real_out_d, fake_out_d)
            # Defense-in-depth: only backward/step on a FINITE discriminator loss; a
            # non-finite loss would otherwise poison the D weights (and then everything).
            if getattr(loss_d, "requires_grad", False) and bool(torch.isfinite(loss_d)):
                loss_d.backward()
                # Clip BEFORE the optimizer step so exploding grads can't reach the params.
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.grad_clip)
                if self._grads_finite(self.discriminator.parameters()):
                    self.opt_d.step()
                else:  # pragma: no cover - defensive; conversions are now NaN-grad-safe
                    warnings.warn(
                        "Trainer: skipped D optimizer step (non-finite gradients).",
                        RuntimeWarning,
                        stacklevel=2,
                    )
            elif not bool(torch.isfinite(loss_d)):  # pragma: no cover - defensive guard
                warnings.warn(
                    "Trainer: skipped D update (non-finite discriminator loss).",
                    RuntimeWarning,
                    stacklevel=2,
                )
            loss_d_val = float(loss_d.detach().item())
        logs["loss_d"] = loss_d_val

        # ---- G update ------------------------------------------------------ #
        self.opt_g.zero_grad(set_to_none=True)

        ctx: Dict[str, Any] = {"step": self.global_step}
        if self.segmenter is not None:
            ctx["segmenter"] = self.segmenter
        if run_gan:
            # Re-score the fake WITH gradients into G; real (detached) for FM.
            fake_in_g = self._disc_input(cond, rgb_fake)
            ctx["discriminator"] = self.discriminator
            ctx["disc_out_fake"] = self.discriminator(fake_in_g)
            with torch.no_grad():
                ctx["disc_out_real"] = self.discriminator(self._disc_input(cond.detach(), rgb_real))

        target = self._loss_target(batch, output)
        total, components = self.criterion(output, target, ctx)
        # Guard the (rare) degenerate case where every active term detached to a constant
        # (e.g. an all-zero-weight config): backward() on a grad-less scalar would raise.
        # Defense-in-depth: only backward/step on a FINITE generator loss, and skip the
        # step if any gradient is non-finite — so a single bad batch can never write
        # NaN/inf into the weights (which is what turned step-1 into an all-NaN step-2).
        if getattr(total, "requires_grad", False) and bool(torch.isfinite(total)):
            total.backward()
            # Clip BEFORE the optimizer step (both G and D) to bound any grad spike.
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self._generator_parameters(), self.grad_clip)
            if self._grads_finite(self._generator_parameters()):
                self.opt_g.step()
                self._step_schedulers()
            else:  # pragma: no cover - defensive; conversions are now NaN-grad-safe
                self.opt_g.zero_grad(set_to_none=True)
                warnings.warn(
                    "Trainer: skipped G optimizer step (non-finite gradients).",
                    RuntimeWarning,
                    stacklevel=2,
                )
        elif not bool(torch.isfinite(total)):  # pragma: no cover - defensive guard
            self.opt_g.zero_grad(set_to_none=True)
            warnings.warn(
                "Trainer: skipped G update (non-finite generator loss).",
                RuntimeWarning,
                stacklevel=2,
            )

        logs["loss_total"] = float(total.detach().item())
        for k, v in components.items():
            logs[k] = float(v)
        self.global_step += 1
        return logs

    # ------------------------------------------------------------------ #
    # Full training loop.
    # ------------------------------------------------------------------ #
    def train(
        self,
        loader: Optional[Any] = None,
        epochs: Optional[int] = None,
        progress: bool = True,
    ) -> Dict[str, Any]:
        """Run the full training loop over a ``DataLoader`` (synthetic by default).

        Args:
          loader:   a ``torch.utils.data.DataLoader`` yielding batched :class:`Sample`
                    dicts; if ``None`` a synthetic loader is built from ``config``.
          epochs:   override ``config.train.epochs``.
          progress: show a tqdm bar (guarded; ignored if tqdm is absent).

        Returns:
          A summary dict ``{"steps", "epochs", "history": [...per-step logs...],
          "final": {...last log...}}``.
        """
        if loader is None:
            from ..data.synthetic import build_synthetic_loader

            loader = build_synthetic_loader(
                self.config, batch_size=self.train_cfg.batch_size, shuffle=True
            )

        n_epochs = int(epochs if epochs is not None else self.train_cfg.epochs)
        steps_per_epoch = len(loader) if hasattr(loader, "__len__") else 0
        planned = n_epochs * max(1, steps_per_epoch)
        if self.max_steps is not None:
            planned = min(planned, int(self.max_steps))
        self._build_schedulers(planned)

        sr_pretrain_epochs = int(self.train_cfg.sr_pretrain_epochs) if self.two_stage else 0

        history: List[Dict[str, float]] = []
        last: Dict[str, float] = {}
        for epoch in range(n_epochs):
            sr_only = epoch < sr_pretrain_epochs
            desc = f"epoch {epoch + 1}/{n_epochs}" + (" [SR]" if sr_only else "")
            it = _maybe_tqdm(loader, steps_per_epoch or None, desc, progress)
            for batch in it:
                last = self.train_step(batch, sr_only=sr_only)
                history.append(last)
                if self.max_steps is not None and self.global_step >= int(self.max_steps):
                    break
            if self.max_steps is not None and self.global_step >= int(self.max_steps):
                break

        return {
            "steps": self.global_step,
            "epochs": n_epochs,
            "history": history,
            "final": last,
        }

    # ------------------------------------------------------------------ #
    # Validation.
    # ------------------------------------------------------------------ #
    def validate(self, loader: Optional[Any] = None, max_batches: int = 4) -> Dict[str, float]:
        """Evaluate the metric suite over up to ``max_batches`` validation batches.

        Builds the suite via :func:`~irchroma.metrics.build_metric_suite` and averages
        each metric over the batches. Metrics whose optional backend is missing record
        ``NaN`` and are skipped from the mean.
        """
        from ..metrics import build_metric_suite

        suite = build_metric_suite(self.config)
        if loader is None:
            from ..data.synthetic import build_synthetic_loader

            loader = build_synthetic_loader(
                self.config, batch_size=self.train_cfg.batch_size, shuffle=False
            )

        self.pipeline.eval()
        sums: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        seen = 0
        with torch.no_grad():
            for batch in loader:
                if seen >= max_batches:
                    break
                batch = self._to_device(batch)
                output = self.pipeline(batch)
                pred = output["rgb"]
                gt = batch.get("rgb")
                results = suite.evaluate(pred, gt, ctx={"segmenter": self.segmenter})
                for name, val in results.items():
                    if val == val:  # not NaN
                        sums[name] = sums.get(name, 0.0) + float(val)
                        counts[name] = counts.get(name, 0) + 1
                seen += 1

        return {name: sums[name] / counts[name] for name in sums if counts.get(name)}

    # ------------------------------------------------------------------ #
    # Checkpoints.
    # ------------------------------------------------------------------ #
    def save_checkpoint(self, path: str) -> str:
        """Save pipeline + discriminator + optimizer states + step to ``path``."""
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        state: Dict[str, Any] = {
            "pipeline": self.pipeline.state_dict(),
            "opt_g": self.opt_g.state_dict(),
            "global_step": self.global_step,
            "config": self.config.to_dict(),
        }
        if self.discriminator is not None:
            state["discriminator"] = self.discriminator.state_dict()
        if self.opt_d is not None:
            state["opt_d"] = self.opt_d.state_dict()
        torch.save(state, path)
        return path

    def load_checkpoint(self, path: str, strict: bool = False) -> None:
        """Load states saved by :meth:`save_checkpoint` (best-effort, ``strict=False``)."""
        state = torch.load(path, map_location=self.device)
        if "pipeline" in state:
            self.pipeline.load_state_dict(state["pipeline"], strict=strict)
        if self.discriminator is not None and "discriminator" in state:
            self.discriminator.load_state_dict(state["discriminator"], strict=strict)
        if "opt_g" in state:
            try:
                self.opt_g.load_state_dict(state["opt_g"])
            except Exception:  # pragma: no cover - optimizer shape drift
                pass
        if self.opt_d is not None and "opt_d" in state:
            try:
                self.opt_d.load_state_dict(state["opt_d"])
            except Exception:  # pragma: no cover
                pass
        self.global_step = int(state.get("global_step", 0))
