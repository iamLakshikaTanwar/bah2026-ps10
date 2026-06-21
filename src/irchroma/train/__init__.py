"""irchroma.train — training loops, schedules, and the synthetic-data demo.

Owner: Builder-7 (training/serving). Implements the two-stage training recipe
(Stage-1 SR, Stage-2 colorization with shared/initialized encoder + light feature
feedback), optimizer/scheduler setup from :class:`irchroma.config.TrainConfig`,
AMP/fp16, checkpointing, geographic-holdout-aware validation, and a runnable
synthetic-data demo (no network / no credentials required).

Public registry:
  * :class:`Trainer`  — the two-player (G + D) training loop (``trainer.py``).

Imports are **resilient**: a torch-less environment (where the trainer cannot be
constructed) does not break ``import irchroma.train`` — ``Trainer`` is simply ``None``.
"""

from __future__ import annotations

from typing import List

__all__: List[str] = []

try:
    from .trainer import Trainer, seed_everything

    __all__.append("Trainer")
    __all__.append("seed_everything")
except Exception:  # pragma: no cover - keep package import safe during parallel dev
    Trainer = None  # type: ignore

    def seed_everything(seed: int = 0) -> int:  # type: ignore[misc]
        """No-op fallback when torch is unavailable (keeps the symbol importable)."""
        import random as _random

        _random.seed(int(seed))
        return int(seed)
