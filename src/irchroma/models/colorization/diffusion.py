"""irchroma.models.colorization.diffusion — BBDM colorization backup (quality ceiling).

Owner: ColorBuilder (Builder-3). Implements the **BACKUP** Stage-2 path: a minimal,
runnable **conditional Brownian-Bridge Diffusion Model (BBDM)** colorizer. A small
conditional UNet (conditioned on the SR'd IR tile) is trained to predict the bridge
target (``x0`` / the RGB endpoint), and a few-step :meth:`sample` runs the reverse
Brownian-bridge to produce RGB ``[B, 3, Hs, Ws]`` in ``[0, 1]``.

> This is the **quality-ceiling backup**, NOT the default. The primary, production
> Stage-2 model is :class:`irchroma.models.colorization.generator.ColorizationGenerator`
> (deterministic single pass, fast per tile). BBDM is kept for maximum FID/realism when
> latency permits, with the published SAR→optical VHR precedent (docs/research/02 §3,§10).

Why a Brownian bridge (vs plain conditional DDPM)?
  The forward process is a *bridge* pinned at both endpoints: ``x_0`` = the target RGB
  and ``x_T`` = (a deterministic function of) the **condition** — here a 3-channel
  projection of the IR. Because the diffusion goes *directly* domain→domain, the
  cross-domain gap is smaller than concat-conditioned DDPM, so fewer steps are needed.
  We use the discrete Brownian-bridge marginals
      ``x_t = (1 - m_t) · x_0 + m_t · y + sqrt(m_t (1 - m_t)) · δ · ε`` ,
  with ``m_t = t / T`` and a small variance scale ``δ`` (``ε ~ N(0, I)``). The UNet
  predicts ``x_0`` from ``(x_t, t, y)``; sampling marches ``t: T → 0`` plugging the
  predicted ``x_0`` back into the bridge posterior.

Operates in **pixel RGB space** by default (the config's VQGAN latent is out of scope
for this small, CPU-runnable backup; we keep it pixel-space and intentionally small).

torch is assumed present at runtime; this file must still ``py_compile`` without it.
"""

from __future__ import annotations

import math
from typing import List, Optional

from irchroma.config import ColorizationConfig
from irchroma.interfaces import BaseModel, TORCH_AVAILABLE, Tensor

if TORCH_AVAILABLE:  # pragma: no cover - exercised only where torch exists
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
else:  # pragma: no cover - torch-less docs/CI environment
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore


__all__ = ["BBDMColorizer"]


# =========================================================================== #
# Time embedding + small conditional UNet building blocks.
# =========================================================================== #
def _sinusoidal_embedding(timesteps: Tensor, dim: int) -> Tensor:
    """Standard sinusoidal timestep embedding -> ``[B, dim]``.

    Args:
      timesteps: ``[B]`` integer/float tensor of diffusion step indices.
      dim:       embedding dimension (even preferred).
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / max(1, half)
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)  # [B, half]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:  # zero-pad odd dim
        emb = F.pad(emb, (0, 1))
    return emb


class _ResBlock(nn.Module if TORCH_AVAILABLE else object):  # type: ignore[misc]
    """GroupNorm-SiLU conv resblock with additive timestep embedding (UNet stage)."""

    def __init__(self, in_ch: int, out_ch: int, temb_dim: int) -> None:
        super().__init__()  # type: ignore[misc]
        groups_in = math.gcd(8, in_ch) or 1
        groups_out = math.gcd(8, out_ch) or 1
        self.norm1 = nn.GroupNorm(groups_in, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.temb = nn.Linear(temb_dim, out_ch)
        self.norm2 = nn.GroupNorm(groups_out, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: Tensor, temb: Tensor) -> Tensor:
        """Apply the resblock with timestep conditioning."""
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.temb(temb).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class _SmallCondUNet(nn.Module if TORCH_AVAILABLE else object):  # type: ignore[misc]
    """A small 2-level conditional UNet predicting the RGB endpoint ``x0``.

    Input is the channel-wise concat of the noisy bridge sample ``x_t`` (3ch) and the
    condition ``y`` (3ch projection of IR) → 6 channels; output is 3-channel ``x0``.
    Two down/up levels keep it CPU-friendly for tiny tiles.
    """

    def __init__(self, in_ch: int = 6, out_ch: int = 3, base: int = 32, temb_dim: int = 128) -> None:
        super().__init__()  # type: ignore[misc]
        self.temb_dim = int(temb_dim)
        self.temb_mlp = nn.Sequential(
            nn.Linear(temb_dim, temb_dim),
            nn.SiLU(),
            nn.Linear(temb_dim, temb_dim),
        )
        self.in_conv = nn.Conv2d(in_ch, base, kernel_size=3, padding=1)

        self.down1 = _ResBlock(base, base, temb_dim)
        self.pool1 = nn.Conv2d(base, base, kernel_size=3, stride=2, padding=1)
        self.down2 = _ResBlock(base, base * 2, temb_dim)
        self.pool2 = nn.Conv2d(base * 2, base * 2, kernel_size=3, stride=2, padding=1)

        self.mid = _ResBlock(base * 2, base * 2, temb_dim)

        self.up2 = nn.ConvTranspose2d(base * 2, base * 2, kernel_size=4, stride=2, padding=1)
        self.upres2 = _ResBlock(base * 2 + base * 2, base * 2, temb_dim)  # skip from down2
        self.up1 = nn.ConvTranspose2d(base * 2, base, kernel_size=4, stride=2, padding=1)
        self.upres1 = _ResBlock(base + base, base, temb_dim)  # skip from down1

        groups_out = math.gcd(8, base) or 1
        self.out_norm = nn.GroupNorm(groups_out, base)
        self.out_conv = nn.Conv2d(base, out_ch, kernel_size=3, padding=1)

    def forward(self, x_t: Tensor, t: Tensor, cond: Tensor) -> Tensor:
        """Predict ``x0`` (RGB) from a noisy bridge sample, timestep, and condition.

        Args:
          x_t:  ``[B, 3, H, W]`` noisy bridge sample.
          t:    ``[B]`` timestep indices.
          cond: ``[B, 3, H, W]`` condition (IR projected to 3 channels).
        """
        temb = self.temb_mlp(_sinusoidal_embedding(t, self.temb_dim))
        h0 = self.in_conv(torch.cat([x_t, cond], dim=1))

        d1 = self.down1(h0, temb)            # [B, base, H, W]
        p1 = self.pool1(d1)                  # [B, base, H/2, W/2]
        d2 = self.down2(p1, temb)            # [B, 2base, H/2, W/2]
        p2 = self.pool2(d2)                  # [B, 2base, H/4, W/4]

        m = self.mid(p2, temb)               # [B, 2base, H/4, W/4]

        u2 = self.up2(m)                     # [B, 2base, H/2, W/2]
        u2 = self._match(u2, d2)
        u2 = self.upres2(torch.cat([u2, d2], dim=1), temb)
        u1 = self.up1(u2)                    # [B, base, H, W]
        u1 = self._match(u1, d1)
        u1 = self.upres1(torch.cat([u1, d1], dim=1), temb)

        return self.out_conv(F.silu(self.out_norm(u1)))

    @staticmethod
    def _match(x: Tensor, ref: Tensor) -> Tensor:
        """Bilinearly match ``x`` spatial size to ``ref`` (guards odd-size transposed convs)."""
        if x.shape[-2:] != ref.shape[-2:]:
            x = F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)
        return x


# =========================================================================== #
# BBDM colorizer.
# =========================================================================== #
class BBDMColorizer(BaseModel):
    """Conditional Brownian-Bridge Diffusion colorizer (Stage-2 backup).

    The bridge is pinned at ``x_0`` (target RGB) and ``y`` (a 3-channel projection of the
    SR'd IR). With ``m_t = t / T`` and variance scale ``δ`` (``max_var``), the discrete
    bridge marginal is::

        x_t = (1 - m_t) · x_0 + m_t · y + sqrt(m_t · (1 - m_t)) · δ · ε ,   ε ~ N(0, I)

    A small UNet predicts ``x_0`` from ``(x_t, t, y)``. Sampling marches ``t: T → 0``,
    re-deriving the next bridge point from the predicted ``x_0`` (DDIM-like, low/zero
    stochasticity to bound hallucination), in a few steps (``sample_steps``).

    Args:
      config:       a :class:`ColorizationConfig` (reads ``bbdm_timesteps`` /
                    ``bbdm_sample_steps`` / ``in_channels`` / ``out_channels``).
      base_channels: UNet base width (kept small for CPU tiny-tile demos).
      max_var:      Brownian-bridge variance scale ``δ`` (small ⇒ low stochasticity).
    """

    name: str = "bbdm_colorizer"

    def __init__(
        self,
        config: Optional[ColorizationConfig] = None,
        base_channels: int = 32,
        max_var: float = 0.1,
    ) -> None:
        super().__init__()  # type: ignore[misc]
        self.config = config if config is not None else ColorizationConfig()
        cfg = self.config

        self.in_channels: int = int(cfg.in_channels)     # IR channels (condition source)
        self.out_channels: int = int(cfg.out_channels)   # RGB
        self.num_timesteps: int = int(cfg.bbdm_timesteps)
        self.sample_steps: int = int(cfg.bbdm_sample_steps)
        self.max_var: float = float(max_var)
        self.base_channels: int = int(base_channels)

        if not TORCH_AVAILABLE:  # pragma: no cover
            self.cond_proj = None
            self.unet = None
            return

        # Project the IR condition (C_ir channels) to a 3-channel bridge endpoint `y`.
        self.cond_proj = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        temb_dim = max(64, self.base_channels * 4)
        self.unet = _SmallCondUNet(
            in_ch=self.out_channels * 2,   # x_t (3) ⊕ cond (3)
            out_ch=self.out_channels,      # predicted x0 (3)
            base=self.base_channels,
            temb_dim=temb_dim,
        )

    # ------------------------------------------------------------------ #
    # Bridge math helpers.
    # ------------------------------------------------------------------ #
    def _m(self, t: Tensor) -> Tensor:
        """Bridge mixing coefficient ``m_t = t / T`` (float tensor, same shape as ``t``)."""
        return t.float() / float(self.num_timesteps)

    def _bridge_sample(self, x0: Tensor, y: Tensor, t: Tensor, noise: Optional[Tensor] = None) -> Tensor:
        """Sample ``x_t`` from the Brownian-bridge marginal (forward / training process).

        Args:
          x0:    ``[B, 3, H, W]`` target RGB endpoint.
          y:     ``[B, 3, H, W]`` condition endpoint (projected IR).
          t:     ``[B]`` integer timesteps in ``[1, T]``.
          noise: optional pre-sampled ``ε`` (same shape as ``x0``); drawn if ``None``.
        """
        if noise is None:
            noise = torch.randn_like(x0)
        m = self._m(t).view(-1, 1, 1, 1)
        var = (m * (1.0 - m)).clamp(min=0.0)
        std = torch.sqrt(var) * self.max_var
        return (1.0 - m) * x0 + m * y + std * noise

    def condition(self, ir: Tensor) -> Tensor:
        """Project the IR input to the 3-channel bridge endpoint ``y`` in ``[0, 1]``."""
        return torch.sigmoid(self.cond_proj(ir))

    # ------------------------------------------------------------------ #
    # Training forward: predict the RGB endpoint x0.
    # ------------------------------------------------------------------ #
    def forward(
        self,
        ir: Tensor,
        rgb: Optional[Tensor] = None,
        semantic: Optional[Tensor] = None,
        t: Optional[Tensor] = None,
    ) -> Tensor:
        """Training step: predict ``x0`` (RGB) from a bridge sample at a random ``t``.

        Args:
          ir:       ``[B, C_ir, Hs, Ws]`` SR'd IR (condition source).
          rgb:      ``[B, 3, Hs, Ws]`` target RGB in ``[0, 1]`` (the bridge ``x0``). If
                    ``None`` (pure inference), this defers to :meth:`sample` and returns a
                    sampled RGB instead — so the module is usable both paired and unpaired.
          semantic: accepted for API symmetry with the primary generator (unused here;
                    the backup conditions on IR only — extend with ControlNet seg later).
          t:        optional ``[B]`` timesteps; random in ``[1, T]`` if ``None``.

        Returns:
          ``[B, 3, Hs, Ws]`` predicted RGB endpoint ``x0`` in ``[0, 1]`` (paired path), or
          a sampled RGB (inference path when ``rgb is None``).
        """
        if not TORCH_AVAILABLE:  # pragma: no cover
            raise RuntimeError("torch is required to run BBDMColorizer.forward().")
        y = self.condition(ir)
        if rgb is None:
            # No target -> behave as inference (few-step sample).
            return self.sample(ir, steps=self.sample_steps)

        b = ir.shape[0]
        if t is None:
            t = torch.randint(1, self.num_timesteps + 1, (b,), device=ir.device)
        x_t = self._bridge_sample(rgb, y, t)
        x0_pred = self.unet(x_t, t, y)
        return torch.clamp(x0_pred, 0.0, 1.0)

    def loss(self, ir: Tensor, rgb: Tensor, t: Optional[Tensor] = None) -> Tensor:
        """Convenience training loss: L1 between predicted and true ``x0`` (scalar).

        The composite loss stack may instead consume ``forward`` output directly; this is
        a self-contained reconstruction objective for standalone BBDM pretraining.
        """
        if not TORCH_AVAILABLE:  # pragma: no cover
            raise RuntimeError("torch is required to run BBDMColorizer.loss().")
        x0_pred = self.forward(ir, rgb=rgb, t=t)
        return F.l1_loss(x0_pred, rgb)

    # ------------------------------------------------------------------ #
    # Few-step reverse sampling.
    # ------------------------------------------------------------------ #
    @torch.no_grad() if TORCH_AVAILABLE else (lambda f: f)  # type: ignore[misc]
    def sample(
        self,
        ir: Tensor,
        steps: Optional[int] = None,
        stochastic: bool = False,
    ) -> Tensor:
        """Few-step reverse Brownian-bridge sampling: IR → RGB.

        Starts from the condition endpoint ``x_T = y`` (deterministic) and marches the
        timestep grid ``T → 0``. At each step the UNet predicts ``x0``; the next bridge
        point is re-derived from that prediction (DDIM-like). With ``stochastic=False``
        (default) the trajectory is deterministic — minimal hallucination, which is the
        whole point of using this as a *bounded* quality-ceiling backup.

        Args:
          ir:         ``[B, C_ir, Hs, Ws]`` SR'd IR (condition source).
          steps:      number of reverse steps (defaults to ``config.bbdm_sample_steps``).
          stochastic: if True, inject the small bridge noise during sampling.

        Returns:
          ``[B, 3, Hs, Ws]`` sampled RGB in ``[0, 1]``.
        """
        if not TORCH_AVAILABLE:  # pragma: no cover
            raise RuntimeError("torch is required to run BBDMColorizer.sample().")
        n_steps = int(steps) if steps is not None else self.sample_steps
        n_steps = max(1, min(n_steps, self.num_timesteps))

        y = self.condition(ir)
        x = y  # x_T == condition endpoint
        # Descending timestep grid from T down toward 0 (inclusive of a small floor).
        grid: List[int] = [
            int(round(self.num_timesteps * (i / n_steps))) for i in range(n_steps, 0, -1)
        ]
        # Ensure strictly-descending, valid (>=1) steps.
        seq = sorted({max(1, g) for g in grid}, reverse=True)

        for idx, t_cur in enumerate(seq):
            t_tensor = torch.full((x.shape[0],), t_cur, device=ir.device, dtype=torch.long)
            x0_pred = torch.clamp(self.unet(x, t_tensor, y), 0.0, 1.0)
            # Next (smaller) timestep on the grid; 0 means we've reached x0.
            t_next = seq[idx + 1] if idx + 1 < len(seq) else 0
            if t_next <= 0:
                x = x0_pred
                break
            t_next_tensor = torch.full((x.shape[0],), t_next, device=ir.device, dtype=torch.long)
            noise = torch.randn_like(x) if stochastic else None
            x = self._bridge_sample(x0_pred, y, t_next_tensor, noise=noise)

        return torch.clamp(x, 0.0, 1.0)
