"""irchroma.data.synthetic — procedural, semantically-consistent IR/RGB data.

This is the **single most important** data-layer module: the end-to-end demo,
the unit tests, the color-LUT build, and the segmentation-consistency audit all
depend on synthetic samples that are *physically and semantically coherent*.

Design (see ARCHITECTURE.md §3.1 tensor flow, §7 semantic integrity):

  1. Draw a random per-pixel **semantic label map** ``L`` :class:`LongTensor`
     ``[B, H, W]`` over the canonical :data:`irchroma.config.LULC_CLASSES`
     (blobby, spatially-coherent regions — not white noise — so neighbouring
     pixels share a class, like real land cover).
  2. Render the **target RGB** ``[B, 3, H*scale, W*scale]`` in ``[0, 1]`` by
     painting each class with its :data:`irchroma.config.DEFAULT_SRGB_PALETTE`
     colour, then adding mild per-class texture + noise. RGB is the *high-
     resolution* product (``scale``× the IR grid).
  3. Derive a single-channel **IR** ``[B, C_ir, H, W]`` as a physically-plausible
     monochrome field: a per-class emissivity/temperature-like base value plus a
     smooth spatial (thermal-gradient) ramp and sensor noise. It is deliberately
     **low-contrast** and at the **lower (IR) resolution** to mimic coarse
     thermal (Landsat TIRS native 100 m → 30 m grid) — the SR target.
  4. Optionally synthesize a co-registered **HR guide band** ``[B, C_g, Hg, Wg]``
     at RGB resolution (a panchromatic-like luminance proxy) for the guided-SR
     branch.

Everything is **deterministic given a seed**, so tests and the demo are
reproducible. Because IR is a (noisy, low-contrast) function of the *same* label
map that paints RGB, the cross-modal mapping the pipeline must learn is real and
the no-hallucination checks are meaningful: a wrong colour shows up as a class
mismatch against ``L``.

``torch`` is the only hard dependency here (the demo path); the module imports
without it only insofar as :mod:`irchroma.interfaces` provides stand-ins, but the
factories below require torch at call time and raise a clear error otherwise.
"""

from __future__ import annotations

import math
from typing import List, Optional

from .. import config as _cfg
from ..config import Config
from ..interfaces import TORCH_AVAILABLE, Sample

# --------------------------------------------------------------------------- #
# Guarded torch import. The module must import even on a torch-less box; the
# factory functions raise a clear RuntimeError only when actually called.
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - exercised on the real (torch) runtime.
    import torch
    from torch import Tensor
    from torch.utils.data import DataLoader, Dataset

    _DatasetBase = Dataset
except Exception:  # pragma: no cover - docs/CI box without torch.
    torch = None  # type: ignore
    Tensor = object  # type: ignore
    DataLoader = None  # type: ignore

    class _DatasetBase:  # minimal stand-in so the class below is *definable*.
        """Placeholder base for ``torch.utils.data.Dataset`` when torch is absent."""

        def __len__(self) -> int:  # pragma: no cover
            raise RuntimeError("torch is not installed.")

        def __getitem__(self, index: int):  # pragma: no cover
            raise RuntimeError("torch is not installed.")


__all__ = [
    "make_synthetic_sample",
    "SyntheticIRRGBDataset",
    "build_synthetic_loader",
]


# --------------------------------------------------------------------------- #
# Internal helpers (all torch; only reached when torch is available).
# --------------------------------------------------------------------------- #
def _require_torch() -> None:
    """Raise a clear error if torch is unavailable on the call path."""
    if not TORCH_AVAILABLE or torch is None:  # pragma: no cover
        raise RuntimeError(
            "irchroma.data.synthetic requires PyTorch. Install torch to generate "
            "synthetic IR/RGB samples (`pip install torch`)."
        )


def _palette_srgb_tensor() -> "Tensor":
    """Return the per-class sRGB palette as a ``[NUM_CLASSES, 3]`` float tensor in [0,1]."""
    rows: List[List[float]] = []
    for name in _cfg.LULC_CLASSES:
        r, g, b = _cfg.DEFAULT_SRGB_PALETTE[name]
        rows.append([r / 255.0, g / 255.0, b / 255.0])
    return torch.tensor(rows, dtype=torch.float32)  # [K, 3]


def _class_ir_levels() -> "Tensor":
    """Per-class monochrome IR base level in [0,1] (emissivity/temperature-like).

    Hand-tuned so the ordering is *physically plausible* for thermal IR while
    staying deliberately **low-contrast** (values clustered in the mid-band):
    water/snow read cool, bare/built read warm, vegetation sits in between. The
    absolute numbers do not matter for the demo; the point is that IR is a real
    (if weak) function of land-cover class — exactly the cross-modal signal the
    pipeline must exploit without hallucinating.
    """
    # name -> base brightness-temperature-like value (low contrast, ~[0.3, 0.7]).
    levels = {
        "water": 0.34,    # cool, thermally inert
        "trees": 0.45,    # evapotranspiration -> cooler than bare
        "grass": 0.50,
        "crops": 0.52,
        "shrub": 0.55,
        "built": 0.66,    # urban heat / high emissivity surfaces
        "bare": 0.64,     # hot dry soil
        "snow": 0.30,     # cold
        "wetland": 0.42,
        "clouds": 0.38,   # cold cloud tops
    }
    vals = [levels[name] for name in _cfg.LULC_CLASSES]
    return torch.tensor(vals, dtype=torch.float32)  # [K]


def _smooth_blob_labels(
    batch_size: int,
    height: int,
    width: int,
    num_classes: int,
    generator: "torch.Generator",
    num_centers: int = 6,
) -> "Tensor":
    """Generate spatially-coherent (blobby) label maps ``[B, H, W]`` (Long).

    Implemented as a per-pixel nearest-centroid assignment over a handful of
    random seed points, each tagged with a random class — a cheap Voronoi-like
    partition that yields contiguous regions (real land cover is contiguous, not
    salt-and-pepper). Fully vectorized; deterministic given ``generator``.
    """
    device = torch.device("cpu")
    # Coordinate grid, normalized to [0, 1].
    ys = torch.linspace(0.0, 1.0, steps=height)
    xs = torch.linspace(0.0, 1.0, steps=width)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # [H, W] each
    coords = torch.stack([grid_y, grid_x], dim=-1)  # [H, W, 2]

    labels = torch.empty((batch_size, height, width), dtype=torch.long, device=device)
    for b in range(batch_size):
        # Random seed-point locations in [0,1]^2 and their class assignments.
        centers = torch.rand((num_centers, 2), generator=generator)  # [C, 2]
        center_cls = torch.randint(
            low=0, high=num_classes, size=(num_centers,), generator=generator
        )  # [C]
        # Squared distance from every pixel to every center -> [H, W, C].
        diff = coords.unsqueeze(2) - centers.view(1, 1, num_centers, 2)
        dist2 = (diff * diff).sum(dim=-1)
        nearest = dist2.argmin(dim=-1)  # [H, W] index into centers
        labels[b] = center_cls[nearest]
    return labels


def _smooth_gradient(
    batch_size: int,
    height: int,
    width: int,
    generator: "torch.Generator",
) -> "Tensor":
    """A smooth low-frequency spatial field ``[B, 1, H, W]`` in ~[-1, 1].

    Models a gentle thermal gradient across a tile (sun angle / topography). Built
    from a couple of random sinusoids so it is smooth and deterministic.
    """
    ys = torch.linspace(0.0, 1.0, steps=height).view(1, height, 1)
    xs = torch.linspace(0.0, 1.0, steps=width).view(1, 1, width)
    field = torch.zeros((batch_size, height, width), dtype=torch.float32)
    for b in range(batch_size):
        # Two random low-frequency components per sample.
        phase = torch.rand((4,), generator=generator) * (2.0 * math.pi)
        freq = 1.0 + torch.rand((2,), generator=generator) * 1.5  # [1, 2.5]
        wave_y = torch.sin(2.0 * math.pi * freq[0].item() * ys + phase[0].item())
        wave_x = torch.cos(2.0 * math.pi * freq[1].item() * xs + phase[1].item())
        field[b] = (wave_y + wave_x).squeeze(0) * 0.5  # ~[-1, 1]
    return field.unsqueeze(1)  # [B, 1, H, W]


def make_synthetic_sample(
    cfg: Config,
    batch_size: int,
    seed: int = 0,
) -> Sample:
    """Build one deterministic, semantically-consistent synthetic :class:`Sample`.

    Args:
      cfg:        A :class:`irchroma.config.Config`. Sizes are read from
                  ``cfg.data`` (``ir_patch_size``) and the SR factor from
                  ``cfg.model.scale`` so the IR/RGB resolutions obey the tensor
                  contract (RGB is ``scale``× the IR grid).
      batch_size: Number of examples ``B`` to stack along the batch dim.
      seed:       RNG seed; identical ``(cfg, batch_size, seed)`` → identical sample.

    Returns:
      A :class:`irchroma.interfaces.Sample` with keys:
        * ``ir``       ``FloatTensor [B, C_ir, H, W]`` in ~[0,1] (low contrast).
        * ``rgb``      ``FloatTensor [B, 3, H*scale, W*scale]`` in [0,1] (GT target).
        * ``guide``    ``FloatTensor [B, C_g, H*scale, W*scale]`` in [0,1] (HR guide)
                       or ``None`` if ``cfg.model.c_guide == 0``.
        * ``semantic`` ``LongTensor  [B, H, W]`` — per-pixel LULC class indices.
        * ``meta``     ``dict`` — provenance + the synthesis parameters.
    """
    _require_torch()
    if batch_size <= 0:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}.")

    data_cfg = cfg.data
    model_cfg = cfg.model

    # ---- Resolve dims from config (honor the synthetic/demo dims) ---------- #
    h = int(data_cfg.ir_patch_size)          # IR (low-res) spatial size
    w = h
    scale = int(model_cfg.scale)             # end-to-end SR factor
    hs, ws = h * scale, w * scale            # RGB / guide (high-res) size
    c_ir = int(model_cfg.c_ir)
    c_guide = int(model_cfg.c_guide)
    num_classes = int(_cfg.NUM_LULC_CLASSES)

    # Deterministic CPU generator (synthetic data is small / CPU-friendly).
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))

    # ---- 1) Semantic label map L : [B, H, W] (Long) ------------------------ #
    semantic = _smooth_blob_labels(batch_size, h, w, num_classes, gen)

    # ---- 2) Target RGB : paint L with the per-class palette, + texture ----- #
    palette = _palette_srgb_tensor()  # [K, 3] in [0,1]
    # Upsample the label map to the RGB grid with nearest-neighbour so class
    # regions stay crisp at high resolution.
    sem_hr = (
        torch.nn.functional.interpolate(
            semantic.unsqueeze(1).float(), size=(hs, ws), mode="nearest"
        )
        .long()
        .squeeze(1)
    )  # [B, Hs, Ws]
    rgb = palette[sem_hr]  # [B, Hs, Ws, 3] gather -> per-pixel base colour
    rgb = rgb.permute(0, 3, 1, 2).contiguous()  # -> [B, 3, Hs, Ws]
    # Mild per-pixel texture + colour noise so it is not a flat paint-by-numbers
    # (real surfaces have intra-class variation); kept small to preserve class id.
    texture = 0.05 * torch.randn((batch_size, 3, hs, ws), generator=gen)
    rgb = (rgb + texture).clamp_(0.0, 1.0)

    # ---- 3) IR : physical monochrome function of class + gradient + noise -- #
    ir_levels = _class_ir_levels()  # [K] in ~[0.3, 0.7]
    ir_base = ir_levels[semantic].unsqueeze(1)  # [B, 1, H, W] per-class level
    gradient = _smooth_gradient(batch_size, h, w, gen)  # [B, 1, H, W] ~[-1,1]
    # Low-contrast composition: small gradient + small sensor noise.
    ir_single = ir_base + 0.08 * gradient + 0.03 * torch.randn(
        (batch_size, 1, h, w), generator=gen
    )
    ir_single = ir_single.clamp_(0.0, 1.0)
    # Replicate across requested IR channels (default 1) so C_ir is honored.
    if c_ir == 1:
        ir = ir_single
    else:
        ir = ir_single.repeat(1, c_ir, 1, 1)

    # ---- 4) Optional HR guide band (panchromatic-like luminance proxy) ----- #
    guide: Optional["Tensor"] = None
    if c_guide > 0:
        # Luminance of the (high-res) RGB target = a realistic co-registered
        # guide band that genuinely carries HR structure for the SR branch.
        lum = (
            0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
        )  # [B, 1, Hs, Ws]
        guide_noise = 0.02 * torch.randn((batch_size, 1, hs, ws), generator=gen)
        guide_single = (lum + guide_noise).clamp_(0.0, 1.0)
        guide = guide_single if c_guide == 1 else guide_single.repeat(1, c_guide, 1, 1)

    # ---- Metadata (provenance + synthesis params for reproducibility) ------ #
    meta = {
        "source": "synthetic",
        "synthetic": True,
        "seed": int(seed),
        "scale": scale,
        "ir_size": [h, w],
        "rgb_size": [hs, ws],
        "c_ir": c_ir,
        "c_guide": c_guide,
        "num_classes": num_classes,
        "crs": None,
        "transform": None,
        "tile_key": f"synthetic/{seed}",
        "date": None,
    }

    sample: Sample = {
        "ir": ir,
        "rgb": rgb,
        "guide": guide,
        "semantic": semantic,
        "meta": meta,
    }
    return sample


class SyntheticIRRGBDataset(_DatasetBase):  # type: ignore[misc]
    """A ``torch.utils.data.Dataset`` of procedural IR↔RGB pairs.

    Each item is a **single** (unbatched) :class:`Sample` — i.e. tensors are
    ``[C, H, W]`` / ``[H, W]`` with no leading batch dim — so the default
    ``DataLoader`` collation stacks them into the ``[B, ...]`` contract tensors.

    The dataset is deterministic: item ``i`` is generated from ``seed + i`` so a
    fixed ``seed`` yields a fixed dataset across runs (reproducible demos/tests).

    Args:
      cfg:  the :class:`irchroma.config.Config` whose ``data``/``model`` sections
            set the tile dims and SR factor.
      n:    number of samples in the dataset (defaults to
            ``cfg.data.synthetic_num_samples``).
      seed: base RNG seed (item ``i`` uses ``seed + i``).
    """

    def __init__(self, cfg: Config, n: Optional[int] = None, seed: int = 0) -> None:
        _require_torch()
        self.cfg = cfg
        self.n = int(n if n is not None else cfg.data.synthetic_num_samples)
        if self.n <= 0:
            raise ValueError(f"Dataset length must be >= 1, got {self.n}.")
        self.seed = int(seed)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int) -> Sample:
        """Return the unbatched sample at ``index`` (tensors without a batch dim)."""
        if index < 0:
            index += self.n
        if not (0 <= index < self.n):
            raise IndexError(index)
        # Generate a batch-of-1, then squeeze the batch dim so DataLoader can
        # re-collate. Keeps a single source of truth for the synthesis logic.
        batched = make_synthetic_sample(self.cfg, batch_size=1, seed=self.seed + index)
        item: Sample = {
            "ir": batched["ir"][0],
            "rgb": None if batched.get("rgb") is None else batched["rgb"][0],
            "guide": None if batched.get("guide") is None else batched["guide"][0],
            "semantic": None
            if batched.get("semantic") is None
            else batched["semantic"][0],
            "meta": dict(batched["meta"], index=index),
        }
        return item


def _collate_samples(batch: List[Sample]) -> Sample:
    """Collate a list of unbatched :class:`Sample` dicts into one batched Sample.

    Stacks the present tensor fields along a new batch dim and gathers ``meta``
    into a list. Robust to optional fields being ``None`` (kept ``None`` only if
    *every* item lacks them; mixed presence is not expected for synthetic data).
    """
    _require_torch()

    def _stack(key: str) -> Optional["Tensor"]:
        vals = [b.get(key) for b in batch]
        if any(v is None for v in vals):
            return None
        return torch.stack(vals, dim=0)  # type: ignore[arg-type]

    collated: Sample = {
        "ir": _stack("ir"),  # type: ignore[typeddict-item]
        "rgb": _stack("rgb"),
        "guide": _stack("guide"),
        "semantic": _stack("semantic"),
        "meta": [b.get("meta", {}) for b in batch],  # type: ignore[typeddict-item]
    }
    return collated


def build_synthetic_loader(
    cfg: Config,
    batch_size: Optional[int] = None,
    n: Optional[int] = None,
    seed: int = 0,
    shuffle: bool = False,
    num_workers: Optional[int] = None,
) -> "DataLoader":
    """Construct a ``DataLoader`` over a :class:`SyntheticIRRGBDataset`.

    Args:
      cfg:         the :class:`irchroma.config.Config`.
      batch_size:  per-batch size (defaults to ``cfg.train.batch_size``).
      n:           dataset length (defaults to ``cfg.data.synthetic_num_samples``).
      seed:        base RNG seed for deterministic data.
      shuffle:     whether to shuffle each epoch (default ``False`` for repeatable demos).
      num_workers: dataloader workers (defaults to ``cfg.data.num_workers``).

    Returns:
      A ``torch.utils.data.DataLoader`` yielding batched :class:`Sample` dicts that
      satisfy the tensor contract (``ir [B,C_ir,H,W]``, ``rgb [B,3,Hs,Ws]``, ...).
    """
    _require_torch()
    bs = int(batch_size if batch_size is not None else cfg.train.batch_size)
    workers = int(num_workers if num_workers is not None else cfg.data.num_workers)
    dataset = SyntheticIRRGBDataset(cfg, n=n, seed=seed)
    return DataLoader(
        dataset,
        batch_size=bs,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=bool(cfg.data.pin_memory) and workers > 0,
        collate_fn=_collate_samples,
        drop_last=False,
    )
