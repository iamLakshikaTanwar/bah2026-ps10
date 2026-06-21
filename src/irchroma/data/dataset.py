"""irchroma.data.dataset — GeoTIFF tile dataset + geographic-holdout split.

  * :class:`GeoTIFFTileDataset` — a ``torch.utils.data.Dataset`` that reads paired
    IR / RGB / guide / semantic GeoTIFF tiles from disk (via ``rasterio``) and emits
    :class:`irchroma.interfaces.Sample` dicts following the tensor contract
    (IR ``[C_ir,H,W]``, RGB ``[3,Hs,Ws]``, guide ``[C_g,Hg,Wg]``, semantic ``[H,W]``).
  * :func:`geographic_split` — the **geographic-holdout** train/val/test split from
    docs/research/06 §7.1 / ARCHITECTURE.md §10.3: partition items by a geographic
    key (WRS-2 path/row, region, scene, or date) so **no group straddles two splits**
    → no near-duplicate leakage from adjacent tiles. Deterministic given a seed.

``rasterio``/``torch``/``numpy`` are import-guarded so the module always imports;
the dataset raises a clear error if used without them, while
:func:`geographic_split` is pure-stdlib and always works.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..config import DataConfig
from ..interfaces import TORCH_AVAILABLE, Sample

# --------------------------------------------------------------------------- #
# Guarded optional deps.
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - present on the real runtime
    import torch
    from torch.utils.data import Dataset

    _DatasetBase = Dataset
except Exception:  # pragma: no cover
    torch = None  # type: ignore

    class _DatasetBase:  # stand-in so the class is definable without torch
        """Placeholder base for ``torch.utils.data.Dataset`` when torch is absent."""

        def __len__(self) -> int:  # pragma: no cover
            raise RuntimeError("torch is not installed.")

        def __getitem__(self, index: int):  # pragma: no cover
            raise RuntimeError("torch is not installed.")


try:  # pragma: no cover
    import numpy as np

    _HAS_NUMPY = True
except Exception:  # pragma: no cover
    np = None  # type: ignore
    _HAS_NUMPY = False

try:  # pragma: no cover
    import rasterio

    _HAS_RASTERIO = True
except Exception:  # pragma: no cover
    rasterio = None  # type: ignore
    _HAS_RASTERIO = False


__all__ = [
    "TileItem",
    "GeoTIFFTileDataset",
    "geographic_split",
]


@dataclass
class TileItem:
    """One paired on-disk tile (paths into GeoTIFF files) + its split/group key.

    Attributes:
      ir_path:       path to the IR GeoTIFF (required).
      rgb_path:      path to the RGB GeoTIFF target (optional at inference).
      guide_path:    path to the HR guide-band GeoTIFF (optional).
      semantic_path: path to the LULC label GeoTIFF (optional).
      path_row:      WRS-2 path/row (or region/scene) key used by :func:`geographic_split`.
      meta:          free-form provenance copied into ``Sample["meta"]``.
    """

    ir_path: str
    rgb_path: Optional[str] = None
    guide_path: Optional[str] = None
    semantic_path: Optional[str] = None
    path_row: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)


class GeoTIFFTileDataset(_DatasetBase):  # type: ignore[misc]
    """Read paired IR/RGB/guide/semantic GeoTIFF tiles → :class:`Sample` dicts.

    Each item yields an **unbatched** Sample (tensors ``[C,H,W]`` / ``[H,W]``) so the
    default ``DataLoader`` collation stacks them. Reflectance/IR rasters are returned
    as floats; semantic rasters as ``Long``. The IR is *not* re-normalized here
    (compose with :class:`irchroma.data.preprocess.IRPreprocessor` upstream if the
    tiles hold raw DN); RGB is scaled to ``[0,1]`` if it looks like 8-bit/uint.

    Args:
      items: the paired tiles to serve.
      cfg:   :class:`DataConfig` (reserved for future tile-size checks). Default ``DataConfig()``.
      to_tensor: if ``True`` (default) return ``torch`` tensors; else numpy arrays.
    """

    def __init__(
        self,
        items: Sequence[TileItem],
        cfg: Optional[DataConfig] = None,
        to_tensor: bool = True,
    ) -> None:
        if to_tensor and not TORCH_AVAILABLE:
            raise RuntimeError(
                "GeoTIFFTileDataset(to_tensor=True) requires PyTorch; install torch "
                "or pass to_tensor=False to get numpy arrays."
            )
        if not _HAS_RASTERIO:
            raise RuntimeError(
                "GeoTIFFTileDataset requires rasterio to read GeoTIFF tiles "
                "(`pip install rasterio`)."
            )
        self.items: List[TileItem] = list(items)
        self.cfg = cfg if cfg is not None else DataConfig()
        self.to_tensor = bool(to_tensor)

    def __len__(self) -> int:
        return len(self.items)

    # ---- IO helpers -------------------------------------------------------- #
    def _read_raster(self, path: str) -> Any:
        """Read a GeoTIFF as ``[bands, H, W]`` (numpy) plus its rasterio profile."""
        with rasterio.open(path) as src:
            arr = src.read()  # [bands, H, W]
            profile = dict(src.profile)
            meta = {
                "crs": str(src.crs) if src.crs else None,
                "transform": tuple(src.transform) if src.transform else None,
                "bounds": tuple(src.bounds) if src.bounds else None,
            }
        return arr, profile, meta

    def _to_float(self, arr: Any) -> Any:
        """Cast to float32; scale obvious 8-bit RGB (uint, max>1) to ``[0,1]``."""
        a = np.asarray(arr)
        if np.issubdtype(a.dtype, np.integer):
            maxv = float(a.max()) if a.size else 0.0
            scale = 255.0 if maxv > 1.0 else 1.0
            return a.astype(np.float32) / scale
        return a.astype(np.float32)

    def _maybe_tensor(self, arr: Any, dtype: str = "float") -> Any:
        """Convert a numpy array to a torch tensor (if ``to_tensor``), else passthrough."""
        if not self.to_tensor:
            return arr
        t = torch.as_tensor(np.ascontiguousarray(arr))
        return t.long() if dtype == "long" else t.float()

    def __getitem__(self, index: int) -> Sample:
        """Read tile ``index`` and assemble an unbatched :class:`Sample`."""
        if index < 0:
            index += len(self.items)
        item = self.items[index]

        ir_arr, _, ir_meta = self._read_raster(item.ir_path)
        ir = self._maybe_tensor(self._to_float(ir_arr), "float")

        rgb = None
        if item.rgb_path is not None:
            rgb_arr, _, _ = self._read_raster(item.rgb_path)
            rgb = self._maybe_tensor(self._to_float(rgb_arr), "float")

        guide = None
        if item.guide_path is not None:
            g_arr, _, _ = self._read_raster(item.guide_path)
            guide = self._maybe_tensor(self._to_float(g_arr), "float")

        semantic = None
        if item.semantic_path is not None:
            s_arr, _, _ = self._read_raster(item.semantic_path)
            # Semantic raster is single-band class indices -> [H, W] Long.
            s2d = np.asarray(s_arr)[0] if np.asarray(s_arr).ndim == 3 else np.asarray(s_arr)
            semantic = self._maybe_tensor(s2d, "long")

        meta = {
            "source": "geotiff",
            "ir_path": item.ir_path,
            "path_row": item.path_row,
            **ir_meta,
            **item.meta,
            "index": index,
        }
        sample: Sample = {
            "ir": ir,
            "rgb": rgb,
            "guide": guide,
            "semantic": semantic,
            "meta": meta,
        }
        return sample


def _group_key(item: Any, by: str) -> str:
    """Resolve the grouping key for an item under strategy ``by``.

    Accepts :class:`TileItem`, mapping, or object with attributes. Falls back to the
    IR filename stem when the requested key is missing so a split is always possible.
    """
    # Attribute access (TileItem / objects).
    if hasattr(item, by) and getattr(item, by) is not None:
        return str(getattr(item, by))
    # Mapping access (dict-like items).
    if isinstance(item, dict) and item.get(by) is not None:
        return str(item[by])
    # Common fallbacks: a 'meta' dict carrying the key.
    meta = getattr(item, "meta", None)
    if meta is None and isinstance(item, dict):
        meta = item.get("meta")
    if isinstance(meta, dict) and meta.get(by) is not None:
        return str(meta[by])
    # Last resort: derive a stable key from the IR path stem.
    ir_path = (
        getattr(item, "ir_path", None)
        or (item.get("ir_path") if isinstance(item, dict) else None)
    )
    if ir_path:
        return os.path.splitext(os.path.basename(str(ir_path)))[0]
    return repr(item)


def geographic_split(
    items: Sequence[Any],
    by: str = "path_row",
    ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 0,
) -> Dict[str, List[Any]]:
    """Geographic-holdout split of ``items`` into train/val/test with no leakage.

    Groups items by a geographic key (``by`` — e.g. WRS-2 ``path_row``, ``region``,
    ``scene``, or ``date``), then assigns **whole groups** to splits so that every
    tile sharing a group lands in the same split. This prevents the leakage that a
    random tile split causes (adjacent tiles are near-duplicates; docs/research/06
    §7.1). Groups are shuffled deterministically by ``seed`` and packed greedily to
    hit the target *item-count* ratios as closely as possible.

    Args:
      items:  the dataset items (:class:`TileItem`, dicts, or objects).
      by:     the grouping attribute/key (default ``"path_row"``).
      ratios: ``(train, val, test)`` fractions (need not sum to exactly 1; they are
              normalized). ``val``/``test`` may be 0.
      seed:   RNG seed for the deterministic group shuffle.

    Returns:
      ``{"train": [...], "val": [...], "test": [...]}`` — disjoint item lists whose
      groups never straddle splits. Group→split membership is also stable across runs.

    Raises:
      ValueError: if ``ratios`` are negative or all zero.
    """
    import random

    tr, va, te = (float(r) for r in ratios)
    if min(tr, va, te) < 0.0:
        raise ValueError(f"ratios must be non-negative, got {ratios!r}.")
    total = tr + va + te
    if total <= 0.0:
        raise ValueError("ratios must not be all zero.")
    tr, va, te = tr / total, va / total, te / total

    # 1) Bucket items by group, preserving first-seen order for stability.
    groups: "OrderedDict[str, List[Any]]" = OrderedDict()
    for it in items:
        groups.setdefault(_group_key(it, by), []).append(it)

    n_items = len(items)
    splits: Dict[str, List[Any]] = {"train": [], "val": [], "test": []}
    if n_items == 0:
        return splits

    # 2) Deterministically shuffle the *groups* (not the items).
    group_keys = list(groups.keys())
    random.Random(seed).shuffle(group_keys)

    # 3) Greedily pack whole groups to approach the target item-count ratios.
    targets = {"train": tr * n_items, "val": va * n_items, "test": te * n_items}
    counts = {"train": 0, "val": 0, "test": 0}
    # Only consider splits with a positive target so zero-ratio splits stay empty.
    active = [s for s in ("train", "val", "test") if targets[s] > 0.0]
    for key in group_keys:
        members = groups[key]
        # Assign this group to the active split with the largest remaining deficit
        # (target minus current count), normalized by target so small splits fill.
        best_split = max(
            active,
            key=lambda s: (targets[s] - counts[s]) / max(targets[s], 1e-9),
        )
        splits[best_split].extend(members)
        counts[best_split] += len(members)

    return splits
