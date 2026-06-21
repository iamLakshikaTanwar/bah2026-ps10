"""irchroma.models.semantic.landcover — LULC label-map construction + frozen checker.

Two pieces (docs/research/04 §A, mechanisms §3; ARCHITECTURE.md §7.1, §7.4):

* :class:`LandCoverLabeler` — builds the fused per-pixel LULC label map ``L`` (the
  ``semantic [B, H, W]`` conditioning signal). The real path (guarded ``earthengine-api``)
  **majority-votes** ESA WorldCover v200 ⊕ Google Dynamic World ⊕ JRC Global Surface
  Water (water hard-prior), each reprojected/aggregated to the target Landsat grid and
  remapped into the canonical :data:`irchroma.config.LULC_CLASSES` taxonomy via the
  class-remap tables below. A ``from_segmenter`` path derives ``L`` from a segmentation
  model when no GEE labels are available (synthetic demo / inference fallback).
* :class:`FrozenSegmenter` — the **frozen, independent** segmentation-consistency
  checker run on the generated RGB for the no-hallucination audit. It prefers a
  transformers SegFormer (guarded import) and otherwise falls back to a lightweight,
  built-in pure-torch CNN segmenter so the demo/tests run **without** transformers. It is
  kept deterministic and frozen (``eval()``, ``requires_grad_(False)``).

Tensor conventions (see :mod:`irchroma.interfaces`):
  * label map ``L`` : ``LongTensor  [B, H, W]`` (indices into ``LULC_CLASSES``).
  * segmenter input : ``FloatTensor [B, 3, H, W]`` RGB in ``[0, 1]``.
  * segmenter output: ``FloatTensor [B, num_classes, H, W]`` class logits.

The module imports even without torch / earthengine / transformers installed.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from irchroma.config import (
    LULC_NAME_TO_INDEX,
    NUM_LULC_CLASSES,
    SemanticConfig,
)
from irchroma.interfaces import BaseModel, TORCH_AVAILABLE

# --------------------------------------------------------------------------- #
# Guarded torch import.
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - exercised only when torch is present
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor
except Exception:  # pragma: no cover - torch-less environments (docs/CI)
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    Tensor = object  # type: ignore


__all__ = [
    "LandCoverLabeler",
    "FrozenSegmenter",
    "WORLDCOVER_TO_LULC",
    "DYNAMIC_WORLD_TO_LULC",
]


# =========================================================================== #
# Class-remap tables (docs/research/04 §A; ARCHITECTURE.md §7.1).
# =========================================================================== #
# ESA WorldCover v200 raw class VALUE -> canonical LULC class NAME.
# WorldCover values: 10 tree, 20 shrub, 30 grass, 40 crop, 50 built, 60 bare,
# 70 snow/ice, 80 permanent water, 90 herbaceous wetland, 95 mangroves, 100 moss/lichen.
# Mangroves (95) fold into wetland; moss/lichen (100) into bare (sparse).
WORLDCOVER_TO_LULC: Dict[int, str] = {
    10: "trees",
    20: "shrub",
    30: "grass",
    40: "crops",
    50: "built",
    60: "bare",
    70: "snow",
    80: "water",
    90: "wetland",
    95: "wetland",   # mangroves -> wetland
    100: "bare",     # moss/lichen -> bare/sparse
}

# Google Dynamic World V1 label-band class INDEX -> canonical LULC class NAME.
# DW indices: 0 water, 1 trees, 2 grass, 3 flooded_veg, 4 crops, 5 shrub_and_scrub,
# 6 built, 7 bare, 8 snow_and_ice.  flooded_veg -> wetland.
DYNAMIC_WORLD_TO_LULC: Dict[int, str] = {
    0: "water",
    1: "trees",
    2: "grass",
    3: "wetland",   # flooded_veg -> wetland
    4: "crops",
    5: "shrub",
    6: "built",
    7: "bare",
    8: "snow",
}


def _remap_table_to_index(name_table: Dict[int, str]) -> Dict[int, int]:
    """Convert a ``{raw_value: class_name}`` table to ``{raw_value: class_index}``."""
    return {raw: LULC_NAME_TO_INDEX[name] for raw, name in name_table.items()}


class LandCoverLabeler:
    """Builds the fused per-pixel LULC label map on the target grid.

    The canonical (real-data) recipe majority-votes ESA WorldCover ⊕ Dynamic World ⊕
    JRC GSW (water hard-prior), each remapped into :data:`LULC_CLASSES` and reprojected
    to the target Landsat grid; gaps/edges may be refined by a Prithvi-EO head (out of
    scope here, see ``from_segmenter`` for the model-based fallback).

    This class is *not* an ``nn.Module``: it orchestrates data construction. The heavy
    Earth Engine dependency is imported lazily inside :meth:`from_gee` and guarded, so a
    plain ``LandCoverLabeler()`` always constructs.

    Args:
      cfg: a :class:`irchroma.config.SemanticConfig` (reads ``water_override_occurrence``).
           Defaults to ``SemanticConfig()``.
    """

    def __init__(self, cfg: Optional[SemanticConfig] = None) -> None:
        self.cfg = cfg or SemanticConfig()
        self.num_classes = NUM_LULC_CLASSES
        self.worldcover_index = _remap_table_to_index(WORLDCOVER_TO_LULC)
        self.dynamic_world_index = _remap_table_to_index(DYNAMIC_WORLD_TO_LULC)
        self.water_index = LULC_NAME_TO_INDEX["water"]

    # ------------------------------------------------------------------ #
    # Real GEE path (guarded earthengine-api).
    # ------------------------------------------------------------------ #
    def from_gee(
        self,
        geometry: Any,
        crs: str,
        scale_m: float,
        date_start: Optional[str] = None,
        date_end: Optional[str] = None,
    ) -> Any:
        """Build the label map from Earth Engine (majority-vote of three products).

        Reprojects ESA WorldCover v200, the date-matched Dynamic World composite, and the
        JRC GSW occurrence layer to ``crs``/``scale_m`` over ``geometry``, remaps each into
        the canonical taxonomy, majority-votes WorldCover ⊕ Dynamic World, then applies the
        JRC water **hard prior** wherever GSW occurrence exceeds
        ``cfg.water_override_occurrence``.

        Requires the optional ``earthengine-api`` dependency (and an initialized EE
        session). Raises :class:`ImportError` with a clear message if it is missing.

        Args:
          geometry:   an ``ee.Geometry`` (the tile footprint).
          crs:        target CRS (e.g. ``"EPSG:32643"``).
          scale_m:    target pixel size in metres (e.g. Landsat 30 m).
          date_start: ISO date (inclusive) for the Dynamic World composite window.
          date_end:   ISO date (exclusive) for the Dynamic World composite window.

        Returns:
          An ``ee.Image`` (single band, ``Long``-valued) of canonical LULC indices on
          the target grid. Bring it to a tensor via ``ee`` export / ``geemap`` / rasterio.
        """
        try:
            import ee  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dep
            raise ImportError(
                "LandCoverLabeler.from_gee requires the optional 'earthengine-api' "
                "package (and an initialized Earth Engine session). Install it with "
                "`pip install earthengine-api` and call `ee.Initialize()`. "
                "For credential-free runs use `from_segmenter`."
            ) from exc

        proj_args = {"crs": crs, "scale": float(scale_m)}

        # ---- ESA WorldCover v200 -> canonical indices. ---------------------- #
        wc_raw = ee.Image("ESA/WorldCover/v200/2021").select("Map")
        wc = self._remap_ee(ee, wc_raw, self.worldcover_index)

        # ---- Dynamic World (date-matched mode composite of the label band). - #
        dw_coll = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1").filterBounds(geometry)
        if date_start is not None and date_end is not None:
            dw_coll = dw_coll.filterDate(date_start, date_end)
        dw_raw = dw_coll.select("label").reduce(ee.Reducer.mode())
        dw = self._remap_ee(ee, dw_raw, self.dynamic_world_index)

        # ---- Majority vote across the two reprojected categorical layers. --- #
        # Stack as bands, reproject to the target grid, then per-pixel mode.
        stacked = (
            ee.Image.cat([wc, dw])
            .reproject(**proj_args)
        )
        voted = stacked.reduce(ee.Reducer.mode()).rename("lulc")

        # ---- JRC GSW water hard-prior override. ----------------------------- #
        occurrence = (
            ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
            .select("occurrence")
            .reproject(**proj_args)
        )
        is_water = occurrence.gte(float(self.cfg.water_override_occurrence))
        lulc = voted.where(is_water, ee.Image.constant(self.water_index)).toInt16()

        return lulc.clip(geometry)

    @staticmethod
    def _remap_ee(ee: Any, image: Any, table: Dict[int, int]) -> Any:
        """Remap raw class values to canonical indices on an ``ee.Image`` (vectorized)."""
        froms = list(table.keys())
        tos = [table[k] for k in froms]
        # default value (unmatched) -> water index 0 is unsafe; use clouds/no-data via -1
        # then callers treat <0 as ignore. ee.Image.remap defaultValue keeps it simple.
        return image.remap(froms, tos, defaultValue=-1).rename("lulc")

    # ------------------------------------------------------------------ #
    # Model-based path (no credentials needed).
    # ------------------------------------------------------------------ #
    def from_segmenter(
        self,
        image: "Tensor",
        segmenter: Optional["FrozenSegmenter"] = None,
    ) -> "Tensor":
        """Derive the label map by argmax of a segmentation model on an RGB image.

        Credential-free fallback used by the synthetic demo and at inference when no GEE
        labels are available. The segmenter's per-pixel class logits are arg-maxed into
        a ``LongTensor`` label map matching :data:`LULC_CLASSES`.

        Args:
          image:     ``FloatTensor [B, 3, H, W]`` RGB in ``[0, 1]`` (e.g. a coarse/naive
                     colorization or a co-registered optical reference).
          segmenter: a :class:`FrozenSegmenter`; a fresh one is created if ``None``.

        Returns:
          ``LongTensor [B, H, W]`` of canonical LULC indices.
        """
        if not TORCH_AVAILABLE:  # pragma: no cover
            raise RuntimeError("LandCoverLabeler.from_segmenter requires torch.")
        if image.dim() != 4 or image.shape[1] != 3:
            raise ValueError(
                f"from_segmenter expects image [B, 3, H, W]; got {tuple(image.shape)}."
            )
        seg = segmenter if segmenter is not None else FrozenSegmenter(self.cfg)
        with torch.no_grad():
            logits = seg(image)  # [B, num_classes, H, W]
        return logits.argmax(dim=1).long()  # [B, H, W]


# =========================================================================== #
# FrozenSegmenter — the no-hallucination consistency checker.
# =========================================================================== #
class _TinyCNNSegmenter(BaseModel):  # type: ignore[misc]
    """Lightweight pure-torch CNN segmenter (no external weights / deps).

    A few dilated conv layers mapping ``[B, 3, H, W]`` -> ``[B, num_classes, H, W]`` at
    full resolution (padding preserves spatial size). Deliberately small; it exists so
    the segmentation-consistency audit and the demo/tests run without ``transformers``.
    """

    name = "tiny_cnn_segmenter"

    def __init__(self, num_classes: int = NUM_LULC_CLASSES, width: int = 32) -> None:
        super().__init__()  # type: ignore[misc]
        self.num_classes = int(num_classes)

        def block(cin: int, cout: int, dilation: int = 1) -> "nn.Module":
            pad = dilation
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=3, padding=pad, dilation=dilation),
                nn.GroupNorm(num_groups=min(8, cout), num_channels=cout),
                nn.ReLU(inplace=True),
            )

        self.features = nn.Sequential(
            block(3, width, dilation=1),
            block(width, width, dilation=2),
            block(width, width * 2, dilation=4),
            block(width * 2, width * 2, dilation=2),
        )
        self.classifier = nn.Conv2d(width * 2, self.num_classes, kernel_size=1)

    def forward(self, rgb: "Tensor") -> "Tensor":
        """``[B, 3, H, W]`` RGB -> ``[B, num_classes, H, W]`` logits."""
        feat = self.features(rgb)
        return self.classifier(feat)


class FrozenSegmenter(BaseModel):  # type: ignore[misc]
    """Frozen, independent segmentation-consistency checker for the no-hallucination audit.

    Prefers a transformers **SegFormer** (guarded ``transformers`` import; uses the
    config ``checker_model`` family, a *different* family from the guidance encoder so the
    generator cannot game its own checker). If ``transformers`` is unavailable, falls
    back to a built-in :class:`_TinyCNNSegmenter` so the demo/tests run regardless.

    The module is always put in ``eval()`` mode with ``requires_grad_(False)`` and its
    forward runs under ``torch.no_grad()`` — deterministic and frozen by construction.

    forward: ``forward(rgb) -> logits`` (``[B, 3, H, W]`` RGB in ``[0,1]`` ->
    ``[B, num_classes, H, W]`` logits resized back to the input resolution).

    Args:
      cfg:        :class:`irchroma.config.SemanticConfig` (reads ``checker_model``,
                  ``num_classes``). Defaults to ``SemanticConfig()``.
      pretrained: if ``True`` and transformers is present, attempt to load pretrained
                  SegFormer weights; otherwise random-init the HF backbone. Ignored by the
                  pure-torch fallback.
      force_fallback: if ``True``, skip transformers and use the built-in CNN (useful for
                  fully reproducible CI).
    """

    name = "frozen_segformer_checker"

    # Map a few config ``checker_model`` aliases to HF SegFormer checkpoints.
    _HF_ALIASES: Dict[str, str] = {
        "segformer_b0": "nvidia/segformer-b0-finetuned-ade-512-512",
        "segformer_b1": "nvidia/segformer-b1-finetuned-ade-512-512",
        "segformer_b2": "nvidia/segformer-b2-finetuned-ade-512-512",
        "segformer_b3": "nvidia/segformer-b3-finetuned-ade-512-512",
    }

    def __init__(
        self,
        cfg: Optional[SemanticConfig] = None,
        pretrained: bool = False,
        force_fallback: bool = False,
    ) -> None:
        super().__init__()  # type: ignore[misc]
        self.cfg = cfg or SemanticConfig()
        self.num_classes = int(self.cfg.num_classes)
        self.backend: str = "tiny_cnn"
        self._hf_model: Optional[Any] = None

        model: Optional[Any] = None
        if not force_fallback:
            model = self._try_build_segformer(pretrained)
        if model is None:
            model = _TinyCNNSegmenter(num_classes=self.num_classes)
            self.backend = "tiny_cnn"
        self.model = model

        # Freeze: eval + no grad. Deterministic by construction.
        self.freeze()

    # ------------------------------------------------------------------ #
    def _try_build_segformer(self, pretrained: bool) -> Optional[Any]:
        """Try to build a transformers SegFormer; return ``None`` if unavailable."""
        try:  # pragma: no cover - optional dep
            from transformers import SegformerForSemanticSegmentation  # type: ignore
        except Exception:
            return None
        try:  # pragma: no cover - network / weights optional
            checker = str(self.cfg.checker_model).lower()
            repo = self._HF_ALIASES.get(checker, self._HF_ALIASES["segformer_b2"])
            if pretrained:
                hf = SegformerForSemanticSegmentation.from_pretrained(
                    repo,
                    num_labels=self.num_classes,
                    ignore_mismatched_sizes=True,
                )
            else:
                from transformers import SegformerConfig  # type: ignore

                hf_cfg = SegformerConfig.from_pretrained(repo, num_labels=self.num_classes)
                hf = SegformerForSemanticSegmentation(hf_cfg)
            self._hf_model = hf
            self.backend = "segformer"
            return hf
        except Exception:
            return None

    def freeze(self) -> None:
        """Put the checker in frozen eval mode (``eval`` + ``requires_grad_(False)``)."""
        self.eval()  # type: ignore[attr-defined]
        for p in self.parameters():  # type: ignore[attr-defined]
            p.requires_grad_(False)

    def train(self, mode: bool = True) -> "FrozenSegmenter":  # type: ignore[override]
        """Override ``train`` so the checker can never leave eval mode (stays frozen)."""
        return super().train(False)  # type: ignore[misc]

    def forward(self, rgb: "Tensor") -> "Tensor":
        """Run the frozen checker.

        Args:
          rgb: ``FloatTensor [B, 3, H, W]`` RGB in ``[0, 1]``.

        Returns:
          ``FloatTensor [B, num_classes, H, W]`` class logits at the input resolution.
        """
        if rgb.dim() != 4 or rgb.shape[1] != 3:
            raise ValueError(f"FrozenSegmenter.forward expects rgb [B, 3, H, W]; got {tuple(rgb.shape)}.")
        out_hw = rgb.shape[-2:]
        with torch.no_grad():
            if self.backend == "segformer":
                # HF SegFormer returns logits at H/4 x W/4; upsample to input size.
                outputs = self.model(pixel_values=rgb)
                logits = outputs.logits  # [B, num_classes, H/4, W/4]
                logits = F.interpolate(
                    logits, size=out_hw, mode="bilinear", align_corners=False
                )
            else:
                logits = self.model(rgb)  # [B, num_classes, H, W]
                if logits.shape[-2:] != out_hw:
                    logits = F.interpolate(logits, size=out_hw, mode="bilinear", align_corners=False)
        return logits
