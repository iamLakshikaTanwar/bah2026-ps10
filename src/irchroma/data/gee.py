"""irchroma.data.gee — Google Earth Engine ingestion (server-side pairing/export).

Implements the GEE access tier (docs/research/01 §15, §17 Recipes 1–2;
ARCHITECTURE.md §4.4) — pairing, cloud-masking, scaling, and reprojection happen
**on Google's servers**; only the final aligned tile is pulled. Used for
prototyping and to export aligned IR↔RGB training tiles.

Two recipes are provided verbatim from doc 01 §17:

  * :meth:`GEEIngestor.landsat_s2_pair` — **Recipe 1 (PRIMARY)**: Landsat-8/9 TIRS
    thermal ``ST_B10`` (IR, 100 m native → 30 m grid) ↔ Sentinel-2 ``B4/B3/B2``
    (RGB target, 10 m), cloud-masked + median-composited + reprojected onto a
    common CRS/origin so a simple integer upsample aligns the grids.
  * :meth:`GEEIngestor.aster_pair` — **Recipe 2 (HIGH-RES INTRA-SENSOR)**: ASTER
    VNIR ``B02/B01/B3N`` (RGB-ish, 15 m) ↔ TIR ``B13/B14`` (thermal, 90 m),
    already co-registered within the scene (zero registration).

Plus :meth:`GEEIngestor.to_sample` — adapt a pulled numpy/array tile dict into the
:class:`irchroma.interfaces.Sample` contract.

``earthengine-api`` (``ee``) and ``numpy`` are import-guarded; methods raise a
clear :class:`RuntimeError` only when invoked without the dependency (and EE must
be authenticated via ``ee.Authenticate()`` / a service account first).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..config import DataConfig

# --------------------------------------------------------------------------- #
# Guarded optional deps (module must import without any of them).
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - present only where EE is installed/authenticated
    import ee  # type: ignore

    _HAS_EE = True
except Exception:  # pragma: no cover
    ee = None  # type: ignore
    _HAS_EE = False

try:  # pragma: no cover
    import numpy as np

    _HAS_NUMPY = True
except Exception:  # pragma: no cover
    np = None  # type: ignore
    _HAS_NUMPY = False


__all__ = [
    "LANDSAT_SR_SCALE",
    "LANDSAT_SR_OFFSET",
    "LANDSAT_ST_SCALE",
    "LANDSAT_ST_OFFSET",
    "GEEIngestor",
]

# Landsat C2 L2 scale factors (docs/research/01 §1.1) — duplicated here so the EE
# server-side recipes are self-contained and auditable.
LANDSAT_SR_SCALE: float = 0.0000275
LANDSAT_SR_OFFSET: float = -0.2
LANDSAT_ST_SCALE: float = 0.00341802
LANDSAT_ST_OFFSET: float = 149.0  # Kelvin


class GEEIngestor:
    """Earth-Engine ingestor for aligned IR↔RGB pairs (server-side compute).

    Args:
      cfg:        :class:`DataConfig` (uses ``source_ids``, ``target_crs``,
                  ``target_resolution_m``, cloud/composite settings). Default ``DataConfig()``.
      initialize: if ``True`` (default), call :meth:`initialize` in ``__init__``
                  (no-op if ``ee`` is unavailable — deferred to first real use).
    """

    def __init__(self, cfg: Optional[DataConfig] = None, initialize: bool = True) -> None:
        self.cfg = cfg if cfg is not None else DataConfig()
        self._initialized = False
        if initialize and _HAS_EE:
            try:  # pragma: no cover - requires credentials
                self.initialize()
            except Exception:
                # Defer auth errors to the first real call so import/construction
                # never fails merely because credentials are not yet set up.
                self._initialized = False

    def initialize(self, project: Optional[str] = None) -> None:
        """Initialize the EE client (``ee.Initialize``). Requires authentication.

        Raises a clear error if ``earthengine-api`` is not installed. Authentication
        (``ee.Authenticate()`` or a service account) must be configured separately.
        """
        if not _HAS_EE:
            raise RuntimeError(
                "GEEIngestor requires earthengine-api (`pip install earthengine-api`) "
                "and an authenticated session (`earthengine authenticate`)."
            )
        if project is not None:  # pragma: no cover - requires credentials
            ee.Initialize(project=project)
        else:  # pragma: no cover - requires credentials
            ee.Initialize()
        self._initialized = True

    def _ensure(self) -> None:
        """Guard: EE present + initialized (lazily initialize if needed)."""
        if not _HAS_EE:
            raise RuntimeError(
                "earthengine-api is not installed; GEE ingestion is unavailable."
            )
        if not self._initialized:  # pragma: no cover - requires credentials
            self.initialize()

    # ---- server-side mask/scale helpers (closures over ee) ----------------- #
    def _mask_scale_landsat(self) -> Any:  # pragma: no cover - requires ee
        """Return a Landsat C2 L2 cloud-mask + scale function (doc 01 §17 Recipe 1)."""

        def _fn(img: Any) -> Any:
            qa = img.select("QA_PIXEL")
            # Bit 3 = cloud, bit 4 = cloud shadow; keep clear pixels.
            cloud = qa.bitwiseAnd(1 << 3).eq(0).And(qa.bitwiseAnd(1 << 4).eq(0))
            sr = img.select("SR_B.").multiply(LANDSAT_SR_SCALE).add(LANDSAT_SR_OFFSET)
            st = img.select("ST_B10").multiply(LANDSAT_ST_SCALE).add(LANDSAT_ST_OFFSET)
            return (
                img.addBands(sr, None, True)
                .addBands(st, None, True)
                .updateMask(cloud)
            )

        return _fn

    def _mask_s2(self) -> Any:  # pragma: no cover - requires ee
        """Return a Sentinel-2 SCL cloud-mask function (doc 01 §17 Recipe 1)."""

        def _fn(img: Any) -> Any:
            scl = img.select("SCL")
            # Drop shadow(3), cloud-medium(8), cloud-high(9), cirrus(10).
            clear = (
                scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
            )
            return img.updateMask(clear)

        return _fn

    # ---- Recipe 1: Landsat thermal IR ↔ Sentinel-2 RGB --------------------- #
    def landsat_s2_pair(
        self,
        bbox: Sequence[float],
        start: str,
        end: str,
        max_cloud: float = 20.0,
    ) -> Dict[str, Any]:
        """Build an aligned Landsat-TIRS(IR) ↔ Sentinel-2(RGB) pair (server-side).

        Recipe 1 (docs/research/01 §17): merge Landsat 8+9 C2 L2, cloud-mask + scale,
        median-composite ``ST_B10`` (thermal IR input); cloud-mask Sentinel-2 and
        median-composite ``B4/B3/B2`` (10 m RGB target); reproject both onto the
        configured CRS at 30 m / 10 m so a simple ×3 upsample aligns IR→RGB grids.

        Args:
          bbox:      ``[minx, miny, maxx, maxy]`` WGS-84 degrees.
          start,end: date strings ``"YYYY-MM-DD"``.
          max_cloud: Sentinel-2 ``CLOUDY_PIXEL_PERCENTAGE`` ceiling.

        Returns:
          ``{"ir": ee.Image, "rgb": ee.Image, "region": ee.Geometry, "crs": str,
             "ir_scale_m": 30, "rgb_scale_m": <target_resolution_m>}`` — server-side
          handles; pull tiles via ``getThumbURL``/``getDownloadURL`` or
          :meth:`export_to_drive`.
        """
        self._ensure()
        crs = self.cfg.target_crs
        rgb_scale = float(self.cfg.target_resolution_m)
        aoi = ee.Geometry.Rectangle(list(bbox))  # type: ignore[union-attr]

        ls = (
            ee.ImageCollection(self.cfg.source_ids.get("landsat_oli_tirs", "LANDSAT/LC08/C02/T1_L2"))
            .merge(ee.ImageCollection("LANDSAT/LC09/C02/T1_L2"))
            .filterBounds(aoi)
            .filterDate(start, end)
            .map(self._mask_scale_landsat())
        )
        s2 = (
            ee.ImageCollection(self.cfg.source_ids.get("sentinel2_msi", "COPERNICUS/S2_SR_HARMONIZED"))
            .filterBounds(aoi)
            .filterDate(start, end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud))
            .map(self._mask_s2())
        )

        ir = ls.select(["ST_B10"]).median().clip(aoi)
        rgb = s2.select(["B4", "B3", "B2"]).median().clip(aoi)
        ir_30 = ir.reproject(crs=crs, scale=30)
        rgb_t = rgb.reproject(crs=crs, scale=rgb_scale)
        return {
            "ir": ir_30,
            "rgb": rgb_t,
            "region": aoi,
            "crs": crs,
            "ir_scale_m": 30,
            "rgb_scale_m": rgb_scale,
        }

    # ---- Recipe 2: ASTER intra-sensor VNIR ↔ TIR --------------------------- #
    def aster_pair(
        self,
        bbox: Sequence[float],
        start: str = "2000-01-01",
        end: str = "2008-12-31",
    ) -> Dict[str, Any]:
        """Build a co-registered ASTER VNIR(RGB-ish) ↔ TIR(thermal) pair.

        Recipe 2 (docs/research/01 §17): ASTER bands are co-registered within the
        scene, so VNIR ``B02(red)/B01(green)/B3N(NIR)`` (15 m) and TIR ``B13/B14``
        (90 m, resampled) slice straight into training pairs with no registration.
        Default date window predates the 2008 SWIR detector failure.

        Args:
          bbox:      ``[minx, miny, maxx, maxy]`` WGS-84 degrees.
          start,end: date strings ``"YYYY-MM-DD"``.

        Returns:
          ``{"ir": ee.Image, "rgb": ee.Image, "region": ee.Geometry,
             "rgb_scale_m": 15, "ir_scale_m": 90}``.
        """
        self._ensure()
        aoi = ee.Geometry.Rectangle(list(bbox))  # type: ignore[union-attr]
        coll = (
            ee.ImageCollection(self.cfg.source_ids.get("aster", "ASTER/AST_L1T_003"))
            .filterBounds(aoi)
            .filterDate(start, end)
        )
        img = coll.first()
        rgb = ee.Image(img).select(["B02", "B01", "B3N"]).clip(aoi)  # red, green, NIR
        tir = ee.Image(img).select(["B13", "B14"]).clip(aoi)  # 10.25 / 11.3 µm
        return {
            "ir": tir,
            "rgb": rgb,
            "region": aoi,
            "rgb_scale_m": 15,
            "ir_scale_m": 90,
        }

    # ---- export ------------------------------------------------------------ #
    def export_to_drive(
        self,
        pair: Dict[str, Any],
        prefix: str = "irchroma_pair",
        folder: str = "irchroma",
    ) -> List[Any]:
        """Start Drive export tasks for the IR and RGB images of a ``pair``.

        Exports both images on the **same CRS** (and the pair's per-band scales) so
        the downloaded tiles are aligned (doc 01 §17 "same CRS + origin" trick).

        Returns:
          The list of started ``ee.batch.Task`` objects.
        """
        self._ensure()
        crs = pair.get("crs", self.cfg.target_crs)
        region = pair["region"]
        tasks: List[Any] = []
        plan: List[Tuple[str, Any, float]] = [
            ("ir", pair["ir"], float(pair.get("ir_scale_m", 30))),
            ("rgb", pair["rgb"], float(pair.get("rgb_scale_m", self.cfg.target_resolution_m))),
        ]
        for name, image, scale in plan:  # pragma: no cover - requires credentials
            task = ee.batch.Export.image.toDrive(
                image=image,
                description=f"{prefix}_{name}",
                folder=folder,
                region=region,
                scale=scale,
                crs=crs,
                maxPixels=int(1e10),
            )
            task.start()
            tasks.append(task)
        return tasks

    # ---- Sample adaptor ---------------------------------------------------- #
    def to_sample(
        self,
        ir: Any,
        rgb: Optional[Any] = None,
        guide: Optional[Any] = None,
        semantic: Optional[Any] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Adapt pulled arrays into the :class:`irchroma.interfaces.Sample` contract.

        Accepts already-downloaded numpy arrays (e.g. from ``ee.Image.getThumbURL`` /
        ``rasterio`` reads of the exported tiles) and arranges them channels-first.
        Expected input layouts: IR ``[H, W]`` or ``[C, H, W]``; RGB ``[H, W, 3]`` or
        ``[3, H, W]``. Returns a plain dict matching the ``Sample`` keys
        (``ir``/``rgb``/``guide``/``semantic``/``meta``); tensor conversion is left to
        the caller's dataset/loader.

        Args:
          ir, rgb, guide, semantic: numpy arrays (RGB/guide HR, IR/semantic LR).
          meta: provenance dict (merged with source defaults).

        Returns:
          A ``Sample``-shaped dict with channels-first arrays.
        """
        if not _HAS_NUMPY:
            raise RuntimeError("to_sample() requires numpy to normalize array layouts.")

        def _chw(arr: Any) -> Any:
            a = np.asarray(arr)
            if a.ndim == 2:  # [H, W] -> [1, H, W]
                return a[None, ...]
            if a.ndim == 3 and a.shape[-1] in (1, 3, 4) and a.shape[0] not in (1, 3, 4):
                return np.transpose(a, (2, 0, 1))  # [H, W, C] -> [C, H, W]
            return a  # assume already [C, H, W]

        sample: Dict[str, Any] = {
            "ir": _chw(ir),
            "rgb": None if rgb is None else _chw(rgb),
            "guide": None if guide is None else _chw(guide),
            "semantic": None if semantic is None else np.asarray(semantic),
            "meta": {"source": "gee", **(meta or {})},
        }
        return sample
