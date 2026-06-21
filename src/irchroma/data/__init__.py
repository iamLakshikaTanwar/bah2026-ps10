"""irchroma.data — multi-satellite ingestion, pairing, co-registration, datasets.

Owner: Builder-1 (Data layer). Implements GEE/STAC/COG access, IR↔RGB pairing &
co-registration (docs/research/01-datasets.md), LULC-aware synthetic data, tiling,
the O(1) tile catalog/cache (docs/research/05), and torch ``Dataset``/``DataLoader``
factories that emit :class:`irchroma.interfaces.Sample` dicts.

Import resilience (the package must always import during parallel development):
the torch/stdlib-safe modules — :mod:`synthetic`, :mod:`preprocess`,
:mod:`pairing`, :mod:`catalog` — are imported eagerly and their registry names are
always exported. Modules that *can* pull heavy optional deps at import time
(:mod:`stac_cog`, :mod:`gee`, :mod:`dataset`) are imported defensively under
``try/except`` so a missing dependency never breaks ``import irchroma.data``; their
public names are exported only if their import succeeds. ``__all__`` is built
dynamically to reflect what is actually available.
"""

from __future__ import annotations

from typing import List

__all__: List[str] = []


# --------------------------------------------------------------------------- #
# Always-available, torch/stdlib-safe modules (eager import; guaranteed export).
# `synthetic` and `preprocess` are required by the demo and tests.
# --------------------------------------------------------------------------- #
from . import synthetic, preprocess, pairing, catalog  # noqa: E402

from .synthetic import (  # noqa: E402
    make_synthetic_sample,
    SyntheticIRRGBDataset,
    build_synthetic_loader,
)
from .preprocess import (  # noqa: E402
    IRPreprocessor,
    standardize_ir,
    to_uint8,
    denormalize,
)
from .pairing import (  # noqa: E402
    pair_ir_rgb,
    coregister,
    coregistration_rmse,
    TileMeta,
    PairCandidate,
)
from .catalog import (  # noqa: E402
    TileCatalog,
    LRUCache,
    quadkey,
    lonlat_to_tile,
    tile_to_lonlat,
)

__all__ += [
    # submodules
    "synthetic",
    "preprocess",
    "pairing",
    "catalog",
    # synthetic registry
    "make_synthetic_sample",
    "SyntheticIRRGBDataset",
    "build_synthetic_loader",
    # preprocess registry
    "IRPreprocessor",
    "standardize_ir",
    "to_uint8",
    "denormalize",
    # pairing registry
    "pair_ir_rgb",
    "coregister",
    "coregistration_rmse",
    "TileMeta",
    "PairCandidate",
    # catalog registry
    "TileCatalog",
    "LRUCache",
    "quadkey",
    "lonlat_to_tile",
    "tile_to_lonlat",
]


# --------------------------------------------------------------------------- #
# Modules that may import heavy optional deps — guard each so the package always
# imports. Public names are exported only when the submodule import succeeds.
# --------------------------------------------------------------------------- #
try:  # STAC + COG streaming reader (pystac-client / rio-tiler / rasterio).
    from . import stac_cog  # noqa: E402
    from .stac_cog import STACCOGReader, STAC_ENDPOINTS  # noqa: E402

    __all__ += ["stac_cog", "STACCOGReader", "STAC_ENDPOINTS"]
except Exception:  # pragma: no cover - optional-dep failure must not break import
    stac_cog = None  # type: ignore
    STACCOGReader = None  # type: ignore

try:  # Google Earth Engine ingestor (earthengine-api).
    from . import gee  # noqa: E402
    from .gee import GEEIngestor  # noqa: E402

    __all__ += ["gee", "GEEIngestor"]
except Exception:  # pragma: no cover
    gee = None  # type: ignore
    GEEIngestor = None  # type: ignore

try:  # GeoTIFF tile dataset (rasterio/torch) + geographic split (pure stdlib).
    from . import dataset  # noqa: E402
    from .dataset import GeoTIFFTileDataset, geographic_split, TileItem  # noqa: E402

    __all__ += ["dataset", "GeoTIFFTileDataset", "geographic_split", "TileItem"]
except Exception:  # pragma: no cover
    dataset = None  # type: ignore
    GeoTIFFTileDataset = None  # type: ignore
    geographic_split = None  # type: ignore
