"""irchroma.data.preprocess — radiometric standardization for IR (and RGB).

Converts raw sensor digital numbers (DN) / physical units into the network's
``~[0, 1]`` convention (ARCHITECTURE.md §4, docs/research/03 §8 preprocessing).

The dominant real-world steps for coarse thermal IR are:

  * **Physical scaling** — apply the sensor's published scale/offset to recover a
    physical quantity (reflectance, or brightness temperature in Kelvin), then map
    that to ``[0, 1]`` over a sensible dynamic range.
  * **Percentile stretch** — robust min/max from low/high percentiles (default
    2/98) so a few hot/cold outliers do not crush contrast.
  * **CLAHE** — Contrast-Limited Adaptive Histogram Equalization to lift the
    notoriously low local contrast of thermal IR *before* super-resolution
    (``DataConfig.use_clahe``). Optional; guarded behind OpenCV.

Landsat Collection-2 Level-2 scale factors (docs/research/01 §1.1), documented
here as constants so they are auditable:

  * Surface reflectance bands  ``SR_B*`` :  ``reflectance = DN * 2.75e-5 - 0.2``
  * Surface temperature band   ``ST_B10`` :  ``kelvin = DN * 3.41802e-3 + 149``

``torch``/``numpy`` are used when present but every method degrades gracefully to
pure-Python on scalars where feasible; OpenCV (``cv2``) is optional and only
required if :meth:`IRPreprocessor.clahe` is actually invoked.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

from ..config import DataConfig
from ..interfaces import TORCH_AVAILABLE

# --------------------------------------------------------------------------- #
# Guarded numeric / vision deps (module must import without any of them).
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - present on the real runtime
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

try:  # pragma: no cover
    import numpy as np

    _HAS_NUMPY = True
except Exception:  # pragma: no cover
    np = None  # type: ignore
    _HAS_NUMPY = False

try:  # OpenCV is optional; only CLAHE needs it.
    import cv2

    _HAS_CV2 = True
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore
    _HAS_CV2 = False


__all__ = [
    "LANDSAT_SR_SCALE",
    "LANDSAT_SR_OFFSET",
    "LANDSAT_ST_SCALE",
    "LANDSAT_ST_OFFSET",
    "SensorRadiometry",
    "SENSOR_RADIOMETRY",
    "IRPreprocessor",
    "standardize_ir",
    "to_uint8",
    "denormalize",
]


# --------------------------------------------------------------------------- #
# Published radiometric constants (auditable; docs/research/01 §1.1).
# --------------------------------------------------------------------------- #
#: Landsat C2 L2 surface-reflectance scale/offset:  reflectance = DN*scale + offset.
LANDSAT_SR_SCALE: float = 2.75e-5
LANDSAT_SR_OFFSET: float = -0.2
#: Landsat C2 L2 surface-temperature (ST_B10) scale/offset:  kelvin = DN*scale + offset.
LANDSAT_ST_SCALE: float = 3.41802e-3
LANDSAT_ST_OFFSET: float = 149.0


@dataclass(frozen=True)
class SensorRadiometry:
    """Per-sensor radiometric recipe to map raw values into a physical range.

    Attributes:
      scale, offset: linear DN→physical transform (``physical = DN*scale + offset``).
      phys_lo, phys_hi: the physical range mapped to ``[0, 1]`` after scaling
        (e.g. brightness-temperature ``[260 K, 330 K]`` for terrestrial thermal,
        reflectance ``[0, 1]``). Values outside are clipped.
      kind: ``"reflectance"`` or ``"brightness_temperature"`` (documentation only).
    """

    scale: float
    offset: float
    phys_lo: float
    phys_hi: float
    kind: str


#: Built-in radiometric recipes keyed by a short sensor/band id. Brightness-
#: temperature ranges are typical terrestrial land-surface bounds; tune per AOI.
SENSOR_RADIOMETRY: dict = {
    # Landsat thermal (the primary IR input). 260–330 K spans cold→hot land.
    "landsat_st_b10": SensorRadiometry(
        LANDSAT_ST_SCALE, LANDSAT_ST_OFFSET, 260.0, 330.0, "brightness_temperature"
    ),
    # Landsat surface reflectance (used for RGB / NIR / SWIR bands).
    "landsat_sr": SensorRadiometry(
        LANDSAT_SR_SCALE, LANDSAT_SR_OFFSET, 0.0, 1.0, "reflectance"
    ),
    # Already-physical brightness temperature in Kelvin (e.g. MODIS/VIIRS LST,
    # Sentinel-3 SLSTR thermal): identity scaling, clip to a land range.
    "kelvin": SensorRadiometry(1.0, 0.0, 260.0, 330.0, "brightness_temperature"),
    # Generic reflectance already in [0,1].
    "reflectance": SensorRadiometry(1.0, 0.0, 0.0, 1.0, "reflectance"),
}


def _is_tensor(x: Any) -> bool:
    return TORCH_AVAILABLE and torch is not None and isinstance(x, torch.Tensor)


def _is_ndarray(x: Any) -> bool:
    return _HAS_NUMPY and isinstance(x, np.ndarray)


def _percentile(x: Any, q: float) -> float:
    """Return the ``q``-th percentile (q in [0,100]) of a tensor/ndarray as a float."""
    if _is_tensor(x):
        # torch.quantile expects a fraction in [0,1].
        return float(torch.quantile(x.flatten().float(), q / 100.0).item())
    if _is_ndarray(x):
        return float(np.percentile(x, q))
    raise TypeError(f"Unsupported array type for percentile: {type(x)!r}")


def _clip01(x: Any) -> Any:
    """Clamp an array/tensor to ``[0, 1]`` (in place where possible)."""
    if _is_tensor(x):
        return x.clamp(0.0, 1.0)
    if _is_ndarray(x):
        return np.clip(x, 0.0, 1.0)
    return max(0.0, min(1.0, float(x)))


class IRPreprocessor:
    """Configurable IR (and RGB) radiometric standardizer.

    Wraps the per-sensor physical scaling, percentile stretch, optional CLAHE, and
    final clip into a single object driven by a :class:`irchroma.config.DataConfig`.
    The same instance handles tensors (``torch``) and numpy arrays; band/channel
    layout is ``[..., H, W]`` (channels-first) for tensors and ``[H, W]`` / ``[H, W, C]``
    for numpy CLAHE.

    Args:
      cfg:    a :class:`DataConfig` (reads ``ir_norm``, percentile bounds, CLAHE
              settings). Defaults to ``DataConfig()``.
      sensor: key into :data:`SENSOR_RADIOMETRY` selecting the physical recipe
              (default ``"landsat_st_b10"`` — the primary thermal input).
    """

    def __init__(
        self,
        cfg: Optional[DataConfig] = None,
        sensor: str = "landsat_st_b10",
    ) -> None:
        self.cfg = cfg if cfg is not None else DataConfig()
        if sensor not in SENSOR_RADIOMETRY:
            raise KeyError(
                f"Unknown sensor radiometry {sensor!r}; "
                f"known: {sorted(SENSOR_RADIOMETRY)}"
            )
        self.sensor = sensor
        self.radiometry = SENSOR_RADIOMETRY[sensor]

    # ---- individual stages ------------------------------------------------- #
    def physical_scale(self, raw: Any) -> Any:
        """Apply the sensor DN→physical transform, then map the physical range to [0,1].

        ``physical = raw*scale + offset``; then linearly rescale ``[phys_lo, phys_hi]``
        to ``[0, 1]`` and clip. For reflectance recipes with ``phys_lo=0, phys_hi=1``
        this is the standard reflectance scaling; for thermal it is the
        brightness-temperature → ``[0,1]`` mapping.
        """
        r = self.radiometry
        physical = raw * r.scale + r.offset
        span = (r.phys_hi - r.phys_lo) or 1.0
        scaled = (physical - r.phys_lo) / span
        return _clip01(scaled)

    def percentile_stretch(
        self,
        x: Any,
        lo: Optional[float] = None,
        hi: Optional[float] = None,
    ) -> Any:
        """Robust contrast stretch using low/high percentiles → ``[0, 1]``.

        Args:
          x:  array/tensor of (ideally already physically-scaled) IR values.
          lo: low percentile (defaults to ``cfg.ir_percentile_lo``).
          hi: high percentile (defaults to ``cfg.ir_percentile_hi``).
        """
        plo = self.cfg.ir_percentile_lo if lo is None else lo
        phi = self.cfg.ir_percentile_hi if hi is None else hi
        vlo = _percentile(x, plo)
        vhi = _percentile(x, phi)
        denom = (vhi - vlo) or 1.0
        stretched = (x - vlo) / denom
        return _clip01(stretched)

    def minmax(self, x: Any) -> Any:
        """Plain min/max normalization to ``[0, 1]`` (non-robust; for completeness)."""
        if _is_tensor(x):
            vlo = float(x.min().item())
            vhi = float(x.max().item())
        elif _is_ndarray(x):
            vlo = float(x.min())
            vhi = float(x.max())
        else:
            raise TypeError(f"Unsupported array type for minmax: {type(x)!r}")
        denom = (vhi - vlo) or 1.0
        return _clip01((x - vlo) / denom)

    def zscore(self, x: Any) -> Any:
        """Standardize to zero-mean/unit-std, then squash to ``[0, 1]`` via a soft map.

        Uses ``0.5 + z/6`` clipped to ``[0,1]`` (≈ ±3σ → full range) so the output
        still satisfies the network's ``[0,1]`` convention.
        """
        if _is_tensor(x):
            mu = float(x.float().mean().item())
            sd = float(x.float().std().item()) or 1.0
        elif _is_ndarray(x):
            mu = float(x.mean())
            sd = float(x.std()) or 1.0
        else:
            raise TypeError(f"Unsupported array type for zscore: {type(x)!r}")
        z = (x - mu) / sd
        return _clip01(0.5 + z / 6.0)

    def clahe(self, x: Any) -> Any:
        """Apply CLAHE (Contrast-Limited Adaptive Histogram Equalization).

        Operates on a single-channel (or HxW) image scaled to ``[0,1]`` and returns
        the equalized image in ``[0,1]``. Requires OpenCV; raises a clear error if
        unavailable. Uses ``cfg.clahe_clip_limit`` and ``cfg.clahe_grid``.

        Note: CLAHE is inherently a numpy/OpenCV op. Tensors are routed through
        numpy and returned as a tensor on the original device/dtype.
        """
        if not _HAS_CV2:
            raise RuntimeError(
                "CLAHE requires OpenCV (cv2). Install opencv-python-headless, or "
                "set DataConfig.use_clahe=False / use percentile_stretch instead."
            )
        clip = float(self.cfg.clahe_clip_limit)
        grid = int(self.cfg.clahe_grid)
        clahe_op = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))

        return_tensor = _is_tensor(x)
        if return_tensor:
            device = x.device
            dtype = x.dtype
            arr = x.detach().cpu().numpy()
        else:
            arr = x
        if not _HAS_NUMPY:  # pragma: no cover - numpy needed to bridge to cv2
            raise RuntimeError("CLAHE path requires numpy to bridge to OpenCV.")

        arr = np.asarray(arr)
        squeeze_axes = arr.shape[:-2]  # any leading dims (e.g. channel)
        flat = arr.reshape((-1,) + arr.shape[-2:]) if arr.ndim > 2 else arr[None]
        out = np.empty_like(flat, dtype=np.float32)
        for i in range(flat.shape[0]):
            plane = flat[i]
            u8 = np.clip(plane * 255.0, 0, 255).astype(np.uint8)
            eq = clahe_op.apply(u8).astype(np.float32) / 255.0
            out[i] = eq
        out = out.reshape(squeeze_axes + arr.shape[-2:]) if arr.ndim > 2 else out[0]

        if return_tensor:
            return torch.as_tensor(out, device=device).to(dtype)
        return out

    # ---- end-to-end -------------------------------------------------------- #
    def __call__(self, raw: Any, apply_physical: bool = True) -> Any:
        """Run the full standardization for the configured strategy.

        Pipeline: optional physical scaling → strategy norm (``cfg.ir_norm`` ∈
        {``percentile``, ``minmax``, ``zscore``, ``physical``}) → optional CLAHE →
        clip to ``[0, 1]``.

        Args:
          raw:            raw IR DN / physical values (tensor or ndarray).
          apply_physical: if ``True`` apply the sensor DN→physical scaling first
                          (set ``False`` if the input is already physical/[0,1]).
        """
        x = self.physical_scale(raw) if apply_physical else raw
        strategy = self.cfg.ir_norm
        if strategy == "percentile":
            x = self.percentile_stretch(x)
        elif strategy == "minmax":
            x = self.minmax(x)
        elif strategy == "zscore":
            x = self.zscore(x)
        elif strategy == "physical":
            x = _clip01(x)  # already physically scaled to [0,1]
        else:
            raise ValueError(
                f"Unknown ir_norm strategy {strategy!r}; "
                "expected one of percentile|minmax|zscore|physical."
            )
        if self.cfg.use_clahe and _HAS_CV2:
            x = self.clahe(x)
        return _clip01(x)


# --------------------------------------------------------------------------- #
# Functional convenience wrappers.
# --------------------------------------------------------------------------- #
def standardize_ir(
    raw: Any,
    cfg: Optional[DataConfig] = None,
    sensor: str = "landsat_st_b10",
    apply_physical: bool = True,
) -> Any:
    """Standardize raw IR to ``~[0, 1]`` using an :class:`IRPreprocessor`.

    Thin functional wrapper around ``IRPreprocessor(cfg, sensor)(raw, apply_physical)``
    for callers that do not want to hold the object.
    """
    return IRPreprocessor(cfg=cfg, sensor=sensor)(raw, apply_physical=apply_physical)


def to_uint8(x: Any) -> Any:
    """Convert a ``[0, 1]`` array/tensor to ``uint8`` ``[0, 255]`` for I/O / display.

    Clips to ``[0,1]`` first. Returns a ``uint8`` tensor for tensor inputs and a
    ``uint8`` ndarray for numpy inputs.
    """
    x = _clip01(x)
    if _is_tensor(x):
        return (x * 255.0 + 0.5).clamp(0, 255).to(torch.uint8)
    if _is_ndarray(x):
        return np.clip(x * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return int(max(0, min(255, round(float(x) * 255.0))))


def denormalize(
    x: Any,
    sensor: str = "landsat_st_b10",
    radiometry: Optional[SensorRadiometry] = None,
) -> Any:
    """Invert :meth:`IRPreprocessor.physical_scale`: map ``[0, 1]`` back to physical units.

    Recovers the physical quantity (reflectance, or Kelvin for thermal) from a
    ``[0, 1]`` standardized value, i.e. the inverse of the physical-scaling stage.
    Note: percentile/CLAHE stretches are *not* invertible and are not undone here.

    Args:
      x:          standardized array/tensor in ``[0, 1]``.
      sensor:     key into :data:`SENSOR_RADIOMETRY` (ignored if ``radiometry`` given).
      radiometry: explicit :class:`SensorRadiometry` to invert.
    """
    r = radiometry if radiometry is not None else SENSOR_RADIOMETRY[sensor]
    span = (r.phys_hi - r.phys_lo) or 1.0
    return x * span + r.phys_lo


def coregistration_note() -> str:  # pragma: no cover - documentation helper
    """Return the documented Landsat scale factors (for logs / provenance)."""
    return (
        f"Landsat C2 L2: SR = DN*{LANDSAT_SR_SCALE} + ({LANDSAT_SR_OFFSET}); "
        f"ST_B10 = DN*{LANDSAT_ST_SCALE} + {LANDSAT_ST_OFFSET} K."
    )
