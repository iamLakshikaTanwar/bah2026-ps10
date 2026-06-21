"""irchroma.config — the typed CONFIGURATION CONTRACT for the whole framework.

This module is the single source of truth for all hyper-parameters, paths, loss
weights, model dimensions, the LULC taxonomy, and the default color palettes that
seed the O(1) color-LUT. Seven parallel builders import from here, so the schema
must stay STABLE and BACKWARD-COMPATIBLE.

Design rules (do not break):
  * Pure standard library only (``dataclasses`` + ``typing``). NO hard dependency
    on torch, numpy, or omegaconf -> ``import irchroma.config`` must never fail.
  * ``OmegaConf`` is used for YAML I/O *if installed*; otherwise a pure-stdlib
    PyYAML / JSON fallback is used (``Config.from_yaml`` / ``Config.to_yaml`` always
    work as long as ``pyyaml`` is present; a JSON fallback covers even that absence).
  * Every config section is a frozen-by-convention ``@dataclass`` with sensible
    defaults so ``Config()`` yields a fully-runnable (synthetic-demo) configuration.

Units & ranges are documented inline next to each field. Tensor conventions live in
:mod:`irchroma.interfaces` (the code contract); this file is the *data* contract.

References: docs/research/01..06 (datasets, colorization, SR, semantic, fast-platform,
evaluation). Default loss weights synthesize docs 02 §7, 03 §9, 04 "losses".
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# Optional YAML backends (guarded so importing this module never fails).
# --------------------------------------------------------------------------- #
try:  # Preferred: OmegaConf (rich interpolation, type coercion).
    from omegaconf import OmegaConf  # type: ignore

    _HAS_OMEGACONF = True
except Exception:  # pragma: no cover - environment without omegaconf
    OmegaConf = None  # type: ignore
    _HAS_OMEGACONF = False

try:  # Fallback: PyYAML.
    import yaml  # type: ignore

    _HAS_YAML = True
except Exception:  # pragma: no cover - environment without pyyaml
    yaml = None  # type: ignore
    _HAS_YAML = False


# =========================================================================== #
# 0. LULC TAXONOMY + DEFAULT PALETTES (seeds for the O(1) class->color LUT)
# =========================================================================== #
# Canonical, ordered land-cover taxonomy used everywhere a semantic label index
# is expected (LongTensor [B, H, W] values index into this list). The ordering is
# the contract: index i  <->  LULC_CLASSES[i]. Do NOT reorder; append only.
#
# This 10-class scheme harmonizes ESA WorldCover v200, Google Dynamic World, and
# JRC Global Surface Water (docs/research/04 §A). Mapping notes per class below.
LULC_CLASSES: Tuple[str, ...] = (
    "water",     # 0  Dynamic World 0 / WorldCover 80 (+ JRC GSW hard prior)
    "trees",     # 1  forest / tree cover.  DW 1 / WC 10
    "grass",     # 2  grassland.            DW 2 / WC 30
    "crops",     # 3  cropland.             DW 4 / WC 40  (season-aware via WorldCereal)
    "shrub",     # 4  shrub & scrub.        DW 5 / WC 20
    "built",     # 5  built-up / urban.     DW 6 / WC 50  (render gray, NOT legend-red)
    "bare",      # 6  bare / sparse / soil. DW 7 / WC 60
    "snow",      # 7  snow & ice.           DW 8 / WC 70
    "wetland",   # 8  herbaceous wetland / flooded veg. DW 3 / WC 90
    "clouds",    # 9  cloud / no-data sentinel (uncertainty -> desaturate + flag)
)

#: Number of canonical LULC classes (== ``len(LULC_CLASSES)``).
NUM_LULC_CLASSES: int = len(LULC_CLASSES)

#: name -> integer index (inverse of ``LULC_CLASSES``).
LULC_NAME_TO_INDEX: Dict[str, int] = {name: i for i, name in enumerate(LULC_CLASSES)}

#: Integer index reserved for "unknown / ignore" in losses & label maps.
#: By convention this aliases ``clouds`` (the no-data sentinel). Builders may also
#: use -1 directly with cross-entropy ``ignore_index``; both are honored.
IGNORE_INDEX: int = -1

# Default *plausible natural-color* palette in 8-bit sRGB (R, G, B), 0..255.
# These are the colors a faithful colorizer should gravitate toward per class.
# IMPORTANT: these are NATURAL-RENDER colors (water->blue, veg->green, built->gray),
# NOT the WorldCover map-legend hex (which uses e.g. red for built-up). Derived by
# anchoring class identity to WorldCover and choosing realistic appearance
# (docs/research/04 "Color-LUT design"). Builders refine ranges from measured
# per-class RGB statistics at LUT-build time.
DEFAULT_SRGB_PALETTE: Dict[str, Tuple[int, int, int]] = {
    "water":   (40, 90, 160),    # blue
    "trees":   (24, 90, 36),     # deep green
    "grass":   (120, 165, 70),   # yellow-green
    "crops":   (150, 160, 80),   # green<->tan (seasonal)
    "shrub":   (140, 130, 70),   # olive / khaki
    "built":   (140, 140, 140),  # neutral gray (NOT red)
    "bare":    (170, 150, 120),  # gray-tan / soil
    "snow":    (240, 240, 245),  # near-white
    "wetland": (60, 130, 130),   # teal
    "clouds":  (200, 200, 200),  # light gray sentinel
}

# Default palette in CIE L*a*b* (perceptually uniform; the LUT clamps chroma here).
# L in [0,100], a in ~[-128,127], b in ~[-128,127]. These are approximate Lab
# centroids of ``DEFAULT_SRGB_PALETTE`` (computed offline, sRGB D65). The runtime
# LUT stores, per class, (mu_Lab, sigma_Lab, [lo,hi]); these seed mu. Chroma (a,b)
# is clamped toward the class centroid while L (luminance) is left freer so SR
# texture/detail survives (docs/research/04: "clamp a,b, leave L freer").
DEFAULT_LAB_PALETTE: Dict[str, Tuple[float, float, float]] = {
    "water":   (37.0, -2.0, -32.0),
    "trees":   (33.0, -30.0, 26.0),
    "grass":   (64.0, -22.0, 44.0),
    "crops":   (63.0, -9.0, 36.0),
    "shrub":   (54.0, -3.0, 33.0),
    "built":   (58.0, 0.0, 0.0),     # neutral: a=b=0
    "bare":    (63.0, 4.0, 22.0),
    "snow":    (95.0, 0.0, -1.0),
    "wetland": (49.0, -20.0, -3.0),
    "clouds":  (81.0, 0.0, 0.0),     # neutral sentinel
}

# Hard semantic priors: classes whose chroma the LUT clamps tightly (small radius),
# because their color is near-deterministic and mislabeling is the most dangerous
# hallucination for analysts. ``water`` additionally gets an override where
# JRC GSW max_extent/occurrence exceeds a threshold (see SemanticConfig).
HARD_PRIOR_CLASSES: Tuple[str, ...] = ("water", "snow")


def class_index(name: str) -> int:
    """Return the integer index for a LULC class name (raises ``KeyError`` if unknown)."""
    return LULC_NAME_TO_INDEX[name]


def class_name(index: int) -> str:
    """Return the LULC class name for an integer index (raises ``IndexError`` if out of range)."""
    return LULC_CLASSES[index]


# =========================================================================== #
# 1. DATA CONFIG
# =========================================================================== #
@dataclass
class DataConfig:
    """Data layer: sources, pairing/co-registration, tiling, normalization, splits.

    See docs/research/01-datasets.md (sources & pairing recipes) and
    docs/research/04 §A (LULC label-map construction).
    """

    # ---- Sources (ranked top sources from doc 01 §16) ---------------------- #
    #: Primary anchor source. Landsat 8/9 OLI+TIRS is the only single platform with
    #: co-registered RGB (OLI SR_B4/3/2) AND thermal (TIRS ST_B10). PS-required.
    primary_source: str = "landsat_oli_tirs"
    #: Additional sources used to cross-verify / fill gaps / add intra-sensor pairs.
    aux_sources: List[str] = field(
        default_factory=lambda: [
            "sentinel2_msi",   # 10 m RGB super-resolution target (cross-sensor)
            "aster",           # high-res intra-sensor VNIR<->TIR (no registration)
            "worldstrat",      # S2 10 m <-> SPOT 1.5 m SR benchmark
            "emit",            # hyperspectral self-paired IR<->RGB (anti-hallucination)
            "modis", "viirs",  # coarse high-volume + night Day/Night Band
            "insat_3d",        # Indian-domain geostationary vis<->thermal
            "resourcesat_liss4",  # Indian 5.8 m RGB target
        ]
    )
    #: GEE collection / STAC IDs keyed by source (subset; extend in data.stac).
    source_ids: Dict[str, str] = field(
        default_factory=lambda: {
            "landsat_oli_tirs": "LANDSAT/LC08/C02/T1_L2",   # also LC09/C02/T1_L2
            "landsat_toa": "LANDSAT/LC08/C02/T1_TOA",       # B10+B11+pan(B8 15 m)
            "sentinel2_msi": "COPERNICUS/S2_SR_HARMONIZED",
            "aster": "ASTER/AST_L1T_003",
            "modis_sr": "MODIS/061/MOD09GA",
            "modis_lst": "MODIS/061/MOD11A1",
            "viirs_sr": "NOAA/VIIRS/001/VNP09GA",
            "esa_worldcover": "ESA/WorldCover/v200",
            "dynamic_world": "GOOGLE/DYNAMICWORLD/V1",
            "jrc_water": "JRC/GSW1_4/GlobalSurfaceWater",
        }
    )
    #: Access backends in preference order (docs/research/01 §15, 05 §A).
    access_backends: List[str] = field(
        default_factory=lambda: ["stac_cog", "planetary_computer", "aws_open_data", "gee"]
    )

    # ---- IR / RGB / guide band definitions --------------------------------- #
    #: IR input band(s). Default = single thermal band (Landsat ST_B10). Multi-band
    #: IR (e.g. add SWIR-1/2 / NIR) is supported by widening this list and ``c_ir``.
    ir_bands: List[str] = field(default_factory=lambda: ["ST_B10"])
    #: RGB target bands (Landsat surface-reflectance red/green/blue).
    rgb_bands: List[str] = field(default_factory=lambda: ["SR_B4", "SR_B3", "SR_B2"])
    #: Optional co-registered HR guide band for guided SR (15 m pan / S2 10 m).
    #: ``None`` disables the guide branch (falls back to blind SISR).
    guide_band: Optional[str] = "SR_B8_PAN"
    #: Number of IR input channels (== len(ir_bands)); mirrors ModelConfig.c_ir.
    c_ir: int = 1
    #: Number of HR-guide channels (0 = no guide).
    c_guide: int = 1

    # ---- Co-registration (docs/research/01 §17 toolbox) -------------------- #
    target_crs: str = "EPSG:32643"          # UTM zone (e.g. 43N for India); per-AOI
    target_resolution_m: float = 10.0       # RGB target GSD (Sentinel-2 grid)
    coregister_method: str = "reproject_match"  # {reproject_match, gee_register, arosics}
    use_arosics_subpixel: bool = True       # final sub-pixel polish (frequency-domain)
    max_coreg_shift_px: float = 2.0         # reject pairs with residual shift > this
    cloud_mask: bool = True                 # QA_PIXEL (Landsat) / SCL (S2)
    composite_window_days: int = 21         # median-composite window to bridge gaps
    max_ndvi_delta: float = 0.2             # drop pairs with large seasonal change

    # ---- Tiling ------------------------------------------------------------ #
    tile_size: int = 512                    # px; fixed-size tile => O(1) inference
    tile_overlap: int = 32                  # px; raised-cosine overlap-blend margin
    ir_patch_size: int = 128                # LR IR crop fed to the network at train
    scale_factor: int = 4                   # SR upscale (mirrors ModelConfig.scale)

    # ---- Normalization / radiometry (docs/research/03 §8 preprocessing) ---- #
    #: IR standardized to ~[0,1]. Strategy: {percentile, minmax, zscore, physical}.
    ir_norm: str = "percentile"
    ir_percentile_lo: float = 2.0
    ir_percentile_hi: float = 98.0
    #: Optional CLAHE contrast normalization for low-contrast IR before SR.
    use_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_grid: int = 8
    rgb_norm: str = "unit"                  # RGB targets scaled to [0,1]

    # ---- LULC label-map construction (docs/research/04 §A, mechanisms) ----- #
    #: Sources majority-voted to build the per-pixel label map L on the Landsat grid.
    lulc_sources: List[str] = field(
        default_factory=lambda: ["esa_worldcover", "dynamic_world", "jrc_water"]
    )
    lulc_majority_vote: bool = True
    use_dynamic_world_uncertainty: bool = True  # (1 - max prob) -> uncertainty map
    num_classes: int = NUM_LULC_CLASSES         # mirrors the taxonomy above

    # ---- Splits (docs/research/06 §7.1 geographic holdout) ----------------- #
    #: Split by geography/scene/path-row, NEVER random tile (prevents leakage).
    split_strategy: str = "geographic_holdout"
    split_unit: str = "wrs2_path_row"           # {wrs2_path_row, region, scene, date}
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15
    holdout_biomes: List[str] = field(
        default_factory=lambda: ["forest", "water", "urban", "desert", "snow"]
    )
    seed: int = 1337

    # ---- Caching / paths --------------------------------------------------- #
    data_root: str = "data"
    cache_dir: str = "data/cache"
    cog_cache: bool = True                       # cache COG IFD headers (LRU)
    num_workers: int = 8
    pin_memory: bool = True

    # ---- Synthetic demo (runs with NO network / NO credentials) ------------ #
    use_synthetic: bool = False                  # train/demo on procedural data
    synthetic_num_samples: int = 256


# =========================================================================== #
# 2. MODEL CONFIG (+ SR / Colorization / Semantic sub-objects)
# =========================================================================== #
@dataclass
class SRConfig:
    """Stage-1 guided super-resolution (docs/research/03)."""

    #: Restoration backbone. {hat_light, nafnet, restormer, rrdb, rfdn, ecbsr}.
    #: Primary = guided HAT-light/NAFNet; fast backups = RFDN/ECBSR/Real-ESRGAN(rrdb).
    backbone: str = "nafnet"
    scale: int = 4                          # upscale factor (== DataConfig.scale_factor)
    in_channels: int = 1                    # C_ir
    out_channels: int = 1                   # SR output keeps IR channel count
    width: int = 64                         # base feature width
    enc_blocks: List[int] = field(default_factory=lambda: [2, 2, 4, 8])
    middle_blocks: int = 12
    dec_blocks: List[int] = field(default_factory=lambda: [2, 2, 2, 2])
    upsampler: str = "pixelshuffle"         # LR-space compute (ESPCN-style)
    channel_attention: bool = True          # RCAN/NAFNet-style
    use_fft_block: bool = True              # SwinFIR FFC global-frequency block

    # ---- Guidance branch (PSRGAN/CoReFusion/DSen2 pattern) ----------------- #
    use_guide: bool = True                  # fuse co-registered HR guide band
    guide_channels: int = 1                 # C_guide
    guide_fusion: str = "cross_attention"   # {concat, cross_attention, deformable}

    # ---- Blind degradation modeling (training only) ------------------------ #
    degradation: str = "realesrgan"         # {realesrgan, bsrgan, bicubic, measured_mtf}
    use_sensor_mtf: bool = True             # measured PSF/MTF (pushbroom)
    add_striping: bool = True               # pushbroom striping artifact
    noise_model: str = "poisson_gaussian"   # photon + read noise

    # ---- Self-supervision / MISR fallbacks --------------------------------- #
    l1bsr_selfsup: bool = False             # L1BSR-style when no HR thermal GT
    misr_frames: int = 1                    # >1 => multi-frame fusion (HighRes-net)


@dataclass
class ColorizationConfig:
    """Stage-2 IR->RGB colorization (docs/research/02)."""

    #: Primary generator family. {pix2pixhd_ddcolor, pix2pixhd, ddcolor, bbdm}.
    generator: str = "pix2pixhd_ddcolor"
    in_channels: int = 1                    # SR'd IR (+ conditioning appended internally)
    out_channels: int = 3                   # RGB
    ngf: int = 64                           # generator base filters
    n_downsample_global: int = 4            # Pix2PixHD coarse-to-fine global G
    n_blocks_global: int = 9                # residual blocks in global G
    n_local_enhancers: int = 1              # local enhancer count (HD detail)

    # ---- DDColor-style query color decoder --------------------------------- #
    use_color_decoder: bool = True
    num_color_queries: int = 100            # learnable color queries (cross-attn)
    color_decoder_layers: int = 3
    encoder: str = "convnext_tiny"          # DDColor encoder (semantics)

    # ---- SPADE semantic conditioning (docs/research/04 mechanisms) --------- #
    use_spade: bool = True                  # spatially-adaptive denorm from label map
    spade_label_nc: int = NUM_LULC_CLASSES  # one-hot label channels into SPADE
    condition_on_palette: bool = True       # append per-class palette prior channels

    # ---- Output color space ------------------------------------------------ #
    #: Predict in Lab (clamp chroma via LUT, free L) then convert to RGB; or direct RGB.
    output_space: str = "lab"               # {lab, rgb}
    color_space_clamp: bool = True          # apply ColorLUT chroma clamp post-decoder

    # ---- Discriminator (Pix2PixHD multi-scale, spectral-norm) -------------- #
    disc_type: str = "multiscale_patchgan"
    num_discriminators: int = 3             # multi-scale (3 resolutions)
    disc_n_layers: int = 3                  # PatchGAN depth
    disc_ndf: int = 64
    spectral_norm: bool = True
    gan_mode: str = "hinge"                 # {hinge, lsgan, vanilla, wgangp}

    # ---- BBDM diffusion backup (docs/research/02 §10 backup) --------------- #
    bbdm_latent: str = "vqgan"              # latent autoencoder for BBDM
    bbdm_timesteps: int = 1000
    bbdm_sample_steps: int = 50             # few-step inference (DDIM/consistency)


@dataclass
class SemanticConfig:
    """Semantic guidance, frozen consistency checker, and O(1) color-LUT (doc 04)."""

    # ---- Guidance encoder (produces the conditioning label map) ------------ #
    guidance_model: str = "segformer_b3"    # or "prithvi_eo_2_300m" (Landsat-native)
    guidance_pretrained: bool = True
    num_classes: int = NUM_LULC_CLASSES

    # ---- Frozen consistency CHECKER (different family => no self-gaming) ---- #
    checker_model: str = "segformer_b2"     # DIFFERENT family to guidance encoder
    checker_frozen: bool = True

    # ---- O(1) class -> CIE-Lab color-LUT (docs/research/04 "Color-LUT") ----- #
    lut_color_space: str = "lab"            # build/clamp in perceptually-uniform Lab
    lut_clamp_sigma: float = 2.0            # clamp chroma to [mu - k*sigma, mu + k*sigma]
    lut_clamp_chroma_only: bool = True      # clamp a,b; leave L (luminance) freer
    lut_hard_prior_sigma: float = 1.0       # tighter clamp for HARD_PRIOR_CLASSES
    #: JRC GSW occurrence threshold (%) above which water LUT is hard-forced.
    water_override_occurrence: float = 80.0

    # ---- Learned 3D-LUT refinement (AdaInt; docs/research/05 §B2) ---------- #
    use_learned_3dlut: bool = True
    lut3d_dim: int = 33                     # lattice size (33^3); trilinear O(1)/pixel
    lut3d_n_basis: int = 3                  # basis LUTs blended by a tiny CNN
    lut3d_adaint: bool = True               # non-uniform sampling intervals (AdaInt)

    # ---- Uncertainty handling (docs/research/04 mechanisms) ---------------- #
    uncertainty_source: str = "dynamic_world"  # {dynamic_world, mc_dropout, ensemble}
    uncertainty_threshold: float = 0.5      # above => desaturate + QA flag
    desaturate_uncertain: bool = True


@dataclass
class ModelConfig:
    """Top-level model contract: tensor dims + the three stage sub-configs."""

    #: Global tensor conventions (MUST match irchroma.interfaces docstring).
    c_ir: int = 1                           # IR input channels  [B, C_ir, H, W]
    c_guide: int = 1                        # HR guide channels  [B, C_g, Hg, Wg]
    c_rgb: int = 3                          # RGB output channels [B, 3, Hs, Ws]
    scale: int = 4                          # end-to-end SR factor (H->H*scale)
    num_classes: int = NUM_LULC_CLASSES     # semantic label range

    #: Two-stage modular design with a shared/initialized restoration encoder
    #: between SR and colorization + light feature feedback (docs/research/03 §10).
    two_stage: bool = True
    share_encoder: bool = True
    feature_feedback: bool = True

    sr: SRConfig = field(default_factory=SRConfig)
    colorization: ColorizationConfig = field(default_factory=ColorizationConfig)
    semantic: SemanticConfig = field(default_factory=SemanticConfig)


# =========================================================================== #
# 3. LOSS CONFIG (weights for EVERY term; fidelity-dominant by design)
# =========================================================================== #
@dataclass
class LossConfig:
    """Composite-loss weights. Synthesizes docs/research/02 §7, 03 §9, 04 "losses".

    Philosophy: fidelity/structure terms DOMINATE (L1+SSIM+edge+FM+seg) and the
    generative terms (adversarial, colorfulness, histogram) are BOUNDED so realism
    never overrides ground truth -> the core anti-hallucination guarantee.

    A weight of 0.0 disables a term. Weights are starting points; tune on a Landsat
    val split. ``CompositeLoss`` (irchroma.interfaces) reads these by name.
    """

    # ---- Stage-1 SR loss stack (docs/research/03 §9) ----------------------- #
    sr_charbonnier: float = 1.0             # robust L1 base (eps below)
    sr_gradient: float = 0.1                # SPSR edge/structure preservation
    sr_fft: float = 0.1                     # Fourier high-frequency emphasis (SwinFIR)
    sr_lpips: float = 0.05                  # light perceptual (optional)
    sr_adversarial: float = 0.0             # U-Net RaGAN; ONLY if perceptual sharpness needed
    sr_contextual: float = 0.0              # for guided SR with imperfect alignment
    charbonnier_eps: float = 1e-3           # sqrt(x^2 + eps^2)

    # ---- Stage-2 colorization loss stack (docs/research/02 §7) ------------- #
    color_l1: float = 10.0                  # PIXEL ANCHOR to GT  <- anti-hallucination core
    color_ms_ssim: float = 5.0              # structure preservation (1 - MS-SSIM)
    color_edge: float = 2.0                 # Sobel/SGA edge alignment (no boundary drift)
    color_lpips: float = 1.0                # perceptual realism + semantics (VGG)
    color_feature_matching: float = 10.0    # Pix2PixHD FM (stabilize / sharpen)
    color_adversarial: float = 1.0          # realism / vividness (kept modest)
    color_lab_chroma: float = 2.0           # a*,b* chrominance fidelity in Lab
    color_histogram: float = 0.5            # color-distribution / palette match (Wasserstein)
    color_colorfulness: float = 0.3         # DDColor colorfulness (counters desaturation)
    color_sam: float = 0.5                  # spectral-angle consistency (multi-band IR)
    color_tv: float = 1e-4                  # mild smoothing only

    # ---- Semantic / no-hallucination terms (docs/research/04) -------------- #
    seg_consistency: float = 1.0            # frozen-SegFormer(G(IR)) vs label map L
    color_lut_outofclass: float = 1.0       # penalize colors outside class LUT range
    task_perceptual: float = 0.5            # frozen YOLO/Mask2Former feature matching
    cycle_consistency: float = 0.0          # RGB->IR'->IR; only if forced unpaired (>0)
    uncertainty_weight: float = 0.1         # down-weight color loss where uncertain

    # ---- GAN bookkeeping --------------------------------------------------- #
    gan_mode: str = "hinge"                 # mirror ColorizationConfig.gan_mode
    r1_gamma: float = 0.0                   # optional R1 gradient penalty on D


# =========================================================================== #
# 4. TRAIN CONFIG
# =========================================================================== #
@dataclass
class TrainConfig:
    """Training loop hyper-parameters (docs/research/03 §10 two-stage recipe)."""

    #: Which stage(s) to train. {sr, colorization, joint, end_to_end}.
    stage: str = "joint"
    epochs: int = 200
    batch_size: int = 16
    accumulate_grad_batches: int = 1

    # ---- Optimizer / schedule --------------------------------------------- #
    optimizer: str = "adamw"
    lr_generator: float = 2.0e-4
    lr_discriminator: float = 2.0e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1.0e-2
    scheduler: str = "cosine"               # {cosine, step, multistep, none}
    warmup_steps: int = 1000
    min_lr: float = 1.0e-6
    grad_clip: float = 1.0
    ema_decay: float = 0.999                # weight EMA for stable eval (0 disables)

    # ---- Precision / performance ------------------------------------------ #
    precision: str = "amp_fp16"             # {fp32, amp_fp16, amp_bf16}
    compile_model: bool = False             # torch.compile graph fusion
    channels_last: bool = True

    # ---- Two-stage curriculum --------------------------------------------- #
    sr_pretrain_epochs: int = 50            # warm SR before colorization joins
    freeze_sr_after_pretrain: bool = False
    d_steps_per_g: int = 1                  # discriminator updates per generator step

    # ---- Checkpoint / logging --------------------------------------------- #
    output_dir: str = "checkpoints"
    log_dir: str = "logs"
    log_every: int = 50                     # steps
    val_every: int = 1                      # epochs
    save_every: int = 10                    # epochs
    keep_last_n: int = 5
    resume: Optional[str] = None            # path to checkpoint to resume
    device: str = "cuda"                    # {cuda, cpu, mps}
    seed: int = 1337


# =========================================================================== #
# 5. INFER CONFIG (fast O(1)-per-tile; docs/research/05)
# =========================================================================== #
@dataclass
class InferConfig:
    """Inference / serving-time engineering (docs/research/05-fast-platform-o1.md)."""

    # ---- Tiled inference with raised-cosine overlap-blend ------------------ #
    tile_size: int = 512                    # fixed => O(1) per tile
    tile_overlap: int = 32                  # px overlap margin
    blend_window: str = "raised_cosine"     # {raised_cosine, gaussian, linear, none}
    batch_tiles: int = 8                    # batched forward to saturate GPU

    # ---- Runtime / precision ---------------------------------------------- #
    runtime: str = "torch"                  # {torch, onnxruntime, tensorrt}
    precision: str = "fp16"                 # {fp32, fp16, int8}
    use_cuda_streams: bool = True           # overlap H2D/compute/D2H
    int8_calibration: Optional[str] = None  # path to calibration set for INT8 PTQ/QAT
    export_path: str = "checkpoints/irchroma.onnx"

    # ---- O(1) cache (H3 / quadkey keyed) ----------------------------------- #
    enable_cache: bool = True
    cache_backend: str = "redis"            # {redis, memory, none}
    cache_key: str = "quadkey"              # {quadkey, h3}  (O(1) bit-encode)
    h3_resolution: int = 9                  # if cache_key == h3
    write_back: bool = True                 # write rendered tile back to cache+CDN

    # ---- 3D-LUT color refinement (applied last, O(1)/pixel) ---------------- #
    apply_learned_3dlut: bool = True
    apply_class_lut_clamp: bool = True      # class->Lab chroma clamp
    emit_uncertainty: bool = True           # return uncertainty map in PipelineOutput
    device: str = "cuda"


# =========================================================================== #
# 6. EVAL CONFIG (6-family, 44-metric suite; docs/research/06)
# =========================================================================== #
@dataclass
class EvalConfig:
    """Evaluation protocol (docs/research/06-evaluation-metrics.md)."""

    # ---- Metric families to run (all six by default) ----------------------- #
    families: List[str] = field(
        default_factory=lambda: [
            "fidelity",       # A: PSNR/SSIM/MS-SSIM/SAM/ERGAS/RASE/SCC/UQI/VIF/CC/RMSE/MAE
            "perceptual",     # B: clean-FID/CLIP-FID/KID/LPIPS(alex,vgg)/DISTS/IS
            "no_reference",   # C: NIQE/BRISQUE/PIQE/MUSIQ/MANIQA/CLIP-IQA
            "color",          # D: CIEDE2000/colorfulness/chroma-PSNR/Lab-hist
            "faithfulness",   # E: seg-consistency mIoU/det count-delta/edge IoU/uncertainty/cycle-IR
            "downstream",     # F: mAP/mIoU + bootstrap CI + Wilcoxon
            "efficiency",     # G: latency/throughput/FLOPs/params/VRAM/energy
        ]
    )

    # ---- SR-specific conventions (docs/research/06 §6) --------------------- #
    y_channel_metrics: bool = True          # report PSNR/SSIM on luminance Y too
    border_shave: int = 4                   # crop border px before metrics

    # ---- FID engineering (docs/research/06 §B) ----------------------------- #
    fid_mode: str = "clean"                 # clean-fid, NOT naive FID
    use_clip_fid: bool = True               # robust for domain-shifted satellite data
    use_kid: bool = True                    # unbiased for small N
    fid_reference_stats: Optional[str] = "landsat_rgb"  # precomputed custom stats name

    # ---- Faithfulness gates (acceptance logic; docs/research/06 §7.3) ------- #
    seg_consistency_min: float = 0.70       # min mIoU(S(output), L_input)
    detection_count_delta_max: float = 0.10 # max fractional new-object delta
    edge_correlation_min: float = 0.80      # min gradient-magnitude correlation

    # ---- Downstream uplift protocol (docs/research/06 §F) ------------------ #
    downstream_detector: str = "yolov11_obb"
    downstream_segmenter: str = "segformer"
    bootstrap_resamples: int = 1000
    significance_test: str = "wilcoxon"     # paired; {wilcoxon, ttest_rel}
    conditions: List[str] = field(
        default_factory=lambda: ["raw_ir", "bicubic_baseline", "ours"]
    )

    # ---- Efficiency measurement (docs/research/06 §G — done correctly) ------ #
    warmup_iters: int = 50                  # GPU power-state ramp before timing
    timing_iters: int = 100                 # average over >=100 runs
    report_fp16: bool = True
    measure_energy: bool = False            # pynvml / codecarbon (optional)

    results_dir: str = "results"


# =========================================================================== #
# 7. TOP-LEVEL CONFIG (nests all sections) + YAML I/O
# =========================================================================== #
@dataclass
class Config:
    """Root configuration object nesting every section.

    ``Config()`` returns a fully-populated, runnable default. Round-trips to/from
    YAML via :meth:`from_yaml` / :meth:`to_yaml` (OmegaConf if available, else
    PyYAML, else JSON). All builders should accept a ``Config`` and read only the
    section(s) they own.
    """

    name: str = "irchroma_default"
    seed: int = 1337
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    infer: InferConfig = field(default_factory=InferConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    # ---- dict <-> dataclass helpers --------------------------------------- #
    def to_dict(self) -> Dict[str, Any]:
        """Return a plain nested ``dict`` (JSON/YAML-serializable)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Build a :class:`Config` from a (possibly partial) nested ``dict``.

        Unknown keys are ignored; missing keys fall back to defaults. This makes
        configs forward-compatible: a YAML written by an older version still loads.
        """
        return _build_dataclass(cls, data or {})

    # ---- YAML I/O ---------------------------------------------------------- #
    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load a :class:`Config` from a YAML file (OmegaConf -> PyYAML -> JSON)."""
        if _HAS_OMEGACONF:
            loaded = OmegaConf.load(path)
            raw = OmegaConf.to_container(loaded, resolve=True)  # -> plain dict
        elif _HAS_YAML:
            with open(path, "r", encoding="utf-8") as fh:
                raw = yaml.safe_load(fh)
        else:  # last-resort: try JSON (a YAML subset)
            with open(path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
        if not isinstance(raw, dict):
            raise ValueError(f"Config YAML at {path!r} did not parse to a mapping.")
        return cls.from_dict(raw)

    def to_yaml(self, path: Optional[str] = None) -> str:
        """Serialize to YAML. Writes to ``path`` if given; always returns the string.

        Uses OmegaConf if available, else PyYAML, else a JSON fallback (still valid
        YAML). Never raises merely because a backend is missing.
        """
        data = self.to_dict()
        if _HAS_OMEGACONF:
            text = OmegaConf.to_yaml(OmegaConf.create(data))
        elif _HAS_YAML:
            text = yaml.safe_dump(data, sort_keys=False, default_flow_style=False)
        else:  # pragma: no cover - JSON is a valid YAML subset
            text = json.dumps(data, indent=2)
        if path is not None:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
        return text

    # ---- ergonomics -------------------------------------------------------- #
    def merge(self, overrides: Dict[str, Any]) -> "Config":
        """Return a NEW Config with ``overrides`` (nested dict) deep-merged in."""
        merged = _deep_merge(self.to_dict(), overrides or {})
        return Config.from_dict(merged)


# --------------------------------------------------------------------------- #
# Internal helpers (stdlib only): recursive dataclass construction + deep merge.
# --------------------------------------------------------------------------- #
def _build_dataclass(cls: Any, data: Dict[str, Any]) -> Any:
    """Recursively instantiate a (possibly nested) dataclass from a dict.

    Nested dataclass fields are recursed into; scalar/list/dict fields are passed
    through. Unknown keys are dropped; missing keys use field defaults. Robust to
    partial configs so older YAML still loads against a newer schema.
    """
    if not dataclasses.is_dataclass(cls):
        return data
    field_types = {f.name: f.type for f in dataclasses.fields(cls)}
    field_objs = {f.name: f for f in dataclasses.fields(cls)}
    kwargs: Dict[str, Any] = {}
    for key, value in (data or {}).items():
        if key not in field_types:
            continue  # forward-compat: ignore unknown keys
        fobj = field_objs[key]
        # Resolve the (possibly default-factory) dataclass type for nested fields.
        nested_cls = _resolve_nested_dataclass(fobj)
        if nested_cls is not None and isinstance(value, dict):
            kwargs[key] = _build_dataclass(nested_cls, value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def _resolve_nested_dataclass(fobj: "dataclasses.Field") -> Optional[Any]:
    """If ``fobj`` is a dataclass-typed field, return that dataclass type, else None.

    Works whether the annotation is a real class or a string (PEP 563 / from
    __future__ import annotations), by consulting the default_factory's product.
    """
    # 1) Annotation is already a dataclass type.
    t = fobj.type
    if dataclasses.is_dataclass(t):
        return t
    # 2) default_factory produces a dataclass instance (covers string annotations).
    if fobj.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
        try:
            produced = fobj.default_factory()  # type: ignore[misc]
            if dataclasses.is_dataclass(produced):
                return type(produced)
        except Exception:
            return None
    # 3) A bare default that is a dataclass instance.
    if fobj.default is not dataclasses.MISSING and dataclasses.is_dataclass(fobj.default):
        return type(fobj.default)
    return None


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base`` (dicts merge, scalars replace)."""
    out = dict(base)
    for key, value in (override or {}).items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


# --------------------------------------------------------------------------- #
# Convenience constructors.
# --------------------------------------------------------------------------- #
def default_config() -> Config:
    """Return the canonical default :class:`Config` (same as ``Config()``)."""
    return Config()


def synthetic_demo_config() -> Config:
    """A tiny, CPU-friendly config that runs end-to-end on procedural data.

    No network, no credentials, no GPU required. Used by ``make demo`` and tests.
    """
    cfg = Config(name="irchroma_synthetic_demo")
    cfg.data.use_synthetic = True
    cfg.data.synthetic_num_samples = 32
    cfg.data.tile_size = 128
    cfg.data.ir_patch_size = 32
    cfg.data.num_workers = 0
    cfg.model.scale = 2
    cfg.model.sr.scale = 2
    cfg.model.sr.width = 16
    cfg.model.colorization.ngf = 16
    cfg.train.stage = "joint"
    cfg.train.epochs = 1
    cfg.train.batch_size = 4
    cfg.train.precision = "fp32"
    cfg.train.device = "cpu"
    cfg.infer.tile_size = 128
    cfg.infer.runtime = "torch"
    cfg.infer.precision = "fp32"
    cfg.infer.enable_cache = False
    cfg.infer.device = "cpu"
    return cfg


__all__ = [
    # taxonomy / palettes
    "LULC_CLASSES",
    "NUM_LULC_CLASSES",
    "LULC_NAME_TO_INDEX",
    "IGNORE_INDEX",
    "DEFAULT_SRGB_PALETTE",
    "DEFAULT_LAB_PALETTE",
    "HARD_PRIOR_CLASSES",
    "class_index",
    "class_name",
    # config sections
    "DataConfig",
    "SRConfig",
    "ColorizationConfig",
    "SemanticConfig",
    "ModelConfig",
    "LossConfig",
    "TrainConfig",
    "InferConfig",
    "EvalConfig",
    "Config",
    # constructors
    "default_config",
    "synthetic_demo_config",
]
