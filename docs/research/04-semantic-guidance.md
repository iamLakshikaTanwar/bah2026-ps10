# 04 — Semantic Guidance, Color-Consistency & No-Hallucination for IR→RGB Colorization

**Project:** BAH 2026 PS-10 — Infrared image colorization and enhancement for improved object interpretation
**This document:** (A) land-cover/semantic data + segmentation models to use as semantic guidance and as a no-hallucination constraint; (B) downstream detection/segmentation datasets to *prove* the colorized output boosts accuracy; (C) a concrete semantic-conditioning + color-consistency + no-hallucination mechanism, with an O(1) color-LUT scheme.

**Method note:** Web research (WebSearch + WebFetch, 2023–2025 SOTA) was the primary source; exact GEE asset IDs and class tables were verified against the Earth Engine Data Catalog. Items that could not be web-verified and rely on the model's training knowledge (cutoff Jan 2026) are labelled `[from internal knowledge]`. Sources are listed at the end.

> **Why semantic guidance matters here.** IR→RGB colorization is *one-to-many and ill-posed*: a single thermal/IR value is consistent with many colors. The grader explicitly rewards "no hallucinations" and "downstream detection/segmentation boost". The robust way to satisfy both is to **stop predicting color freely and instead predict color conditioned on a land-cover class label per pixel**, then **clamp** the predicted color into a per-class plausible range via a Look-Up Table. Land-cover is the right conditioning signal because the target categories (water→blue, vegetation→green, built-up→gray) *are* land-cover classes.

---

## Part A — Land-cover / semantic data (guidance + no-hallucination constraint)

These provide (i) per-pixel class labels to condition generation and (ii) ground-truth class identity to verify the output did not change semantics. For Landsat 8/9 tiles (the PS-10 training source), all GEE products below can be sampled at the tile footprint and reprojected with `rasterio`/GDAL.

### A.1 Global 10 m products (primary — best match to PS-10)

| Dataset | GEE / STAC ID | Res. | Classes | Role in our pipeline |
|---|---|---|---|---|
| **ESA WorldCover v200 (2021)** | `ESA/WorldCover/v200` (band `Map`); v100 = `ESA/WorldCover/v100` (2020) | 10 m | **11**: 10 Tree cover, 20 Shrubland, 30 Grassland, 40 Cropland, 50 Built-up, 60 Bare/sparse veg, 70 Snow/ice, 80 Permanent water, 90 Herbaceous wetland, 95 Mangroves, 100 Moss/lichen | **Primary static label map.** Ships an official hex palette (see Color-LUT). Stable, global, one-pass. |
| **Google Dynamic World V1** | `GOOGLE/DYNAMICWORLD/V1` (ImageCollection) | 10 m | **9** label band (0 water,1 trees,2 grass,3 flooded_veg,4 crops,5 shrub_and_scrub,6 built,7 bare,8 snow_and_ice) **+ 9 per-class probability bands** | **Primary near-real-time label + uncertainty.** The probability bands give a free **confidence/uncertainty map** (1 − max-prob) for flagging unreliable regions. Temporally matchable to the acquisition date. |
| **ESRI / Impact Observatory 10 m Annual LULC (2017–2025)** | `projects/sat-io/open-datasets/landcover/ESRI_Global-LULC_10m_TS` (awesome-gee-community-catalog) | 10 m | **9** (water, trees, flooded veg, crops, built area, bare, snow/ice, clouds, rangeland) | Cross-check / ensemble vote against WorldCover + Dynamic World. |

### A.2 Regional / thematic products (cross-checks, specialized masks)

| Dataset | GEE / source ID | Res. | Classes / contents | Role |
|---|---|---|---|---|
| **JRC Global Surface Water (Pekel et al. 2016)** | `JRC/GSW1_4/GlobalSurfaceWater` (latest; `_1_4`/`_1_2` historic), bands incl. `occurrence`, `seasonality`, `recurrence`, `transition`, `max_extent` | 30 m | Water occurrence % (0–100) | **Authoritative water mask** → hard prior "this pixel is water → blue". `max_extent` = binary ever-water. |
| **MODIS MCD12Q1.061** | `MODIS/061/MCD12Q1` (band `LC_Type1` = IGBP) | 500 m | **17 IGBP** (11 natural, 3 developed/mixed, 3 non-vegetated); also UMD, LAI, BGC, PFT schemes | Coarse fallback where 10 m maps unavailable; long temporal record. |
| **Copernicus Global Land Cover (CGLS-LC100)** | `COPERNICUS/Landcover/100m/Proba-V-C3/Global` | 100 m | 23 classes + per-class cover fractions | Fractional cover priors (soft labels) for mixed pixels. |
| **CORINE Land Cover (Europe)** | `COPERNICUS/CORINE/V20/100m` | 100 m | **44** classes (3-level hierarchy) | High-detail EU validation region. `[from internal knowledge]` for class count granularity. |
| **NLCD (CONUS)** | `USGS/NLCD_RELEASES/2021_REL/NLCD` (Annual NLCD also available) | 30 m | ~16 Anderson-style classes (open water, developed low/med/high, deciduous/evergreen/mixed forest, shrub, grassland, pasture, crops, wetlands…) | High-quality US validation labels (Landsat-native, perfectly co-registered to PS-10 Landsat). |
| **Chesapeake Conservancy / "Open" high-res landcover** | `projects/sat-io/open-datasets/...` (community) | 1 m | ~6–13 classes (water, tree canopy, low veg, barren, impervious roads/surfaces) | Very high-res supervision for fine built-up/road structure. `[from internal knowledge]`. |
| **ESA WorldCereal** | `ESA-WorldCereal` STAC / `projects/...` (community mirrors) | 10 m | Temporary crops, maize, winter/spring cereals (binary seasonal maps) | Refines the "cropland" class (seasonally), avoiding green-on-bare-soil errors. |
| **ISRO Bhuvan LULC (India)** | NRSC Bhuvan Thematic Services (WMS/downloads; not native GEE) | 1:50k (≈ pixel ~tens of m); also 1:10k SIS-DP at 5.8 m LISS-IV | **Level-I: 8, Level-II: 31, Level-III: 54** classes; built-up/agriculture/natural | **India AOI label source** (mandatory if jury expects Indian scenes). Years 2005-06/2010-11/2015-16. |
| **OpenStreetMap (roads/buildings/water)** | OSM via `osmnx`/Overpass; rasterize with `rasterio.features.rasterize` | vector | `highway=*` (roads), `building=*`, `natural=water`/`waterway=*` | **Sharp structural masks** for roads/buildings/water that raster LULC misses; great for edge/structure constraints and built-up color. |

**Co-registration note (PS-10-specific):** PS-10 trains on Landsat 8/9 (30 m optical, 100 m TIR resampled to 30 m). NLCD (30 m, Landsat-native) and JRC GSW align cleanly; 10 m products (WorldCover, Dynamic World) should be `reproject`-ed/aggregated to the Landsat grid (majority filter) to make the per-pixel label map. Use `GDAL`/`rasterio` `WarpedVRT` with nearest-neighbour for categorical layers.

---

## Part B — Segmentation models (semantic guidance + the no-hallucination "checker")

Two distinct uses:
1. **Guidance encoder** — produces the conditioning map fed into the generator.
2. **Consistency checker** — a *frozen* segmenter run on the colorized RGB output; its prediction must match the input-derived labels (segmentation-consistency loss). Using a different architecture for (1) vs (2) avoids the model gaming its own checker.

### B.1 General-purpose segmentation

| Model | Year | Type | Notes for us |
|---|---|---|---|
| **SAM** (Segment Anything) | 2023 | Promptable, class-agnostic masks | Region proposals / instance masks; no semantics on its own. |
| **SAM 2** | 2024 | Promptable image+video, streaming memory | **Hiera-B+ ~130 FPS @1024², ~6× faster than SAM for images**, 3× fewer interactions. Best for fast, tile-scale instance masks; pairs well with the O(1) angle. |
| **SegFormer** (B0–B5) | 2021 | MiT transformer encoder + lightweight MLP decoder | Strong speed/accuracy; B4 ≈ 50.3% mIoU ADE20K @64M params. **Recommended lightweight guidance/checker.** |
| **Mask2Former** | 2022 | Masked-attention universal (sem/inst/pan) | SOTA-class accuracy; heavier. Good high-accuracy checker. |
| **UPerNet** (+Swin) | 2018/21 | Unified perceptual parsing, multi-scale | Repeatedly cited as top performer on RS LULC with Swin backbone. |
| **DeepLabv3+** | 2018 | Atrous + encoder-decoder | Reliable baseline, easy to train on custom LULC. |
| **U-Net / U-Net++** | 2015/18 | Encoder-decoder, skip connections | Default for satellite LULC; cheap, strong on small datasets; U-Net ensembles competitive on RS. |
| **HRNet** | 2019 | High-res parallel branches | Excellent for fine roads/edges (built-up). |
| **PSPNet** | 2017 | Pyramid pooling | Classic baseline. |

### B.2 Remote-sensing / EO foundation models (best for satellite LULC)

| Model | Year | Backbone / data | Why relevant |
|---|---|---|---|
| **Prithvi-EO-2.0** (IBM-NASA) | 2024 | ViT-MAE, 300M & **600M**; HLS (Landsat+Sentinel-2), ~4.2M samples, multi-temporal | **Top pick for Landsat-native LULC.** **75.6% avg on GEO-Bench (+8% over v1)**; HLS pretraining = same sensors as PS-10. 300M/600M checkpoints on HF. |
| **Clay v1** | 2024 | MAE-ViT, ~70M chips, 7 sensors | Open, multi-sensor; strong general EO embeddings. |
| **Scale-MAE** | 2023 (ICCV) | Scale-aware MAE, GSD-aware positional encoding | Best when fusing IR tiles across resolutions/scales. |
| **SatMAE / SatMAE++** | 2022/24 | MAE w/ spectral + temporal embeddings | Multispectral-aware pretraining. |
| **DOFA** | 2024 | Dynamic one-for-all; adapts to *any* #channels/modalities | Unique: can ingest variable IR/MS band counts directly. |
| **SpectralGPT** | 2024 | Spectral transformer (3D) | Spectral-cube modelling for multi-band IR. `[from internal knowledge]` for exact specs. |

**Recommendation (models):** Guidance = **Prithvi-EO-2.0 (300M) fine-tuned on the WorldCover label scheme** (Landsat-native, best LULC). Lightweight/real-time alternative and the **independent consistency checker** = **SegFormer-B2/B3** (different family ⇒ unbiased check). **SAM 2** for fast instance/region masks where sharp object boundaries help (built-up, vehicles).

---

## Part C — Downstream detection/segmentation (to *prove* the boost)

Protocol: run a **frozen** detector/segmenter on (i) raw IR, (ii) naive colorized, (iii) our semantically-constrained colorized; report deltas. Improvement on (iii) over (i)/(ii) is the proof.

### C.1 Detectors / segmenters to evaluate with

| Family | Members | Use |
|---|---|---|
| **YOLO** | YOLOv8 / v9 / v10 / **v11** (and YOLOv11-RGBT for RGB-T fusion) | Fast oriented/HBB detection on aerial tiles; primary downstream metric. |
| **DETR family** | DETR, **DINO-DETR**, Deformable-DETR | Transformer detection; robust mAP reference. |
| **Faster R-CNN** | + FPN | Classic two-stage baseline. |
| **Segmenters** | Mask2Former / SegFormer / DeepLabv3+ | Downstream *segmentation* boost (iSAID, SpaceNet, Inria). |

### C.2 Aerial/satellite detection & segmentation datasets

| Dataset | Task | Scale / classes | Use |
|---|---|---|---|
| **DOTA** (v1/v1.5/v2) | Oriented OD | **11,268 images, ~1.79M instances, 18 categories**, OBB | Headline oriented-detection benchmark. |
| **iSAID** | Instance seg | Instance-seg extension of DOTA-v1.0, 15 classes | Downstream **segmentation** proof. |
| **DIOR** | OD (HBB) | 23,463 images, 20 classes, ~192k instances | Broad-class detection. |
| **FAIR1M** | Fine-grained OBB | >1M instances, 5 categories/37 sub-classes | Fine-grained boost test. |
| **xView** | OD | 1,413 images, 60 fine classes, ~1M instances | Large-scale, small objects. |
| **VEDAI** | Vehicle OD | Aerial vehicles, multi-class, includes IR channel | **Vehicle detection + has IR** — ideal IR↔RGB ablation. |
| **SpaceNet** (2/3/5…) | Buildings/roads | WV-2/3 @0.3–0.5 m; SN2 = Vegas/Paris/Shanghai/Khartoum, 650² | Building/road **segmentation** boost. |
| **Inria Aerial** | Building seg | 360 tiles, 10 cities, 0.3 m, building/not-building; train/test cities disjoint | Cross-city **generalization** test. |

### C.3 Thermal/IR RGB-paired detection datasets (most on-topic for IR→RGB)

| Dataset | Pairing | Scale / classes | Use |
|---|---|---|---|
| **LLVIP** | RGB+IR, strictly aligned | **15,488 pairs (30,976 imgs)**, low-light, pedestrians | **Best aligned IR↔RGB pair set** for colorization + detection eval. |
| **M3FD** | RGB+IR | **4,200 pairs, 6 classes** (people/cars/trucks…), adverse weather | Fusion + detection; adverse-condition robustness. |
| **FLIR ADAS** | Thermal (+ref RGB) | >26k imgs, ~520k boxes, ~15 classes | Standard thermal-detection benchmark. |
| **KAIST Multispectral** | Color+thermal, beam-split aligned | 95,328 pairs @20 fps | Pedestrian, day/night; large aligned corpus. |

> For PS-10, **VEDAI** (aerial + IR) and **LLVIP** (aligned IR↔RGB) are the two most directly usable to *quantify* the colorization-driven detection boost. Aerial LULC datasets (DOTA/iSAID/SpaceNet) prove the satellite-scale segmentation boost.

---

## Mechanisms

### 1) Semantic-conditioned generation

| Mechanism | How it conditions on the label map | Fit for IR→RGB |
|---|---|---|
| **SPADE** (spatially-adaptive normalization, NVlabs 2019) | Learns per-pixel affine (γ,β) **from the segmentation map**, injected at every decoder block, so semantic info isn't "washed out" by normalization (which is exactly pix2pixHD's failure on flat masks). | **Primary recommendation.** Inject the per-pixel land-cover label so each class gets its own normalization → class-consistent color by construction. |
| **OASIS** (2021) | Discriminator is itself a **semantic-segmentation network** → strong label-alignment, high diversity, no need for VGG perceptual loss. | Strong alternative/complement; its segmenter-discriminator *is* a built-in semantic-consistency signal. `[from internal knowledge]`. |
| **Pix2PixHD (seg-conditioned)** | Concatenate label map (and instance/edge maps) as input; coarse-to-fine G, multi-scale D. | Solid baseline; **inferior to SPADE on uniform regions** (water, fields) per the SPADE paper. |
| **ControlNet (segmentation conditioning)** | Adds a trainable copy of a frozen diffusion UNet conditioned on a seg map; high-fidelity layout control. | Use if moving to a diffusion backbone; strongest spatial control, heaviest compute. |
| **Label-guided diffusion** | Class map as conditioning channel / cross-attention key. | Highest realism (good FID), slowest inference — tension with the PS-10 inference-time metric. |

### 2) Color-consistency enforcement (the O(1) core)

- **Per-class color priors via Look-Up Table (LUT) — genuine O(1).** Precompute, once, a table `class_id → (μ_Lab, Σ_Lab, [lo,hi] per channel)` from real RGB statistics per land-cover class (e.g., from co-located Sentinel-2/Landsat RGB over WorldCover labels). At inference, every pixel does a single array index `LUT[class_id]` (O(1), hashable; vectorizes to one gather over the whole tile). Seed ranges from the **official WorldCover palette** (below) and widen using measured per-class color distributions so output stays natural, not flat.
- **Histogram matching to class palettes.** Per class, match the output color histogram to the class's reference palette (e.g., `skimage.exposure.match_histograms` applied per-class-mask) so global tone is realistic while staying in-class.
- **Constrained / out-of-class penalty loss.** `L_color = Σ_p w(class_p) · dist(color_p, range[class_p])` — penalize colors outside the class range (e.g., distance outside `[μ−kσ, μ+kσ]` in Lab). Heaviest weight on hard priors (water, dense vegetation).

### 3) No-hallucination guarantees

- **Segmentation-consistency loss (key anti-hallucination signal).** Run a **frozen** segmenter `S` on the generated RGB; require `S(G(IR)) ≈ L_input` (the input-derived label map). Cross-entropy/Dice between predicted and reference labels. If colorization invents an object, `S` mislabels it → loss rises. Use a *different* architecture than the guidance encoder (e.g., guidance = Prithvi/SPADE, checker = SegFormer) so the generator can't trivially fool its own segmenter.
- **Structural / edge preservation.** Gradient/edge loss (Sobel or learned) + **SSIM** between IR structure and output luminance ensures geometry (roads, building edges) is preserved — colorization may add chroma but must not move/erase structure. Aligns with the PS-10 SSIM metric.
- **Cycle / identity constraints.** CycleGAN-style cycle loss (RGB→IR' must reconstruct input IR) + identity loss on already-color-like inputs; bounds the mapping and curbs fabrication. (PS-10 explicitly lists CycleGAN/Pix2Pix as baselines.)
- **Uncertainty maps flagging unreliable regions.** Use **Dynamic World per-class probabilities** (1 − max-prob) and/or MC-dropout/ensemble variance of the generator. High-uncertainty pixels are (a) down-weighted in the color loss and (b) rendered desaturated/greyscale + flagged in a QA overlay → "honest" output instead of confident hallucination. Directly answers the grader's hallucination concern.

### 4) Downstream-aware training

- **Detection/segmentation perceptual loss (task-driven).** Following *Task-Driven Perceptual Loss* (CVPR 2024) and frozen-detector fine-tuning: pass `G(IR)` through a **frozen** detector/segmenter and add a feature-matching loss vs. the GT-RGB pass, so the generator restores **task-relevant high-frequency content** (not just pixel similarity). Optional CQMix/alternate-training trick mitigates domain gap.
- **Task-driven evaluation protocol** (the proof): freeze one detector + one segmenter; evaluate raw-IR vs naive-color vs ours; report mAP / mIoU deltas (see protocol below).

---

## Recommended semantic stack (concrete, for PS-10)

```
                 ┌──────────────────────────── conditioning ─────────────────────────────┐
 IR tile ──► SR/sharpen module ──► Generator G (SPADE decoder) ──► colorized RGB ─┐
   │                                     ▲                                         │
   │                      per-pixel land-cover label map L                         │
   │            (WorldCover v200 ⊕ Dynamic World ⊕ JRC-water, majority-voted,      │
   │             reprojected to Landsat grid; Prithvi-EO-2.0 fills gaps)           │
   │                                     │                                         │
   │                              O(1) Color-LUT clamp  ◄── LUT[class]→Lab range   │
   │                                                                               ▼
   └─────────────────────────────── losses ───────────────────────────────────────┤
        L = λ1·L1/percep(vs GT-RGB)  + λ2·L_color(out-of-class, LUT)               │
          + λ3·L_segcons( SegFormer(G(IR)) vs L )      ← no-hallucination          │
          + λ4·L_edge/SSIM(IR, G(IR))                  ← structure preserve         │
          + λ5·L_cycle(RGB→IR'→IR)                     ← anti-fabrication           │
          + λ6·L_taskperc( frozen YOLOv11/Mask2Former feats )  ← downstream-aware   │
        uncertainty U = 1 − max DynamicWorld prob → desaturate + QA flag           │
```

1. **Label map L:** majority-vote of **ESA WorldCover v200** (`ESA/WorldCover/v200`) ⊕ **Dynamic World** (`GOOGLE/DYNAMICWORLD/V1`, date-matched) ⊕ **JRC GSW** water hard-prior, all reprojected to the Landsat grid; gaps/edges refined by a **Prithvi-EO-2.0 (300M)** segmentation head fine-tuned to the WorldCover scheme.
2. **Generator:** SR/sharpen front-end → **SPADE**-conditioned decoder (label map drives per-pixel normalization). (ControlNet-seg only if a diffusion backbone is adopted later.)
3. **Color-LUT clamp:** O(1) `class→Lab range` table applied post-decoder (details below).
4. **Checker:** frozen **SegFormer-B2** for the segmentation-consistency loss (different family from Prithvi).
5. **Uncertainty:** Dynamic World probabilities → desaturate + flag low-confidence pixels.

---

## Color-LUT design (O(1) class→color)

**Build (offline, once):**
1. Sample co-located real RGB (Landsat/Sentinel-2 surface reflectance, stretched) over each **WorldCover class** across many tiles/biomes.
2. Per class, compute color distribution in **CIE-Lab** (perceptually uniform): mean `μ`, covariance `Σ`, and clamp range `[μ−kσ, μ+kσ]` per channel (k≈2). Store as a fixed array indexed by class id.
3. Seed/sanity-anchor with the **official ESA WorldCover palette**:

| Class (value) | WorldCover hex | Plausible-color intent |
|---|---|---|
| Tree cover (10) | `#006400` | deep green |
| Shrubland (20) | `#ffbb22` | olive/khaki |
| Grassland (30) | `#ffff4c` | yellow-green |
| Cropland (40) | `#f096ff` | green↔tan (season-aware via WorldCereal) |
| Built-up (50) | `#fa0000` (legend) → **render gray** | gray/desaturated |
| Bare/sparse (60) | `#b4b4b4` | gray-tan |
| Snow/ice (70) | `#f0f0f0` | white |
| Permanent water (80) | `#0064c8` | blue |
| Herbaceous wetland (90) | `#0096a0` | teal |
| Mangroves (95) | `#00cf75` | blue-green |
| Moss/lichen (100) | `#fae6a0` | pale tan |

> Note: the WorldCover **legend** hex (e.g., built-up = red, cropland = magenta) is for *map visualization*, **not** natural color. The LUT's *plausible-color* ranges (right column) come from **measured real RGB statistics**, with the legend used only to anchor class identity. (Water→blue, vegetation→green, built-up→gray map directly to the PS-10 examples.)

**Apply (inference, O(1) per pixel):**
```
lab_out = generator_lab                      # raw prediction in Lab
lo, hi  = LUT[L]                             # single gather, O(1)/pixel (vectorized over tile)
lab_out = clip(lab_out, lo, hi)             # keep chroma in-class, free luminance for texture
# optional: per-class histogram match to class palette, then Lab→RGB
```
- **Clamp `a,b` (chroma), leave `L` (luminance) freer** → colors stay class-correct while SR/texture detail survives.
- Pure table lookup + elementwise clip ⇒ **true O(1) per pixel**, hash/array indexable, GPU-vectorized over the whole tile; adds negligible inference time (helps the PS-10 inference-time metric).
- Hard-prior override: where **JRC GSW `max_extent`/occurrence>τ** ⇒ force water LUT regardless of generator (guarantees water→blue).

---

## Downstream evaluation protocol (the "proof of boost")

**Setup.** Freeze: one detector (**YOLOv11**) + one segmenter (**Mask2Former** or **SegFormer**). Three input conditions per test tile:
(i) raw IR (1-ch → 3-ch replicated), (ii) naive colorization (no semantic constraint), (iii) **ours** (SPADE + LUT + consistency).

**Datasets.**
- IR↔RGB paired detection: **VEDAI** (aerial, has IR), **LLVIP**, **M3FD**, **FLIR ADAS**.
- Aerial detection: **DOTA**, **DIOR**, **FAIR1M**.
- Aerial segmentation: **iSAID**, **SpaceNet**, **Inria Aerial**.

**Metrics.**
- *Detection:* mAP@0.5, mAP@[.5:.95], per-class AP (esp. vehicles/buildings); report **Δ(iii−i)** and **Δ(iii−ii)**.
- *Segmentation:* mIoU, per-class IoU (water/veg/built-up), boundary F-score.
- *Image quality (PS-10 required):* **PSNR, SSIM, FID** vs GT-RGB.
- *Anti-hallucination (quantified):* **segmentation-consistency score** = mIoU between `S(output)` and `L_input` (higher = fewer invented/altered semantics); plus % pixels flagged by uncertainty map.
- *Speed (PS-10 required):* **inference time per tile** (must hold up after LUT — it will, since LUT is O(1)).

**Headline claim to demonstrate:** ours yields **higher mAP/mIoU than raw IR and naive colorization**, **higher segmentation-consistency** (fewer hallucinations), with **negligible added latency** from the O(1) LUT.

---

## Sources

- ESA WorldCover v200 — Earth Engine Data Catalog: https://developers.google.com/earth-engine/datasets/catalog/ESA_WorldCover_v200 ; project: https://esa-worldcover.org/en/data-access ; Zenodo: https://zenodo.org/records/7254221
- Dynamic World V1 — Earth Engine: https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_DYNAMICWORLD_V1 ; Nature Sci. Data: https://www.nature.com/articles/s41597-022-01307-4
- ESRI/Impact Observatory 10 m LULC — community catalog: https://gee-community-catalog.org/projects/S2TSLULC/
- JRC Global Surface Water — Earth Engine: https://developers.google.com/earth-engine/datasets/catalog/JRC_GSW1_2_GlobalSurfaceWater (Pekel et al., Nature 2016)
- MODIS MCD12Q1.061 — Earth Engine: https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MCD12Q1 ; User Guide: https://lpdaac.usgs.gov/documents/438/MCD12Q1_User_Guide_V51.pdf
- ISRO Bhuvan LULC — NRSC: https://bhuvan-app1.nrsc.gov.in/2dresources/thematic/LULC503/lulc.pdf ; https://www.nrsc.gov.in/EO_LULC_Application/
- SAM 2 — arXiv 2408.00714: https://arxiv.org/abs/2408.00714 ; Ultralytics: https://docs.ultralytics.com/models/sam-2 ; survey: https://arxiv.org/pdf/2503.12781
- Prithvi-EO-2.0 — arXiv 2412.02732: https://arxiv.org/abs/2412.02732 ; HF: https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M ; IBM: https://research.ibm.com/blog/prithvi2-geospatial
- Scale-MAE — ICCV 2023: https://openaccess.thecvf.com/content/ICCV2023/papers/Reed_Scale-MAE_A_Scale-Aware_Masked_Autoencoder_for_Multiscale_Geospatial_Representation_Learning_ICCV_2023_paper.pdf ; Clay: https://developmentseed.org/projects/clay/ ; DOFA-CLIP: https://arxiv.org/html/2503.06312v2 ; RS-FM list: https://github.com/Jack-bo1220/Awesome-Remote-Sensing-Foundation-Models
- SPADE — arXiv 1903.07291: https://arxiv.org/pdf/1903.07291 ; project: https://nvlabs.github.io/SPADE/
- SegFormer / Mask2Former / UPerNet for RS — https://arxiv.org/html/2410.01092v1 ; https://www.mdpi.com/2227-7390/12/5/765 ; https://www.mdpi.com/2072-4292/16/12/2077
- DOTA — arXiv 2102.12219: https://arxiv.org/pdf/2102.12219 ; project: https://captain-whu.github.io/DOTA/ ; aerial-OD list: https://github.com/visionxiang/awesome-object-detection-in-aerial-images
- SpaceNet / Inria — https://www.kaggle.com/datasets/sagar100rathod/inria-aerial-image-labeling-dataset ; sat-imagery DL: https://github.com/satellite-image-deep-learning
- LLVIP — arXiv 2108.10831: https://arxiv.org/html/2108.10831 ; multispectral-detection resource: https://github.com/CalayZhou/Multispectral-Pedestrian-Detection-Resource ; YOLOv11-RGBT: https://www.arxiv.org/pdf/2506.14696v1
- Task-Driven Perceptual Loss — CVPR 2024: https://openaccess.thecvf.com/content/CVPR2024/papers/Kim_Beyond_Image_Super-Resolution_for_Image_Recognition_with_Task-Driven_Perceptual_Loss_CVPR_2024_paper.pdf (arXiv 2404.01692)
- Semantic/structured consistency for segmentation: https://arxiv.org/pdf/2001.04647 ; pixelated semantic colorization: https://link.springer.com/article/10.1007/s11263-019-01271-4
