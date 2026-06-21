# 01 — Dataset & Sensor Catalog for IR → RGB Colorization + Super-Resolution

> **Project:** BAH 2026 PS-10 — *Infrared image colorization and enhancement for improved object interpretation.*
> **Goal of this document:** an exhaustive, cross-verified catalog of satellite/sensor datasets usable to **train and validate** an IR → RGB colorization + super-resolution model, plus a concrete **pairing + co-registration recipe** and an **O(1) fast-access** analysis for the data layer.
> **Author:** DataScout · **Date:** 2026-06-21 · **Sources:** primarily live web research (2024–2025 catalog pages); a few items fall back to internal knowledge and are explicitly tagged `[from internal knowledge]`.
> **Coverage:** **31 distinct data sources** (13 sensor families + 18 benchmark/curated datasets and access platforms). See the numbered summary at the end.

---

## 0. TL;DR — How to read this for the model

The problem statement *names Landsat 8/9 as the required dataset*, but a robust IR→RGB model needs **many** IR↔RGB pairings to fuse and cross-verify against. The architecture should:

1. **Train the colorization head** mostly on *intra-sensor* IR↔RGB pairs where IR and RGB are captured by the **same instrument at the same instant** (perfect co-registration, no temporal gap): **ASTER (VNIR↔SWIR↔TIR)**, **Landsat OLI+TIRS**, **Sentinel-3 SLSTR (vis↔thermal)**, **MODIS**, **VIIRS**, **INSAT-3D/3DR**, **GOES/Himawari**. These give *physically consistent* IR/RGB without registration error.
2. **Train the super-resolution head** on *cross-sensor* pairs where a **low-res IR/source** is matched to a **high-res RGB target**: **Landsat TIRS (100 m) → Sentinel-2 (10 m)**, **WorldStrat (S2 10 m → SPOT 1.5 m)**, **PRISMA/EMIT (30–60 m) → Sentinel-2 (10 m)**.
3. **Use ground-truth benchmark pairs** (LLVIP, KAIST, M3FD, FLIR, SEN12MS) for sanity-checking the *visible↔infrared translation* objective and for the downstream-detection metric.

Key practical truth: **Landsat is the only single platform that natively carries good RGB (OLI) AND thermal TIR (TIRS) co-registered on the same satellite** — making it the backbone for this PS. Everything else fuses around it.

---

## 1. Landsat 8 / 9 (OLI + TIRS) — **primary backbone**

Landsat 8 (2013) and Landsat 9 (2021) each carry **OLI** (Operational Land Imager, reflective/visible/NIR/SWIR) + **TIRS** (Thermal Infrared Sensor, 2 thermal bands). They are co-registered on the same platform, 8 days out of phase → effective ~8-day revisit for the constellation. **This is the canonical IR↔RGB co-registered source named in the PS.**

### 1.1 Bands — Collection 2 Level 2 (Surface Reflectance + Surface Temperature)

| GEE band | Description | Wavelength (µm) | Native res | Notes |
|---|---|---|---|---|
| `SR_B1` | Ultra-blue / coastal aerosol | 0.435–0.451 | 30 m | reflectance |
| `SR_B2` | **Blue** | 0.452–0.512 | 30 m | RGB-B |
| `SR_B3` | **Green** | 0.533–0.590 | 30 m | RGB-G |
| `SR_B4` | **Red** | 0.636–0.673 | 30 m | RGB-R |
| `SR_B5` | NIR | 0.851–0.879 | 30 m | IR |
| `SR_B6` | SWIR-1 | 1.566–1.651 | 30 m | IR |
| `SR_B7` | SWIR-2 | 2.107–2.294 | 30 m | IR |
| `ST_B10` | **Surface temperature (TIRS-1)** | 10.60–11.19 | 30 m (resampled; TIRS native 100 m) | **thermal IR target/source** |

Scale factors (apply before use): SR bands `DN*0.0000275 - 0.2` → reflectance; `ST_B10` `DN*0.00341802 + 149` → Kelvin. Extra QA/intermediate bands: `SR_QA_AEROSOL, ST_QA, ST_EMIS, ST_TRAD, ST_URAD, QA_PIXEL, QA_RADSAT`.

> **L1 (TOA / radiance)** keeps the **two** raw thermal bands separately: **B10 (10.6–11.19 µm)** and **B11 (11.5–12.51 µm)**, plus a 15 m **panchromatic B8 (0.503–0.676 µm)** that is extremely useful for pan-sharpening RGB to 15 m. TIRS native resolution is **100 m**, resampled/delivered at 30 m.

### 1.2 Access — exact collection IDs

| Platform | Collection ID | Level | License |
|---|---|---|---|
| **Google Earth Engine** | `LANDSAT/LC08/C02/T1_L2`, `LANDSAT/LC09/C02/T1_L2` | C2 L2 SR+ST (Tier 1) | Public domain (USGS) |
| GEE (TOA, has B10+B11+pan) | `LANDSAT/LC08/C02/T1_TOA`, `LANDSAT/LC09/C02/T1_TOA` | C2 L1 TOA | Public domain |
| GEE (raw DN) | `LANDSAT/LC08/C02/T1`, `…/T2` | C2 L1 | Public domain |
| **AWS Open Data** | `s3://usgs-landsat/collection02/` (requester-pays, COG) | C2 L1/L2 | Public domain |
| **Microsoft Planetary Computer** | STAC collections `landsat-c2-l2`, `landsat-c2-l1` | C2 | Public domain |
| **USGS EarthExplorer / M2M API** | "Landsat 8-9 OLI/TIRS C2 L1 / L2" | all | Public domain |

**Co-registration note:** OLI and TIRS are pre-aligned in C2 L1T/L2 products (terrain-corrected, sub-pixel geolocation). For IR↔RGB you can treat `SR_B4/B3/B2` (RGB) and `ST_B10` (thermal) as **already pixel-aligned** in the same asset — *no extra registration needed*. This is the single biggest reason Landsat is the backbone.

---

## 2. Sentinel-2 MSI — **high-res RGB super-resolution target**

Sentinel-2A/B/C MSI: 13 bands, 10/20/60 m, 5-day revisit (2-sat constellation), available from 2017-03-28. **No thermal**, but the **best free 10 m RGB + NIR** → ideal *target* for super-resolving Landsat-thermal-derived RGB, and for cross-sensor pairing.

| Band | Name | Center (nm) | Res | Role |
|---|---|---|---|---|
| B2 | **Blue** | 490 | 10 m | RGB-B target |
| B3 | **Green** | 560 | 10 m | RGB-G target |
| B4 | **Red** | 665 | 10 m | RGB-R target |
| B8 | NIR | 842 | 10 m | IR |
| B5/B6/B7 | Red-edge | 705/740/783 | 20 m | IR |
| B8A | Narrow NIR | 865 | 20 m | IR |
| B11 / B12 | SWIR-1 / SWIR-2 | 1610 / 2190 | 20 m | IR |
| B1 / B9 / B10 | Aerosol / WV / Cirrus | 443 / 945 / 1375 | 60 m | atmos (B10 absent in L2A) |

**Access:** GEE `COPERNICUS/S2_SR_HARMONIZED` (L2A surface reflectance; bands scaled ×10000, no B10) and `COPERNICUS/S2_HARMONIZED` (L1C TOA). Planetary Computer `sentinel-2-l2a` (COG). AWS `s3://sentinel-s2-l2a` (requester-pays) + Element84 `sentinel-2-l2a` STAC. Copernicus Data Space Ecosystem (CDSE). **License:** free, open (Copernicus / CC-BY-like attribution). *Harmonized* fixes the Jan-2022 baseline-04.00 radiometric offset.

---

## 3. Sentinel-3 SLSTR (+ OLCI) — **intra-platform vis ↔ thermal pairs**

### 3.1 SLSTR (Sea & Land Surface Temperature Radiometer)
Carries **visible/SWIR AND thermal** on the same instrument → native IR↔RGB pairs (coarse, but perfectly co-registered). Dual-view (nadir + oblique). ~1–2 day revisit (S3A+S3B).

| Band | Center (µm) | Res | Type |
|---|---|---|---|
| S1 | 0.555 | 500 m | **vis (green)** |
| S2 | 0.659 | 500 m | **vis (red)** |
| S3 | 0.865 | 500 m | NIR |
| S4 | 1.375 | 500 m | cirrus |
| S5 | 1.61 | 500 m | SWIR |
| S6 | 2.25 | 500 m | SWIR |
| S7 | 3.74 | **1 km** | **MWIR (thermal)** |
| S8 | 10.85 | **1 km** | **TIR (thermal)** |
| S9 | 12.0 | **1 km** | **TIR (thermal)** |
| F1/F2 | 3.74 / 10.85 | 1 km | fire channels |

Vis/SWIR (S1–S6) delivered as TOA radiance; thermal (S7–S9, F1/F2) as TOA brightness temperature. **Access:** GEE `COPERNICUS/S3/OLCI` exists for OLCI; SLSTR via Copernicus Data Space / EUMETSAT / Planetary Computer `sentinel-3-slstr-lst-l2-netcdf` and `sentinel-3-slstr-wst-l2-netcdf`. **License:** free/open (Copernicus).

### 3.2 OLCI (Ocean & Land Colour Instrument)
21 bands, 400–1020 nm (400, 412, 442, 490, 510, 560, 620, 665, 674, 681, 709, 754, 779, 865, 885, 1024 nm), **300 m**, ~2-day global, swath 1270 km. MERIS heritage → rich **visible color**. **Access:** GEE `COPERNICUS/S3/OLCI` (EFR TOA radiances), Planetary Computer `sentinel-3-olci-lfr-l2-netcdf` / `sentinel-3-olci-efr-l1-netcdf`. **License:** free/open.

**Role:** SLSTR gives a clean *coarse* IR(S7/S8/S9)↔RGB(S1/S2/S3 + OLCI) supervision signal with zero registration error — good for pre-training the colorization head on physically-consistent data, then fine-tuning at higher res.

---

## 4. Sentinel-1 SAR — **all-weather complement to IR**

Dual-pol C-band SAR (5.405 GHz), day/night, all-weather. Not optical, but the PS explicitly motivates "night-time / adverse weather"; SAR is the canonical weather-independent modality and pairs well with optical (BigEarthNet/SEN12MS/SEN1-2 use S1↔S2 pairs).

- **Modes/res:** GRD scenes at **10 / 25 / 40 m**; polarizations VV, HH, VV+VH, HH+HV (+ incidence-angle band).
- **Access:** GEE `COPERNICUS/S1_GRD` (already thermal-noise-removed, radiometrically calibrated, terrain-corrected, dB-scaled). Planetary Computer `sentinel-1-grd` and **`sentinel-1-rtc`** (radiometric-terrain-corrected, analysis-ready). AWS `s3://sentinel-s1-l1c`. **License:** free/open (Copernicus). Updated daily.

**Role:** optional SAR→RGB or SAR-as-auxiliary-channel branch; strongest as a *fusion input* proving the night/all-weather story.

---

## 5. MODIS (Terra / Aqua) — **36-band thermal + visible, daily**

36 bands, 0.41–14.4 µm, daily global. Resolutions: **250 m (B1–2)**, **500 m (B3–7)**, **1 km (B8–36)**. 20 reflective solar bands (0.41–2.1 µm) + 16 thermal-emissive bands (>3.7 µm). Native vis↔thermal on one instrument.

| Product | GEE asset | Content |
|---|---|---|
| Surface reflectance daily | `MODIS/061/MOD09GA` (Terra), `MYD09GA` (Aqua) | B1–B7, 500 m + 1 km geometry |
| Surface reflectance bands | `MODIS/061/MOD09A1` | 8-day composite B1–B7 |
| **LST/Emissivity** | `MODIS/061/MOD11A1`, `MYD11A1` | **thermal (TIR) 1 km daily** |
| Calibrated radiances (all 36 bands) | `MODIS/MOD021KM` family (L1B, via LAADS) | full thermal incl. MWIR/LWIR |

**Access:** GEE (above), AWS Open Data `s3://modis-pds`, NASA LAADS/LP DAAC. **License:** public domain (NASA). **Role:** coarse but huge-volume IR↔RGB pre-training; LST product gives clean thermal target.

---

## 6. VIIRS (Suomi-NPP / NOAA-20 / NOAA-21) — **Day-Night Band + thermal**

22 bands: 16 **M-bands @ 750 m**, 5 **I-bands @ 375 m**, 1 **Day/Night Band (DNB) @ 750 m**. The **DNB** literally produces "visible-like" night imagery from moonlight/airglow/city lights — *directly relevant* to the PS's night-time IR→RGB motivation. Thermal: I4 (3.74 µm), I5 (11.45 µm), M12–M16 (3.7–12 µm).

| Product | GEE asset | Content |
|---|---|---|
| Surface reflectance daily | `NOAA/VIIRS/001/VNP09GA` | I & M reflectance, 375/750 m |
| **Nighttime lights / DNB** | `NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG` | **DNB radiance** |
| LST | `NASA/VIIRS/002/VNP21A1D` (day), `…A1N` (night) | **thermal LST** |

**Access:** GEE, LAADS DAAC (VNP02IMG L1B 375 m, VNP02MOD 750 m), LP DAAC. **License:** public domain (NASA/NOAA). **Role:** night DNB↔thermal is a *unique* supervision pair for the "interpret IR at night" objective.

---

## 7. ASTER (Terra) — **best intra-sensor VNIR ↔ SWIR ↔ TIR pairing**

14 bands on one instrument from VIS→TIR (0.52–11.65 µm), all co-registered, 60×60 km scenes. **The single best source for *high-res* intra-sensor IR↔RGB pairs** because it has visible AND thermal at moderate res with no temporal gap.

| Subsystem | Bands | Wavelength (µm) | Res |
|---|---|---|---|
| **VNIR** | B1 (green 0.52–0.60), B2 (red 0.63–0.69), B3N/B3B (NIR 0.76–0.86) | 0.52–0.86 | **15 m** |
| **SWIR** | B4–B9 | 1.60–2.43 | 30 m |
| **TIR** | B10–B14 | 8.125–11.65 | **90 m** |

**Access:** GEE `ASTER/AST_L1T_003` (calibrated at-sensor radiance, ortho/terrain-corrected — SWIR detectors failed after ~2008 so SWIR may be empty in later scenes). Surface products `AST_07` (SR VNIR/SWIR), `AST_05`/`AST_08` (emissivity/LST) at LP DAAC. NASA Earthdata / LP DAAC Cloud, USGS EarthExplorer. **License:** public domain (NASA/METI; some on-demand products need a (free) order). **Co-registration:** VNIR↔TIR intra-scene alignment is built in (TIR resampled to common grid). **Role:** *primary high-res colorization-training source alongside Landsat.*

---

## 8. ECOSTRESS (ISS) — **high-res thermal (70 m)**

PHyTIR imaging radiometer on ISS: **5 LWIR/TIR channels (8–12 µm) + 1 SWIR** at **70 m**, variable time-of-day, coverage 52°N–52°S. Highest-res *spaceborne thermal* widely available → great for thermal-SR targets and IR-source diversity.

- **GEE:** `NASA/ECOSTRESS/L2T_LSTE/V2` (LST&E, 70 m) — **note: as of catalog, only tiles covering the LA metro were ingested into GEE**; for global use go to LP DAAC.
- **LP DAAC:** `ECO_L2T_LSTE` v2 (tiled LST&E), `ECO_L2G_LSTE` v2 (gridded), `ECO_L1B_RAD` (radiance, 5 TIR bands). (V1 `ECO2LSTE` decommissioned 2025-05-21 → use V2.)
- **License:** public domain (NASA). **Pairing:** ECOSTRESS thermal ↔ same-area Sentinel-2/Landsat RGB (different platform → needs co-registration; ISS orbit gives varied local times — useful for day/night robustness).

---

## 9. EMIT (ISS) — **imaging spectroscopy, 285 bands**

Earth-surface Mineral dust source InvesTigation: VSWIR imaging spectrometer, **285 contiguous bands, 381–2493 nm, ~7.5 nm spectral, 60 m, 75 km swath**, launched 2022-07-14. You can synthesize *any* RGB and many NIR/SWIR "IR" bands from the same cube → **self-paired IR↔RGB with zero registration error** and full spectral control.

- **Access:** LP DAAC — `EMITL2ARFL` (L2A surface reflectance, 285 bands) and `EMITL1BRAD` (L1B radiance). Earthdata Cloud (S3, NetCDF). Not yet a standard GEE asset (community efforts exist). **License:** public domain (NASA).
- **Role:** generate *physically exact* IR↔RGB training pairs (slice red/green/blue vs NIR/SWIR from one cube) to teach the colorization head the true spectral mapping; excellent for the "no hallucination / semantic integrity" objective.

---

## 10. PRISMA (ASI) — **hyperspectral VNIR–SWIR + 5 m pan**

Italian hyperspectral mission (2019): **~240 bands, 400–2500 nm** (VNIR 400–1010 nm/66 bands; SWIR 920–2505 nm/173 bands), **30 m hyperspectral + co-registered 5 m panchromatic**, 30 km swath, <12 nm spectral resolution. Pan is *already co-registered* to the hypercube.

- **Access:** ASI PRISMA portal (free for registered research; user must request acquisitions/tasking). Community Earth-Engine ingests exist under `awesome-gee-community-catalog` (search "PRISMA"). **License:** open for research, registration + project required.
- **Role:** like EMIT, gives self-consistent IR↔RGB pairs at higher spatial res; the **5 m pan** enables true hyperspectral→pan **super-resolution** supervision (sharpen 30 m to 5 m). Hyper-sharpening with Sentinel-2 is a documented technique.

---

## 11. Geostationary IR + Visible — GOES-R ABI, Himawari-8/9 AHI, Meteosat SEVIRI

Continuous (10-min) full-disk imaging; native vis↔IR on one instrument; huge temporal volume; ideal for *coarse, perfectly-registered, day↔night-paired* colorization pre-training and temporal augmentation.

| Sensor | Bands | Vis/NIR | IR | Res | GEE / access |
|---|---|---|---|---|---|
| **GOES-R ABI** (16, 17, 18, 19) | 16 | 2 vis + 4 NIR (0.47–2.2 µm) | 10 IR (3.9–13.3 µm) | 0.5 km (red B2) / 1 km / **2 km IR** | GEE `NOAA/GOES/16/MCMIPF`, `…/16/MCMIPC`, `…/18/MCMIPF`, `…/18/MCMIPC` (Cloud & Moisture Imagery, all 16 CMI bands); AWS `s3://noaa-goes16`, `noaa-goes18` |
| **Himawari-8/9 AHI** | 16 | 3 vis + 3 NIR | 10 IR | 0.5–2 km | AWS `s3://noaa-himawari8/9`; JMA; GEE community `[from internal knowledge]` |
| **Meteosat SEVIRI** (MSG/MTG) | 12 (SEVIRI) | HRV vis 1 km + vis/NIR | several IR incl. 10.8 µm | 3 km (1 km HRV) | EUMETSAT Data Store; AWS `s3://eumetsat-…` `[from internal knowledge]` |

**License:** GOES/Himawari public domain (NOAA/NASA/JMA open); Meteosat free/open via EUMETSAT. **Bands 1–6 reflective, 7–16 emissive** on ABI. **Role:** pre-train colorization on millions of registered vis↔IR full-disks; capture diurnal IR variation (same scene, day RGB + night IR).

---

## 12. ISRO / Indian satellites — **important for this Indian hackathon**

> Indian IR↔RGB data is highly relevant to BAH (ISRO hackathon) and lets the model show domain fit on Indian geographies. Primary portals: **Bhuvan** (NRSC geoportal), **MOSDAC** (meteo/ocean), **ISRO/NRSC** order desks.

### 12.1 INSAT-3D / 3DR / 3DS — geostationary TIR + visible imager & sounder
**Imager (6 bands):** VIS 0.52–0.72 µm (**1 km**), SWIR 1.55–1.70 µm (1 km), **MWIR 3.80–4.00 µm (4 km)**, **WV 6.50–7.00 µm (8 km)**, **TIR-1 10.2–11.2 µm (4 km)**, **TIR-2 11.5–12.5 µm (4 km)**.
**Sounder (19 ch):** 7 LWIR (14.7–12.0 µm) + 5 MWIR (11.0–6.5 µm) + 6 SWIR (4.57–3.74 µm) + 1 visible (0.695 µm), 10 km.
**Access:** **MOSDAC** (`mosdac.gov.in`) — L1B/L1C imager & sounder, free with registration. **License:** free for research (ISRO/MOSDAC terms). **Role:** Indian-domain geostationary vis↔thermal pairs; directly aligns with "IR at night over India."

### 12.2 Resourcesat-2 / 2A — LISS-III, LISS-IV, AWiFS (VNIR + SWIR)
| Sensor | Bands | Wavelengths | Res | Swath |
|---|---|---|---|---|
| **LISS-IV** | Green, Red, NIR | 0.52–0.59 / 0.62–0.68 / 0.77–0.86 µm | **5.8 m** | 70 km |
| **LISS-III** | Green, Red, NIR, **SWIR** | +1.55–1.70 µm | 23.5 m | 140 km |
| **AWiFS** | Green, Red, NIR, SWIR | same as LISS-III | 56 m | 740 km |

**Access:** **Bhuvan** / NRSC (`bhuvan.nrsc.gov.in`); also **USGS EROS archive** mirrors LISS-3 & AWiFS for some periods. **License:** Bhuvan/NRSC open-data terms (registration). **Role:** Indian high-res VNIR RGB+NIR targets; LISS-IV 5.8 m is a strong SR target over India.

### 12.3 Cartosat-2 / 3 — pan + multispectral, very high-res
Cartosat-3: **0.25 m panchromatic**, **1 m multispectral (MX)**; MX bands B (0.45–0.52), G (0.52–0.59), R (0.62–0.68), NIR (0.77–0.86 µm). Cartosat-2 series ~0.65 m pan. **Access:** NRSC/Bhuvan order desk (mostly priced/restricted; some samples on Bhuvan). **Role:** ultra-high-res RGB *target* for extreme super-resolution; pan for pan-sharpening.

### 12.4 RISAT-1 / 2B — SAR (all-weather)
RISAT-1 C-band SAR (multiple modes 1–50 m); RISAT-2B/2BR X-band SAR (up to ~0.35 m in high-res spotlight). **Access:** NRSC (restricted/priced). **Role:** Indian all-weather SAR complement (night/cloud story).

### 12.5 Oceansat-2 / 3 (EOS-06) — OCM
OCM-2: 8-band VNIR multispectral, 360 m IFOV, 1420 km swath; OCM-3 (Oceansat-3/EOS-06): 13-channel VNIR–NIR spectro-radiometer. **Access:** MOSDAC / Bhuvan. **Role:** broad visible/NIR coverage (coastal/ocean color); supplementary.

### 12.6 Other EOS / IRS
EOS-04 (RISAT-1A, SAR), EOS-06 (Oceansat-3), legacy Resourcesat-1, IRS-1C/1D. `[from internal knowledge]` for the legacy ones. **Access:** Bhuvan/NRSC/MOSDAC.

---

## 13. Commercial — Planet & Maxar (availability notes)

| Source | Sensor | Bands | Res | Access / license |
|---|---|---|---|---|
| **PlanetScope** | Dove CubeSats | RGB + NIR (some 8-band SuperDove: coastal, blue, green I/II, yellow, red, red-edge, NIR) | ~3 m (≤4.2 m) | Planet API/QGIS; commercial license; **research/education via Planet's Education & Research program**; daily revisit |
| **Planet NICFI basemaps** | Doves (mosaics) | B, G, R, NIR | 4.77 m | **Free for non-commercial** via GEE `projects/planet-nicfi/assets/basemaps/{africa,americas,asia}` — *NICFI program ended new tiles ~April 2025*; existing tiles still in GEE |
| **SkySat** | SkySat-C | 4-band MS + pan | **~0.5 m** (50 cm orthorectified) | Planet; commercial/tasking |
| **Maxar WorldView-1/2/3, GeoEye, Legion** | — | pan + 4–8 MS (+ SWIR on WV-3) | **0.3–0.5 m** (WV-3 pan 0.31 m) | Maxar; commercial license/NDA; some via AWS/SecureWatch; **Open Data Program** releases for disasters (free) |

**Role:** highest-quality RGB *targets* if budget/NDA allows; otherwise rely on free Sentinel-2 / WorldStrat-SPOT / LISS-IV. Treat as optional premium target tier.

---

## 14. Benchmark / curated paired datasets (IR/thermal colorization & SR)

| # | Dataset | What it pairs | Size / res | Why it matters here | Access |
|---|---|---|---|---|---|
| A | **WorldStrat** | Sentinel-2 (10 m, multi-temporal LR) ↔ **SPOT 6/7 (1.5 m HR)** | ~10,000 km², global, incl. humanitarian/under-represented sites | **Best open SR benchmark** (LR→HR RGB); baseline MFSR code included | Zenodo `record/6810792`; GitHub `worldstrat/worldstrat`; CC-BY-NC |
| B | **SEN12MS** | **Sentinel-1 SAR ↔ Sentinel-2 MS ↔ MODIS land cover** | 180,662 triplets, global, 4 seasons | Multimodal IR/optical/SAR fusion + land-cover labels (semantic constraint) | TUM / mediaTUM; CC-BY |
| C | **SEN1-2** | Sentinel-1 SAR ↔ Sentinel-2 RGB | 282,384 patches | SAR↔optical translation (GAN colorization heritage) | TUM; open |
| D | **BigEarthNet** | Sentinel-1 ↔ Sentinel-2 patches (+ multi-labels) | 590,326 pairs (S1+S2) | Large multimodal pretraining + classification labels | `bigearth.net`; CDLA-permissive |
| E | **Major TOM** | Global S2 / S1 grid (expandable EO datasets) | continent-scale | Massive standardized EO tiles for pretraining | HuggingFace; open `[from internal knowledge]` |
| F | **fMoW** (Functional Map of the World) | RGB + multispectral, temporal, 62 classes | ~1M images | Downstream detection/classification context | AWS Open Data; community license `[from internal knowledge]` |
| G | **DOTA** | Aerial RGB, oriented bboxes | 2,806 imgs / 188k instances | **Downstream object-detection metric** target | captain-whu DOTA; academic `[from internal knowledge]` |
| H | **LLVIP** | **Visible ↔ infrared image pairs**, time/space synchronized | 15,488 pairs, low-light | **Direct supervised IR→RGB translation** ground truth | GitHub `bupt-ai-cz/LLVIP`; research |
| I | **KAIST Multispectral** | Color ↔ thermal, beam-splitter aligned | day+night traffic | Well-aligned color↔thermal pairs (driving) | KAIST RCV; research |
| J | **M3FD** | Visible ↔ infrared, detection labels | 4,200 pairs | IR/vis fusion + detection benchmark | GitHub (TarDAL); research `[from internal knowledge]` |
| K | **FLIR ADAS (Free) Thermal** | Thermal (+ ref RGB, **unregistered**) | ~14k frames | Thermal detection pretraining (note: not pixel-aligned) | FLIR/Teledyne; free w/ EULA |
| L | **VEDAI** | Aerial RGB + NIR, vehicle bboxes | 1,210 imgs | Small-vehicle detection in aerial IR/RGB | Univ. Caen; research `[from internal knowledge]` |
| M | **RIT thermal / RIT-18 etc.** | Multispectral/thermal aerial | — | Thermal SR/colorization research | RIT DIRS `[from internal knowledge]` |
| N | **Sen2Venµs / SEN2VENUS** | Sentinel-2 ↔ VENµS (5 m) reference | ~130k patches | **S2→5 m super-resolution** reference pairs | Zenodo; CC-BY `[from internal knowledge]` |
| O | **PROBA-V Super Resolution (ESA)** | 300 m ↔ 100 m multi-frame | Kelvin challenge set | Classic MFSR benchmark | Kelvin/ESA; open `[from internal knowledge]` |

> Items E, F, G, J, L, M, N, O are tagged `[from internal knowledge]` where the live search did not directly confirm 2024-25 hosting URLs — verify the access link before use, but the datasets are well-established.

---

## 15. O(1) / Fast server-side access analysis (for the data layer)

The PS scores **inference time per tile** and scalability, so the *data layer* must avoid bulk downloads. Rank by "near-instant, range-request / server-side" access:

| Tier | Platform | Mechanism | Why it's ~O(1) |
|---|---|---|---|
| ⭐⭐⭐ | **Google Earth Engine** | Server-side compute graph; `ee.Image`, `getThumbURL`, `getPixels`, tiles | No download — pairing/co-reg/reproject run **on Google's servers**; pull only the final tile. Ideal for on-the-fly Landsat-TIRS↔S2 pairs |
| ⭐⭐⭐ | **STAC + COG (range requests)** | `pystac-client` search → `rasterio`/`rioxarray` reads **byte ranges** from Cloud-Optimized GeoTIFFs | Read only the overview/window you need (HTTP Range GET) — no full-scene fetch |
| ⭐⭐⭐ | **Microsoft Planetary Computer** | STAC API + signed COG URLs + hosted JupyterHub/Dask | COG range reads next to the data (Azure); `planetary-computer.sign()` |
| ⭐⭐ | **AWS Open Data (COG buckets)** | `s3://usgs-landsat`, `sentinel-s2-l2a`, `noaa-goes16`, `modis-pds` + Element84 STAC | COG range reads; requester-pays for some; great with `odc-stac`/`stackstac` |
| ⭐ | **MOSDAC / Bhuvan / EarthExplorer / Copernicus** | Order + download (HDF/NetCDF/GeoTIFF), some WMS/WMTS | Bulk/file-based; pre-stage to local COG; Bhuvan/CDSE offer WMS tiles for viewing |

**Design recommendation:** Build the training pipeline on **GEE (export aligned IR↔RGB tile pairs)** + **STAC/COG (Planetary Computer & AWS) for streaming**. Convert any HDF/NetCDF (MODIS, INSAT, ECOSTRESS, EMIT, PRISMA) to **COG** once, store in an object bucket, and serve via STAC for fast windowed reads at train/inference time.

---

## 16. ⭐ Recommended dataset stack (ranked for THIS problem)

Ranked by *fit to IR→RGB colorization + SR + Indian-domain + fast access*:

| Rank | Source | Use in the model | Pairing type | Registration cost |
|---|---|---|---|---|
| **1** | **Landsat 8/9 OLI+TIRS (C2 L2)** — `LANDSAT/LC0{8,9}/C02/T1_L2` | **Core IR(`ST_B10`)→RGB(`SR_B4/3/2`)**; the PS-required source | *intra-platform*, native | **none** (pre-aligned) |
| **2** | **ASTER** — `ASTER/AST_L1T_003` | High-res intra-sensor VNIR(15 m)↔TIR(90 m) colorization | intra-platform | minimal (built-in) |
| **3** | **Sentinel-2 MSI** — `COPERNICUS/S2_SR_HARMONIZED` | **10 m RGB SR target**; cross-pair with Landsat thermal | cross-sensor target | AROSICS/GEE register |
| **4** | **WorldStrat** (S2↔SPOT 1.5 m) | Super-resolution head training (LR→HR RGB) | curated cross-sensor | pre-aligned in dataset |
| **5** | **EMIT** + **PRISMA** (hyperspectral) | Spectrally-exact self-paired IR↔RGB (anti-hallucination) | intra-sensor (synth bands) | none (same cube) |
| **6** | **Sentinel-3 SLSTR/OLCI**, **MODIS**, **VIIRS DNB** | Coarse, high-volume vis↔thermal + **night DNB** pre-training | intra-platform | none |
| **7** | **GOES/Himawari ABI/AHI** | Diurnal vis↔IR pre-training (day RGB + night IR, same scene) | intra-platform | none |
| **8** | **INSAT-3D/3DR (MOSDAC)** + **Resourcesat LISS-IV (Bhuvan)** | **Indian-domain** vis↔thermal + 5.8 m RGB target | mixed | mixed |
| **9** | **LLVIP / KAIST / M3FD** | Ground-truth visible↔IR translation + downstream-detection metric | curated, registered | pre-aligned |
| **10** | **Sentinel-1 SAR** + **SEN12MS/BigEarthNet** | All-weather/night fusion branch; semantic land-cover labels | cross-modal | terrain-corrected |

**Minimum viable stack for a hackathon submission:** #1 (Landsat) + #3 (Sentinel-2) via GEE for the headline SR+colorization, #2 (ASTER) for extra intra-sensor pairs, #9 (LLVIP) to validate the translation objective and report the detection metric, and #8 (one Indian source) to demonstrate domain relevance.

---

## 17. ⭐ Pairing recipe — concrete steps + code-level hints

### Recipe 1 (PRIMARY): Landsat-TIRS thermal ↔ Sentinel-2 RGB super-res pairs (Google Earth Engine)
*IR source = Landsat thermal (100 m native, 30 m grid); RGB target = Sentinel-2 (10 m).* Same area, near-same date, ≤ few days apart.

```python
import ee; ee.Initialize()
aoi = ee.Geometry.Rectangle([77.4, 12.8, 77.8, 13.1])      # e.g. Bengaluru
start, end = '2024-01-01', '2024-03-31'

def mask_l2(img):                                          # scale + cloud mask Landsat C2 L2
    qa = img.select('QA_PIXEL')
    cloud = qa.bitwiseAnd(1<<3).eq(0).And(qa.bitwiseAnd(1<<4).eq(0))
    sr = img.select('SR_B.').multiply(0.0000275).add(-0.2)
    st = img.select('ST_B10').multiply(0.00341802).add(149)  # Kelvin
    return img.addBands(sr, None, True).addBands(st, None, True).updateMask(cloud)

ls = (ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
        .merge(ee.ImageCollection('LANDSAT/LC08/C02/T1_L2'))
        .filterBounds(aoi).filterDate(start, end).map(mask_l2))

def mask_s2(img):
    scl = img.select('SCL')
    clear = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
    return img.updateMask(clear)

s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(aoi).filterDate(start, end)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)).map(mask_s2))

ir   = ls.select(['ST_B10']).median().clip(aoi)            # thermal IR (input)
rgb  = s2.select(['B4','B3','B2']).median().clip(aoi)      # high-res RGB (target)

# Reproject/register IR onto a common grid, then export aligned tiles:
ir_30 = ir.reproject(crs='EPSG:32643', scale=30)
rgb_10 = rgb.reproject(crs='EPSG:32643', scale=10)         # keep target at 10 m
for name, img, scale in [('ir', ir_30, 30), ('rgb', rgb_10, 10)]:
    ee.batch.Export.image.toDrive(image=img, description=f'pair_{name}',
        region=aoi, scale=scale, crs='EPSG:32643', maxPixels=1e10).start()
```
*Key alignment trick:* export both on the **same CRS + origin** so a simple integer upsample (×3) maps 30 m IR onto the 10 m RGB grid. GEE does the reproject server-side (O(1)-ish). For sub-pixel polish, run **AROSICS** on the exported pair.

### Recipe 2 (HIGH-RES INTRA-SENSOR): ASTER VNIR ↔ TIR (no registration needed)
```python
aster = (ee.ImageCollection('ASTER/AST_L1T_003')
           .filterBounds(aoi).filterDate('2005-01-01','2008-12-31'))  # SWIR alive pre-2008
img = aster.first()
rgb = img.select(['B02','B01','B3N'])    # red, green, NIR(as proxy)  -> visible-ish RGB @15 m
tir = img.select(['B13','B14'])          # thermal 10.25 / 11.3 µm    @90 m (resampled)
# Already co-registered within the scene -> slice straight into training pairs.
```
Use ASTER `B02(red)/B01(green)` + synthesize blue (or use NIR false-color) as RGB; `B10–B14` as the thermal IR input. Perfect pixel alignment → clean colorization labels.

### Recipe 3 (STREAMING / COG): Planetary Computer STAC + rasterio windowed read
```python
import planetary_computer as pc, pystac_client, rioxarray
cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
items = cat.search(collections=["landsat-c2-l2"], bbox=[77.4,12.8,77.8,13.1],
                   datetime="2024-01-01/2024-03-31",
                   query={"eo:cloud_cover":{"lt":20}}).item_collection()
it = pc.sign(items[0])
thermal = rioxarray.open_rasterio(it.assets["lwir11"].href)   # COG range read (ST/B10)
red     = rioxarray.open_rasterio(it.assets["red"].href)
# Reproject thermal onto red's grid for a co-registered pair:
thermal_on_rgb = thermal.rio.reproject_match(red)             # rasterio/GDAL warp
```
COG `open_rasterio` only fetches the byte ranges you read → near-instant, scalable.

### Recipe 4 (SELF-PAIRED SPECTRAL TRUTH): EMIT / PRISMA hyperspectral slicing
From one EMIT L2A reflectance cube (285 bands, 381–2493 nm): pick **~660/560/490 nm → RGB target** and **~860 nm (NIR) / 1610 nm / 2200 nm (SWIR) → "IR" input**. Zero co-registration error (same pixels), full control of the band definition → ideal to enforce the *semantic-integrity / no-hallucination* constraint and to pretrain the spectral mapping.

### Recipe 5 (NIGHT / VALIDATION + DOWNSTREAM METRIC): VIIRS DNB + LLVIP
- **VIIRS DNB↔thermal** (`NOAA/VIIRS/...DNB...` + `VNP21A1N`) for the literal *night-time IR→visible* objective at scale.
- **LLVIP** (15,488 registered visible↔IR pairs) as held-out **ground truth** for PSNR/SSIM/FID *and* to compute the **downstream object-detection accuracy** metric the PS asks for (run a detector on generated RGB vs. real RGB).

### Co-registration toolbox (use in this order)
1. **GEE `reproject()` / `register()`** — server-side, set common CRS + scale + origin (handles 90 % of cross-sensor cases on export).
2. **`rasterio`/`rioxarray.rio.reproject_match()`** or **`gdalwarp -t_srs EPSG:326XX -tr 10 10 -r cubic`** — local COG warp onto the target grid.
3. **AROSICS** (`from arosics import COREG_LOCAL`) — automatic **sub-pixel** shift detection + correction between the warped IR and RGB (frequency-domain matching, robust to clouds, outputs an affine warp). Run it last to remove residual misregistration before writing the final training pair.
4. **Cloud/temporal-gap handling:** mask via `QA_PIXEL` (Landsat) / `SCL` (S2) / `Cloud_Mask` (MODIS), then **median composite** over a short window (≤2–4 weeks) so IR and RGB describe the same scene state; drop pairs where seasonal land-cover change is large (NDVI delta threshold).

---

## 18. Numbered summary of distinct data sources covered

**Sensor / satellite families (13):**
1. **Landsat 8/9 OLI+TIRS** (C2 L2/L1) — `LANDSAT/LC0{8,9}/C02/T1_L2`
2. **Sentinel-2 MSI** — `COPERNICUS/S2_SR_HARMONIZED`
3. **Sentinel-3 SLSTR + OLCI** — `COPERNICUS/S3/OLCI`, PC `sentinel-3-slstr-*`
4. **Sentinel-1 SAR (GRD/RTC)** — `COPERNICUS/S1_GRD`
5. **MODIS (Terra/Aqua)** — `MODIS/061/MOD09GA`, `MOD11A1`, `MOD021KM`
6. **VIIRS (SNPP/NOAA-20/21)** incl. **Day/Night Band** — `NOAA/VIIRS/001/VNP09GA`, DNB, `VNP21A1`
7. **ASTER** — `ASTER/AST_L1T_003`
8. **ECOSTRESS (ISS)** — `NASA/ECOSTRESS/L2T_LSTE/V2`, LP DAAC `ECO_L2T_LSTE`
9. **EMIT (ISS)** — LP DAAC `EMITL2ARFL` / `EMITL1BRAD`
10. **PRISMA (ASI)** — ASI portal + GEE community catalog
11. **GOES-R ABI / Himawari AHI / Meteosat SEVIRI** — `NOAA/GOES/{16,18}/MCMIPF`, AWS Himawari/EUMETSAT
12. **ISRO/Indian:** INSAT-3D/3DR/3DS (MOSDAC), Resourcesat-2/2A LISS-III/IV+AWiFS (Bhuvan/USGS), Cartosat-2/3, RISAT-1/2B, Oceansat-2/3 OCM, EOS series
13. **Commercial:** Planet (PlanetScope/SkySat/NICFI), Maxar (WorldView/GeoEye/Legion)

**Benchmark / curated paired datasets (18):**
14. WorldStrat · 15. SEN12MS · 16. SEN1-2 · 17. BigEarthNet · 18. Major TOM · 19. fMoW · 20. DOTA · 21. **LLVIP** · 22. KAIST Multispectral · 23. M3FD · 24. FLIR ADAS Thermal · 25. VEDAI · 26. RIT thermal · 27. Sen2Venµs · 28. PROBA-V SR (ESA Kelvin) · 29. **AROSICS** (co-registration tooling) · 30. **STAC/COG + Planetary Computer + AWS Open Data** (fast-access platforms) · 31. **Google Earth Engine** (server-side compute platform)

> **Distinct data sources catalogued: 31** (≥ the 18 requested), spanning intra-sensor, cross-sensor, hyperspectral, geostationary, Indian, commercial, and curated-benchmark categories.

### Top-5 recommended pairing recipes (ranked)
1. **Landsat TIRS `ST_B10` (IR) ↔ Sentinel-2 `B4/B3/B2` (RGB)** via GEE — primary colorization + super-resolution training set (matches the PS-required Landsat source).
2. **ASTER intra-sensor VNIR `B02/B01/B3N` ↔ TIR `B10–B14`** — high-res, zero-registration colorization pairs.
3. **Planetary-Computer / AWS STAC + COG windowed reads** (Landsat `landsat-c2-l2`, S2 `sentinel-2-l2a`) with `rio.reproject_match` — scalable, near-O(1) streaming pipeline.
4. **EMIT / PRISMA hyperspectral self-pairing** (slice RGB vs NIR/SWIR from one cube) — spectrally-exact, anti-hallucination supervision.
5. **VIIRS Day/Night Band + thermal, validated on LLVIP** — covers the literal night-time IR→RGB objective and yields the downstream object-detection metric.

---

## Sources

- USGS Landsat 8/9 C2 L2 — https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_L2 ; https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC09_C02_T1_L2 ; https://www.usgs.gov/centers/eros/science/usgs-eros-archive-landsat-archives-landsat-8-9-olitirs-collection-2-level-2
- Sentinel-2 — https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED ; https://sentinel.esa.int/en/web/sentinel/user-guides/sentinel-2-msi/resolutions/spatial
- Sentinel-3 SLSTR — https://sentiwiki.copernicus.eu/web/s3-slstr-instrument ; https://user.eumetsat.int/resources/user-guides/sentinel-3-slstr-level-1-data-guide ; OLCI — https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S3_OLCI ; https://planetarycomputer.microsoft.com/dataset/sentinel-3-olci-lfr-l2-netcdf
- Sentinel-1 — https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD ; https://documentation.dataspace.copernicus.eu/Data/SentinelMissions/Sentinel1.html
- MODIS — https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD09GA ; https://modis.gsfc.nasa.gov/data/ ; https://terra.nasa.gov/about/terra-instruments/modis
- VIIRS — https://en.wikipedia.org/wiki/Visible_Infrared_Imaging_Radiometer_Suite ; https://lpdaac.usgs.gov/documents/124/VNP09_User_Guide_V1.6.pdf ; https://ladsweb.modaps.eosdis.nasa.gov/missions-and-measurements/products/VNP02IMG/
- ASTER — https://developers.google.com/earth-engine/datasets/catalog/ASTER_AST_L1T_003 ; https://www.earthdata.nasa.gov/data/instruments/aster ; https://lpdaac.usgs.gov/documents/1319/ASTER_User_Handbook_v4.pdf
- ECOSTRESS — https://developers.google.com/earth-engine/datasets/catalog/NASA_ECOSTRESS_L2T_LSTE_V2 ; https://lpdaac.usgs.gov/products/eco_l2t_lstev002/ ; https://www.eoportal.org/satellite-missions/iss-ecostress
- EMIT — https://www.earthdata.nasa.gov/data/instruments/emit-imaging-spectrometer/spectral-bands ; https://www.earthdata.nasa.gov/data/catalog/lpcloud-emitl2arfl-001
- PRISMA — https://www.eoportal.org/satellite-missions/prisma-hyperspectral ; https://en.wikipedia.org/wiki/PRISMA_(spacecraft)
- GOES-R ABI / Himawari — https://www.goes-r.gov/spacesegment/abi.html ; https://developers.google.com/earth-engine/datasets/catalog/NOAA_GOES_16_MCMIPF ; https://developers.google.com/earth-engine/datasets/catalog/NOAA_GOES_18_MCMIPF ; https://jstnbraaten.medium.com/goes-in-earth-engine-53fbc8783c16
- INSAT-3D/3DR/3DS — https://www.mosdac.gov.in/insat-3d-payloads ; https://www.mosdac.gov.in/insat-3dr-payloads ; https://www.eoportal.org/satellite-missions/insat-3d
- Resourcesat-2 — https://www.eoportal.org/satellite-missions/resourcesat-2 ; https://earth.esa.int/eogateway/missions/resourcesat-2 ; https://www.usgs.gov/centers/eros/science/usgs-eros-archive-isro-resourcesat-1-and-resourcesat-2-liss-3
- Cartosat-3 — https://www.eoportal.org/satellite-missions/cartosat-3 ; https://www.isro.gov.in/Cartosat_3.html
- RISAT / Oceansat — https://www.eoportal.org/satellite-missions/risat-1 ; https://en.wikipedia.org/wiki/RISAT-2B ; https://www.eoportal.org/satellite-missions/oceansat-3
- Bhuvan — https://bhuvan.nrsc.gov.in/ngmaps
- Planet / Maxar — https://docs.planet.com/data/imagery/skysat/ ; https://www.planet.com/nicfi/ ; https://developers.google.com/earth-engine/datasets/catalog/projects_planet-nicfi_assets_basemaps_africa
- WorldStrat — https://zenodo.org/records/6810792 ; https://arxiv.org/pdf/2207.06418 ; https://github.com/worldstrat/worldstrat
- SEN12MS — https://isprs-annals.copernicus.org/articles/IV-2-W7/153/2019/isprs-annals-IV-2-W7-153-2019.pdf ; BigEarthNet — https://bigearth.net/ ; Major TOM — https://arxiv.org/html/2402.12095
- LLVIP — https://arxiv.org/abs/2108.10831 ; https://openaccess.thecvf.com/content/ICCV2021W/RLQ/papers/Jia_LLVIP_A_Visible-Infrared_Paired_Dataset_for_Low-Light_Vision_ICCVW_2021_paper.pdf
- AROSICS — https://pypi.org/project/arosics/ ; https://github.com/GFZ/arosics ; Landsat–S2 co-registration — https://www.mdpi.com/2072-4292/10/2/160
- Planetary Computer / STAC — https://planetarycomputer.microsoft.com/ ; https://stacindex.org/ecosystem
