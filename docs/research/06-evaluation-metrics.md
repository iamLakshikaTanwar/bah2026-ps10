# 06 — Evaluation & Validation Metrics Protocol

**Project:** Satellite IR → RGB Super-Resolution + Colorization (BAH 2026, Problem Statement 10)
**Scope of this document:** the complete, rigorous, multi-metric validation framework for the pipeline. It defines *what to measure, with which exact library/function, which way is "better", and the end-to-end evaluation protocol* (splits, goalposts, statistical tests).
**Author role:** EvalMetrics (IQA & validation researcher).
**Date:** 2026-06-21.
**Research basis:** primary web sources (clean-fid CVPR'22, Blau & Michaeli CVPR'18, torchmetrics docs, sewar/pyiqa/piq docs, BasicSR/RCAN SR convention, DOTA/DIOR detection benchmarks, recent IR-colorization & RS-SR papers 2024–2025). Items that could not be web-confirmed and rely on the Jan-2026 internal knowledge cutoff are explicitly tagged **[from internal knowledge]**.

---

## 0. Why so many metrics? (The core design principle)

The problem statement asks for PSNR, SSIM, FID, inference time, and a no-hallucination check. That minimal set is **not sufficient and is internally contradictory** for a GAN/diffusion colorization+SR system. The reason is the **Perception–Distortion Tradeoff** (Blau & Michaeli, CVPR 2018): mathematically, lowering distortion (improving PSNR/SSIM) and improving perceptual realism (lowering FID/LPIPS) are *in tension* — you cannot maximize both, and this holds for *any* distortion measure, though it is milder for deep-feature distances (VGG/LPIPS) than for pixel MSE. A GAN can score an excellent FID while *hallucinating* fake objects, and a regression model can score excellent PSNR while producing blurry, desaturated, useless output.

Therefore the validation strategy is deliberately **redundant and cross-checking**: we report **distortion metrics** (PSNR/SSIM/...) *together with* **perceptual metrics** (LPIPS/FID/DISTS/...) *together with* **semantic-faithfulness metrics** (segmentation/detection consistency, edge IoU) *together with* **color-specific metrics** (ΔE2000, colorfulness) *together with* **downstream-task gains** (mAP/mIoU) *together with* **efficiency** (latency/VRAM/FLOPs). A claim ("the model is good and does not hallucinate") is only accepted if it survives **multiple independent families** of metrics. This yields the ≥30-method robustness target.

The six families and their counts:

| # | Family | Metrics in this doc | Purpose |
|---|--------|--------------------|---------|
| A | Reconstruction fidelity (full-reference) | 11 | pixel/structural/spectral correctness vs ground-truth RGB |
| B | Perceptual / realism | 7 | how "real" the RGB looks (deep features) |
| C | No-reference IQA | 6 | quality on **real IR** where no RGB ground truth exists |
| D | Color-specific | 4 | colorization correctness (the heart of PS-10) |
| E | Hallucination / semantic faithfulness | 6 | the critical "no fake objects" check |
| F | Downstream-task uplift | 4 | proof that colorization *helps* detection/segmentation |
| G | Efficiency | 6 | scalability per tile (warmup/sync/fp16/FLOPs/VRAM/energy) |

**Total distinct metrics/checks: 44** (see the master table in §8). This comfortably exceeds the 25+/30+ robustness target.

---

## A. Reconstruction Fidelity (full-reference, vs ground-truth RGB)

These require an aligned ground-truth RGB tile. Use on the **paired Landsat 8/9 test split**. For SR specifically, follow the **Y-channel convention** (see §6).

1. **PSNR (Peak Signal-to-Noise Ratio)** — pixel-level reconstruction error in dB. Higher better. Typical good range for SR/colorization ≈ 20–35 dB. *Pitfall:* nearly insensitive to perceptually obvious changes; a blurry image can have high PSNR. RS-SR literature notes PSNR is "insensitive and unsuited" as a *sole* criterion.
   - `torchmetrics.image.PeakSignalNoiseRatio` · `skimage.metrics.peak_signal_noise_ratio` · `sewar.full_ref.psnr` · `piq.psnr`
2. **SSIM (Structural Similarity Index)** — local luminance/contrast/structure similarity. Higher better, range [-1, 1] (≈[0,1] in practice). More reliable than PSNR for structure.
   - `torchmetrics.image.StructuralSimilarityIndexMeasure` · `skimage.metrics.structural_similarity` · `sewar.full_ref.ssim` · `piq.ssim`
3. **MS-SSIM (Multi-Scale SSIM)** — SSIM over a Gaussian pyramid; more robust to scale/viewing conditions than single-scale SSIM. Higher better, ≈[0,1].
   - `torchmetrics.image.MultiScaleStructuralSimilarityIndexMeasure` · `sewar.full_ref.msssim` · `piq.multi_scale_ssim`
4. **RMSE (Root Mean Squared Error)** — sqrt of mean squared pixel error. Lower better.
   - `sewar.full_ref.rmse` · `skimage.metrics.normalized_root_mse` (NRMSE variant) · trivially from torch.
5. **MAE (Mean Absolute Error / L1)** — mean absolute pixel error; less outlier-sensitive than RMSE. Lower better.
   - `torch.nn.functional.l1_loss` · `numpy.mean(np.abs(a-b))` · `sklearn.metrics.mean_absolute_error`.
6. **UQI (Universal Quality Image Index)** — Wang–Bovik precursor to SSIM (loss of correlation + luminance + contrast distortion). Higher better, [-1,1].
   - `sewar.full_ref.uqi` · `torchmetrics.image.UniversalImageQualityIndex` · `piq.vsi`/`piq` family.
7. **VIF / VIFp (Visual Information Fidelity)** — information-theoretic fidelity in a natural-scene-statistics + HVS model. Higher better; 1.0 = perfect, >1 possible after enhancement.
   - `sewar.full_ref.vifp` · `torchmetrics.image.VisualInformationFidelity` · `piq.vif_p`.
8. **SAM (Spectral Angle Mapper)** — angle (radians/degrees) between predicted and reference per-pixel *spectral vectors* (across the 3 RGB channels here, the standard multispectral fidelity metric). **Lower better** (0 = identical spectra). *Critically important for multispectral/colorization* because it measures hue/spectral direction independent of brightness — directly relevant to "is the color right?".
   - `sewar.full_ref.sam` · `torchmetrics.image.SpectralAngleMapper`.
9. **ERGAS (Erreur Relative Globale Adimensionnelle de Synthèse)** — global relative dimensionless synthesis error, the canonical **pansharpening / RS fusion** quality index (normalizes RMSE per band by the band mean and the resolution ratio). Lower better; <3 is generally "good" in fusion literature **[from internal knowledge for the <3 rule of thumb]**.
   - `sewar.full_ref.ergas` · `torchmetrics.image.ErrorRelativeGlobalDimensionlessSynthesis`.
10. **SAM's spatial cousins — RASE & SCC.** RASE (Relative Average Spectral Error, lower better) and SCC (Spatial Correlation Coefficient, higher better) round out the RS-fusion suite.
    - `sewar.full_ref.rase` / `torchmetrics.image.RelativeAverageSpectralError`; `sewar.full_ref.scc` / `torchmetrics.image.SpatialCorrelationCoefficient`.
11. **CC (Correlation Coefficient)** — Pearson correlation between predicted and reference pixels (per-channel, then averaged). Higher better, [-1,1]. Simple, complements SSIM/UQI.
    - `numpy.corrcoef` / `scipy.stats.pearsonr` (flatten per channel) **[from internal knowledge]**. (SCC above is the *spatial-gradient* correlation variant.)

> **Library note:** `sewar` expects `H×W×C` numpy arrays; `torchmetrics` expects `N×C×H×W` tensors; `piq` expects `N×C×H×W` in `[0,1]`. Keep a single conversion utility to avoid layout bugs. Compute a *full-reference* metric only on the **paired** split — never on real IR with no RGB GT.

---

## B. Perceptual / Realism (deep-feature, distribution-level)

These quantify "does the output look like a real RGB satellite image". Report **alongside** family A.

12. **LPIPS (Learned Perceptual Image Patch Similarity)** — distance in deep-net feature space; correlates with human perception of similarity. **Lower better**, ≈[0,1]. Report **both backbones**: AlexNet (the original, fastest) and VGG (often used as a loss). 
    - `lpips` pkg: `lpips.LPIPS(net='alex')` / `net='vgg')`. Also `torchmetrics.image.LearnedPerceptualImagePatchSimilarity(net_type='alex'|'vgg')`, `piq.LPIPS`. Input range must be `[-1,1]` for the `lpips` package, `[0,1]`/`[-1,1]` (config-dependent) for torchmetrics — *verify normalization, this is a common bug*.
13. **FID (Fréchet Inception Distance)** — Fréchet distance between Inception feature *distributions* of real vs generated sets. **Lower better**, ≥0. **Use `clean-fid`, not a naive FID.** 
    - **Pitfalls (clean-fid, CVPR 2022):** (a) **aliased resizing** — PyTorch/TF FID use fixed-width bilinear that aliases when downsampling; clean-fid uses an adaptive (PIL-bicubic-matched) antialiased resize. (b) **JPEG quantization** — JPEG quality can swing FID dramatically even when images look identical; keep compression consistent (ideally lossless PNG for both sets). (c) **N matters** — FID is biased upward for small N; keep both sets large and equal.
    - `from cleanfid import fid; fid.compute_fid(fdir_real, fdir_fake, mode="clean")`. To reproduce old numbers use `mode="legacy_pytorch"` / `"legacy_tensorflow"`. Precompute reference stats once: `fid.make_custom_stats("landsat_rgb", path, mode="clean")` then `fid.compute_fid(fdir_fake, dataset_name="landsat_rgb", dataset_split="custom", mode="clean")`. Also `torchmetrics.image.FrechetInceptionDistance` for an in-loop estimate (still prefer clean-fid for the headline number).
14. **CLIP-FID** — FID computed on **CLIP ViT-B/32** features instead of Inception. More robust and better-behaved for **limited / domain-shifted data** (satellite ≠ ImageNet), where Inception features are a poor fit. Lower better.
    - `fid.compute_fid(fdir_real, fdir_fake, mode="clean", model_name="clip_vit_b_32")`.
15. **KID (Kernel Inception Distance)** — squared MMD between Inception features with a polynomial kernel. **Unbiased for small samples** (unlike FID) and comes with a variance estimate — preferred when the test set is small. Lower better (≈0).
    - `fid.compute_kid(fdir_real, fdir_fake)` · `torchmetrics.image.KernelInceptionDistance` (reports mean ± std; mind the `subset_size` for small sets).
16. **DISTS (Deep Image Structure and Texture Similarity)** — unifies structure sensitivity with **texture-resampling tolerance**; invariant to texture substitution, so it rewards realistic texture without demanding pixel-exact match. **Lower better**, ≈[0,1]. Excellent companion to LPIPS for SR texture.
    - `torchmetrics.image.DeepImageStructureAndTextureSimilarity` (a.k.a. DISTS) · `piq.DISTS` · available in `pyiqa` (`pyiqa.create_metric('dists')`).
17. **IS (Inception Score)** — class-confidence × diversity of generated set. Higher better. **Report but de-emphasize:** IS is weak here because (a) it uses no reference/real set, (b) it is tuned to ImageNet classes irrelevant to satellite tiles, (c) it is insensitive to intra-class mode collapse and to fidelity. Include it only for completeness / reviewer expectations.
    - `torchmetrics.image.InceptionScore` · `torch-fidelity` (`isc=True`).
18. **(Aggregator) torch-fidelity** — one call that emits FID + KID + IS + PR together for a sanity cross-check against clean-fid.
    - `torch_fidelity.calculate_metrics(input1=..., input2=..., fid=True, kid=True, isc=True, prc=True)`.

> **Recommendation:** headline realism numbers = **clean-FID + CLIP-FID + KID + LPIPS(alex&vgg) + DISTS**. Treat IS as secondary.

---

## C. No-Reference IQA (for **real IR** tiles with NO RGB ground truth)

Real operational IR scenes have *no* paired RGB, so families A/B are impossible there. No-reference (NR) IQA scores the *output's* quality blindly. Use the **`pyiqa` (IQA-PyTorch)** toolbox — uniform API `m = pyiqa.create_metric(name); score = m(img_tensor)`; each metric exposes `m.lower_better`.

19. **NIQE (Natural Image Quality Evaluator)** — distance of output NSS features from a pristine-image model. **Lower better.** Opinion-unaware, no training labels needed → good default.
    - `pyiqa.create_metric('niqe')` · also in `piq.brisque`-family / `sewar`-adjacent. 
20. **BRISQUE** — spatial NSS-based blind quality. **Lower better.** Classic, fast.
    - `pyiqa.create_metric('brisque')` · `piq.brisque` · `skimage`/`imquality` variants.
21. **PIQE (Perception-based Image Quality Evaluator)** — block-wise distortion/noticeable-artifact estimator. **Lower better.** Complements NIQE/BRISQUE.
    - `pyiqa.create_metric('piqe')` **[verify exact key in installed pyiqa version]**.
22. **MUSIQ (Multi-scale Image Quality Transformer)** — deep multi-scale NR-IQA, strong human correlation. **Higher better** (default KonIQ weights). 
    - `pyiqa.create_metric('musiq')`.
23. **MANIQA (Multi-dimension Attention NR-IQA)** — attention-based deep NR-IQA, SOTA on several NR benchmarks. **Higher better.**
    - `pyiqa.create_metric('maniqa')`.
24. **CLIP-IQA / CLIP-IQA+** — vision-language NR quality ("a good photo" vs "a bad photo" prompt similarity via CLIP). **Higher better.** Robust to domain shift, useful for satellite.
    - `pyiqa.create_metric('clipiqa')` / `'clipiqa+'`. (Also `torchmetrics.multimodal.CLIPImageQualityAssessment`.)

> **Use:** report NR metrics on (a) **real IR→RGB** outputs to demonstrate quality without GT, and (b) as an extra cross-check on the paired test set (NR should *improve* vs the raw IR input). NR-IQA models are themselves imperfect — use the **panel** (NIQE+BRISQUE+PIQE+MUSIQ+MANIQA+CLIP-IQA), not any single one.

---

## D. Color-Specific Metrics (the heart of colorization correctness)

PSNR/SSIM under-weight chroma (they are dominated by luminance). These directly measure *color* correctness — essential for PS-10's "water→blue, forest→green without artifacts".

25. **CIEDE2000 (ΔE₀₀)** — perceptually-uniform color difference in CIE L\*a\*b\*. **Lower better.** Rule of thumb: ΔE<1 imperceptible, 1–2 perceptible on close inspection, >5 clearly different **[from internal knowledge for thresholds]**. The single most important *color-accuracy* number.
    - Convert RGB→Lab (`skimage.color.rgb2lab`), then `skimage.color.deltaE_ciede2000(lab_pred, lab_gt)` → mean over pixels. Signature: `deltaE_ciede2000(lab1, lab2, kL=1, kC=1, kH=1, channel_axis=-1)`. Also `colour.delta_E(..., method='CIE 2000')` (colour-science, Sharma 2005 impl).
26. **Colorfulness (Hasler–Süsstrunk M3)** — single-number colorfulness from mean/std of opponent channels `rg = R−G`, `yb = 0.5(R+G)−B`: `M3 = sqrt(σ_rg²+σ_yb²) + 0.3·sqrt(μ_rg²+μ_yb²)`. Correlates >90% with human ratings. **Report as a match to GT** (|colorfulness_pred − colorfulness_GT|, lower better) — a desaturated GAN output flags here even if PSNR is fine.
    - Implement directly with OpenCV/numpy (PyImageSearch reference impl), or `HSM3` repo. No single canonical pip name — ~10-line function **[from internal knowledge for the exact formula constant 0.3]**.
27. **Chrominance PSNR (PSNR on Cb/Cr, or on a\*/b\*)** — PSNR computed on the *color* channels only, isolating color reconstruction from luminance. Higher better. Pairs with Y-channel PSNR (§6) to separate "structure" vs "color" errors.
    - Convert to YCbCr (`skimage.color.rgb2ycbcr` / OpenCV `cv2.cvtColor(...COLOR_RGB2YCrCb)`), compute PSNR on Cb & Cr separately **[from internal knowledge]**.
28. **Lab histogram correlation / EMD** — correlation (or Earth Mover's Distance) between predicted vs GT 2D a\*–b\* chroma histograms; captures *global palette* match independent of spatial alignment. Higher correlation / lower EMD better.
    - `cv2.calcHist` on a\*,b\* → `cv2.compareHist(h1,h2,cv2.HISTCMP_CORREL)`; EMD via `scipy.stats.wasserstein_distance` or `cv2.EMD` **[from internal knowledge]**.

---

## E. Hallucination / Semantic-Faithfulness Checks (the critical "no fake objects" requirement)

This is the make-or-break section for PS-10 ("ensure the process does not distort or misrepresent ground-truth objects" / "no hallucinations"). Pixel and perceptual metrics **cannot** catch a plausible-but-fake vehicle. We need **task-level consistency** checks. The web literature on "hallucination metrics" is mostly NLP-centric; the image-domain protocol below is synthesized **[from internal knowledge]** and grounded in the Perception–Distortion framing (Blau & Michaeli).

29. **Segmentation-consistency mIoU (input-vs-output).** Run a frozen land-cover/semantic segmenter (e.g., a model trained on the *real RGB* domain, or a class-agnostic SAM) on (a) the **ground-truth RGB** and (b) the **generated RGB**; compute **mIoU / pixel-accuracy** between the two label maps. High agreement ⇒ colorization preserved semantics; large drops localize where the model *changed the scene's meaning*. **Higher better.**
    - `torchmetrics.JaccardIndex` (a.k.a. `MulticlassJaccardIndex`) for mIoU; segmenter via `mmsegmentation`/`segmentation-models-pytorch`/SAM.
30. **Detection-consistency (object count & matched mAP, input-derivable).** Run a detector (trained on RGB satellite, e.g. on DOTA/DIOR) on GT-RGB and on generated-RGB; compare **per-class object counts** and **box agreement**. *New* boxes appearing only in the generated image that are absent in GT-RGB are **candidate hallucinations**; *missing* boxes are **erasures**. **Higher agreement / smaller count-delta better.**
    - Detector: `ultralytics` YOLO (OBB for DOTA), `mmrotate`. Matching via IoU + Hungarian **[from internal knowledge]**.
31. **Edge / structure preservation — gradient correlation.** Correlation between Sobel/Scharr gradient magnitude maps of input-IR (upsampled) vs output. SR/colorization must *not* invent edges. **Higher better.** This is a *reference-free fabrication detector*: the output's structural edges should derive from the IR input, not appear from nowhere.
    - Sobel via `cv2.Sobel` / `skimage.filters.sobel`; Pearson corr of magnitudes **[from internal knowledge]**. (`SCC` in family A is a quantitative sibling.)
32. **Edge IoU (Canny edge maps).** Binarize Canny edges of GT-RGB (or IR-derived structure) and generated-RGB; compute IoU of edge pixels. Catches *added/removed* structures. **Higher better.**
    - `cv2.Canny` → boolean IoU **[from internal knowledge]**.
33. **Uncertainty / disagreement maps (ensemble or MC-dropout).** Run the generator N times (dropout on, or an ensemble); per-pixel variance of the output highlights *low-confidence / likely-fabricated* regions. Report the **mean predictive uncertainty** and overlay maps in the qualitative panel. High variance co-located with new objects ⇒ hallucination evidence. **Lower mean uncertainty better; spatial map is the deliverable.**
    - MC-dropout / deep ensembles; variance over the N forward passes **[from internal knowledge]**.
34. **Feature-inversion / cycle-consistency check.** If an IR↔RGB cycle is available (CycleGAN-style) or a separate RGB→IR predictor exists, re-derive IR from the generated RGB and compare to the *original IR* (PSNR/SSIM/SAM on the reconstructed IR). Large cycle error ⇒ the RGB encodes content not supported by the IR ⇒ hallucination. **Lower cycle error better.**
    - Reuse the SR/colorization stack + an auxiliary RGB→IR head; metrics from family A on the IR domain **[from internal knowledge]**.

> **Decision rule for "no hallucination":** accept only if (i) segmentation-consistency mIoU ≥ threshold, (ii) detection count-delta ≈ 0 with no systematic *new* high-confidence objects, (iii) gradient/edge correlation high, (iv) uncertainty not spiking on novel structures. Any single perceptual metric (FID) looking good is **not** sufficient — this is the Blau–Michaeli warning operationalized.

---

## F. Downstream-Task Uplift (proving colorization *helps*)

PS-10 explicitly requires that outputs "significantly improve" detection/segmentation. The protocol is **paired before-vs-after**: run the *same* frozen downstream model on **(baseline) raw IR / grayscale-replicated** vs **(ours) colorized-SR RGB**, on a held-out labeled set, and report the delta with significance.

35. **mAP / mAP50 / mAP50-95 (object detection).** Evaluate a fixed detector on baseline vs enhanced inputs. **Higher better.** Use oriented boxes for aerial (DOTA/DIOR-R) or VEDAI for vehicles. Report the *uplift* Δ.
    - `torchmetrics.detection.MeanAveragePrecision` (COCO-style) · `ultralytics` `model.val()` · `pycocotools`. Goalpost context: YOLO-OBB on DOTA-v1 reaches mAP50 ≈ 79–82, mAP50-95 ≈ 52–57; Oriented R-CNN ≈ 77.5 mAP on DOTA-v1.5.
36. **mIoU / pixel-accuracy / Dice (semantic segmentation).** Same paired protocol for a land-cover segmenter. **Higher better.**
    - `torchmetrics.JaccardIndex` (mIoU), `torchmetrics.Dice`, `torchmetrics.classification.MulticlassAccuracy`.
37. **Before-vs-after comparison protocol (the experimental design).** Three conditions minimum: **(B0)** raw IR (1→3 channel replicate) into the RGB-trained downstream model; **(B1)** simple baselines (bicubic SR / classical pseudo-color) ; **(Ours)** full SR+colorization. Same model, same weights, same test tiles → the only variable is our pipeline. Report each condition's mAP/mIoU and Δ(Ours−B0).
38. **Statistical significance (bootstrap CIs + paired test).** Per-tile (or per-image) scores are *paired*. Report **95% bootstrap confidence intervals** (≈1,000 resamples) on the mean uplift, and a **paired significance test**: **Wilcoxon signed-rank** (non-parametric, no Gaussian assumption — preferred for IQA/mAP differences) and/or **paired t-test**. Significance ⇒ the uplift is real, not noise.
    - `scipy.stats.wilcoxon(scores_ours, scores_baseline)`; `scipy.stats.ttest_rel(...)`; bootstrap via `scipy.stats.bootstrap((diffs,), np.mean, n_resamples=1000, method='BCa')` or numpy resampling.

---

## G. Efficiency / Scalability (per-tile, done correctly)

PS-10 requires inference time per tile. Measure it **rigorously** — naive `time.time()` around a CUDA call is wrong because GPU ops are asynchronous.

39. **Inference latency per tile (ms).** Lower better. **Protocol:** (1) **warmup** (≥10–50 dummy forward passes — GPU power states ramp up); (2) use **`torch.cuda.Event(enable_timing=True)`** start/end *or* wrap with **`torch.cuda.synchronize()`** before stopping a CPU timer; (3) average over ≥100 runs, report **mean ± std and p50/p95**; (4) report separately for **fp32 and fp16/AMP** (fp16 is the deployment-relevant number). Fix tile size (e.g., 256×256 or 512×512) and state batch size.
    - `torch.cuda.synchronize()` + `time.perf_counter()`, or CUDA events.
40. **Throughput (tiles/s).** Higher better. `throughput = (num_batches × batch_size) / total_time` at the largest batch that fits VRAM (saturates parallelism). Report at the batch used.
41. **FLOPs / MACs.** Lower better (efficiency). Per-tile multiply-adds in the forward pass.
    - `fvcore.nn.FlopCountAnalysis(model, input).total()` · `thop.profile` · `ptflops`. (Note fvcore counts MACs; ×2 for FLOPs — state which.)
42. **Parameter count.** Lower better. `sum(p.numel() for p in model.parameters())` (and trainable-only).
43. **Peak VRAM (MB).** Lower better. `torch.cuda.reset_peak_memory_stats()` before, `torch.cuda.max_memory_allocated()` after a forward pass at the target tile/batch.
44. **Energy per tile (J) / power (W).** Lower better. Optional but strong for a "scalable solution" claim. Sample `nvidia-smi --query-gpu=power.draw` during a sustained run, or use `pynvml` / `codecarbon` (also gives gCO₂) **[from internal knowledge for codecarbon]**.

---

## 6. Super-Resolution–specific evaluation notes

- **Y-channel convention.** SR results are conventionally reported on the **luminance (Y) channel** of YCbCr (human vision is more luminance-sensitive; this is the BasicSR/RCAN standard). For PS-10 report **both** RGB-domain metrics *and* **Y-channel PSNR/SSIM** so numbers are comparable to SR literature.
  - `skimage.color.rgb2ycbcr(img)[..., 0]` → PSNR/SSIM on Y. (Mirror `MATLAB`/BasicSR exactly if comparing to published numbers.)
- **Border shaving.** Crop a border equal to the scale factor (commonly 4–6 px) before computing metrics to avoid boundary artifacts inflating/deflating scores ("shave"). State the shave width.
- **Perceptual SR metric.** Always pair Y-PSNR/SSIM with **LPIPS** (and DISTS) — SR-GANs trade PSNR for perceptual quality (Perception–Distortion tradeoff).
- **Evaluate on REAL data, not only bicubic-degraded.** The single biggest SR evaluation pitfall: training/testing only on **bicubic-downsampled** pairs overstates performance because real sensor degradation (blur kernel, noise, MTF, atmosphere) differs. Report on (a) synthetic bicubic pairs *and* (b) **real LR–HR pairs** (e.g., true native-resolution Landsat tiles, or a realistic degradation model / kernel-aware pipeline). Recent RS-SR work (KANet, MuS²) is explicit that simulated data "does not fully reflect operating conditions."

---

## 7. THE EVALUATION PROTOCOL (end-to-end)

### 7.1 Data splits — **geographic holdout to prevent leakage**
- **Do NOT do random tile split.** Adjacent tiles from the same scene are highly correlated; random splitting leaks near-duplicate content into the test set and inflates every metric.
- **Split by geography/scene/time:** hold out **entire Landsat scenes / WRS-2 path-rows / geographic regions** (and/or distinct acquisition dates) for validation and test. Example: train on a set of path-rows, validate on disjoint path-rows, test on a *third* disjoint set spanning *different biomes* (forest, water, urban, desert, snow) and seasons. This tests true generalization across land-cover types relevant to PS-10.
- Recommended ratio ≈ 70 / 15 / 15 by *scene area*, never by random tile.
- **Co-registration check** is part of validation: any residual IR↔RGB misalignment caps achievable PSNR/SSIM/SAM — verify alignment (phase-correlation / GCP RMSE in pixels) and report it, otherwise low scores may reflect mis-registration, not the model.
- Keep a separate **real-IR-only** set (no RGB GT) for the NR-IQA (family C) and qualitative hallucination panels.

### 7.2 What to report (the scorecard)
For every model/ablation, emit one row with:
- **Distortion:** PSNR(RGB), SSIM(RGB), **Y-PSNR, Y-SSIM**, MS-SSIM, RMSE, MAE, UQI, VIF, **SAM**, ERGAS, RASE, SCC, CC.
- **Perceptual:** **clean-FID, CLIP-FID, KID**, LPIPS(alex), LPIPS(vgg), DISTS, (IS for completeness).
- **Color:** **ΔE2000**, |Δcolorfulness|, chroma-PSNR(Cb/Cr), Lab-hist correlation.
- **No-reference (on real IR outputs):** NIQE, BRISQUE, PIQE, MUSIQ, MANIQA, CLIP-IQA.
- **Faithfulness:** seg-consistency mIoU, detection count-delta / matched-mAP, gradient correlation, edge IoU, mean uncertainty, cycle-IR error.
- **Downstream uplift:** mAP50 & mAP50-95 (det), mIoU & Dice (seg), each with **Δ vs baseline + 95% bootstrap CI + Wilcoxon p**.
- **Efficiency:** latency/tile (fp32 & fp16, mean±std, p95), throughput tiles/s, FLOPs, params, peak VRAM, (energy/tile).
- **Qualitative panel:** side-by-side IR | baseline | ours | GT-RGB, **plus** uncertainty heatmap and seg/detection-overlay diffs — for the human no-hallucination sign-off.

### 7.3 Acceptance logic (robustness via cross-checks)
A configuration is "validated" only if it is **Pareto-good across families**: competitive distortion **and** competitive perceptual **and** passing faithfulness gates **and** positive, significant downstream uplift **and** within the efficiency budget per tile. A win in one family that *regresses* another (e.g., great FID, failing seg-consistency) is flagged as **probable hallucination**, not success.

---

## 8. Literature goalposts (targets / baselines)

These set realistic expectations. (Datasets/domains differ from Landsat 8/9, so treat as *order-of-magnitude goalposts*, not hard targets.)

**IR / thermal colorization (RGB-domain):**
- FCNet (reference-driven + contrastive, 2024): **PSNR 28.67 / SSIM 0.542 (KAIST)**, **PSNR 30.42 / SSIM 0.726 (FLIR)** — current strong baseline; also leads on FID & NIQE.
- Unsupervised contrastive (UGCI/NIR): ~**57% FID reduction** vs prior NIR→RGB.
- cGAN NIR→RGB: ~**+12.6% PSNR, +7.4% SSIM, +9.5% color-histogram-similarity** vs baselines.
- Practical interpretation: aim **PSNR ≳ 25–30 dB, SSIM ≳ 0.6–0.75, low FID, low ΔE2000** on the paired Landsat test set; do not over-index on PSNR alone.

**Satellite / RS super-resolution:**
- 2025 SOTA: **PSNR 29.17 dB / SSIM 0.8958 (DFC-2019)**, **PSNR 31.08 dB / SSIM 0.9442 (RSI-CB)**.
- Real-world degradation (KANet): outperforms by **0.2–0.8 dB PSNR**; **≥30–34% NIQE** improvement vs others on real cross-sensor data.
- Note: PSNR gains in RS-SR are *small* (tenths of a dB); SSIM/LPIPS/NIQE often more discriminative.

**Downstream detection (aerial, oriented):**
- DOTA-v1 YOLO-OBB: **mAP50 ≈ 79–82, mAP50-95 ≈ 52–57** (model-size dependent).
- DOTA-v1.5 Oriented R-CNN: **≈ 77.5 mAP**.
- Goal for PS-10: demonstrate a **statistically significant +Δ mAP / +Δ mIoU** of the colorized-SR input over the raw-IR baseline using the *same* frozen detector/segmenter.

**FID/perceptual sanity:** there is no universal "good" FID (it is dataset/N-dependent) — always report it **relative to** a baseline and the real-set self-FID floor, computed with **clean-fid** under identical preprocessing.

---

## 9. Recommended Python metrics stack (exact packages)

Install:
```bash
pip install torch torchvision torchmetrics[image]   # core FR + FID/KID/IS/LPIPS/DISTS/SAM/ERGAS/VIF/UQI/RASE/SCC
pip install clean-fid                                # clean-FID, CLIP-FID, KID (the headline realism numbers)
pip install lpips                                    # canonical LPIPS (alex & vgg)
pip install pyiqa                                    # NR-IQA: NIQE/BRISQUE/PIQE/MUSIQ/MANIQA/CLIP-IQA (+ FR cross-checks)
pip install sewar                                    # RS suite: SAM/ERGAS/UQI/VIF/RASE/SCC/MS-SSIM/PSNR-B/Q2N (numpy)
pip install piq                                      # extra FR/NR + losses (DISTS, BRISQUE, VIF, etc.)
pip install scikit-image                             # SSIM/PSNR/NRMSE, rgb2lab, deltaE_ciede2000, rgb2ycbcr
pip install opencv-python                            # color spaces, histograms, Sobel/Canny, colorfulness
pip install colour-science                           # CIEDE2000 (Sharma 2005 ref), color science
pip install torch-fidelity                           # aggregate FID+KID+IS+PR cross-check
pip install fvcore thop ptflops                      # FLOPs / MACs
pip install pycocotools ultralytics                  # downstream detection mAP (COCO + YOLO/OBB)
pip install scipy                                    # Wilcoxon, paired t-test, bootstrap, wasserstein
pip install pynvml codecarbon                        # VRAM/power/energy (optional)
pip install rasterio                                 # GEO IO for Landsat tiles / scene-level splits
```

**Minimal call cheat-sheet:**
```python
# Full-reference (torchmetrics, N×C×H×W in [0,1])
from torchmetrics.image import (PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure,
    MultiScaleStructuralSimilarityIndexMeasure, SpectralAngleMapper,
    ErrorRelativeGlobalDimensionlessSynthesis, VisualInformationFidelity,
    UniversalImageQualityIndex, LearnedPerceptualImagePatchSimilarity,
    FrechetInceptionDistance, KernelInceptionDistance)
# RS numpy suite (H×W×C)
from sewar.full_ref import sam, ergas, uqi, vifp, rase, scc, msssim, rmse, psnr, ssim
# Color
from skimage.color import rgb2lab, rgb2ycbcr, deltaE_ciede2000
# Realism (preferred headline)
from cleanfid import fid
fid.compute_fid(real_dir, fake_dir, mode="clean")                                  # clean-FID
fid.compute_fid(real_dir, fake_dir, mode="clean", model_name="clip_vit_b_32")      # CLIP-FID
fid.compute_kid(real_dir, fake_dir)                                                 # KID
# No-reference
import pyiqa
for n in ["niqe","brisque","musiq","maniqa","clipiqa"]:
    m = pyiqa.create_metric(n); print(n, m.lower_better, m(img_tensor))
# Significance
from scipy.stats import wilcoxon, ttest_rel, bootstrap
```

---

## 10. Pitfall checklist (quick reference)

- [ ] **clean-fid, not naive FID** — antialiased resize + consistent JPEG/PNG; equal & large N.
- [ ] **CLIP-FID + KID** for the domain-shifted / small satellite test set (Inception-FID is biased there).
- [ ] **Y-channel PSNR/SSIM + border shave** for SR comparability; also report RGB-domain.
- [ ] **Geographic / scene / date holdout** — never random tile split (leakage inflates everything).
- [ ] **Verify IR↔RGB co-registration** before trusting PSNR/SSIM/SAM (mis-registration caps them).
- [ ] **Evaluate on real degraded data**, not only bicubic pairs.
- [ ] **Normalization ranges** for LPIPS (`[-1,1]`) vs piq/torchmetrics (`[0,1]`) — a silent-bug source.
- [ ] **Pair distortion + perceptual + semantic** — a great FID with failing seg-consistency = hallucination, not success (Blau–Michaeli).
- [ ] **Color metrics (ΔE2000, colorfulness, chroma-PSNR)** — PSNR/SSIM under-weight color; mandatory for a *colorization* task.
- [ ] **Efficiency done right** — warmup + `cuda.synchronize()`/CUDA events, report fp16, p95, peak VRAM, FLOPs.
- [ ] **Significance** — bootstrap CIs + Wilcoxon on *paired* downstream scores; report Δ, not just absolutes.

---

### Sources (primary web research)
- Clean-FID / aliased resizing pitfalls — Parmar et al., CVPR 2022: https://github.com/GaParmar/clean-fid · https://arxiv.org/pdf/2104.11222 · https://pypi.org/project/clean-fid/
- Perception–Distortion Tradeoff — Blau & Michaeli, CVPR 2018: https://arxiv.org/abs/1711.06077 · https://openaccess.thecvf.com/content_cvpr_2018/papers/Blau_The_Perception-Distortion_Tradeoff_CVPR_2018_paper.pdf
- sewar (SAM/ERGAS/UQI/VIF/RASE/SCC/MS-SSIM): https://github.com/andrewekhalel/sewar · https://pypi.org/project/sewar/
- torchmetrics image API: https://lightning.ai/docs/torchmetrics/stable/ · DISTS: https://lightning.ai/docs/torchmetrics/stable/image/dists.html
- pyiqa / IQA-PyTorch (NIQE/BRISQUE/MUSIQ/MANIQA/CLIP-IQA): https://github.com/chaofengc/IQA-PyTorch · https://iqa-pytorch.readthedocs.io/
- piq: https://github.com/photosynthesis-team/piq
- Hasler–Süsstrunk colorfulness: https://infoscience.epfl.ch/record/33994/files/HaslerS03.pdf · https://pyimagesearch.com/2017/06/05/computing-image-colorfulness-with-opencv-and-python/
- CIEDE2000 (skimage / colour-science): https://scikit-image.org/docs/dev/api/skimage.color.html · https://github.com/colour-science/colour
- DISTS — Ding et al.: https://arxiv.org/pdf/2004.07728
- Y-channel / border-shave SR convention (BasicSR/RCAN): https://github.com/yulunzhang/RCAN
- IR/thermal colorization goalposts (FCNet etc.): https://www.sciencedirect.com/science/article/abs/pii/S1350449524005590 · https://link.springer.com/chapter/10.1007/978-3-031-58181-6_5
- RS super-resolution goalposts & real-degradation: https://www.sciencedirect.com/science/article/abs/pii/S0924271622001824 · https://www.nature.com/articles/s41597-023-02538-9 (MuS²) · https://arxiv.org/pdf/2103.06270
- DOTA / DIOR detection benchmarks: https://docs.ultralytics.com/datasets/obb/dota-v2 · https://openaccess.thecvf.com/content_cvpr_2018/papers/Xia_DOTA_A_Large-Scale_CVPR_2018_paper.pdf
- Inference-time measurement (warmup/sync/CUDA events/FLOPs): https://medium.com/data-science/the-correct-way-to-measure-inference-time-of-deep-neural-networks-304a54e5187f · https://leimao.github.io/blog/PyTorch-Benchmark/
- Bootstrap CIs + Wilcoxon for IQA significance: https://arxiv.org/pdf/2509.13150
