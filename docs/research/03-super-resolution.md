# 03 — Super-Resolution & Restoration for Infrared Satellite Imagery

**Project context:** Enhancement stage of an IR→RGB pipeline (Problem Statement 10). Inputs are single-channel, low-contrast, often natively low-resolution IR/thermal satellite bands (e.g., Landsat 8/9 TIRS thermal at 100 m native, resampled to 30 m; OLI SWIR/NIR at 30 m). Goal: super-resolve and sharpen faint edges/textures **without hallucinating fake structures**, then hand off to colorization. Evaluation cares about PSNR, SSIM, FID (on the colorized output), and **per-tile inference time**.

**Method note:** Web research (WebSearch/WebFetch, 2023–2025 sources) was the primary source; where the web was thin on exact numbers/repos, figures are filled from internal knowledge (cutoff Jan 2026) and marked `[from internal knowledge]`. PSNR/SSIM benchmark numbers are RGB natural-image benchmarks (Set5/Set14/BSD100/Urban100/Manga109, DIV2K) unless stated otherwise — treat them as *relative ranking* signals, not absolute predictors for single-channel IR, whose statistics differ (lower texture entropy, lower contrast, heavier sensor noise, strong low-frequency content).

---

## 0. TL;DR Recommendation

- **Primary enhancement approach:** **Guided / reference-based SR** — super-resolve the IR band using a co-registered higher-resolution band (Landsat OLI 15 m panchromatic, or Sentinel-2 10 m VIS/NIR) as a guidance image, with a modern restoration backbone (**HAT-light or NAFNet/Restormer-style encoder-decoder** with a guidance branch à la **PSRGAN / CoReFusion / guided thermal SR**). This is the single biggest lever because pure single-image SR of low-native-resolution thermal data is fundamentally ill-posed — there is little real high-frequency content to recover, so a real HR cross-sensor guide beats hallucination. Add **Real-ESRGAN/BSRGAN-style synthetic degradation** for blind robustness.
- **Fast backup (no guide / latency-critical):** **Real-ESRGAN (RRDB ×4)** for perceptual blind SR, or **RFDN/ECBSR/IMDN** reparameterized lightweight nets for hard real-time per-tile latency, exported to **ONNX/TensorRT FP16** with tiling.
- **Loss stack for IR:** `L1 (Charbonnier) + edge/gradient (SPSR-style) + FFT/Fourier high-frequency + light LPIPS` for the PSNR/SSIM-oriented stage; add a **U-Net relativistic GAN** term only if you need perceptual sharpness and can tolerate hallucination risk. Down-weight or drop VGG-perceptual (VGG is trained on RGB; gradient + FFT transfer better to single-channel).
- **Joint vs two-stage:** **Two-stage with a shared/transferable encoder backbone** (SR first, then colorization), not one monolithic network. Strong evidence that joint multi-task SR+colorization helps (Sun et al., *Single satellite imagery simultaneous SR + colorization*), but for this task a **decoupled pipeline with a shared restoration backbone and feature feedback** gives better debuggability, lets you swap the guided-SR module, and avoids color bleeding into the SR loss. See §10.

---

## 1. CNN-based SR (foundational, fast, low hallucination)

| Method | Year | Arch idea | Scales | Train data / degradation | Key losses | Benchmark (PSNR) | Repo | Speed |
|---|---|---|---|---|---|---|---|---|
| **SRCNN** | 2014/16 | 3-layer CNN on bicubic-upsampled input | ×2–4 | T91/ImageNet, bicubic | L2 (MSE) | Set5 ×4 ≈ 30.5 dB `[internal]` | `github.com/...` (many) | Tiny, but pre-upsampling = slow per-MAC |
| **FSRCNN** | 2016 | LR-space conv + deconv upsample (fast SRCNN) | ×2–4 | T91+General100, bicubic | L2 | Set5 ×4 ≈ 30.7 dB `[internal]` | yjn870/FSRCNN-pytorch | Very fast; real-time CPU-feasible |
| **VDSR** | 2016 | 20 layers, global residual, gradient clipping | ×2–4 (multi-scale 1 model) | bicubic | L2 | Set5 ×4 ≈ 31.35 dB `[internal]` | — | Moderate |
| **ESPCN** | 2016 | **Sub-pixel (pixel-shuffle)** upsample in LR space | ×2–4 | bicubic | L2 | — | leftthomas/ESPCN | Fast; pixel-shuffle is the standard upsampler now |
| **EDSR** | 2017 | Deep ResNet, **BN removed**, residual scaling | ×2,3,4 | DIV2K, bicubic | L1 | Urban100 ×4 ≈ 26.64 dB `[internal]` | sanghyun-son/EDSR-PyTorch | Heavy (43M), strong PSNR baseline |
| **RDN** | 2018 | Residual Dense Blocks, dense local+global feature fusion | ×2,3,4 | DIV2K | L1 | Urban100 ×4 ≈ 26.61 dB `[internal]` | yulunzhang/RDN | Heavy |
| **RCAN** | 2018 | **Channel attention** + Residual-in-Residual, very deep (400+ layers) | ×2,3,4,8 | DIV2K | L1 | Urban100 ×4 ≈ 26.82 dB `[internal]` | yulunzhang/RCAN | Heavy (~16M); CA is a reusable building block |
| **RRDB (ESRGAN backbone)** | 2018 | **Residual-in-Residual Dense Block**, no BN | ×4 | DIV2K(+Flickr2K) | L1 (PSNR variant) | basis for ESRGAN/Real-ESRGAN | xinntao/ESRGAN | ~16.7M; workhorse generator |

**Takeaways for IR:** EDSR/RCAN/RRDB are reliable, low-hallucination PSNR-oriented backbones. **Pixel-shuffle (ESPCN) upsampling** is the de-facto efficient head. RCAN's **channel attention** and RDN's dense fusion are worth importing into a custom IR net. CNN SR rarely invents structure → good for the "no fake objects" constraint, but limited at recovering genuinely missing high-frequency from coarse thermal.

---

## 2. GAN-based SR (perceptual sharpness; hallucination risk)

| Method | Year | Arch / idea | Scales | Degradation | Key losses | Notes | Repo |
|---|---|---|---|---|---|---|---|
| **SRGAN** | 2017 | SRResNet generator + adversarial + VGG perceptual | ×4 | bicubic | L2/L1 + VGG + adversarial | First photo-realistic SR | tensorlayer/SRGAN |
| **ESRGAN** | 2018 | RRDB generator, **relativistic avg GAN (RaGAN)**, VGG-before-activation | ×4 | bicubic | L1 + VGG + RaGAN | NTIRE PIRM winner | xinntao/ESRGAN |
| **Real-ESRGAN** | 2021 | RRDB + **U-Net SN discriminator**, **high-order synthetic degradation** (blur→noise→JPEG→resize, ×2) | ×2,×4 (+anime) | pure synthetic blind | L1 + VGG + U-Net GAN | De-facto **real-world blind SR baseline**; handles unknown blur/noise/compression | xinntao/Real-ESRGAN |
| **BSRGAN** | 2021 | RRDB + **randomly shuffled degradation** model (blur, down, noise, JPEG in random order) | ×4 | random-shuffle synthetic | L1 + VGG + GAN | Alternative degradation philosophy to Real-ESRGAN; strong generalization | cszn/BSRGAN |
| **SPSR** | 2020 | **Structure-Preserving SR**: gradient-domain branch + gradient loss to stop geometric distortion | ×4 | bicubic | L1 + VGG + GAN + **gradient-map loss** | Directly relevant: preserves edges/structure → less hallucination of geometry | Maclory/SPSR |

**Takeaways for IR:** **Real-ESRGAN and BSRGAN degradation pipelines are the key transferable asset** — synthesize realistic LR-IR from HR-IR (blur kernels, sensor/Gaussian+Poisson noise, downsampling, optional sensor MTF) so the network learns blind robustness. **SPSR's gradient-domain supervision** is exactly what low-contrast IR needs (it preserves edges without inventing texture). GAN sharpness must be balanced against the **no-fake-objects** evaluation criterion — keep adversarial weight low or gate it behind a structure/gradient constraint.

---

## 3. Transformer-based SR (current SOTA quality)

| Method | Year | Arch idea | Scales | Key losses | Quality vs SwinIR | Params | Speed | Repo |
|---|---|---|---|---|---|---|---|---|
| **SwinIR** | 2021 | Swin (shifted-window) self-attention, RSTB blocks; SR/denoise/JPEG variants | ×2,3,4,8 | L1 (+GAN var.) | baseline SOTA-2021 | ~11.8M (SR) | Moderate (windowed attn) | JingyunLiang/SwinIR |
| **HAT** | 2023 | **Hybrid Attention Transformer**: channel attn + window self-attn + **overlapping cross-attention**; "activates more pixels" | ×2,3,4 | L1 (+ same-task pretrain) | **+0.48–0.64 dB Urban100 ×4**, +0.34–0.45 dB Manga109 over SwinIR; highest PSNR among surveyed | ~20.8M (HAT), ~40M (HAT-L) | Slower than SwinIR; HAT-L heavy | XPixelGroup/HAT |
| **DAT** | 2023 | **Dual Aggregation Transformer**: alternating spatial + channel attention, adaptive interaction | ×2,3,4 | L1 | ≥ SwinIR, competitive with HAT | ~14.8M | Moderate | zhengchen1999/DAT |
| **DRCT** | 2024 | **Dense-Residual-Connected Transformer**; mitigates info bottleneck in deep SwinIR | ×2,3,4 | L1 | ≥ HAT on several sets, **near-zero prominent artifacts** | ~14M `[internal]` | Moderate | ming053l/DRCT |
| **GRL** | 2023 | **Global-Regional-Local** anchored attention; efficient long-range modeling | ×4 | L1 | Strong, efficient variant | scalable | Efficient tiers available | ofsoundof/GRL-Image-Restoration |
| **SwinFIR** | 2022 | SwinIR + **Fast Fourier Convolution** block (global frequency receptive field) + improved training | ×2,3,4 | L1 (+ FFT-aware) | > SwinIR | ~similar to SwinIR | Moderate | Zdafeng/SwinFIR |
| **Restormer** | 2022 | **MDTA** (channel-wise transposed attn, linear in resolution) + **GDFN**; restoration (deblur/denoise/derain) | full-res | L1 (+ Charbonnier) | SOTA restoration; deblur GoPro ~32.9 dB | ~26M | Efficient at high-res (linear attn) | swz30/Restormer |
| **Uformer** | 2022 | U-shaped **window-attention transformer** with modulators; restoration | full-res | Charbonnier | Strong denoise/deblur | ~20M | Moderate | ZhendongWang6/Uformer |

**2025 frontier (web):** NTIRE-2025 ×4 SR teams integrated **Mamba** (state-space, linear long-range) into HAT for global context; HAT-L/DRCT remain the artifact-clean quality leaders.

**Takeaways for IR:** **HAT / DRCT = best quality** if latency budget allows. **Restormer / Uformer** are the right choice if the task is dominated by **deblurring + denoising + sharpening** (which IR enhancement largely is) rather than large upscale factors — their attention is **linear in resolution**, so they tile better for large satellite scenes. **SwinFIR's FFT block** is attractive because IR detail lives in specific frequency bands. For a fast deployable transformer, use **GRL-tiny or a small SwinIR/HAT** config.

---

## 4. Diffusion-based SR (top perceptual realism; watch latency + hallucination)

| Method | Year | Idea | Steps / speed | Strengths | Repo |
|---|---|---|---|---|---|
| **SR3** | 2021 | DDPM conditioned on LR (iterative denoising) | ~100–1000 steps (slow) | High perceptual quality, set the paradigm | (unofficial: Janspiry/Image-Super-Resolution-via-Iterative-Refinement) |
| **SRDiff** | 2022 | Diffusion on **residual** (HR−upsampled LR) | ~100 steps | Stable, diverse outputs | LeiaLi/SRDiff |
| **StableSR** | 2023 | Fine-tune **pretrained Stable Diffusion** prior + time-aware encoder + CFW | many steps (slow) | Rich texture from large prior | IceClear/StableSR |
| **DiffBIR** | 2023 | **Two-stage**: degradation removal (restoration net) → SD-based generative refinement | many steps | Blind general restoration | XPixelGroup/DiffBIR |
| **ResShift** | 2023 (TPAMI'24) | **Residual-shifting** Markov chain between HR and LR | **~15 steps** | Big speedup over DDPM-SR, strong quality | zsyOAOA/ResShift |
| **SinSR** | 2024 | Distill ResShift to **1 step** via deterministic + consistency | **1 step** | Near-real-time diffusion SR | wyf0912/SinSR |
| **OSEDiff** | 2024 (NeurIPS) | **One-step** SD distillation via VSD loss | **1 step** | ~6× faster than ResShift, ~105× vs StableSR | cswry/OSEDiff |
| **EDiffSR** | 2023 (TGRS) | **Remote-sensing** efficient DPM: EANet (simple gate + simplified channel attn) noise predictor + Conditional Prior Enhancement Module | efficient (RS-tuned) | Beats CNN/GAN/DPM on 4 RS datasets; **built for satellite** | XY-boy/EDiffSR |

**Inference-speed note (web):** one-step distilled diffusion (SinSR, OSEDiff, TSD-SR) closes most of the latency gap — TSD-SR ~4× faster than ResShift and ~90× vs StableSR; OSEDiff ~6× vs ResShift. Still heavier per tile than CNN/lightweight nets.

**Takeaways for IR:** Diffusion gives the most realistic texture but **hallucinates** — risky against the "no fake objects" criterion for an analyst-facing product. If used, prefer **EDiffSR** (designed for remote sensing) or **ResShift/SinSR** (few/one-step) and keep them as a *quality-max optional path*, not the default. The **residual-prediction idea (SRDiff/ResShift)** is also useful conceptually: predict the high-freq residual on top of a deterministic guided-SR base.

---

## 5. Blind / real-world SR & degradation modeling (the practical core)

| Method | Year | Idea | Why it matters for IR |
|---|---|---|---|
| **Real-ESRGAN degradation** | 2021 | **High-order** synthetic pipeline (two rounds of blur→resize→noise→JPEG, + sinc ringing) | Build LR-IR from HR-IR with realistic sensor blur + noise; train blind model |
| **BSRGAN degradation** | 2021 | **Random-shuffle** order of degradations | Alternative, broader degradation coverage |
| **KernelGAN** | 2019 | **Estimate the image-specific blur kernel** internally (zero-shot) | Useful when true sensor MTF/PSF unknown; estimate per-scene kernel |
| **DASR** | 2021 | **Degradation-Aware SR** via contrastive degradation representation | Adapt to varying IR sensor/acquisition conditions implicitly |
| **DAN / IKC** | 2020 | Iterative kernel correction / unfolding for blind SR | Classical blind-SR baselines |
| **Mixed/probabilistic degradation (2025)** | 2025 | Probabilistic mix of degradations incl. ringing/overshoot | Latest refinement of the synthetic-LR philosophy |

**Recommendation:** model the **real Landsat/Sentinel degradation explicitly** — sensor **PSF/MTF** (Gaussian or measured kernel), **bicubic/area downsample** to bridge native→target GSD, **Poisson+Gaussian** photon/read noise, optional **striping** artifacts characteristic of pushbroom IR sensors. Combine measured kernels with Real-ESRGAN/BSRGAN randomization so the SR net is blind-robust.

---

## 6. Satellite / remote-sensing-specific SR

| Method | Year | Type | Idea | Repo / source |
|---|---|---|---|---|
| **HighRes-net** | 2020 | **MISR** (multi-frame) | Recursive fusion: learns co-registration + fusion + upsample + registration-at-loss end-to-end; won ESA PROBA-V MFSR | ElementAI/HighRes-net |
| **RAMS** | 2020 | MISR | **3D-conv residual attention** multi-image SR; strong on PROBA-V | EscVM/RAMS |
| **TR-MISR** | 2022 | MISR | **Transformer** fusion of multiple LR frames | (paper; IEEE JSTARS) |
| **DSen2 / Sen2SR** | 2018–25 | Guided SISR | Residual CNN upscales 20 m Sentinel-2 bands **guided by 10 m bands** | up42/DSen2 |
| **L1BSR** | 2023 | **Self-supervised** | Self-supervised SR + band alignment from a **single Sentinel-2 L1B** product (uses detector overlap) — no HR ground truth needed | centreborelli/L1BSR |
| **WorldStrat (baselines)** | 2022 | Dataset + MISR/SISR baselines | SPOT 6/7 HR (~1.5 m) paired with Sentinel-2 LR; HighRes-net variant trains in ~30 min/V100 | worldstrat/worldstrat |
| **EDiffSR** | 2023 | Diffusion (RS) | Efficient DPM for RS SR (see §4) | XY-boy/EDiffSR |
| **PSRGAN** | 2021 | **Thermal/IR GAN** | **Transfer learning**: main path extracts features from **visible images**, branch path models IR patterns → guided IR SR | (paper, IEEE GRSL) |
| **Thermal IR SR review** | 2022 (v5 2024) | Survey | Systematic review of IR SR; emphasizes **guided** (visible-guided) SR because IR has fewer patterns | arXiv:2212.12322 |
| **CoReFusion** | 2023 | Guided thermal SR | **Contrastive-regularized fusion** of visible guide + LR thermal; robust to missing guide | aniketgsd/CoReFusion `[internal]` |
| **PBVS Thermal SR Challenge** | 2023–24 | Benchmark | ×8 thermal SR **guided by HR visible**, cross-spectral (Balser/TAU2) dataset | PBVS workshop |

**Takeaways for IR (critical):** The satellite/thermal literature **strongly favors guidance** — both **MISR** (fuse multiple revisits of the same scene) and **cross-sensor guided SR** (visible/pan guide for thermal). For Landsat, this directly maps to using **OLI 15 m pan / VIS-NIR bands to guide TIRS thermal SR**, or fusing multi-date acquisitions (MISR). **L1BSR** offers a self-supervised recipe when you lack HR thermal ground truth.

---

## 7. Lightweight / fast SR (the latency backup tier)

| Method | Year | Idea | Speed lever | Repo |
|---|---|---|---|---|
| **IMDN** | 2019 | **Information multi-distillation** blocks (split + progressive refinement) | small, AIM-2019 winner | Zheng222/IMDN |
| **RFDN** | 2020 | **Residual feature distillation** (refines IMDN, lighter) | AIM-2020 efficient SR winner | njulj/RFDN |
| **ECBSR** | 2021 | **Edge-oriented conv block** + **reparameterization** (train multi-branch, infer single 3×3) | **zero extra inference cost**, edge-device targeted | xindongzhang/ECBSR |
| **ABPN / XLSR** | 2021 | Anchored / extremely-light nets | Won **Mobile Real-Time SISR 2021**; ~tens-of-ms on Galaxy S21 | (challenge repos) |
| **NAFNet** | 2022 | **Nonlinear-Activation-Free** blocks (simple gate + simplified channel attn) | **GoPro deblur 33.69 dB at 8.4% of prior compute**; SIDD denoise 40.30 dB | megvii-research/NAFNet |
| **RLFN** | 2022 | Residual Local Feature Network | NTIRE-2022 efficient SR winner | bytedance/RLFN `[internal]` |

**SR-vs-latency tradeoff:** quality ranking roughly **diffusion (HAT-L) > transformer (HAT/DRCT) > RRDB-GAN > deep CNN (RCAN/EDSR) > lightweight (RFDN/ECBSR/IMDN)**, while **per-tile latency is the inverse**. **Reparameterization (ECBSR)** is the best "free lunch": train an expressive multi-branch net, fold it to a single conv for inference. **NAFNet** is the sweet spot for combined **sharpen+denoise** at low compute.

---

## 8. Sharpening / deblurring / contrast (IR enhancement specifics)

| Method | Type | Use for IR |
|---|---|---|
| **NAFNet** | Learned deblur/denoise | Primary learned **sharpening + denoising** backbone (cheap, SOTA restoration) |
| **Restormer** | Learned deblur (transformer) | High-res deblur, linear-in-resolution attention; good for large tiles |
| **Uformer** | Learned restoration | Alternative U-shaped restoration |
| **Unsharp masking** | Classical | Cheap edge boost baseline; risk of halos/noise amplification |
| **CLAHE** | Classical contrast | **Strong fit for low-contrast IR** — adaptive local histogram equalization; standard preprocessing for thermal |
| **Retinex (e.g., Retinexformer / classical MSR)** | Illumination-reflectance | Low-light/low-contrast enhancement; separates illumination for IR contrast lift |
| **Gamma / percentile stretch** | Classical | Per-tile radiometric normalization before SR |

**Recommendation:** Treat **contrast/radiometric normalization (CLAHE or percentile stretch) as preprocessing**, do **SR + learned sharpening (NAFNet/Restormer-style)** in the network, and keep **unsharp masking** only as a cheap post-step if needed. Retinex-based modules are worth testing for very low-contrast night IR.

---

## 9. Loss functions — recommended stack for IR

| Loss | What it does | IR relevance |
|---|---|---|
| **L1 / Charbonnier** (√(x²+ε²)) | Pixel fidelity; Charbonnier = robust smooth-L1 | **Base loss.** Charbonnier preferred (robust to IR noise/outliers, better than L2 which over-smooths) |
| **L2 / MSE** | Pixel fidelity, maximizes PSNR | Tends to blur; use only if PSNR is the sole metric |
| **Gradient / edge loss (SPSR)** | Penalize gradient-map difference | **High value** — preserves faint IR edges/structure, discourages geometric hallucination |
| **FFT / Fourier loss (SwinFIR, FDPL)** | Match frequency spectra; emphasize missing high-freq | **High value** — IR detail is band-specific; directly targets high-freq recovery, complements spatial loss |
| **Perceptual VGG** | Match deep RGB features | **Down-weight/skip** — VGG is RGB-trained; weak transfer to single-channel IR (replicate-to-3ch is a hack) |
| **LPIPS** | Learned perceptual similarity | Light weight; better human-perception match than VGG, but still RGB-prior |
| **Adversarial (RaGAN / U-Net SN-disc)** | Realism / sharp texture | **Optional, low weight** — improves sharpness but risks fake objects; gate behind gradient/structure loss |
| **Contextual loss** | Feature-distribution match (non-aligned) | Useful when guide and target are not pixel-perfectly aligned (cross-sensor guided SR) |

**Recommended IR loss stack (PSNR/SSIM-oriented enhancement):**
```
L = 1.0·L1_Charbonnier
  + 0.1·L_gradient            # SPSR-style edge/structure preservation
  + 0.1·L_FFT                 # Fourier high-frequency emphasis (SwinFIR/FDPL)
  + 0.05·L_LPIPS              # light perceptual (optional)
  ( + 0.01·L_adv  U-Net RaGAN # ONLY if perceptual sharpness needed; risks hallucination )
  ( + 0.1·L_contextual        # ONLY for guided/cross-sensor SR with imperfect alignment )
```
Rationale: IR has **lower texture entropy, lower contrast, heavier noise, strong low-frequency content**. Pixel + **gradient + FFT** losses (both spatial and frequency domain — shown to be complementary) recover edges/high-freq without RGB-perceptual bias and without inventing objects. Keep adversarial small and structure-constrained.

---

## 10. Joint (SR+colorization) vs two-stage — verdict

**Evidence both ways (web):**
- **Pro-joint:** Sun et al., *"Single satellite imagery simultaneous super-resolution and colorization using multi-task deep neural networks"* — two concurrent task networks where **colorization features feed back into the SR network**, with **combined losses**; reports mutual cooperation improving both. Older work (Brown et al., *Colorization for Single Image SR*) shows color cues can aid SR.
- **Pro-decoupled:** Real-world blind-restoration systems (DiffBIR, two-stage pipelines) deliberately **separate degradation removal from generative synthesis** for controllability and to prevent error/bias coupling.

**Verdict — TWO-STAGE with a shared/transferable restoration backbone (SR → colorization), not a single monolithic network:**
1. **Debuggability & metrics separation** — SR is graded by PSNR/SSIM vs an IR HR target; colorization by FID vs RGB. A joint loss entangles them and makes the **"no hallucination"** audit harder (you can't tell whether a fake edge came from SR or color synthesis).
2. **Modularity** — the **guided-SR module** (which needs a cross-sensor guide and explicit degradation modeling) is architecturally different from the **IR→RGB translation** module (pix2pix/diffusion). Swapping or retraining one shouldn't destabilize the other.
3. **Reuse without coupling** — capture most of the joint-learning benefit by **sharing/initializing the encoder** between SR and colorization and optionally adding **light feature feedback** (as in Sun et al.) — i.e., a shared backbone with two heads — rather than one fused loss. This keeps the cooperation benefit while preserving stage-wise control.
4. **Latency control** — two stages let you run a **fast SR tier** independently and budget colorization separately; a fused net forces one latency profile.

If a single model is mandated for deployment simplicity, use a **shared-encoder / two-decoder multi-task net** with separately weighted SR and color losses and a structure-preservation constraint — but the **default recommendation is the decoupled pipeline**.

---

## 11. O(1) / fast-inference engineering (per-tile latency)

- **Architecture:** prefer **pixel-shuffle upsampling** (LR-space compute), **reparameterized** blocks (ECBSR — multi-branch train, single-conv infer), **NAFNet** simple-gate blocks. Avoid pre-upsampling (SRCNN/VDSR) and 1000-step diffusion for the default path.
- **Precision:** **FP16/AMP** inference (≈2× throughput, ~half memory on tensor-core GPUs); INT8 PTQ for edge if quality holds.
- **Tiling:** process large scenes in **overlapping tiles** (e.g., 256–512 px, 8–16 px overlap) with feathered blending to avoid seams; tiling also bounds peak memory and lets you parallelize tiles. (Real-ESRGAN/SwinIR ship tiled inference.)
- **Export/runtime:** **ONNX → TensorRT** (fused kernels, FP16/INT8) or **torch.compile**; static shapes per tile size for best TRT plans. Batch tiles. CNN/lightweight nets convert cleanly; transformers/diffusion are heavier to optimize.
- **One-step diffusion** (SinSR/OSEDiff) if a diffusion-quality path is required under latency limits.

---

## 12. Recommended SR approach (final)

**PRIMARY — Guided / reference-based SR with explicit degradation modeling:**
- **Backbone:** restoration-grade net — **HAT-light or NAFNet/Restormer-style encoder–decoder** with **channel attention** (RCAN/NAFNet) and an **FFC/SwinFIR frequency block**.
- **Guidance branch (PSRGAN/CoReFusion/DSen2 pattern):** ingest co-registered **HR guide** (Landsat OLI 15 m pan / VIS-NIR, or Sentinel-2 10 m). Fuse guide features into the IR reconstruction; use **contextual loss** for imperfect alignment. Where no HR thermal ground truth exists, adopt **L1BSR-style self-supervision** and/or **MISR** fusion of multi-date revisits.
- **Degradation:** train blind with **measured sensor PSF/MTF + Real-ESRGAN/BSRGAN randomization + Poisson-Gaussian noise + striping**.
- **Loss:** stack from §9 (Charbonnier + gradient + FFT + light LPIPS; adversarial optional/low).
- **Why:** thermal/IR is **low native resolution with little real high-frequency content** — pure SISR must hallucinate, violating the no-fake-objects rule. A **real cross-sensor HR guide** supplies genuine structure → higher fidelity *and* lower hallucination. This is the consensus of the IR-SR survey and thermal-SR challenges.

**BACKUP — Fast blind SISR (no guide / latency-critical):**
- **Quality backup:** **Real-ESRGAN (RRDB ×4)** retrained on synthetic-degraded IR.
- **Speed backup:** **RFDN / ECBSR / IMDN** (reparameterized, pixel-shuffle), or **NAFNet-small** for sharpen+denoise — exported to **TensorRT FP16 + tiling** for hard per-tile latency.
- **Optional quality-max path:** **EDiffSR** (RS diffusion) or **SinSR/OSEDiff** (1-step) when realism matters and latency permits — used with a structure-preservation check.

**Pipeline placement:** preprocess (CLAHE/percentile stretch) → **guided SR + learned sharpening** → (shared-encoder handoff) → IR→RGB colorization. **Two-stage, shared/transferable backbone.**

---

## Sources

- State-of-the-Art Transformer Models for Image Super-Resolution (2025) — https://arxiv.org/pdf/2501.07855
- NTIRE 2025 Challenge on Image SR (×4) — https://arxiv.org/html/2504.14582v1
- HAT: Activating More Pixels in Image SR Transformer — https://arxiv.org/pdf/2205.04437 ; HAT (restoration) — https://arxiv.org/pdf/2309.05239
- SwinFIR — https://arxiv.org/pdf/2208.11247
- Real-ESRGAN — https://arxiv.org/pdf/2107.10833 ; topic overview — https://www.emergentmind.com/topics/real-esrgan
- Real-world blind SR via mixed/probabilistic degradation (2025) — https://www.sciencedirect.com/science/article/abs/pii/S095070512501682X
- Efficient Diffusion (ResShift), TPAMI 2024 — https://dl.acm.org/doi/abs/10.1109/TPAMI.2024.3461721
- OSEDiff (One-Step Effective Diffusion), NeurIPS 2024 — https://proceedings.neurips.cc/paper_files/paper/2024/file/a8223b0ad64007423ffb308b0dd92298-Paper-Conference.pdf
- TSD-SR (One-Step), CVPR 2025 — https://arxiv.org/html/2411.18263v3
- EDiffSR (RS diffusion), TGRS 2023 — https://arxiv.org/abs/2310.19288
- HighRes-net (MISR) — https://arxiv.org/pdf/2002.06460
- WorldStrat dataset — https://arxiv.org/html/2207.06418v2
- Cross-sensor SR of Sentinel-2 time series (L1BSR-related) — https://arxiv.org/pdf/2404.16409
- Sentinel-2 SR via channel attention + high-freq enhancement (DSen2 lineage) — https://www.frontiersin.org/journals/remote-sensing/articles/10.3389/frsen.2025.1644460/full
- Advancing Image SR in Remote Sensing: A Survey (2025) — https://arxiv.org/pdf/2505.23248
- Infrared Image SR: Systematic Review & Future Trends — https://arxiv.org/html/2212.12322v5
- Infrared Image SR via GAN — https://arxiv.org/html/2312.00689
- Enhancement of guided thermal image SR — https://www.sciencedirect.com/science/article/abs/pii/S0925231223013206
- CoReFusion: Contrastive Regularized Fusion for Guided Thermal SR — https://arxiv.org/pdf/2304.01243
- RFDN (Residual Feature Distillation) — https://arxiv.org/pdf/2009.11551 ; RLFN — https://arxiv.org/pdf/2205.07514
- NAFNet / Simple Baselines for Image Restoration — https://arxiv.org/abs/2204.04676 ; https://github.com/megvii-research/NAFNet
- Real-Time SR on Mobile Devices — https://ar5iv.labs.arxiv.org/html/2206.01777
- Frequency Domain Perceptual Loss for SR — https://arxiv.org/abs/2007.12296 ; Fourier Space Losses for SR — https://arxiv.org/pdf/2106.00783
- Investigating Loss Functions for Extreme SR — https://openaccess.thecvf.com/content_CVPRW_2020/papers/w31/Jo_Investigating_Loss_Functions_for_Extreme_Super-Resolution_CVPRW_2020_paper.pdf
- Single satellite imagery simultaneous SR + colorization (multi-task) — https://www.sciencedirect.com/science/article/abs/pii/S1047320318300488
- Colorization for Single Image SR (Brown et al.) — http://www.cse.yorku.ca/~mbrown/pdf/eccv10_SR.pdf

*Numbers/repos marked `[internal]` are from internal knowledge (cutoff Jan 2026) and should be re-verified against the linked primary sources before publication.*
