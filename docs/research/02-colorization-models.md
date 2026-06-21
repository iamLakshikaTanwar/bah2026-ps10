# 02 — Colorization & IR→RGB Translation Models: SOTA Survey

**Project:** Infrared (IR/thermal) satellite imagery → realistic RGB, **without hallucinating fake objects**, while preserving semantic ground truth (PS-10).
**Target data:** Landsat 8/9 thermal + IR bands, paired & co-registered with RGB; per-tile fast inference; metrics PSNR / SSIM / FID + downstream detection/segmentation; "no hallucination" is a hard constraint.
**Date:** 2026-06-21. **Method:** web research (2023–2025 sources) cross-checked with internal knowledge (cutoff Jan 2026). Items not confirmable on the web in this session are tagged **[from internal knowledge]**.

> **Bottom line up front.**
> - **Primary recommendation:** a **paired, conditional GAN built on the Pix2PixHD generator (or a Restormer/Uformer restoration backbone) with a DDColor-style query-based color decoder**, trained with a **multi-term fidelity loss** (L1 + multi-scale SSIM + LPIPS + feature-matching + LSGAN-hinge adversarial + Lab chrominance + histogram/color-consistency + edge/gradient + semantic-segmentation guidance + light TV). Deterministic, single forward pass, fast per tile, low hallucination because output is pixel-anchored to the IR input via L1/SSIM/edge terms and semantic masks.
> - **Backup recommendation:** a **conditional latent diffusion / Brownian-Bridge Diffusion Model (BBDM)** conditioned on the IR tile (+ optional ControlNet edge/semantic conditioning), used when maximum perceptual realism/FID is needed and slightly higher latency is acceptable; constrained with classifier-free guidance + low stochasticity + inference-time color/structure guidance to bound hallucination.

---

## 0. How this problem differs from generic colorization

Classic colorization predicts chrominance (a*b*) from a known luminance L that is *already the correct RGB luminance*. Here the input is a **thermal/IR radiance**, which is **not** the RGB luminance — a hot road and a cool road can look identical in RGB, and vegetation that is "bright" in NIR is mid-tone green in RGB. So this is genuinely an **image-to-image translation** problem (cross-modal), closer to SAR→optical than to "old-photo colorization."

Consequences that drive every choice below:
1. **Paired + co-registered supervision is available** (Landsat bands are inherently aligned). This is the single biggest lever against hallucination — we should *exploit pairing*, not throw it away with unpaired methods.
2. The mapping is **one-to-many / ambiguous** (many RGB appearances share one IR signature). Pure L2 regression → desaturated "sepia" mean; pure GAN/diffusion → vivid but **invented** detail. The art is balancing the two.
3. **Hallucination = inventing structure or mislabeling land cover.** Anti-hallucination must be *designed in* via structure-preserving losses, semantic conditioning, and (optionally) uncertainty, not hoped for.

---

## 1. GAN-based image-to-image translation (general backbones)

| Method | Paired? | Core architecture | Key losses | Strengths | Weaknesses | Hallucination risk | Repo |
|---|---|---|---|---|---|---|---|
| **Pix2Pix** | Paired | U-Net generator + PatchGAN (70×70) discriminator; conditional GAN | cGAN (adversarial) + **L1** | Simple, strong baseline, deterministic, fast | 256² native, mild blur, limited high-freq detail | **Low** (L1 anchors to input) | phillipi/pix2pix |
| **Pix2PixHD** | Paired | Coarse-to-fine **global+local** generator; **multi-scale** discriminators (3) | cGAN + **feature-matching (FM)** + VGG perceptual | High-res (2k), crisp, FM loss stabilizes & sharpens | Heavier; still single-modal-ish | **Low** | NVIDIA/pix2pixHD |
| **CycleGAN** | **Unpaired** | Two generators + two PatchGAN discriminators | adversarial + **cycle-consistency** + identity | No pairs needed; good when alignment is poor | **Cycle loss tolerates geometry change → can move/invent objects**; mode issues | **High** | junyanz/CycleGAN |
| **DualGAN** | Unpaired | Same dual-generator idea; WGAN objective | WGAN adversarial + reconstruction (cycle) | WGAN improves training stability vs vanilla | Same structural-drift risk as CycleGAN | **High** | duxingren14/DualGAN |
| **UNIT** | Unpaired | Shared-latent VAE-GAN (two encoders share latent) | VAE (KL) + adversarial + cycle | Shared latent gives coherent translation | Shared-latent assumption strong; blurry | Med–High | mingyuliu/UNIT |
| **MUNIT** | Unpaired | **Disentangled** content (shared) + style (per-domain) codes | adv + content/style reconstruction + cycle | **Multimodal** outputs; controllable style | Stochastic style → more invented appearance; complex | **High** (stochastic) | NVlabs/MUNIT |
| **CUT / FastCUT** | **Unpaired** | One-sided generator; **PatchNCE** contrastive loss (mutual-info between input/output patches) | adversarial + **PatchNCE** (+identity for CUT) | **No cycle network** → faster, half memory (FastCUT), better structure preservation than CycleGAN; one image possible | Still unpaired, no pixel anchor to GT color | Med (better structure than CycleGAN, but no GT-color anchor) | taesungp/contrastive-unpaired-translation |

**Reading for our task.** Because we **have aligned pairs**, the **paired** Pix2Pix family is strictly preferable: the L1/FM/perceptual terms directly pin output structure to the GT, which is exactly the no-hallucination requirement. CycleGAN/DualGAN/MUNIT are unpaired and use cycle-consistency, which is *known to permit geometric edits and object invention* (the network can "cheat" the cycle by hiding/adding content) — a poor fit. **CUT/FastCUT** is the best *unpaired* option (its PatchNCE preserves input structure far better than cycle-consistency and trains faster), and is a sensible fallback if some Landsat tiles cannot be perfectly co-registered. **Pix2PixHD's multi-scale discriminator + feature-matching loss** is the most directly reusable generator/discriminator design for high-resolution satellite tiles.

- CUT details confirmed: one-sided, PatchNCE cross-entropy contrastive loss with temperature τ=0.07; CUT uses identity loss with λ_NCE=1, FastCUT drops identity with λ_NCE=10, ~half GPU memory and ~2× faster than CycleGAN.
- Pix2PixHD confirmed: multi-scale discriminator + robust feature-matching loss + more powerful coarse-to-fine generator over Pix2Pix.

---

## 2. Thermal / IR-specific translation & colorization

| Method | Paired? | Architecture | Key losses | Domain | Notes / metrics | Hallucination risk | Repo |
|---|---|---|---|---|---|---|---|
| **TIC-CGAN** (Thermal IR Colorization via cGAN) | Paired | Conditional GAN (Pix2Pix-style, U-Net + PatchGAN) on LWIR 8–15 µm | cGAN + L1 | Traffic-scene LWIR→RGB | First cGAN to colorize thermal traffic IR; KAIST/FLIR-style data | **Low** | (Kuang et al.; arXiv 1810.05399) |
| **PearlGAN** | **Unpaired** | ToDayGAN/CycleGAN base + **Top-Down Guided Attention (TDGA)** | cycle + adv + **attentional loss (AD + ACCS)** + **Structured Gradient Alignment (SGA)** + SSIM | Nighttime TIR → daytime color (NTIR2DC) | Explicitly tackles **semantic-encoding entanglement & geometric distortion**; FLIR + KAIST; new APCE edge metric, mIoU, mAP | Med (attention+SGA reduce it vs CycleGAN) | FuyaLuo/PearlGAN |
| **ToDayGAN** | Unpaired | CycleGAN variant, 3 separate discriminators (RGB/blur/grad) | adversarial + cycle | Night→day RGB; reused for TIR colorization | Backbone PearlGAN builds on | High (cycle) | AAnoosheh/ToDayGAN |
| **I2V-GAN** | Unpaired | Infrared→visible **video**; perceptual + spatio-temporal | adv + cycle + perceptual + **temporal consistency** | IR→visible video | Adds temporal smoothness; ROB dataset | High | BIT-DA/I2V-GAN |
| **IR2VI / ThermalGAN** | Unpaired | Cross-modality color↔thermal (person re-ID, surveillance) | adv + L1/feature | RGB↔thermal | ThermalGAN generates thermal *from* RGB (reverse dir) for re-ID; useful as augmentation, not our forward task | n/a | vlkniaz/ThermalGAN |
| **MUGAN** | Paired | **Mixed-skipping U-Net** generator + GAN | adv + L1 (+ perceptual) | TIR→RGB colorization | Dense/mixed skip connections improve detail transfer; reported PSNR/SSIM gains over Pix2Pix on KAIST | **Low** | (Liang et al., 2022/2023) |
| **MCU-GAN** (multi-convolution fusion) | Paired/mixed | Multi-conv fusion generator | adv + L1 + perceptual | IR colorization | 2024-era; multi-receptive-field fusion for texture | Low–Med | [from internal knowledge] |
| **Top-Down Guided Attention (Luo 2021)** | Unpaired | = PearlGAN line | attentional + SGA | NTIR | See PearlGAN row | Med | FuyaLuo/PearlGAN |
| **Memory-Guided Collaborative Attention (MornGAN)** | Unpaired | Memory module + collaborative attention | adv + cycle + memory/online-semantic-distillation | NTIR colorization | Reduces semantic mismatch in unpaired night IR | Med | FuyaLuo/MornGAN [from internal knowledge] |
| **FoalGAN** (Feedback Object-Appearance) | Unpaired | Adds **feedback-based object-appearance learning** + dual-feedback | adv + cycle + appearance feedback | NTIR colorization | 2023; improves small-object color fidelity, less mislabeling | Med | (arXiv 2310.15688) |
| **Implicit Multi-Spectral Transformer (IMSTrans)** | Paired | Lightweight transformer, implicit neural representation | L1 + perceptual | Visible↔IR translation | 2024; **lightweight & fast** — relevant to per-tile latency | Low | (arXiv 2404.07072) |
| **Edge-guided IR colorization** | Paired | cGAN + explicit Canny/edge branch as condition | cGAN + L1 + **edge** | TIR colorization | Edge conditioning curbs structure drift | **Low** | various |

**Reading for our task.** The thermal-specific literature is dominated by *nighttime driving* scenes (FLIR/KAIST), which are **unpaired** because day-RGB and night-IR can't be co-captured — hence the heavy reliance on CycleGAN-style cycle-consistency and clever attention/gradient losses (PearlGAN's **SGA** and **TDGA**, MornGAN's memory, FoalGAN's appearance feedback) to fight the resulting distortion/hallucination. **Our Landsat case is fundamentally easier and safer because the bands are paired and co-registered**, so we can use a *paired cGAN* (TIC-CGAN/MUGAN style) and **borrow PearlGAN's structured-gradient-alignment and attention ideas as auxiliary losses** rather than as a crutch for missing pairs. **MUGAN's mixed-skip U-Net** and **edge-guided conditioning** are directly transferable. IMSTrans is worth noting for **fast inference**.

---

## 3. Diffusion-based colorization / translation (highest fidelity, controllable hallucination)

| Method | Paired? | Architecture | Conditioning | Key loss | Strengths | Weaknesses | Halluc. risk | Repo |
|---|---|---|---|---|---|---|---|---|
| **Palette** | Paired | Image-to-image **conditional DDPM** (U-Net), input concatenated | grayscale/source image | denoising (L1/L2 on ε or x0) | Unified i2i (colorize/inpaint/uncrop); strong FID | Slow (iterative); stochastic | (Saharia et al.; community repos) |
| **SR3** | Paired | Conditional DDPM **super-resolution** by repeated refinement | LR image | denoising MSE | SOTA SR realism; **directly relevant to the SR/sharpening sub-task** | Slow; can add fine detail not in GT | (Saharia et al.) |
| **BBDM** | Paired | **Brownian-Bridge** diffusion in **VQGAN latent** space | endpoints = source & target (bridge), *not* conditional concat | bridge ELBO / x0 prediction | Models i2i *directly* domain-to-domain (less domain gap than conditional DM); strong FID/LPIPS on translation benchmarks | Latent autoencoder can blur fine geo detail; iterative | xuekt98/BBDM |
| **Conditional latent diffusion (LDM/Stable-Diffusion-based)** | Paired | LDM U-Net in VAE latent, cross-attention/concat conditioning | source image (+text/palette) | latent denoising | Efficient (latent), leverages pretrained priors | Latent compression risks small-object loss/hallucination; needs careful conditioning | CompVis/latent-diffusion |
| **ControlNet** | Paired | Frozen diffusion + trainable **condition encoder** (edge/seg/depth/grayscale) | **edge / segmentation / luminance maps** | denoising | **Best structural control** — propagates spatial structure from condition; excellent anti-drift lever | Inherits base-model bias; slow | lllyasviel/ControlNet |
| **DDColor** | Paired | **Dual decoders**: pixel decoder (spatial) + **query-based color decoder**; cross-attention to multi-scale features; ConvNeXt encoder | grayscale | L1 + perceptual + adversarial + **colorfulness loss** | **SOTA single-pass colorization**, semantic-aware, *deterministic* (not iterative → fast), strongly reduces **color bleeding** | Designed for L→ab (true luminance); needs adaptation for IR-as-input | piddnad/DDColor |
| **ColorizeDiffusion / DiffColor / "Colorize at Will"** | Paired | Diffusion prior + control (text/reference) | reference / text / palette | denoising + guidance | Flexible, high realism, reference-guided | Stochastic, heavy, text-bias hallucination | various (2024) |
| **Cold Diffusion** | Paired | Generalized (deterministic) degradation→restoration diffusion | source | restoration | Can be **deterministic** (less random hallucination) | Less mature for colorization | arpitbansal297/Cold-Diffusion |
| **Conditional BBDM for VHR SAR→Optical** | Paired | BBDM specialized to very-high-res SAR→optical | SAR tile | bridge | **Direct remote-sensing precedent** for diffusion i2i with fidelity focus | New, compute-heavy | (arXiv 2408.07947) |

**Reading for our task.** Diffusion gives the **best FID/realism** and, with **ControlNet-style edge/segmentation conditioning + classifier-free guidance + low/zero stochasticity**, hallucination can be *bounded*. **DDColor** is the standout for a **deterministic, single-pass, semantic-aware** design — its query-based color decoder and colorfulness loss are exactly what enforces "water→blue, veg→green" while killing color bleeding, and it is fast (no iterative sampling). **BBDM** (and its SAR→optical VHR variant) is the most principled *translation* diffusion model and is the strongest pure-diffusion backup. Plain SD-based LDM is risky for satellites (latent compression can drop or invent small objects) unless tightly conditioned.

- DDColor confirmed: ICCV 2023, pixel decoder + query-based color decoder, cross-attention over multi-scale semantics to reduce color bleeding, colorfulness loss; deterministic feed-forward.
- BBDM confirmed: CVPR 2023, first Brownian-bridge i2i; bidirectional bridge process instead of conditional generation to reduce the cross-domain gap; latent-space (VQGAN) operation.

---

## 4. Transformer / hybrid backbones

| Method | Paired? | Architecture | Notes | Halluc. risk |
|---|---|---|---|---|
| **ColTran (Colorization Transformer)** | Paired | Conditional **autoregressive** axial-transformer: coarse low-res color → 2 parallel upsamplers | Diverse but **aggressive sampling → counterintuitive colors**; slow AR sampling | Med–High |
| **CT2 (Colorization via Color Tokens)** | Paired | ViT + discretized **color tokens**, query attention | Improves ColTran's odd colors; token vocabulary constrains palette | Med |
| **Restormer** | Paired | Efficient transformer for restoration (MДTA + GDFN), multi-scale U-shaped | **Excellent restoration/SR backbone**; under-saturated if used as plain regressor | Low (conservative) |
| **Uformer** | Paired | U-shaped local-window transformer (LeWin blocks) | Strong restoration; tends **under-saturated** as pure regression | Low (conservative) |
| **ViT-based encoders (ConvNeXt/Swin)** | — | Used as encoders in DDColor/others | Better global semantics → fewer semantic mistakes | Low |

**Reading for our task.** Pure regression transformers (Restormer/Uformer) are **conservative (low hallucination) but desaturated** — perfect as the **super-resolution/sharpening front-end** and as a **strong generator backbone** that we then push toward vivid-but-faithful color with adversarial + colorfulness + chrominance losses. ColTran/CT2's autoregressive sampling is too slow and color-erratic for fast, faithful per-tile satellite work.

---

## 5. Reference / exemplar / palette-/LUT-guided colorization

| Method | Idea | Why relevant here |
|---|---|---|
| **Deep Exemplar-based Colorization** (He et al.) | Warp colors from a reference image via semantic correspondence | Could supply a *real* RGB tile of similar land cover as reference → colors come from real data, not invention |
| **Reference-driven + contrastive TIR colorization (2024)** | Reference RGB conditions thermal colorization; contrastive alignment | Directly in our domain; reduces fabricated color by anchoring to a real exemplar |
| **Palette-/LUT-guided & palette-conditioned diffusion** | Condition on a target color palette/distribution (concat palette tokens) | **Enforce land-cover palette priors**: water=blue band, vegetation=green band, built-up=grey — strong, cheap anti-mislabeling control |
| **Pixel-level semantics-guided colorization** (Zhao et al.) | Per-pixel semantic map drives color | Same principle as our semantic-mask constraint |

**Reading for our task.** Exemplar/palette guidance is a **first-class anti-hallucination tool**: by conditioning on (a) a **land-cover segmentation map** and (b) a **per-class palette/LUT prior**, we *constrain* the output color distribution to physically plausible land-cover colors and prevent e.g. water turning green. Recommend incorporating **palette/segmentation conditioning** into whichever primary model we pick.

---

## 6. Spectral / physics-aware remote-sensing translation (analogues)

| Method | Task | Takeaways for IR→RGB |
|---|---|---|
| **pix2pix on SEN1-2** | SAR→optical (Sentinel-1→2) | Establishes paired cGAN as the RS baseline; SEN1-2 = 282,384 paired patches |
| **SAR2Opt benchmark** (Zhao et al., GRSL 2022) | Benchmark + heterogeneous dataset for RS i2i | Standard eval protocol; pix2pix/CycleGAN/CUT baselines |
| **SAR-to-Optical via interpretable network** (RS 2024) | Physics-interpretable translation | Interpretability → trust/no-hallucination |
| **AWM-GAN** (RS 2024) | SAR→optical with **adaptive weight maps (attribution+uncertainty)** | **Uncertainty/attribution weighting** amplifies loss in important regions, guides uncertain ones — strong anti-hallucination idea |
| **Multi-conditional SAR→EO** (arXiv 2207.13184) | Extra conditions (e.g., NDVI, classes) | Multi-condition inputs improve fidelity |
| **Optimal-transport RS synthesis** (PMC 2024) | OT-regularized translation | Distribution matching without object invention |
| **Multispectral→RGB / pansharpening** | Band synthesis & sharpening | **Spectral-angle (SAM) loss** + band-ratio priors map naturally to Landsat bands |

**Reading for our task.** SAR→optical is the closest published analogue (cross-modal, paired, satellite) and **confirms the paired-cGAN / conditional-diffusion playbook**. Two ideas to *steal*: **(1) uncertainty/attribution-weighted losses (AWM-GAN)** to focus fidelity where it matters and flag where the model is guessing; **(2) spectral-angle loss** from multispectral→RGB to keep spectral relationships physically consistent across Landsat bands.

---

## 7. Loss functions — ranked for *this* no-hallucination, semantically-faithful task

| Loss | What it does | Fit for our task | Rank |
|---|---|---|---|
| **L1 (MAE)** | Pixel fidelity to GT; sharper than L2 | **Essential** — primary pixel anchor to ground truth; backbone of anti-hallucination | ★★★★★ |
| **Multi-scale SSIM / SSIM** | Structural similarity (luminance/contrast/structure) | **Essential** — preserves edges/structure, penalizes invented structure | ★★★★★ |
| **Edge / gradient loss** (Sobel/Canny; PearlGAN-SGA) | Aligns output edges to input edges | **Very high** — directly stops object boundary drift/invention | ★★★★★ |
| **Semantic / segmentation-guided loss** | Penalize via pretrained land-cover seg (run seg on output, compare to GT/mask) | **Very high** — enforces water/veg/built-up correctness; core anti-mislabel | ★★★★★ |
| **Color-consistency / histogram (Wasserstein) & Lab a*b* chrominance** | Match color *distribution* / chroma to GT or palette prior | **High** — enforces palette priors (water=blue), curbs sepia & wrong hues | ★★★★☆ |
| **Perceptual (LPIPS / VGG)** | Feature-space similarity → realism + semantic fidelity | **High** — adds realism without raw-pixel overfit; LPIPS-VGG best in studies | ★★★★☆ |
| **Feature-matching (FM, Pix2PixHD)** | Match discriminator features GT vs gen | **High** — stabilizes GAN, sharpens, reduces artifacts | ★★★★☆ |
| **Adversarial (LSGAN / hinge / WGAN-GP)** | Realism / texture; escape desaturated mean | **Medium-high** — needed for vividness, but the *main hallucination source* → keep weight modest, prefer **LSGAN or hinge** (stable) over vanilla; WGAN-GP if instability | ★★★☆☆ |
| **Colorfulness loss (DDColor)** | Boost saturation/color richness | **Medium** — counters under-saturation from L1/SSIM; bound it | ★★★☆☆ |
| **Spectral-angle (SAM) loss** | Preserve spectral relationships across bands | **Medium** (physics) — useful with multi-band Landsat input | ★★★☆☆ |
| **Total variation (TV)** | Smoothness/denoise | **Low-medium** — small weight only; over-use → blur | ★★☆☆☆ |
| **Cycle-consistency** | Round-trip A→B→A | **Low** — only if forced unpaired; *tolerates geometry change → enables hallucination* | ★★☆☆☆ |
| **Identity loss** | Output≈input when fed target domain | **Low** — minor color-stability aid | ★★☆☆☆ |
| **L2 (MSE)** | Pixel fidelity | **Low** — drives to blurry desaturated mean; prefer L1 | ★★☆☆☆ |

### Recommended loss stack (concrete)

```
L_total =  λ_L1 · L1                         (10.0)   # pixel anchor to GT  ← anti-hallucination core
         + λ_ssim · (1 − MS-SSIM)            (5.0)    # structure preservation
         + λ_edge · L_grad(Sobel/SGA)        (2.0)    # edge alignment, no boundary drift
         + λ_lpips · LPIPS_vgg               (1.0)    # perceptual realism + semantics
         + λ_fm · L_featmatch                (10.0)   # Pix2PixHD feature matching (stabilize/sharpen)
         + λ_adv · L_adv(LSGAN or hinge)     (1.0)    # realism / vividness (kept modest)
         + λ_lab · L_chroma(a*,b*)           (2.0)    # chrominance fidelity in Lab
         + λ_hist · L_color_hist(Wasserstein)(0.5)    # palette/color-distribution match
         + λ_seg · L_semseg(land-cover)      (1.0)    # water→blue, veg→green correctness
         + λ_tv · TV                          (1e-4)  # mild smoothing only
   (optional, multi-band input) + λ_sam · L_SAM       (0.5)   # spectral-angle consistency
```

Weights are starting points (tune on a Landsat val split); the deliberate design is **fidelity/structure terms dominate** (L1+SSIM+edge+FM+seg) and **generative terms (adv, colorfulness, hist) are bounded** so realism never overrides ground truth.

---

## 8. Anti-hallucination techniques (dedicated)

1. **Paired supervision >> unpaired.** Use Landsat's inherently co-registered bands and train a *conditional* (paired) model. Avoid cycle-consistency-only methods (CycleGAN/DualGAN/MUNIT) — cycle losses are satisfied even when the network *moves or invents* structure (steganographic "cheating"), a documented source of fabricated content in RS translation.
2. **Pixel/structure anchors.** Heavy **L1 + MS-SSIM + edge/gradient (SGA)** keep output geometry locked to the IR input; the model cannot relocate roads or grow buildings without paying a large penalty.
3. **Semantic-mask conditioning + segmentation loss.** Feed a **land-cover segmentation** as an extra input channel and penalize a frozen segmenter run on the output. This enforces class→color rules (water→blue, vegetation→green) and prevents *semantic mislabeling*, the most dangerous hallucination type for analysts.
4. **Palette / LUT priors (reference/exemplar conditioning).** Condition on per-class color palettes or a real RGB exemplar tile so colors are *sampled from real distributions*, not invented.
5. **Deterministic > stochastic.** Prefer a **single-pass deterministic** generator (DDColor-style / Pix2PixHD) or **low-/zero-noise, fixed-seed** diffusion. Stochastic style models (MUNIT, vanilla diffusion, ColTran's aggressive sampling) generate *different* content per run — inherently higher hallucination.
6. **Classifier-free guidance control (diffusion).** If using diffusion, keep CFG scale **low** (high CFG amplifies invented, "more confident" detail) and add **inference-time loss-guided color/structure preservation** (gradient guidance toward input edges/luminance during sampling).
7. **Registration-aware / misalignment-tolerant losses.** Even Landsat bands have sub-pixel misregistration; train with **co-registration QA** and use *robust* (e.g., locally-shifted / contextual / patch) losses or AWM-GAN-style **attribution+uncertainty weighting** so small misalignments don't get "learned" as fake texture.
8. **Uncertainty estimation.** Predict an **aleatoric uncertainty map** (heteroscedastic loss) or use MC-dropout/ensemble variance to **flag low-confidence pixels**; surface these to analysts and down-weight color there (don't fabricate confident color where the model is guessing).
9. **Bounded generative terms.** Keep adversarial/colorfulness weights modest; use **LSGAN/hinge** (stable) over vanilla GAN; optionally **spectral-normalized** discriminator.
10. **Frequency/structure regularization.** Position-aware attention + frequency-domain regularization (noted in RS literature) suppress high-frequency hallucinated texture.

---

## 9. Compute & inference-time considerations (fast per-tile)

| Family | Inference | Per-tile latency | Verdict |
|---|---|---|---|
| Paired cGAN (Pix2Pix/Pix2PixHD/DDColor-style) | **Single forward pass** | **Fast (ms–tens of ms/tile on GPU)** | ✅ best for scalable tiling |
| CUT/FastCUT | Single pass (generator only at test) | Fast | ✅ (unpaired fallback) |
| Restormer/Uformer SR front-end | Single pass | Fast–moderate | ✅ for the SR/sharpen stage |
| Diffusion (Palette/SR3/BBDM/LDM) | **Iterative (10–1000 steps)** | **Slow** unless distilled (DDIM/consistency/LCM) | ⚠ backup; needs step-distillation for throughput |
| ColTran (autoregressive) | Sequential token sampling | Very slow | ❌ |

For large-area Landsat mosaics processed tile-by-tile, **deterministic single-pass models win on throughput**; diffusion only becomes practical with **step distillation (DDIM/consistency/LCM)**.

---

## 10. Recommended architecture

### Primary — **Paired conditional GAN: restoration-transformer/Pix2PixHD generator + DDColor-style color decoder, with semantic & palette conditioning**

- **Front-end (SR/sharpen):** a **Restormer or Uformer** (or RRDB/ESRGAN) module to upscale/sharpen the IR bands — conservative, low-hallucination, directly satisfies the super-resolution objective.
- **Colorization generator:** **Pix2PixHD coarse-to-fine generator** (or ConvNeXt/Restormer encoder) + **DDColor-style dual decoder** (pixel decoder + query-based color decoder with cross-attention to multi-scale semantics) to produce vivid, semantically-aware color **without color bleeding**, in a **single deterministic pass**.
- **Conditioning:** concatenate **(IR band(s) + land-cover segmentation map + per-class palette prior)**; optional NDVI/extra bands.
- **Discriminator:** **multi-scale PatchGAN** (Pix2PixHD) with **spectral norm**, **LSGAN/hinge** objective.
- **Loss:** the **§7 recommended stack** (fidelity/structure-dominant, bounded generative terms, semantic + chroma + histogram + edge).
- **Why:** paired + structure-anchored + semantic/palette-constrained ⇒ **lowest hallucination**; single-pass ⇒ **fast per tile**; DDColor decoder + colorfulness/adversarial ⇒ **realistic, not sepia**; reuses mature, well-supported code (Pix2PixHD, DDColor, Restormer).

### Backup — **Conditional latent / Brownian-Bridge Diffusion (BBDM) with ControlNet edge+semantic conditioning**

- **Model:** **BBDM** (direct domain-to-domain bridge in latent space) *or* a conditional LDM, conditioned on the IR tile via **ControlNet** (edge + segmentation + luminance), **low CFG**, **few-step (DDIM/consistency-distilled)** sampling, **fixed seed**, plus **inference-time color/structure guidance**.
- **Why:** highest **FID/perceptual realism** and a published **SAR→optical / VHR remote-sensing precedent (Conditional BBDM)**; ControlNet + low stochasticity bounds hallucination. **Cost:** higher latency (mitigate via step distillation). Use when realism/FID must be maximized and throughput allows.

> **Practical plan:** ship the **Primary** as the production pipeline (fast, faithful, scalable), and keep the **Backup diffusion** for a high-realism mode / quality ceiling and FID benchmarking. Evaluate both with PSNR/SSIM/FID **plus** downstream detection/segmentation accuracy and explicit **hallucination audits** (edge/semantic-consistency checks + uncertainty maps).

---

## Appendix — Sources

- InfraGAN / thermal colorization GAN survey — IEEE Xplore 9418801; MUGAN (ResearchGate 365102845); TIC-CGAN (arXiv 1810.05399); A Conditional GAN for TIR (IEEE 10174589).
- PearlGAN / Top-Down Guided Attention (NTIR2DC) — arXiv 2104.14374 (ar5iv).
- MornGAN / Memory-Guided Collaborative Attention — arXiv 2208.02960. FoalGAN feedback object-appearance — arXiv 2310.15688.
- I2V-GAN — arXiv 2108.00913. IR2VI — arXiv 1806.09565. IMSTrans (lightweight visible↔IR transformer) — arXiv 2404.07072.
- CUT / FastCUT — github.com/taesungp/contrastive-unpaired-translation; arXiv 2007.15651. Dual Contrastive Learning — arXiv 2104.07689.
- DDColor — ICCV 2023 (arXiv 2212.11613); github.com/piddnad/DDColor.
- BBDM — CVPR 2023 (arXiv 2205.07680); github.com/xuekt98/BBDM. Conditional BBDM for VHR SAR→Optical — arXiv 2408.07947.
- Diffusion colorization — Palette/SR3/ControlNet (lllyasviel/ControlNet); "Colorize at Will: Harnessing Diffusion Prior" (ResearchGate 382666347); palette-guidance (arXiv 2508.08754); inference-time loss-guided color (arXiv 2601.17259); TIR colorization via Stable Diffusion shadow network (IOPscience 2631-8695/ae4445).
- ColTran — arXiv 2102.04432. CT2 — ECCV 2022.
- SAR→optical — SAR2Opt benchmark github.com/MarsZhaoYT/SAR2Opt-Heterogeneous-Dataset; pix2pix SAR2Optical github.com/yuuIind/SAR2Optical; AWM-GAN (RS 17/23/3878); interpretable SAR→optical (MDPI rs16020242); generative SAR–optical review (ScienceDirect S1569843225006569); multi-conditional SAR→EO (arXiv 2207.13184).
- Colorization losses / color spaces — arXiv 2204.02850 (Influence of Color Spaces); ResearchGate 359814230 (Analysis of Different Losses); pixel-level semantics-guided colorization arXiv 1808.01597.
- Pix2Pix/Pix2PixHD/CycleGAN/MUNIT/UNIT/DualGAN/ToDayGAN/ThermalGAN — original repos (phillipi/pix2pix, NVIDIA/pix2pixHD, junyanz/CycleGAN, NVlabs/MUNIT, mingyuliu/UNIT, AAnoosheh/ToDayGAN, vlkniaz/ThermalGAN).

*Items tagged **[from internal knowledge]** (MCU-GAN specifics, MornGAN repo) were not fully re-verified on the web in this session and should be confirmed before citing in a paper.*
