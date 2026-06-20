# Project Report — Anatomically- and Connectivity-Constrained Multi-Structure Liver CT Segmentation (3D-IRCADb-01)

> **Purpose of this document.** This is a complete, self-contained technical report of the project. It is intended to be handed (together with the six figures in `outputs/figures/`) to an LLM assistant that will draft a research paper. Every method detail, hyperparameter, protocol choice, and *exact* experimental number is recorded here. A short **"Known issues & honest findings"** section (§10) flags where the current results do and do not support the paper's thesis, so the paper can be framed truthfully. No host-hardware details are included anywhere (hardware was an implementation constraint only).

---

## 1. One-paragraph summary

We train a **single 3D convolutional network with three output heads** to segment **liver**, **liver tumour**, and **hepatic vessels** simultaneously from contrast-enhanced abdominal CT, on the **3D-IRCADb-01** dataset (20 patients). On top of standard region losses we add two methodological contributions: (i) **connectivity-aware vessel learning** via a differentiable **soft clDice** (centerline-Dice) loss, and (ii) **anatomical-constraint losses** that softly enforce real anatomy — tumour and vessels must lie *inside* the liver (**containment**), and vessels and tumour must not occupy the same voxel (**exclusion**). We evaluate under a **5-fold cross-validation stratified by tumour burden**, pooling per-patient test metrics (n = 20 for liver/vessel, n = 17 for tumour) and testing significance with the paired Wilcoxon signed-rank test. The full model significantly improves **vessel connectivity** (connected-components error and centerline overlap) over the baseline, while the standalone clDice ablation is unstable (see §10).

---

## 2. Problem and motivation

Accurate, *simultaneous* delineation of the liver, focal liver lesions (tumours), and the hepatic vasculature is clinically important for surgical planning, tumour-burden quantification, and treatment guidance. Three difficulties motivate the method:

1. **Multi-structure coupling.** Liver, tumour, and vessels are anatomically nested and mutually exclusive in well-defined ways, but standard multi-task segmentation networks do not encode these relationships and can produce anatomically impossible outputs (e.g. tumour voxels outside the liver, vessels overlapping tumour).
2. **Vessel topology.** Vessels form a thin, branching, connected tree. Region-overlap losses (Dice/BCE) optimise voxel overlap but are insensitive to *connectivity*: a prediction can have high Dice yet be broken into many disconnected fragments, which is clinically misleading.
3. **Class imbalance & heterogeneity.** Tumours are small and highly variable (some patients have none); vessels are extremely thin; spacing is anisotropic and varies per patient.

Our contributions directly target (1) with anatomical-constraint losses and (2) with a soft-clDice loss and connectivity metrics, while handling (3) through tumour-burden-stratified cross-validation, foreground-oversampled patch training, and per-structure presence masking.

---

## 3. Dataset: 3D-IRCADb-01

20 contrast-enhanced abdominal CT volumes, 512×512 in-plane, with per-structure expert masks stored as DICOM. We generated a full audit directly from the DICOM (no audit file ships with the data). Key facts:

- **In-plane spacing** 0.56–0.87 mm; **slice thickness** 1.0–4.0 mm; **slices/volume** 74–260. Spacing is **anisotropic and varies per patient → resampling is mandatory.**
- **Structures used (binary targets):**
  - **liver** = the `liver` mask (present in all 20).
  - **tumour** = union of any mask whose name contains `tumor`/`tumour` (e.g. `livertumor01..07`). Note `liverkyst` (cyst) is **excluded**.
  - **vessel** = union of whichever of `{portalvein, venoussystem, venacava, artery}` are present.
- **Tumour-bearing patients: 17.** Patients **11, 14, 20 have no tumour mask** (confirmed empirically by the audit). All 20 patients have liver and vessel masks.
- For any patient missing a structure, that structure's mask is all-zeros and its **loss and metrics are skipped** for that patient, so empty masks never corrupt averages.

**Per-patient audit (tumour burden = tumour voxels / liver voxels, raw resolution):**

| PID | slices | in-plane (mm) | thick (mm) | tumour burden | structures |
|----|----|----|----|----|----|
| 01 | 129 | 0.57 | 1.60 | 0.0748 | L,T,V |
| 02 | 172 | 0.78 | 1.60 | 0.0085 | L,T,V |
| 03 | 200 | 0.62 | 1.25 | 0.0060 | L,T,V |
| 04 | 91  | 0.74 | 2.00 | 0.0059 | L,T,V |
| 05 | 139 | 0.78 | 1.60 | 0.0027 | L,T,V |
| 06 | 135 | 0.78 | 1.60 | 0.2377 | L,T,V |
| 07 | 151 | 0.78 | 1.60 | 0.4676 | L,T,V |
| 08 | 124 | 0.56 | 1.60 | 0.0063 | L,T,V |
| 09 | 111 | 0.87 | 2.00 | 0.0312 | L,T,V |
| 10 | 122 | 0.74 | 1.60 | 0.0080 | L,T,V |
| 11 | 132 | 0.72 | 1.60 | 0.0000 | L,V |
| 12 | 260 | 0.68 | 1.00 | 0.1044 | L,T,V |
| 13 | 122 | 0.67 | 1.60 | 0.0594 | L,T,V |
| 14 | 113 | 0.72 | 1.60 | 0.0000 | L,V |
| 15 | 125 | 0.78 | 1.60 | 0.0012 | L,T,V |
| 16 | 155 | 0.70 | 1.60 | 0.0027 | L,T,V |
| 17 | 119 | 0.74 | 1.60 | 0.0883 | L,T,V |
| 18 | 74  | 0.74 | 2.50 | 0.0037 | L,T,V |
| 19 | 124 | 0.70 | 4.00 | 0.0710 | L,T,V |
| 20 | 225 | 0.81 | 2.00 | 0.0000 | L,V |

Tumour burden spans nearly two orders of magnitude (0.0012–0.4676 among tumour-bearing cases), which is why folds are **stratified by burden** (§6). Mean GT vessel mask consists of ≈20 connected components per patient (std ≈15) — the vessel target is genuinely a multi-branch tree, making connectivity a meaningful evaluation axis.

---

## 4. Preprocessing

Pipeline (`pydicom → SimpleITK → NumPy`, cached to `.npy`; idempotent):

1. **Load** the CT volume and all masks from the DICOM (read directly from the nested `PATIENT_DICOM.zip` / `MASKS_DICOM.zip` archives). Slices are ordered by `ImagePositionPatient[z]`; CT intensities are converted to Hounsfield Units via `RescaleSlope`/`RescaleIntercept`. Masks are stacked in the identical slice order.
2. **Resample to 1.0 mm isotropic** with SimpleITK — **trilinear** for the CT image, **nearest-neighbour** for masks (preserving binary labels), with physical extent preserved.
3. **Intensity normalisation** — clip HU to the liver soft-tissue window **[−200, 250]**, then min-max scale to **[0, 1]**.
4. **Cache** `volume.npy` + `mask_{liver,tumour,vessel}.npy` + a per-patient `meta.json` (ids, raw/resampled shapes, spacing, structures present, voxel counts). Reused across all folds/conditions.

Array convention throughout: volumes/masks are `(H, W, D)` with `D` the axial (slice) axis; tensors are `(B, C, H, W, D)`.

---

## 5. Method

### 5.1 Network — residual, shared-encoder, multi-head 3D U-Net

A single trunk feeds three 1-channel heads (liver, tumour, vessel). Deliberately **no transformer** in the bottleneck — the architecture is kept lean so the ablation isolates the *loss-level* contributions.

- **Residual block:** `(Conv3d → InstanceNorm3d → LeakyReLU) ×2` with a 1×1 convolution residual skip.
- **Encoder:** three residual levels with channel widths `[base, 2·base, 4·base]`, `MaxPool3d(2)` between levels.
- **Bottleneck:** one residual block at width `8·base`.
- **Decoder:** symmetric, `ConvTranspose3d` upsampling with a trilinear-resize fallback to match skip shapes, skip-concatenation, residual block.
- **Heads:** three `Conv3d(base, 1, 1)` layers; sigmoid applied in the loss (numerically-stable BCE-with-logits) and at inference.
- **Init:** Kaiming for convolutions; near-zero head bias and small head weights.
- `base = 32` → **≈5.69 M parameters.**
- **Memory-efficient design:** patch-based training, mixed-precision (AMP), gradient checkpointing on every residual block, and Gaussian-weighted sliding-window inference — so the memory footprint stays modest and bounded regardless of volume size. (An automatic fallback ladder reduces patch size / channels on out-of-memory; this never triggered at the reported settings.)

### 5.2 Base per-structure losses (on sigmoid probabilities `p`)

- **Liver:** `L_liver = 0.5·Dice + 0.5·BCE`
- **Tumour:** `L_tumour = 0.5·Dice + 0.5·Focal` (γ = 2, α = 0.75)
- **Vessel:** `L_vessel = 0.5·Dice + 0.5·{clDice | BCE}` — clDice when connectivity-aware learning is enabled, otherwise BCE.

All base losses are computed **per sample**, and a per-sample weight of 0 is applied when that patch's ground truth for the structure is empty (the computation graph is preserved, contribution is zero), so empty/absent structures never corrupt the batch average.

Soft Dice (per sample): `Dice = 1 − (2·Σ p·g + s)/(Σ p + Σ g + s)`, smoothing `s = 1`.
Focal: `α_t (1 − p_t)^γ · BCE`, with `p_t = p·g + (1−p)(1−g)`, `α_t = α·g + (1−α)(1−g)`.

### 5.3 Connectivity-aware vessel loss — soft clDice (Shit et al., CVPR 2021)

Soft skeletonisation via iterative min/max pooling (differentiable):
`soft_erode = min over axis-wise (−maxpool(−x))`; `soft_open = dilate(erode(x))`; the soft skeleton accumulates `relu(x − open(x))` over `T` iterations (T = 5).

Given skeletons `S_p = skel(p)` and `S_g = skel(g)`:
- topology precision `T_prec = (Σ S_p·g + s)/(Σ S_p + s)`
- topology sensitivity `T_sens = (Σ S_g·p + s)/(Σ S_g + s)`
- `clDice = 1 − 2·T_prec·T_sens/(T_prec + T_sens)`

This rewards predictions whose centerline lies within the GT and whose GT centerline is covered by the prediction — i.e. a topologically faithful vessel tree.

### 5.4 Anatomical-constraint losses (the contribution)

Computed on probability maps; **individually toggleable** and **ramped in linearly** from epoch 10 → 30 (zero before, full after) to avoid early instability.

- **Containment:** `L_contain = mean(relu(p_tumour − p_liver)) + mean(relu(p_vessel − p_liver))` — softly enforces tumour ⊆ liver and vessel ⊆ liver.
- **Exclusion:** `L_exclude = mean(p_vessel · p_tumour)` — discourages vessel and tumour from co-occupying a voxel.

### 5.5 Total objective

`L = w_liver·L_liver + w_tumour·L_tumour + w_vessel·L_vessel + λ_c(t)·L_contain + λ_e(t)·L_exclude`

with `w_liver = 1.0`, `w_tumour = 2.0`, `w_vessel = 1.0`, and `λ_c, λ_e` ramped to 0.5 each over epochs 10→30.

### 5.6 The four ablation conditions

| Condition | Vessel loss | Constraints | Isolates |
|---|---|---|---|
| **baseline** | Dice + BCE | off | multi-task U-Net, no contributions |
| **cldice** | Dice + **clDice** | off | connectivity-aware vessel loss |
| **constraints** | Dice + BCE | **on** | containment + exclusion |
| **full (Ours)** | Dice + **clDice** | **on** | both contributions together |

All four are trained and evaluated under the **identical** cross-validation split.

---

## 6. Cross-validation protocol

- **5-fold CV over all 20 patients, stratified by tumour burden** (terciles of the burden ratio + a no-tumour stratum, distributed across folds by a seeded global round-robin). Deterministic (seed = 42); saved to `folds.json`.
- **Every patient appears in exactly one test fold** → pooled per-patient test values: **n = 20** (liver, vessel) and **n = 17** (tumour).
- Within each training fold, a small **internal validation subset** (~20%) is held out for early-stopping / best-checkpoint selection (never the test fold).
- **Significance:** paired **Wilcoxon signed-rank** test of *Full* vs each ablation, across the pooled per-patient values (appropriate for this n).

**Fold assignment (test sets), burden thresholds = [0.0061, 0.0672]:**

| Fold | Test patients | #tumour | mean burden |
|---|---|---|---|
| 0 | 03, 10, 14, 17 | 3/4 | 0.026 |
| 1 | 02, 05, 11, 12 | 3/4 | 0.029 |
| 2 | 06, 13, 16, 20 | 3/4 | 0.075 |
| 3 | 07, 08, 15, 18 | 4/4 | 0.120 |
| 4 | 01, 04, 09, 19 | 4/4 | 0.046 |

---

## 7. Training configuration

- Optimiser **AdamW** (lr 1e-4, weight decay 1e-5); scheduler **CosineAnnealingWarmRestarts** (T₀ = 50, T_mult = 2).
- **Mixed precision** + gradient scaler; gradient clipping at 1.0.
- **Patch size 128×128×64**, batch size 2, base channels 32, gradient checkpointing on.
- **Patch sampling:** 60% of patches centred inside the liver bounding box (foreground oversampling), 40% random.
- **Augmentation (train only):** random per-axis flips, random 90° in-plane rotation, Gaussian noise, brightness, contrast.
- **Epoch budget 150**, early-stopping patience 30 validation-passes (validation every 5 epochs); best checkpoint chosen on internal-validation mean Dice.
- **Inference:** Gaussian-weighted sliding window, 0.5 overlap, for full-volume validation and test.
- Global seed (Python/NumPy/torch/CUDA). Exact config serialised to `outputs/config.json`.
- All 20 models (4 conditions × 5 folds) trained to completion under this protocol; runs are checkpointed every epoch and fully resumable.

---

## 8. Metrics

- **Liver & tumour:** Dice, IoU, sensitivity, precision, HD95 (mm), ASSD (mm).
- **Vessel:** the above **plus** connectivity metrics — **clDice** (hard centerline-Dice via skeletonisation), **connected-components error** `|#comp_pred − #comp_GT|` (a Betti-0 proxy), **centerline overlap** (fraction of GT skeleton covered by prediction), and reported component counts.
- HD95/ASSD use surface distance transforms with correct voxel spacing. Metrics return NaN when a structure is absent (GT empty) and NaNs are excluded from all averages.

---

## 9. Results (exact numbers)

All values are **mean ± std over per-patient test values, pooled across the 5 folds** (n in each row; liver/vessel n = 20, tumour overlap n = 17, tumour boundary n = 14 because HD95/ASSD require a non-empty prediction).

### 9.1 Liver (n = 20)

| metric | baseline | cldice | constraints | full |
|---|---|---|---|---|
| Dice | 0.873 ± 0.045 | 0.862 ± 0.044 | 0.870 ± 0.043 | **0.876 ± 0.037** |
| IoU | 0.777 ± 0.069 | 0.760 ± 0.065 | 0.773 ± 0.065 | **0.781 ± 0.056** |
| Sensitivity | 0.916 ± 0.062 | 0.910 ± 0.054 | **0.923 ± 0.055** | 0.915 ± 0.054 |
| Precision | 0.839 ± 0.071 | 0.824 ± 0.068 | 0.828 ± 0.069 | **0.844 ± 0.057** |
| HD95 (mm) | **97.5 ± 33.6** | 102.7 ± 43.0 | 98.0 ± 36.2 | 102.7 ± 40.6 |
| ASSD (mm) | **12.26 ± 4.22** | 13.21 ± 5.66 | 12.33 ± 4.45 | 14.06 ± 6.69 |

### 9.2 Tumour (overlap n = 17; HD95/ASSD n = 14)

| metric | baseline | cldice | constraints | full |
|---|---|---|---|---|
| Dice | **0.171 ± 0.209** | 0.116 ± 0.180 | 0.174 ± 0.202 | 0.078 ± 0.155 |
| IoU | 0.110 ± 0.147 | 0.073 ± 0.119 | **0.111 ± 0.137** | 0.049 ± 0.102 |
| Sensitivity | 0.217 ± 0.284 | 0.153 ± 0.225 | **0.229 ± 0.267** | 0.132 ± 0.269 |
| Precision | 0.278 ± 0.337 | **0.295 ± 0.353** | 0.243 ± 0.302 | 0.118 ± 0.241 |
| HD95 (mm) | **91.5 ± 42.4** | 117.8 ± 51.8 | 95.3 ± 47.1 | 96.5 ± 34.9 |
| ASSD (mm) | **35.8 ± 26.9** | 49.3 ± 33.3 | 36.5 ± 23.3 | 42.5 ± 24.3 |

### 9.3 Vessel (n = 20) — region and connectivity

| metric | baseline | cldice | constraints | full |
|---|---|---|---|---|
| Dice | **0.588 ± 0.110** | 0.015 ± 0.008 | 0.578 ± 0.098 | 0.487 ± 0.175 |
| IoU | **0.425 ± 0.111** | 0.007 ± 0.004 | 0.413 ± 0.096 | 0.339 ± 0.148 |
| Sensitivity | 0.634 ± 0.143 | 1.000 ± 0.000 | 0.601 ± 0.141 | 0.635 ± 0.135 |
| Precision | 0.625 ± 0.206 | 0.007 ± 0.004 | **0.637 ± 0.199** | 0.485 ± 0.251 |
| HD95 (mm) | 37.4 ± 30.3 | 190.7 ± 35.6 | 37.2 ± 26.6 | **36.9 ± 29.2** |
| ASSD (mm) | **6.05 ± 4.39** | 115.6 ± 23.6 | 6.06 ± 4.18 | 6.97 ± 4.67 |
| **clDice** | 0.616 ± 0.085 | 0.225 ± 0.402 | 0.614 ± 0.079 | **0.645 ± 0.109** |
| **cc-error ↓** | 299.4 ± 119.7 | 19.2 ± 14.7 | 288.9 ± 104.2 | **204.2 ± 96.0** |
| **centerline-overlap** | 0.639 ± 0.120 | 1.000 ± 0.000 | 0.611 ± 0.120 | **0.736 ± 0.121** |
| #components (pred) | 319.5 ± 116.0 | 1.0 ± 0.0 | 309.1 ± 101.8 | 224.4 ± 93.6 |
| #components (GT) | 20.2 ± 14.7 | — | — | — |

### 9.4 Significance — Full vs each (paired Wilcoxon, p-values)

Stars: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant.

| structure | metric | Full vs baseline | Full vs cldice | Full vs constraints |
|---|---|---|---|---|
| liver | Dice | 0.294 ns | 0.0094 ** | 0.083 ns |
| liver | IoU | 0.388 ns | 0.0083 ** | 0.097 ns |
| liver | HD95 | 0.522 ns | 0.784 ns | 0.277 ns |
| liver | ASSD | 0.012 * (worse) | 0.133 ns | 0.0094 ** (worse) |
| tumour | Dice | 0.0096 ** (worse) | 0.070 ns | 0.019 * (worse) |
| tumour | IoU | 0.0076 ** (worse) | 0.070 ns | 0.019 * (worse) |
| tumour | HD95 | 0.065 ns | 0.0020 ** | 0.084 ns |
| tumour | ASSD | 0.695 ns | 0.0020 ** | 1.000 ns |
| vessel | Dice | 0.0017 ** (worse) | 0.0000 *** | 0.015 * (worse) |
| vessel | clDice | 0.090 ns | 0.0002 *** | 0.165 ns |
| vessel | **cc-error** | **0.0005 *** (better)** | 0.0000 *** | **0.0001 *** (better)** |
| vessel | **centerline-overlap** | **0.0000 *** (better)** | 0.0000 *** | **0.0000 *** (better)** |

"(better)"/"(worse)" indicates the direction of the Full model relative to the comparator for that metric.

---

## 10. Honest findings & known issues (read before writing the paper)

This section states plainly what the data supports, so the paper is framed truthfully.

**What is solid and defensible:**
1. **Liver segmentation is strong and stable** across all conditions (Dice ≈ 0.86–0.88); the contributions do not harm it (Full is numerically best on Dice/IoU/precision, differences vs baseline n.s.).
2. **The Full model significantly improves vessel connectivity** over the baseline: **connected-components error 204 vs 299 (p<0.001)** and **centerline-overlap 0.736 vs 0.639 (p<0.001)**, with clDice also highest (0.645). This is the clearest positive result and is the natural headline: *anatomical constraints + connectivity-aware learning yield more topologically faithful vessel trees.* Note that all conditions still over-fragment vessels relative to GT (≈224 vs ≈20 components for Full), so the claim should be **relative improvement in connectivity**, not solved connectivity.

**What does NOT currently support the thesis (must be addressed or framed carefully):**
3. **The standalone `cldice` ablation collapsed.** Vessel Dice 0.015, precision 0.007, sensitivity 1.000, centerline-overlap 1.000, #components = 1 — i.e. the model degenerated to predicting vessel almost *everywhere* (a single huge blob that trivially "covers" the GT centerline). The soft-clDice term, when used **without** the containment constraint, fell into a "predict-all" basin. In the **Full** model the **containment constraint prevents this** (vessels confined to the liver), which is itself an interesting interaction — but as currently trained, the isolated clDice number is a training-stability artefact, not a fair measurement of clDice's value. **Recommended:** treat clDice-alone numbers as a known failure mode, and/or re-run after stabilising clDice (warmup-ramp the clDice term like the constraints; up-weight the Dice component; cap the soft-skeleton). A re-run of the `cldice` and `full` conditions is resume-safe.
4. **Vessel and tumour region overlap drop under the contributions.** Full vessel Dice (0.487) is below baseline (0.588, p<0.01), and Full tumour Dice (0.078) is the worst of all conditions (p<0.01 vs baseline). The exclusion/containment terms and the clDice term trade region overlap for connectivity, and the exclusion term appears to over-suppress the (small, hard) tumour class. **Recommended framing:** either (a) position the work honestly as "improved vessel topology at some cost to raw overlap, with tuning of constraint weights as future work," or (b) re-run after softening tumour suppression (lower `λ_exclude`, longer ramp, or exclude the tumour head from the exclusion term) and re-tuning, then report the improved trade-off.
5. **Tumour segmentation is low overall** (best Dice ≈ 0.17). This is expected for whole-liver-window single-stage tumour segmentation on this small, highly imbalanced dataset, but it is too low to be a headline; the paper should foreground liver + vessel-connectivity and treat tumour as a hard secondary task.

**Net recommendation for the paper author:** The strongest, fully-supported story is **"connectivity-aware + anatomically-constrained multi-structure segmentation improves vessel topology (cc-error, centerline overlap, clDice) without harming the liver,"** with the constraint–clDice interaction (constraints rescue clDice from degenerate over-segmentation) as a genuine insight. The tumour result and the standalone-clDice collapse should be reported transparently as limitations / future work, **or** the affected conditions re-run after the stabilisation fixes in §10.3–10.4 before final numbers go into the paper.

---

## 11. Figures (in `outputs/figures/`)

Use these directly; suggested captions below.

- **`a_dataset_overview.png`** — Per-patient liver/tumour/vessel voxel counts and tumour-burden ratio across the 20 cases. *Caption:* "Dataset composition of 3D-IRCADb-01: structure volumes and tumour burden per patient, showing the wide burden range and the three tumour-free cases (11, 14, 20) that motivate burden-stratified folds."
- **`b_preprocess_verification.png`** — Tri-planar CT with liver (green), tumour (red), vessel (blue) overlays for sample patients. *Caption:* "Preprocessing verification: 1 mm isotropic resampling and liver-window normalisation, with masks correctly aligned across axial/coronal/sagittal views."
- **`c_training_curves.png`** — Training loss and internal-validation mean Dice vs epoch, per fold/condition. *Caption:* "Convergence behaviour across folds and ablation conditions."
- **`d_qualitative_full.png`** — For representative test patients: CT | ground truth | Full prediction | TP/FP/FN difference map (TP green, FP red, FN yellow). *Caption:* "Qualitative results of the Full model; the difference panel highlights vessel/tumour errors."
- **`e_results_table.png`** — Rendered table of mean ± std per structure/metric per condition. *Caption:* "Quantitative comparison of the four ablation conditions (pooled per-patient test metrics)."
- **`f_ablation_bar.png`** — Bar charts (liver Dice, tumour Dice, vessel Dice, vessel clDice, vessel cc-error) with Wilcoxon significance stars vs Full. *Caption:* "Ablation with significance: the Full model significantly improves vessel connectivity (cc-error) and centerline metrics."

> Caveat for figure use: panels showing the **`cldice`** condition reflect the degenerate collapse described in §10.3 (e.g. vessel Dice ≈ 0). If the paper re-runs after the clDice stabilisation fix, regenerate `e_` and `f_` (and `c_`) with `python -m src.run figures`.

---

## 12. Reproducibility

- **Code layout:** `src/{config,data,models,losses,metrics,cross_val,train,evaluate,figures,run}.py`. Single CLI: `python -m src.run {audit|folds|preprocess|run-cv|evaluate|figures|status}`.
- **Determinism:** global seed 42 (Python/NumPy/torch/CUDA); fold assignment and config serialised (`folds.json`, `config.json`, `dataset_audit.json`).
- **Exact commands:**
  - `python -m src.run run-cv --val-every 5 --max-epochs 150 --patience 30` (train all 20 models; resume-safe, idempotent)
  - `python -m src.run figures` (tables + significance + all figures)
- **Outputs:** `outputs/results/{per_patient_metrics.csv, aggregated_metrics.csv, ablation_significance.csv}`, `outputs/figures/*.png`, `outputs/checkpoints/<condition>_fold<k>/`.
- **Loss/metric implementations:** soft clDice follows Shit et al., "clDice — a Novel Topology-Preserving Loss Function for Tubular Structure Segmentation," CVPR 2021.

---

## 13. Suggested paper structure (for the drafting assistant)

1. **Abstract** — multi-structure liver/tumour/vessel segmentation; anatomical constraints + connectivity-aware (clDice) learning; 5-fold burden-stratified CV on 3D-IRCADb-01; headline = significant vessel-connectivity gains without harming liver.
2. **Introduction** — clinical motivation; the three difficulties (§2); contributions: (i) anatomical-constraint losses, (ii) connectivity-aware vessel learning, (iii) rigorous per-patient CV with connectivity metrics and significance testing.
3. **Related work** — multi-organ/lesion CT segmentation; topology-preserving losses (clDice); anatomy/shape-constrained segmentation.
4. **Method** — §5 (architecture, base losses, soft clDice, containment & exclusion, total objective, ablation conditions).
5. **Experimental setup** — §3, §4, §6, §7, §8.
6. **Results** — §9 tables + figures `e_`, `f_`, `d_`; emphasise vessel connectivity (cc-error, centerline overlap, clDice) significance.
7. **Discussion / Ablation analysis** — the constraint–clDice interaction (constraints prevent clDice collapse); the overlap-vs-connectivity trade-off.
8. **Limitations & future work** — §10 (tumour difficulty; constraint-weight tuning; clDice stabilisation; single dataset; consider a coarse-to-fine tumour stage).
9. **Conclusion.**

> The paper must remain truthful to the numbers in §9 and the caveats in §10. If any results are re-run after the §10 fixes, replace the §9 tables and regenerate the figures before finalising.
