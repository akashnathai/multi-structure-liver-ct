<div align="center">

# 🫀 Anatomically- & Connectivity-Constrained Multi-Structure Liver CT Segmentation

**Simultaneous liver + tumour + hepatic-vessel segmentation from CT with anatomical-constraint losses and connectivity-aware (soft-clDice) vessel learning — on 3D-IRCADb-01.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![SimpleITK](https://img.shields.io/badge/SimpleITK-2.3%2B-005f9e)](https://simpleitk.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#-license)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](#-contributing)

### ⭐ If this project helps you, please **[Star](#)** and **[Fork](#)** it — it helps others find the work and keeps it growing!

</div>

---

A single 3D CNN with three output heads segments the **liver**, **liver tumour**, and **hepatic vessels** at once, trained with two methodological contributions:

1. **🧩 Anatomical-constraint losses** — soft, differentiable penalties that enforce real anatomy: the tumour and vessels must lie *inside* the liver (**containment**), and vessels and tumour must not occupy the same voxel (**exclusion**).
2. **🌿 Connectivity-aware vessel learning** — a differentiable **soft clDice** (centerline-Dice) loss so the predicted vessel tree stays topologically connected, plus connectivity metrics (clDice, connected-components error, centerline overlap) for evaluation.

The codebase is **fully reproducible** and produces the tables, figures, and ablation a paper needs — automatically.

---

## 📑 Table of contents

- [Highlights](#-highlights)
- [Method](#-method)
- [Results](#-results)
- [Project structure](#-project-structure)
- [Installation](#-installation)
- [Dataset](#-dataset)
- [Quickstart](#-quickstart)
- [Long runs & resuming](#-long-runs--resuming)
- [Configuration](#-configuration)
- [Outputs](#-outputs)
- [Citation](#-citation)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)

---

## ✨ Highlights

- **One network, three structures** — shared-encoder, residual, multi-head 3D U-Net (~5.7 M params).
- **Anatomy-aware training** — containment + exclusion constraints, ramped in after a warmup.
- **Topology-aware vessels** — soft clDice loss + connectivity metrics (Betti-0 proxy).
- **Rigorous protocol** — 5-fold cross-validation **stratified by tumour burden**; per-patient pooled metrics; **paired Wilcoxon** significance tests.
- **Runs on a modest GPU** — patch-based training, mixed precision (AMP), gradient checkpointing, Gaussian sliding-window inference, and an automatic out-of-memory fallback ladder.
- **Reproducible & resumable** — fixed seed, cached preprocessing, per-epoch checkpoints, idempotent CLI; re-running skips finished work.
- **Single command** generates every CSV, table, and figure.

---

## 🧠 Method

A shared encoder/decoder trunk feeds three sigmoid heads. Each structure has a tailored base loss, and two constraint terms are added on top:

| Structure | Base loss |
|---|---|
| Liver | `0.5·Dice + 0.5·BCE` |
| Tumour | `0.5·Dice + 0.5·Focal` (γ=2, α=0.75) |
| Vessel | `0.5·Dice + 0.5·clDice` |

**Constraints** (ramped in over epochs 10→30):

```
L_contain = mean(relu(p_tumour − p_liver)) + mean(relu(p_vessel − p_liver))   # ⊆ liver
L_exclude = mean(p_vessel · p_tumour)                                          # no overlap
L_total   = L_liver + 2·L_tumour + L_vessel + λ_c·L_contain + λ_e·L_exclude
```

Training is patch-based with foreground oversampling; full volumes are evaluated with Gaussian-weighted sliding-window inference. Patients missing a structure have that structure's loss and metrics skipped, so empty masks never corrupt averages.

### The four ablation conditions

| Condition | Vessel loss | Constraints | Isolates |
|---|---|---|---|
| `baseline` | Dice + BCE | ✗ | multi-task U-Net, no contributions |
| `cldice` | Dice + **clDice** | ✗ | connectivity-aware vessel loss |
| `constraints` | Dice + BCE | **✓** | containment + exclusion |
| `full` *(Ours)* | Dice + **clDice** | **✓** | both contributions |

All four are trained and evaluated under the **identical** cross-validation split, so differences are attributable to the loss terms alone.

---

## 📊 Results

5-fold cross-validation on 3D-IRCADb-01 (mean over per-patient test values; liver/vessel *n*=20, tumour *n*=17). Selected metrics:

| Metric | baseline | `full` (Ours) |
|---|---|---|
| Liver Dice | 0.873 | **0.876** |
| Vessel clDice ↑ | 0.616 | **0.645** |
| Vessel connected-components error ↓ | 299.4 | **204.2** (p<0.001) |
| Vessel centerline-overlap ↑ | 0.639 | **0.736** (p<0.001) |

**Takeaway:** the full model **significantly improves vessel connectivity** (fewer broken components, higher centerline coverage) without harming liver segmentation.

> 📄 A complete technical write-up with all metric tables, significance values, ablation analysis, and an honest discussion of limitations is in **[`report.md`](report.md)**. Figures are in `outputs/figures/`.

---

## 📁 Project structure

```
.
├── src/
│   ├── config.py      # typed Config dataclass; loss weights; ablation conditions; memory ladder
│   ├── data.py        # zip-aware DICOM loader, target composition, resample/cache, PatchDataset
│   ├── models.py      # residual shared-encoder multi-head 3D U-Net (+ gradient checkpointing)
│   ├── losses.py      # Dice, BCE, Focal, soft clDice, containment, exclusion, CombinedLoss
│   ├── metrics.py     # Dice, IoU, HD95, ASSD, clDice, components-error, centerline overlap
│   ├── cross_val.py   # dataset audit, burden-stratified folds, CV orchestration
│   ├── train.py       # AMP training loop, checkpoint/resume, early stopping, OOM ladder
│   ├── evaluate.py    # Gaussian sliding-window inference + per-patient metric collection
│   ├── figures.py     # aggregation, Wilcoxon significance, all figures/tables
│   └── run.py         # CLI entry point
├── requirements.txt
├── report.md          # full technical report
└── README.md
```

---

## 🛠 Installation

> Requires **Python 3.10+** and a **CUDA-capable GPU** (the model is memory-efficient and targets a modest single-GPU budget).

```bash
# 1. Clone
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. Create an environment (conda or venv)
conda create -n liverseg python=3.11 -y
conda activate liverseg

# 3. Install PyTorch for YOUR CUDA version (see https://pytorch.org/get-started/locally/)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 4. Install the rest
pip install -r requirements.txt
```

---

## 💾 Dataset

This project uses **3D-IRCADb-01** (20 contrast-enhanced abdominal CT volumes with expert masks).

1. Download it from IRCAD: <https://www.ircad.fr/research/data-sets/liver-segmentation-3d-ircadb-01/>
2. You get 20 patient folders `3Dircadb1.1 … 3Dircadb1.20`. Each may contain either **extracted** `PATIENT_DICOM/` + `MASKS_DICOM/<organ>/` directories **or** the original nested `PATIENT_DICOM.zip` / `MASKS_DICOM.zip` archives — **the loader reads both**, no manual unzip needed.
3. Point the pipeline at the folder that *directly contains* those 20 patient folders:

```bash
# Linux / macOS
export IRCADB_DATA_ROOT="/path/to/3Dircadb1"

# Windows (PowerShell)
$env:IRCADB_DATA_ROOT = "C:\path\to\3Dircadb1"
```

…or pass it per-command: `python -m src.run audit --data-root /path/to/3Dircadb1`.
(You can also change the default in `src/config.py`.)

**Targets** are composed from the raw masks: `liver` = the `liver` mask; `tumour` = union of any mask containing `tumor`/`tumour` (e.g. `livertumor01..07`); `vessel` = union of `portalvein`, `venoussystem`, `venacava`, `artery` where present. `dataset_audit.json` is **generated** from the DICOM by the `audit` command (it does not ship with the data).

---

## 🚀 Quickstart

Run from the project root. Every step is **resume-safe and idempotent**.

```bash
# 0. (Recommended) end-to-end wiring check on 2 patients / 3 epochs (~minutes)
python -m src.run smoke-test

# 1. Audit the dataset  -> outputs/dataset_audit.json (+ prints the per-patient table)
python -m src.run audit

# 2. Build the tumour-burden-stratified 5-fold split -> outputs/folds.json
python -m src.run folds

# 3. Preprocess all 20 patients to 1 mm isotropic .npy caches
python -m src.run preprocess

# 4. Full 5-fold cross-validation over all four ablation conditions
python -m src.run run-cv

# 5. Build every table and figure from the collected metrics
python -m src.run figures

# Check progress any time (read-only, safe during training)
python -m src.run status
```

Useful flags:

```bash
# Subset of conditions / folds
python -m src.run run-cv --conditions full baseline --folds 0 1

# Throttle epoch budget / validation frequency (faster)
python -m src.run run-cv --val-every 5 --max-epochs 150 --patience 30

# See all options for any subcommand
python -m src.run run-cv -h
```

> **Note:** `run-cv` auto-runs `audit` → `folds` → `preprocess` if you skip them, so step 4 alone is enough to go from raw data to trained models.

---

## ♻️ Long runs & resuming

Training 20 models is a multi-hour/multi-day job. Checkpoints are written **every epoch**, so a run can be interrupted and resumed at any time — just re-run the same command and it continues where it stopped (and skips finished folds/conditions).

For unattended runs, a self-restarting launcher rides out transient GPU/driver hiccups:

<details>
<summary><b>PowerShell auto-resume launcher</b></summary>

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"   # only needed if you hit an OpenMP duplicate-lib warning
$ok = $false
for ($i=1; $i -le 100 -and -not $ok; $i++) {
  Write-Host "===== run-cv attempt $i ====="
  python -m src.run run-cv --val-every 5 --max-epochs 150 --patience 30 2>&1 | Tee-Object -Append outputs\logs\cv.log
  if ($LASTEXITCODE -eq 0) { $ok = $true } else { Start-Sleep 20 }
}
python -m src.run figures 2>&1 | Tee-Object outputs\logs\figures.log
```

</details>

<details>
<summary><b>Bash auto-resume launcher</b></summary>

```bash
until python -m src.run run-cv --val-every 5 --max-epochs 150 --patience 30 2>&1 | tee -a outputs/logs/cv.log; do
  echo "crashed; resuming in 20s..."; sleep 20
done
python -m src.run figures 2>&1 | tee outputs/logs/figures.log
```

</details>

---

## ⚙️ Configuration

All hyperparameters live in [`src/config.py`](src/config.py) as a typed `Config` dataclass: HU window (`hu_min=-200`, `hu_max=250`), 1 mm isotropic resampling, patch size, batch size, base channels, AMP & gradient-checkpointing toggles, loss weights (`w_tumour=2.0`), constraint weights (`lambda_contain=0.5`, `lambda_exclude=0.5`) with a linear ramp over epochs 10→30, soft-clDice iterations, AdamW + CosineAnnealingWarmRestarts, early-stopping patience, and sliding-window inference settings. A global seed is set for `random`, `numpy`, and `torch` (CPU + CUDA), and the exact config is serialised to `outputs/config.json`.

If the GPU runs out of memory, an automatic **fallback ladder** (smaller patch → fewer channels → batch 1) activates and the active setting is logged.

---

## 📦 Outputs

```
outputs/
├── dataset_audit.json                 per-patient slices, spacing, HU, masks, tumour burden
├── folds.json                         fold assignment (stratified, seeded)
├── config.json                        exact config used
├── preprocessed/<pid>/                cached volume.npy + mask_{liver,tumour,vessel}.npy + meta.json
├── checkpoints/<cond>_fold<k>/        best.pt, latest.pt, history.json
├── results/
│   ├── per_patient_metrics.csv        one row per patient × structure × condition × fold
│   ├── aggregated_metrics.csv         mean ± std per structure per condition
│   └── ablation_significance.csv      Wilcoxon p-values, Full vs each
├── figures/                           a_dataset_overview · b_preprocess_verification ·
│                                      c_training_curves · d_qualitative_full ·
│                                      e_results_table · f_ablation_bar
└── logs/
```

---

## 📝 Citation

If you use this code, please cite the repository and the clDice loss it builds on:

```bibtex
@misc{liverseg_constrained,
  title  = {Anatomically- and Connectivity-Constrained Multi-Structure Liver CT Segmentation},
  author = {Akash Nath, Sagnikta Saha},
  year   = {2026},
  howpublished = {\url{https://github.com/akashnathai/multi-structure-liver-ct}}
}

@inproceedings{shit2021cldice,
  title     = {{clDice} -- A Novel Topology-Preserving Loss Function for Tubular Structure Segmentation},
  author    = {Shit, Suprosanna and Paetzold, Johannes C. and others},
  booktitle = {CVPR},
  year      = {2021}
}
```

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

1. Fork the repo and create your branch: `git checkout -b feature/my-feature`
2. Commit your changes and open a Pull Request.
3. For bugs or ideas, please open an [issue](../../issues).

If you build on this work or get good results, I'd love to hear about it.

---

## 📜 License

Released under the **MIT License** — add a `LICENSE` file to the repository root. Note that the **3D-IRCADb-01 dataset** has its own licence/terms from IRCAD; please review and comply with them separately.

---

## 🙏 Acknowledgements

- **3D-IRCADb-01** dataset by [IRCAD](https://www.ircad.fr/).
- **clDice** topology-preserving loss by Shit *et al.* (CVPR 2021).
- Built with [PyTorch](https://pytorch.org/), [SimpleITK](https://simpleitk.org/), [pydicom](https://pydicom.github.io/), [scikit-image](https://scikit-image.org/), and [SciPy](https://scipy.org/).

---

<div align="center">

### ⭐ Found this useful? **Star** the repo and **Fork** it to build your own experiments!

Made with care for reproducible medical-imaging research.

</div>
