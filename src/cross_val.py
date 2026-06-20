

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .config import Config, STRUCTURES, ABLATION_CONDITIONS, ABLATION_LABELS
from .data import DICOMVolumeLoader, compose_targets, patient_dir_for


# +------------------------- #
# Dataset audit
# +------------------------- #


def build_audit(cfg: Config, pids: Optional[Sequence[int]] = None,
                force: bool = False) -> Dict:
    """Audit every patient straight from DICOM and write dataset_audit.json."""
    if cfg.audit_path.exists() and not force:
        with open(cfg.audit_path, encoding="utf-8") as fh:
            return json.load(fh)

    pids = list(pids) if pids is not None else list(range(1, 21))
    patients: List[Dict] = []
    for pid in pids:
        pdir = patient_dir_for(cfg, pid)
        if not pdir.exists():
            print(f"  [SKIP] patient {pid:02d} not found at {pdir}")
            continue
        data = DICOMVolumeLoader(pdir).load()
        vol = data["volume"]
        targets, sources = compose_targets(data["masks"], vol.shape)
        liver_vox = int(targets["liver"].sum())
        tumour_vox = int(targets["tumour"].sum())
        vessel_vox = int(targets["vessel"].sum())
        burden = float(tumour_vox / liver_vox) if liver_vox > 0 else 0.0
        rec = {
            "patient_id": f"{pid:02d}",
            "patient_index": pid,
            "n_slices": int(vol.shape[2]),
            "in_plane": [int(vol.shape[0]), int(vol.shape[1])],
            "spacing_mm": [round(float(s), 4) for s in data["spacing"]],
            "hu_range": [float(vol.min()), float(vol.max())],
            "raw_mask_names": data["mask_names"],
            "sources": sources,
            "present": {s: bool(len(sources[s]) > 0) for s in STRUCTURES},
            "voxels": {"liver": liver_vox, "tumour": tumour_vox, "vessel": vessel_vox},
            "tumour_burden_ratio": burden,
        }
        patients.append(rec)
        print(f"  [AUDIT] P{pid:02d}  slices={rec['n_slices']:>3}  "
              f"spacing={rec['spacing_mm']}  HU=[{rec['hu_range'][0]:.0f},"
              f"{rec['hu_range'][1]:.0f}]  tumour_vox={tumour_vox:>6}  "
              f"burden={burden:.4f}  present="
              f"{[s for s in STRUCTURES if rec['present'][s]]}")

    audit = {"dataset": "3D-IRCADb-01", "n_patients": len(patients),
             "structures": list(STRUCTURES), "patients": patients}
    cfg.audit_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.audit_path, "w", encoding="utf-8") as fh:
        json.dump(audit, fh, indent=2)
    print(f"\n[AUDIT] wrote {cfg.audit_path}")
    return audit


def summarise_audit(audit: Dict) -> None:
    """Print a compact per-patient availability/burden table."""
    print("\n" + "=" * 92)
    print(f"{'PID':>4} {'slices':>7} {'spacing (mm)':>20} {'HU range':>16} "
          f"{'liver':>8} {'tumour':>8} {'vessel':>8} {'burden':>8}  structures")
    print("-" * 92)
    n_tumour = 0
    for p in audit["patients"]:
        sp = "x".join(f"{s:.2f}" for s in p["spacing_mm"])
        hu = f"[{p['hu_range'][0]:.0f},{p['hu_range'][1]:.0f}]"
        present = ",".join(s for s in STRUCTURES if p["present"][s])
        n_tumour += int(p["present"]["tumour"])
        print(f"{p['patient_id']:>4} {p['n_slices']:>7} {sp:>20} {hu:>16} "
              f"{p['voxels']['liver']:>8} {p['voxels']['tumour']:>8} "
              f"{p['voxels']['vessel']:>8} {p['tumour_burden_ratio']:>8.4f}  {present}")
    print("-" * 92)
    n = audit["n_patients"]
    print(f"  patients={n}  tumour-bearing={n_tumour}  "
          f"no-tumour={[p['patient_id'] for p in audit['patients'] if not p['present']['tumour']]}")
    print("=" * 92)


# +------------------------- #
# Tumour-burden-stratified folds
# +------------------------- #


def _stratum(burden: float, thresholds: Sequence[float]) -> int:
    """Map a tumour-burden ratio to a stratum index (0 = none)."""
    if burden <= 0.0:
        return 0
    for i, t in enumerate(thresholds, start=1):
        if burden <= t:
            return i
    return len(thresholds) + 1


def build_folds(cfg: Config, audit: Dict, force: bool = False) -> Dict:
    """Assign every patient to exactly one test fold, stratified by tumour burden."""
    if cfg.folds_path.exists() and not force:
        with open(cfg.folds_path, encoding="utf-8") as fh:
            return json.load(fh)

    patients = audit["patients"]
    burdens = [p["tumour_burden_ratio"] for p in patients if p["tumour_burden_ratio"] > 0]
    # Tercile thresholds over tumour-bearing burdens -> strata none/low/med/high.
    if burdens:
        t1, t2 = np.percentile(burdens, [33.3, 66.7])
        thresholds = [float(t1), float(t2)]
    else:
        thresholds = [0.0, 0.0]

    # Group patient ids by stratum (deterministic order within each).
    strata: Dict[int, List[str]] = {}
    for p in sorted(patients, key=lambda r: (r["tumour_burden_ratio"], r["patient_index"])):
        s = _stratum(p["tumour_burden_ratio"], thresholds)
        strata.setdefault(s, []).append(p["patient_id"])

    rng = random.Random(cfg.seed)
    for s in strata:
        rng.shuffle(strata[s])

    # Global round-robin across the concatenated stratified order spreads each
    # stratum evenly across folds.
    n = cfg.n_folds
    fold_of: Dict[str, int] = {}
    counter = 0
    for s in sorted(strata):
        for pid in strata[s]:
            fold_of[pid] = counter % n
            counter += 1

    folds = {str(k): sorted([pid for pid, f in fold_of.items() if f == k])
             for k in range(n)}

    out = {
        "seed": cfg.seed,
        "n_folds": n,
        "stratify_by": "tumour_burden_ratio",
        "thresholds": thresholds,
        "fold_of_patient": fold_of,
        "folds": folds,
    }
    cfg.folds_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.folds_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    return out


def summarise_folds(folds: Dict, audit: Dict) -> None:
    burden_of = {p["patient_id"]: p["tumour_burden_ratio"] for p in audit["patients"]}
    tumour_of = {p["patient_id"]: p["present"]["tumour"] for p in audit["patients"]}
    print("\n" + "=" * 72)
    print(f"5-fold assignment (stratified by tumour burden, seed={folds['seed']})")
    print(f"burden tercile thresholds = {folds['thresholds']}")
    print("-" * 72)
    for k in sorted(folds["folds"], key=int):
        ids = folds["folds"][k]
        n_tum = sum(int(tumour_of[i]) for i in ids)
        mean_b = float(np.mean([burden_of[i] for i in ids]))
        print(f"  fold {k}: test={ids}  (n={len(ids)}, tumour={n_tum}, "
              f"mean_burden={mean_b:.4f})")
    print("=" * 72)


def fold_split(folds: Dict, fold_idx: int, cfg: Config):
    """Return (train_ids, internal_val_ids, test_ids) for one fold."""
    test_ids = folds["folds"][str(fold_idx)]
    train_pool = [pid for pid in folds["fold_of_patient"] if pid not in test_ids]
    train_pool = sorted(train_pool)
    rng = random.Random(cfg.seed + fold_idx)
    rng.shuffle(train_pool)
    n_val = max(1, int(round(len(train_pool) * cfg.internal_val_fraction)))
    internal_val = sorted(train_pool[:n_val])
    train_ids = sorted(train_pool[n_val:])
    return train_ids, internal_val, test_ids


# +------------------------- #
# Cross-validation orchestration
# +------------------------- #


def run_cross_validation(cfg: Config, conditions: Optional[Sequence[str]] = None,
                         folds_to_run: Optional[Sequence[int]] = None) -> None:
    """Train + evaluate every (condition x fold) under the identical CV split.

    Idempotent: a (condition, fold) whose best checkpoint and per-patient
    metrics already exist is skipped. Per-patient metrics are appended to
    ``outputs/results/per_patient_metrics.csv``.
    """
    from .train import train_one_fold
    from .evaluate import evaluate_fold, append_per_patient_rows
    from .data import preprocess_all

    audit = build_audit(cfg)
    folds = build_folds(cfg, audit)
    # Ensure every patient is preprocessed (idempotent; skips cached volumes).
    print("\n[CV] ensuring all patients are preprocessed ...")
    preprocess_all(cfg)
    conditions = list(conditions) if conditions else list(ABLATION_CONDITIONS)
    folds_to_run = list(folds_to_run) if folds_to_run is not None else list(range(cfg.n_folds))

    for cond in conditions:
        switches = ABLATION_CONDITIONS[cond]
        for fidx in folds_to_run:
            train_ids, val_ids, test_ids = fold_split(folds, fidx, cfg)
            tag = f"{cond}_fold{fidx}"
            print("\n" + "#" * 72)
            print(f"# CONDITION={cond} ({ABLATION_LABELS[cond]})  FOLD={fidx}")
            print(f"#   train={train_ids}\n#   val  ={val_ids}\n#   test ={test_ids}")
            print("#" * 72)

            ckpt = train_one_fold(cfg, tag, train_ids, val_ids, switches)
            rows = evaluate_fold(cfg, tag, cond, fidx, test_ids, ckpt, switches)
            append_per_patient_rows(cfg, rows)
