

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from .config import Config, STRUCTURES, ABLATION_LABELS
from .data import load_preprocessed
from .metrics import evaluate_structure


# +------------------------- #
# Sliding-window inference
# +------------------------- #


def _gaussian_weight(patch: Sequence[int], sigma_scale: float) -> np.ndarray:
    coords = [np.arange(p) - (p - 1) / 2.0 for p in patch]
    grids = np.meshgrid(*coords, indexing="ij")
    g = np.zeros(patch, dtype=np.float32)
    for axis, gr in enumerate(grids):
        sigma = max(patch[axis] * sigma_scale, 1e-3)
        g = g + (gr ** 2) / (2.0 * sigma ** 2)
    w = np.exp(-g).astype(np.float32)
    w = np.maximum(w, w.max() * 1e-3)  # avoid zero weight at corners
    return w


def _starts(dim: int, patch: int, step: int) -> List[int]:
    if dim <= patch:
        return [0]
    s = list(range(0, dim - patch + 1, step))
    if s[-1] != dim - patch:
        s.append(dim - patch)
    return s


def sliding_window_predict(cfg: Config, model: torch.nn.Module, volume: np.ndarray,
                           device: torch.device,
                           overlap: Optional[float] = None) -> Dict[str, np.ndarray]:
    """Return per-structure probability volumes (H,W,D float32) for one volume."""
    model.eval()
    ov = cfg.sw_overlap if overlap is None else overlap
    pH, pW, pD = cfg.patch_size
    vol = np.asarray(volume, dtype=np.float32)
    H, W, D = vol.shape

    # Pad so every dim is at least one patch.
    pad = [(0, max(pH - H, 0)), (0, max(pW - W, 0)), (0, max(pD - D, 0))]
    if any(p[1] > 0 for p in pad):
        vol = np.pad(vol, pad, mode="constant")
    Hp, Wp, Dp = vol.shape

    step = (max(int(pH * (1 - ov)), 1),
            max(int(pW * (1 - ov)), 1),
            max(int(pD * (1 - ov)), 1))
    ys = _starts(Hp, pH, step[0])
    xs = _starts(Wp, pW, step[1])
    zs = _starts(Dp, pD, step[2])

    weight = _gaussian_weight((pH, pW, pD), cfg.sw_sigma_scale)
    w_t = torch.from_numpy(weight).to(device)

    acc = {s: torch.zeros((Hp, Wp, Dp), dtype=torch.float32, device=device)
           for s in STRUCTURES}
    wsum = torch.zeros((Hp, Wp, Dp), dtype=torch.float32, device=device)

    coords = [(y, x, z) for y in ys for x in xs for z in zs]
    use_amp = cfg.use_amp and device.type == "cuda"
    with torch.no_grad():
        for i in range(0, len(coords), cfg.sw_batch_size):
            chunk = coords[i:i + cfg.sw_batch_size]
            batch = np.stack([vol[y:y + pH, x:x + pW, z:z + pD] for (y, x, z) in chunk])
            t = torch.from_numpy(batch).unsqueeze(1).to(device)  # (b,1,pH,pW,pD)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                logits = model(t)
            for s in STRUCTURES:
                prob = torch.sigmoid(logits[s].float())[:, 0]  # (b,pH,pW,pD)
                for j, (y, x, z) in enumerate(chunk):
                    acc[s][y:y + pH, x:x + pW, z:z + pD] += prob[j] * w_t
                    if s == STRUCTURES[0]:
                        wsum[y:y + pH, x:x + pW, z:z + pD] += w_t

    wsum = torch.clamp(wsum, min=1e-6)
    out = {}
    for s in STRUCTURES:
        p = (acc[s] / wsum).cpu().numpy()
        out[s] = p[:H, :W, :D]  # crop padding away
    return out


def predict_masks(cfg: Config, model: torch.nn.Module, volume: np.ndarray,
                  device: torch.device, threshold: float = 0.5,
                  overlap: Optional[float] = None) -> Dict[str, np.ndarray]:
    probs = sliding_window_predict(cfg, model, volume, device, overlap=overlap)
    return {s: (probs[s] >= threshold).astype(np.uint8) for s in STRUCTURES}


# +------------------------- #
# Internal-validation metric (early stopping)
# +------------------------- #


def validation_dice(cfg: Config, model: torch.nn.Module, val_ids: Sequence[str],
                    device: torch.device) -> float:
    """Mean Dice over present structures across the internal-validation set."""
    dices: List[float] = []
    for pid in val_ids:
        rec = load_preprocessed(cfg, pid)
        preds = predict_masks(cfg, model, rec["volume"], device, overlap=cfg.sw_overlap_val)
        for s in STRUCTURES:
            if rec["meta"]["nonempty"][s]:
                d = evaluate_structure(preds[s], np.asarray(rec["masks"][s]), s)["dice"]
                if not np.isnan(d):
                    dices.append(d)
        del preds
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return float(np.mean(dices)) if dices else 0.0


# +------------------------- #
# Test-set evaluation
# +------------------------- #


PER_PATIENT_FIELDS = ["condition", "condition_label", "fold", "patient_id", "structure",
                      "present", "dice", "iou", "sensitivity", "precision", "hd95",
                      "assd", "cldice", "cc_error", "centerline_overlap",
                      "n_components_pred", "n_components_gt"]


def evaluate_fold(cfg: Config, tag: str, condition: str, fold_idx: int,
                  test_ids: Sequence[str], ckpt_path: Path,
                  switches: Dict[str, bool]) -> List[Dict]:
    """Run inference on the fold's test patients and return per-(patient,structure) rows."""
    from .models import load_model_from_ckpt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(str(ckpt_path), map_location=device)
    model = load_model_from_ckpt(cfg, state, device)
    model.eval()

    rows: List[Dict] = []
    for pid in test_ids:
        rec = load_preprocessed(cfg, pid)
        preds = predict_masks(cfg, model, rec["volume"], device)
        for s in STRUCTURES:
            present = bool(rec["meta"]["nonempty"][s])
            m = evaluate_structure(preds[s], np.asarray(rec["masks"][s]), s) if present \
                else {k: float("nan") for k in
                      ["dice", "iou", "sensitivity", "precision", "hd95", "assd",
                       "cldice", "cc_error", "centerline_overlap",
                       "n_components_pred", "n_components_gt"]}
            row = {"condition": condition, "condition_label": ABLATION_LABELS[condition],
                   "fold": fold_idx, "patient_id": pid, "structure": s,
                   "present": int(present)}
            for f in PER_PATIENT_FIELDS:
                if f not in row:
                    row[f] = m.get(f, float("nan"))
            rows.append(row)
            if present:
                print(f"   [EVAL] {tag} P{pid} {s:>6}  dice={row['dice']:.4f}"
                      + (f"  cldice={row['cldice']:.4f} cc_err={row['cc_error']:.0f}"
                         if s == "vessel" else ""))
    return rows


def append_per_patient_rows(cfg: Config, rows: List[Dict]) -> None:
    """Append rows to per_patient_metrics.csv, de-duplicating by key (resume-safe)."""
    path = cfg.results_dir / "per_patient_metrics.csv"
    key = lambda r: (r["condition"], int(r["fold"]), r["patient_id"], r["structure"])
    existing: Dict = {}
    if path.exists():
        with open(path, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                existing[(r["condition"], int(r["fold"]), r["patient_id"], r["structure"])] = r
    for r in rows:
        existing[key(r)] = r
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=PER_PATIENT_FIELDS)
        w.writeheader()
        for r in sorted(existing.values(),
                        key=lambda r: (r["condition"], int(r["fold"]), r["patient_id"], r["structure"])):
            w.writerow({k: r.get(k, "") for k in PER_PATIENT_FIELDS})
    print(f"   [CSV] updated {path} ({len(existing)} rows)")
