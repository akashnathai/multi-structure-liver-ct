"""Aggregation, significance testing, and every paper-facing figure/table.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.stats import wilcoxon

from .config import (Config, STRUCTURES, ABLATION_CONDITIONS, ABLATION_LABELS)

# Key metrics highlighted in tables / bar charts / significance tests.
KEY_METRICS = {
    "liver": ["dice", "iou", "hd95", "assd"],
    "tumour": ["dice", "iou", "hd95", "assd"],
    "vessel": ["dice", "cldice", "cc_error", "centerline_overlap"],
}
HIGHER_IS_BETTER = {"dice", "iou", "sensitivity", "precision", "cldice", "centerline_overlap"}


# +------------------------- #
# Aggregation & significance
# +------------------------- #


def _load_per_patient(cfg: Config) -> pd.DataFrame:
    path = cfg.results_dir / "per_patient_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run the cross-validation first.")
    df = pd.read_csv(path)
    return df


def aggregate_metrics(cfg: Config) -> pd.DataFrame:
    """Pool per-patient values across folds -> mean/std per condition/structure/metric."""
    df = _load_per_patient(cfg)
    metric_cols = [c for c in df.columns if c not in
                   ("condition", "condition_label", "fold", "patient_id",
                    "structure", "present")]
    rows = []
    for cond in df["condition"].unique():
        for s in STRUCTURES:
            sub = df[(df["condition"] == cond) & (df["structure"] == s) & (df["present"] == 1)]
            for m in metric_cols:
                vals = pd.to_numeric(sub[m], errors="coerce").dropna()
                if len(vals) == 0:
                    continue
                rows.append({"condition": cond, "structure": s, "metric": m,
                             "mean": float(vals.mean()), "std": float(vals.std(ddof=0)),
                             "n": int(len(vals))})
    out = pd.DataFrame(rows)
    out.to_csv(cfg.results_dir / "aggregated_metrics.csv", index=False)
    print(f"[FIG] wrote {cfg.results_dir / 'aggregated_metrics.csv'}")
    return out


def compute_significance(cfg: Config) -> pd.DataFrame:
    """Paired Wilcoxon signed-rank, Full vs each other condition, per structure/metric."""
    df = _load_per_patient(cfg)
    rows = []
    for s in STRUCTURES:
        for m in KEY_METRICS[s]:
            full = df[(df["condition"] == "full") & (df["structure"] == s) & (df["present"] == 1)]
            full = full.set_index("patient_id")[m].apply(pd.to_numeric, errors="coerce")
            for cond in ABLATION_CONDITIONS:
                if cond == "full":
                    continue
                other = df[(df["condition"] == cond) & (df["structure"] == s) & (df["present"] == 1)]
                other = other.set_index("patient_id")[m].apply(pd.to_numeric, errors="coerce")
                common = full.dropna().index.intersection(other.dropna().index)
                a, b = full.loc[common], other.loc[common]
                p = float("nan")
                try:
                    if len(common) >= 3 and float((a - b).abs().sum()) > 0:
                        p = float(wilcoxon(a, b, zero_method="wilcox").pvalue)
                except Exception:
                    p = float("nan")
                rows.append({"structure": s, "metric": m, "comparison": f"full_vs_{cond}",
                             "n_pairs": int(len(common)),
                             "full_mean": float(a.mean()) if len(a) else float("nan"),
                             "other_mean": float(b.mean()) if len(b) else float("nan"),
                             "p_value": p})
    out = pd.DataFrame(rows)
    out.to_csv(cfg.results_dir / "ablation_significance.csv", index=False)
    print(f"[FIG] wrote {cfg.results_dir / 'ablation_significance.csv'}")
    return out


def _stars(p: float) -> str:
    if np.isnan(p):
        return "n/s"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


# +------------------------- #
# (a) Dataset overview
# +------------------------- #


def fig_dataset_overview(cfg: Config, audit: Dict) -> Path:
    pats = audit["patients"]
    ids = [p["patient_id"] for p in pats]
    liver = [p["voxels"]["liver"] for p in pats]
    tumour = [p["voxels"]["tumour"] for p in pats]
    vessel = [p["voxels"]["vessel"] for p in pats]
    burden = [p["tumour_burden_ratio"] for p in pats]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    x = np.arange(len(ids))
    axes[0, 0].bar(x, liver, color="tab:green")
    axes[0, 0].set_title("Liver volume (voxels)")
    axes[0, 1].bar(x, tumour, color="tab:red")
    axes[0, 1].set_title("Tumour volume (voxels)")
    axes[1, 0].bar(x, vessel, color="tab:blue")
    axes[1, 0].set_title("Vessel volume (voxels)")
    axes[1, 1].bar(x, burden, color="tab:purple")
    axes[1, 1].set_title("Tumour burden ratio (tumour / liver)")
    for ax in axes.ravel():
        ax.set_xticks(x)
        ax.set_xticklabels(ids, rotation=90, fontsize=7)
        ax.set_xlabel("patient")
    fig.suptitle("3D-IRCADb-01 dataset overview", fontsize=13)
    fig.tight_layout()
    out = cfg.figures_dir / f"a_dataset_overview.{cfg.fig_format}"
    fig.savefig(out, dpi=cfg.fig_dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[FIG] {out}")
    return out


# +------------------------- #
# (b) Preprocessing verification
# +------------------------- #


def _triplanar(ax_row, vol, masks, pid):
    H, W, D = vol.shape
    liver = np.asarray(masks["liver"])
    if liver.sum() > 0:
        cy, cx, cz = [int(c) for c in ndimage.center_of_mass(liver)]
    else:
        cy, cx, cz = H // 2, W // 2, D // 2
    overlays = [(1, 0, 0, 0.5), (0, 0.4, 1, 0.6)]  # tumour red, vessel blue; liver green
    views = [
        (vol[:, :, cz], {s: np.asarray(masks[s])[:, :, cz] for s in STRUCTURES}, "Axial"),
        (vol[:, cx, :].T, {s: np.asarray(masks[s])[:, cx, :].T for s in STRUCTURES}, "Coronal"),
        (vol[cy, :, :].T, {s: np.asarray(masks[s])[cy, :, :].T for s in STRUCTURES}, "Sagittal"),
    ]
    for col, (img, msk, name) in enumerate(views):
        ax = ax_row[col]
        ax.imshow(img, cmap="gray", origin="lower")
        for s, color in zip(STRUCTURES, [(0, 1, 0, 0.3), (1, 0, 0, 0.5), (0, 0.4, 1, 0.6)]):
            m = msk[s]
            if m.sum() > 0:
                rgba = np.zeros((*m.shape, 4))
                rgba[m > 0] = color
                ax.imshow(rgba, origin="lower")
        ax.set_title(f"P{pid} {name}", fontsize=9)
        ax.axis("off")


def fig_preprocess_verification(cfg: Config, pids: Optional[Sequence[str]] = None) -> Path:
    from .data import load_preprocessed
    avail = sorted(d.name for d in cfg.preprocessed_dir.iterdir() if d.is_dir())
    if not avail:
        raise FileNotFoundError("No preprocessed patients -- run preprocess first.")
    if pids is None:
        rng = np.random.RandomState(cfg.seed)
        pids = list(rng.choice(avail, size=min(3, len(avail)), replace=False))
    fig, axes = plt.subplots(len(pids), 3, figsize=(13, 4.3 * len(pids)))
    if len(pids) == 1:
        axes = axes[None, :]
    for row, pid in enumerate(pids):
        rec = load_preprocessed(cfg, pid)
        _triplanar(axes[row], np.asarray(rec["volume"]), rec["masks"], pid)
    fig.suptitle("Preprocessing verification  (green=liver, red=tumour, blue=vessel)",
                 fontsize=12)
    fig.tight_layout()
    out = cfg.figures_dir / f"b_preprocess_verification.{cfg.fig_format}"
    fig.savefig(out, dpi=cfg.fig_dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[FIG] {out}")
    return out


# +------------------------- #
# (c) Training / validation curves
# +------------------------- #


def fig_training_curves(cfg: Config, conditions: Optional[Sequence[str]] = None) -> Optional[Path]:
    conditions = list(conditions) if conditions else list(ABLATION_CONDITIONS)
    tags = []
    for cond in conditions:
        for f in range(cfg.n_folds):
            d = cfg.checkpoints_dir / f"{cond}_fold{f}"
            if (d / "history.json").exists():
                tags.append((cond, f, d / "history.json"))
    if not tags:
        print("[FIG] no training histories found -- skipping training curves.")
        return None
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for cond, f, hp in tags:
        with open(hp, encoding="utf-8") as fh:
            h = json.load(fh)["history"]
        ep = [r["epoch"] for r in h]
        ax1.plot(ep, [r["train_loss"] for r in h], alpha=0.6, label=f"{cond} f{f}")
        ax2.plot(ep, [r["val_dice"] for r in h], alpha=0.6, label=f"{cond} f{f}")
    ax1.set_title("Training loss"); ax1.set_xlabel("epoch"); ax1.set_ylabel("loss")
    ax2.set_title("Internal-validation mean Dice"); ax2.set_xlabel("epoch"); ax2.set_ylabel("Dice")
    ax2.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    out = cfg.figures_dir / f"c_training_curves.{cfg.fig_format}"
    fig.savefig(out, dpi=cfg.fig_dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[FIG] {out}")
    return out


# +------------------------- #
# (d) Qualitative overlays (CT | GT | pred | TP/FP/FN diff)
# +------------------------- #


def fig_qualitative(cfg: Config, condition: str = "full",
                    pids: Optional[Sequence[str]] = None, max_patients: int = 3) -> Optional[Path]:
    import torch
    from .data import load_preprocessed
    from .models import load_model_from_ckpt
    from .evaluate import predict_masks

    with open(cfg.folds_path, encoding="utf-8") as fh:
        folds = json.load(fh)
    fold_of = folds["fold_of_patient"]

    avail = sorted(d.name for d in cfg.preprocessed_dir.iterdir() if d.is_dir())
    if pids is None:
        # prefer patients with both tumour and vessel for a richer panel
        cand = []
        for pid in avail:
            rec = load_preprocessed(cfg, pid)
            score = sum(rec["meta"]["nonempty"][s] for s in STRUCTURES)
            cand.append((score, pid))
        pids = [p for _, p in sorted(cand, reverse=True)[:max_patients]]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for pid in pids:
        ckpt = cfg.checkpoints_dir / f"{condition}_fold{fold_of[pid]}" / "best.pt"
        if not ckpt.exists():
            print(f"[FIG] qualitative: missing checkpoint {ckpt} -- skipping P{pid}")
            continue
        rec = load_preprocessed(cfg, pid)
        model = load_model_from_ckpt(cfg, torch.load(str(ckpt), map_location=device), device)
        preds = predict_masks(cfg, model, rec["volume"], device)
        rows.append((pid, rec, preds))
    if not rows:
        print("[FIG] qualitative: nothing to draw.")
        return None

    fig, axes = plt.subplots(len(rows), 4, figsize=(16, 4.2 * len(rows)))
    if len(rows) == 1:
        axes = axes[None, :]
    colors = {"liver": (0, 1, 0), "tumour": (1, 0, 0), "vessel": (0, 0.4, 1)}
    for r, (pid, rec, preds) in enumerate(rows):
        vol = np.asarray(rec["volume"])
        liver = np.asarray(rec["masks"]["liver"])
        cz = int(ndimage.center_of_mass(liver)[2]) if liver.sum() else vol.shape[2] // 2
        ct = vol[:, :, cz]
        for c in range(4):
            axes[r, c].imshow(ct, cmap="gray", origin="lower")
            axes[r, c].axis("off")
        axes[r, 0].set_title(f"P{pid} CT", fontsize=9)
        # GT
        for s, col in colors.items():
            m = np.asarray(rec["masks"][s])[:, :, cz]
            if m.sum() > 0:
                rgba = np.zeros((*m.shape, 4)); rgba[m > 0] = (*col, 0.5)
                axes[r, 1].imshow(rgba, origin="lower")
        axes[r, 1].set_title("Ground truth", fontsize=9)
        # Pred
        for s, col in colors.items():
            m = preds[s][:, :, cz]
            if m.sum() > 0:
                rgba = np.zeros((*m.shape, 4)); rgba[m > 0] = (*col, 0.5)
                axes[r, 2].imshow(rgba, origin="lower")
        axes[r, 2].set_title("Prediction", fontsize=9)
        # Diff (tumour+vessel TP/FP/FN)
        diff = np.zeros((*ct.shape, 4))
        for s in ("tumour", "vessel"):
            g = np.asarray(rec["masks"][s])[:, :, cz] > 0
            p = preds[s][:, :, cz] > 0
            diff[np.logical_and(g, p)] = (0, 1, 0, 0.6)      # TP green
            diff[np.logical_and(p, ~g)] = (1, 0, 0, 0.6)     # FP red
            diff[np.logical_and(g, ~p)] = (1, 1, 0, 0.6)     # FN yellow
        axes[r, 3].imshow(diff, origin="lower")
        axes[r, 3].set_title("Diff (TP=g, FP=r, FN=y)", fontsize=9)
    fig.suptitle(f"Qualitative results -- {ABLATION_LABELS[condition]}", fontsize=12)
    fig.tight_layout()
    out = cfg.figures_dir / f"d_qualitative_{condition}.{cfg.fig_format}"
    fig.savefig(out, dpi=cfg.fig_dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[FIG] {out}")
    return out


# +------------------------- #
# (e) Main results table as a figure
# +------------------------- #


def fig_results_table(cfg: Config, agg: Optional[pd.DataFrame] = None) -> Path:
    if agg is None:
        agg = aggregate_metrics(cfg)
    conds = [c for c in ABLATION_CONDITIONS if c in set(agg["condition"])]
    col_labels = ["Structure", "Metric"] + conds
    cell_text, row_labels = [], []
    for s in STRUCTURES:
        for m in KEY_METRICS[s]:
            row = [s, m]
            for cond in conds:
                sel = agg[(agg["condition"] == cond) & (agg["structure"] == s) & (agg["metric"] == m)]
                if len(sel):
                    row.append(f"{sel['mean'].values[0]:.3f}±{sel['std'].values[0]:.3f}")
                else:
                    row.append("-")
            cell_text.append(row[2:])
            row_labels.append(f"{s}/{m}")
    fig, ax = plt.subplots(figsize=(2.4 + 2.2 * len(conds), 0.5 + 0.42 * len(row_labels)))
    ax.axis("off")
    tbl = ax.table(cellText=cell_text, rowLabels=row_labels,
                   colLabels=conds, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.4)
    ax.set_title("Mean ± std over per-patient test metrics (pooled across folds)", fontsize=11)
    out = cfg.figures_dir / f"e_results_table.{cfg.fig_format}"
    fig.savefig(out, dpi=cfg.fig_dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[FIG] {out}")
    return out


# +------------------------- #
# (f) Ablation bar chart with significance stars
# +------------------------- #


def fig_ablation_bar(cfg: Config, agg: Optional[pd.DataFrame] = None,
                     sig: Optional[pd.DataFrame] = None) -> Path:
    if agg is None:
        agg = aggregate_metrics(cfg)
    if sig is None:
        sig = compute_significance(cfg)
    panels = [("liver", "dice"), ("tumour", "dice"), ("vessel", "dice"),
              ("vessel", "cldice"), ("vessel", "cc_error")]
    conds = [c for c in ABLATION_CONDITIONS if c in set(agg["condition"])]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.0 * len(panels), 5))
    if len(panels) == 1:
        axes = [axes]
    for ax, (s, m) in zip(axes, panels):
        means, stds = [], []
        for cond in conds:
            sel = agg[(agg["condition"] == cond) & (agg["structure"] == s) & (agg["metric"] == m)]
            means.append(sel["mean"].values[0] if len(sel) else np.nan)
            stds.append(sel["std"].values[0] if len(sel) else 0.0)
        x = np.arange(len(conds))
        bars = ax.bar(x, means, yerr=stds, capsize=3,
                      color=["#bbbbbb", "#88bb88", "#8888bb", "#bb6666"][:len(conds)])
        ax.set_xticks(x); ax.set_xticklabels(conds, rotation=30, ha="right", fontsize=8)
        ax.set_title(f"{s} / {m}", fontsize=10)
        # significance stars: full vs each other
        if "full" in conds:
            top = np.nanmax([m_ + s_ for m_, s_ in zip(means, stds)])
            yo = top * 0.04 + 1e-3
            for i, cond in enumerate(conds):
                if cond == "full":
                    continue
                row = sig[(sig["structure"] == s) & (sig["metric"] == m) &
                          (sig["comparison"] == f"full_vs_{cond}")]
                if len(row):
                    ax.text(i, means[i] + stds[i] + yo, _stars(float(row["p_value"].values[0])),
                            ha="center", fontsize=9)
    fig.suptitle("Ablation: Full vs baseline / +clDice / +constraints  "
                 "(stars = Wilcoxon vs Full)", fontsize=12)
    fig.tight_layout()
    out = cfg.figures_dir / f"f_ablation_bar.{cfg.fig_format}"
    fig.savefig(out, dpi=cfg.fig_dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[FIG] {out}")
    return out


# +------------------------- #
# Driver
# +------------------------- #


def make_all_figures(cfg: Config) -> None:
    from .cross_val import build_audit
    audit = build_audit(cfg)
    fig_dataset_overview(cfg, audit)
    try:
        fig_preprocess_verification(cfg)
    except FileNotFoundError as e:
        print(f"[FIG] skip preprocessing verification: {e}")
    fig_training_curves(cfg)
    try:
        agg = aggregate_metrics(cfg)
        sig = compute_significance(cfg)
        fig_results_table(cfg, agg)
        fig_ablation_bar(cfg, agg, sig)
        fig_qualitative(cfg)
    except FileNotFoundError as e:
        print(f"[FIG] skip results figures (no metrics yet): {e}")
