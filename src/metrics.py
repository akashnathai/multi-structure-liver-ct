

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, label
from skimage.morphology import skeletonize

NAN = float("nan")
Spacing = Tuple[float, float, float]


def _as_bool(a: np.ndarray) -> np.ndarray:
    return np.asarray(a) > 0


# +------------------------- #
# Overlap metrics
# +------------------------- #


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = _as_bool(pred), _as_bool(gt)
    if gt.sum() == 0:
        return NAN
    denom = pred.sum() + gt.sum()
    return float(2.0 * np.logical_and(pred, gt).sum() / denom) if denom > 0 else NAN


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = _as_bool(pred), _as_bool(gt)
    if gt.sum() == 0:
        return NAN
    union = np.logical_or(pred, gt).sum()
    return float(np.logical_and(pred, gt).sum() / union) if union > 0 else NAN


def sensitivity(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = _as_bool(pred), _as_bool(gt)
    if gt.sum() == 0:
        return NAN
    return float(np.logical_and(pred, gt).sum() / gt.sum())


def precision(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = _as_bool(pred), _as_bool(gt)
    if gt.sum() == 0:
        return NAN
    if pred.sum() == 0:
        return 0.0
    return float(np.logical_and(pred, gt).sum() / pred.sum())


# +------------------------- #
# Surface distance metrics
# +------------------------- #


def _surface(mask: np.ndarray) -> np.ndarray:
    er = binary_erosion(mask, iterations=1, border_value=0)
    return mask & (~er)


def _surface_distances(pred: np.ndarray, gt: np.ndarray, spacing: Spacing
                       ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    pred, gt = _as_bool(pred), _as_bool(gt)
    if pred.sum() == 0 or gt.sum() == 0:
        return None
    sp = _surface(pred)
    sg = _surface(gt)
    if sp.sum() == 0 or sg.sum() == 0:
        return None
    dt_to_gt = distance_transform_edt(~sg, sampling=spacing)
    dt_to_pred = distance_transform_edt(~sp, sampling=spacing)
    d_pred_to_gt = dt_to_gt[sp]
    d_gt_to_pred = dt_to_pred[sg]
    return d_pred_to_gt, d_gt_to_pred


def hd95(pred: np.ndarray, gt: np.ndarray, spacing: Spacing = (1.0, 1.0, 1.0)) -> float:
    res = _surface_distances(pred, gt, spacing)
    if res is None:
        return NAN
    d1, d2 = res
    return float(max(np.percentile(d1, 95), np.percentile(d2, 95)))


def assd(pred: np.ndarray, gt: np.ndarray, spacing: Spacing = (1.0, 1.0, 1.0)) -> float:
    res = _surface_distances(pred, gt, spacing)
    if res is None:
        return NAN
    d1, d2 = res
    return float((d1.sum() + d2.sum()) / (len(d1) + len(d2)))


# +------------------------- #
# Connectivity / topology metrics (vessels)
# +------------------------- #


def cldice_metric(pred: np.ndarray, gt: np.ndarray) -> float:
    """Hard centerline-Dice: harmonic mean of topology precision & sensitivity."""
    pred, gt = _as_bool(pred), _as_bool(gt)
    if gt.sum() == 0:
        return NAN
    if pred.sum() == 0:
        return 0.0
    skel_p = skeletonize(pred)
    skel_g = skeletonize(gt)
    if skel_p.sum() == 0 or skel_g.sum() == 0:
        return 0.0
    tprec = np.logical_and(skel_p, gt).sum() / skel_p.sum()
    tsens = np.logical_and(skel_g, pred).sum() / skel_g.sum()
    if (tprec + tsens) == 0:
        return 0.0
    return float(2.0 * tprec * tsens / (tprec + tsens))


def connected_components_error(pred: np.ndarray, gt: np.ndarray) -> float:
    """|#components(pred) - #components(gt)| -- a Betti-0 (connectivity) proxy."""
    pred, gt = _as_bool(pred), _as_bool(gt)
    if gt.sum() == 0:
        return NAN
    n_pred = label(pred)[1]
    n_gt = label(gt)[1]
    return float(abs(n_pred - n_gt))


def centerline_overlap(pred: np.ndarray, gt: np.ndarray) -> float:
    """Fraction of the GT centerline (skeleton) covered by the prediction."""
    pred, gt = _as_bool(pred), _as_bool(gt)
    if gt.sum() == 0:
        return NAN
    skel_g = skeletonize(gt)
    if skel_g.sum() == 0:
        return NAN
    return float(np.logical_and(skel_g, pred).sum() / skel_g.sum())


def n_components(mask: np.ndarray) -> int:
    return int(label(_as_bool(mask))[1])


# +------------------------- #
# Per-structure metric bundles
# +------------------------- #


def evaluate_structure(pred: np.ndarray, gt: np.ndarray, structure: str,
                       spacing: Spacing = (1.0, 1.0, 1.0)) -> Dict[str, float]:
    """Compute the metric set appropriate to *structure*.

    Returns NaNs throughout if the ground truth is empty (structure absent).
    """
    m = {
        "dice": dice(pred, gt),
        "iou": iou(pred, gt),
        "sensitivity": sensitivity(pred, gt),
        "precision": precision(pred, gt),
        "hd95": hd95(pred, gt, spacing),
        "assd": assd(pred, gt, spacing),
    }
    if structure == "vessel":
        m.update({
            "cldice": cldice_metric(pred, gt),
            "cc_error": connected_components_error(pred, gt),
            "centerline_overlap": centerline_overlap(pred, gt),
            "n_components_pred": float(n_components(pred)),
            "n_components_gt": float(n_components(gt)),
        })
    return m
