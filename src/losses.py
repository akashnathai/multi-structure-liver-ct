"""Loss functions: base per-structure losses, soft-clDice, and the anatomical
constraint losses (containment + exclusion), composed by ``CombinedLoss``.

All base losses are computed per-sample so that empty-structure patches can be
masked out (weight 0) without corrupting the batch average — the graph is kept
intact so the optimiser step is still well-defined.

Constraint and clDice terms are individually toggleable (for the ablation) and
the constraint weights are ramped in linearly after a warmup to avoid early
instability.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

_DIMS = (1, 2, 3, 4)  # channel + 3 spatial dims (per-sample reduction keeps B)
_SMOOTH = 1.0


# +------------------------- #
# Per-sample base losses (operate on probabilities unless noted)
# +------------------------- #


def dice_loss_per_sample(prob: torch.Tensor, target: torch.Tensor,
                         smooth: float = _SMOOTH) -> torch.Tensor:
    inter = (prob * target).sum(_DIMS)
    denom = prob.sum(_DIMS) + target.sum(_DIMS)
    return 1.0 - (2.0 * inter + smooth) / (denom + smooth)


def bce_loss_per_sample(logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logit, target, reduction="none").mean(_DIMS)


def focal_loss_per_sample(logit: torch.Tensor, target: torch.Tensor,
                          gamma: float = 2.0, alpha: float = 0.75) -> torch.Tensor:
    prob = torch.sigmoid(logit)
    ce = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
    p_t = prob * target + (1.0 - prob) * (1.0 - target)
    alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
    focal = alpha_t * (1.0 - p_t).clamp(min=1e-6).pow(gamma) * ce
    return focal.mean(_DIMS)


# +------------------------- #
# Soft clDice (Shit et al., CVPR 2021) -- differentiable centerline Dice
# +------------------------- #


def _soft_erode(img: torch.Tensor) -> torch.Tensor:
    p1 = -F.max_pool3d(-img, (3, 1, 1), 1, (1, 0, 0))
    p2 = -F.max_pool3d(-img, (1, 3, 1), 1, (0, 1, 0))
    p3 = -F.max_pool3d(-img, (1, 1, 3), 1, (0, 0, 1))
    return torch.min(torch.min(p1, p2), p3)


def _soft_dilate(img: torch.Tensor) -> torch.Tensor:
    return F.max_pool3d(img, (3, 3, 3), 1, (1, 1, 1))


def _soft_open(img: torch.Tensor) -> torch.Tensor:
    return _soft_dilate(_soft_erode(img))


def soft_skeleton(img: torch.Tensor, iters: int) -> torch.Tensor:
    """Differentiable soft skeletonisation via iterative min/max pooling."""
    img1 = _soft_open(img)
    skel = F.relu(img - img1)
    for _ in range(iters):
        img = _soft_erode(img)
        img1 = _soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


def soft_cldice_per_sample(prob: torch.Tensor, target: torch.Tensor,
                           iters: int = 5, smooth: float = _SMOOTH) -> torch.Tensor:
    skel_p = soft_skeleton(prob, iters)
    skel_t = soft_skeleton(target, iters)
    tprec = (torch.sum(skel_p * target, _DIMS) + smooth) / (torch.sum(skel_p, _DIMS) + smooth)
    tsens = (torch.sum(skel_t * prob, _DIMS) + smooth) / (torch.sum(skel_t, _DIMS) + smooth)
    return 1.0 - 2.0 * (tprec * tsens) / (tprec + tsens)


# +------------------------- #
# Anatomical constraint losses (the contribution)
# +------------------------- #


def containment_loss(p_liver: torch.Tensor, p_tumour: torch.Tensor,
                     p_vessel: torch.Tensor) -> torch.Tensor:
    """Penalise sub-structure probability exceeding liver probability.

    L = mean(relu(p_tumour - p_liver)) + mean(relu(p_vessel - p_liver)).
    Softly enforces tumour ⊆ liver and vessel ⊆ liver.
    """
    return (F.relu(p_tumour - p_liver).mean()
            + F.relu(p_vessel - p_liver).mean())


def exclusion_loss(p_vessel: torch.Tensor, p_tumour: torch.Tensor) -> torch.Tensor:
    """Penalise vessel and tumour co-occupying a voxel: L = mean(p_vessel * p_tumour)."""
    return (p_vessel * p_tumour).mean()


# +------------------------- #
# Helpers
# +------------------------- #


def _weighted_mean(per_sample: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Mean of per-sample loss weighted by presence; 0 (graph-preserving) if none."""
    denom = weight.sum()
    if float(denom) <= 0.0:
        return (per_sample * 0.0).sum()
    return (per_sample * weight).sum() / denom


def constraint_ramp(epoch: int, start: int, end: int) -> float:
    """Linear ramp in [0,1]: 0 at/below start, 1 at/above end."""
    if epoch <= start:
        return 0.0
    if epoch >= end:
        return 1.0
    return (epoch - start) / float(end - start)


# +------------------------- #
# Combined loss ------- main for loss and aggrigation
# +------------------------- #


class CombinedLoss:
    """Total multi-task loss with toggleable clDice and constraint terms.

    L = w_liver * L_liver + w_tumour * L_tumour + w_vessel * L_vessel
        + lambda_c(epoch) * L_contain + lambda_e(epoch) * L_exclude
    """

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def __call__(self, logits: Dict[str, torch.Tensor],
                 targets: Dict[str, torch.Tensor],
                 present: Dict[str, torch.Tensor],
                 epoch: int) -> Tuple[torch.Tensor, Dict[str, float]]:
        cfg = self.cfg
        probs = {s: torch.sigmoid(v) for s, v in logits.items()}
        dev = logits["liver"].device
        w_present = {s: present[s].to(dev) for s in present}

        # ---- Liver: 0.5 Dice + 0.5 BCE ----
        liver_ps = 0.5 * dice_loss_per_sample(probs["liver"], targets["liver"]) \
            + 0.5 * bce_loss_per_sample(logits["liver"], targets["liver"])
        l_liver = _weighted_mean(liver_ps, w_present["liver"])

        # ---- Tumour: 0.5 Dice + 0.5 Focal ----
        tumour_ps = 0.5 * dice_loss_per_sample(probs["tumour"], targets["tumour"]) \
            + 0.5 * focal_loss_per_sample(logits["tumour"], targets["tumour"],
                                          cfg.focal_gamma, cfg.focal_alpha)
        l_tumour = _weighted_mean(tumour_ps, w_present["tumour"])

        # ---- Vessel: 0.5 Dice + 0.5 (clDice | BCE) ----
        vessel_dice = dice_loss_per_sample(probs["vessel"], targets["vessel"])
        if cfg.use_cldice:
            vessel_conn = soft_cldice_per_sample(probs["vessel"], targets["vessel"],
                                                 iters=cfg.cldice_iters)
        else:
            vessel_conn = bce_loss_per_sample(logits["vessel"], targets["vessel"])
        vessel_ps = 0.5 * vessel_dice + 0.5 * vessel_conn
        l_vessel = _weighted_mean(vessel_ps, w_present["vessel"])

        total = cfg.w_liver * l_liver + cfg.w_tumour * l_tumour + cfg.w_vessel * l_vessel

        comp = {
            "liver": float(l_liver.detach()),
            "tumour": float(l_tumour.detach()),
            "vessel": float(l_vessel.detach()),
            "contain": 0.0,
            "exclude": 0.0,
            "lambda_c": 0.0,
            "lambda_e": 0.0,
        }

        # ---- Constraints (ramped) ----
        if cfg.use_constraints:
            ramp = constraint_ramp(epoch, cfg.constraint_warmup_start,
                                   cfg.constraint_warmup_end)
            lam_c = cfg.lambda_contain * ramp
            lam_e = cfg.lambda_exclude * ramp
            l_contain = containment_loss(probs["liver"], probs["tumour"], probs["vessel"])
            l_exclude = exclusion_loss(probs["vessel"], probs["tumour"])
            total = total + lam_c * l_contain + lam_e * l_exclude
            comp.update({
                "contain": float(l_contain.detach()),
                "exclude": float(l_exclude.detach()),
                "lambda_c": lam_c,
                "lambda_e": lam_e,
            })

        comp["total"] = float(total.detach())
        return total, comp
