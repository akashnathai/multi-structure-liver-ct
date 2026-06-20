"""The actual training loop - AMP, cosine warm restarts, grad clipping,
checkpoint/resume, early stopping, plus the OOM fallback ladder.

The fallback ladder only exists because we're trying to fit this in an 8 GB
budget. Whatever rung ends up active gets printed to the console/log, but it
never goes into anything paper-facing - it's purely a "did this even run"
detail, not a result.
"""

from __future__ import annotations

import copy
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

from .config import Config
from .data import PatchDataset, collate_patches
from .losses import CombinedLoss
from .models import build_model
from .evaluate import validation_dice


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def _save_ckpt(path: Path, **state) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, str(path))


def _is_completed(ckpt_dir: Path) -> bool:
    f = ckpt_dir / "history.json"
    if not f.exists():
        return False
    with open(f, encoding="utf-8") as fh:
        h = json.load(fh)
    return bool(h.get("completed", False))


# ---------------------------------------------------------------------------
# A single training attempt, at whatever memory rung we're currently on
# ---------------------------------------------------------------------------


def _train_attempt(cfg: Config, tag: str, train_ids: Sequence[str],
                   val_ids: Sequence[str]) -> Path:
    device = get_device()
    set_all_seeds(cfg.seed)

    ckpt_dir = cfg.checkpoints_dir / tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest_path = ckpt_dir / "latest.pt"
    best_path = ckpt_dir / "best.pt"
    hist_path = ckpt_dir / "history.json"

    model = build_model(cfg).to(device)
    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = CosineAnnealingWarmRestarts(opt, T_0=cfg.cosine_T0, T_mult=cfg.cosine_T_mult)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.use_amp and device.type == "cuda"))
    criterion = CombinedLoss(cfg)

    train_ds = PatchDataset(cfg, train_ids, train=True)
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                          num_workers=cfg.num_workers, collate_fn=collate_patches,
                          pin_memory=(device.type == "cuda"), drop_last=False)

    start_epoch, best_metric, patience = 0, -1.0, 0
    history: List[Dict] = []

    # pick up where we left off, if there's anything to pick up
    if latest_path.exists():
        ck = torch.load(str(latest_path), map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch = ck["epoch"] + 1
        best_metric = ck["best_metric"]
        patience = ck["patience"]
        history = ck.get("history", [])
        print(f"   [RESUME] {tag} from epoch {start_epoch} (best={best_metric:.4f})")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"   [MODEL] params={n_params:,}  base_ch={cfg.base_channels}  "
          f"patch={cfg.patch_size}  batch={cfg.batch_size}  amp={cfg.use_amp}  "
          f"ckpt={cfg.use_checkpointing}  cldice={cfg.use_cldice}  "
          f"constraints={cfg.use_constraints}")

    for epoch in range(start_epoch, cfg.max_epochs):
        model.train()
        t0 = time.time()
        agg: Dict[str, float] = {}
        n_batches = 0
        for batch in train_dl:
            image = batch["image"].to(device, non_blocking=True)
            targets = {s: v.to(device, non_blocking=True) for s, v in batch["targets"].items()}
            present = batch["present"]

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda",
                                enabled=(cfg.use_amp and device.type == "cuda")):
                logits = model(image)
                loss, comp = criterion(logits, targets, present, epoch)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            scaler.step(opt)
            scaler.update()

            for k, v in comp.items():
                agg[k] = agg.get(k, 0.0) + v
            n_batches += 1

        sched.step()
        agg = {k: v / max(n_batches, 1) for k, v in agg.items()}

        arch = {"base_channels": cfg.base_channels, "patch_size": list(cfg.patch_size)}

        # Write the checkpoint for THIS epoch before we touch validation.
        # Validation is by far the heaviest GPU op we run, and on a laptop GPU
        # that's where a driver hiccup / TDR is most likely to take everything
        # down. Saving first means a crash there just costs a re-run, not the
        # whole epoch - we resume cleanly at epoch+1 either way.
        def _write_latest():
            _save_ckpt(latest_path, model=model.state_dict(), opt=opt.state_dict(),
                       sched=sched.state_dict(), scaler=scaler.state_dict(),
                       epoch=epoch, best_metric=best_metric, patience=patience,
                       history=history, arch=arch)
        _write_latest()

        is_val_epoch = (epoch % cfg.val_every == 0) or (epoch == cfg.max_epochs - 1)
        if is_val_epoch:
            val_metric = validation_dice(cfg, model, val_ids, device)
            improved = val_metric > best_metric
            if improved:
                best_metric = val_metric
                patience = 0
            else:
                patience += 1
        else:
            val_metric = float("nan")
            improved = False
        if device.type == "cuda":
            torch.cuda.empty_cache()

        rec = {"epoch": epoch, "train_loss": agg.get("total", 0.0),
               "val_dice": val_metric, "lr": opt.param_groups[0]["lr"],
               "components": agg, "best": best_metric}
        history.append(rec)

        if epoch % cfg.log_every == 0 or improved:
            print(f"   [E{epoch:03d}] loss={agg.get('total', 0):.4f} "
                  f"(L={agg.get('liver', 0):.3f} T={agg.get('tumour', 0):.3f} "
                  f"V={agg.get('vessel', 0):.3f} C={agg.get('contain', 0):.3f} "
                  f"X={agg.get('exclude', 0):.4f})  val_dice={val_metric:.4f}"
                  f"{'  *best*' if improved else ''}  ({time.time()-t0:.1f}s)")

        # Now that validation's done, write latest again (best/patience/history
        # are current) and drop a fresh best.pt if this epoch earned it.
        _write_latest()
        if improved:
            _save_ckpt(best_path, model=model.state_dict(), epoch=epoch,
                       best_metric=best_metric, arch=arch)
        with open(hist_path, "w", encoding="utf-8") as fh:
            json.dump({"tag": tag, "history": history, "completed": False,
                       "best_metric": best_metric}, fh, indent=2)

        if patience >= cfg.early_stop_patience:
            print(f"   [EARLY STOP] {tag} at epoch {epoch} (best={best_metric:.4f})")
            break

    if not best_path.exists():  # edge case: it never once improved, so just keep the last epoch
        _save_ckpt(best_path, model=model.state_dict(), epoch=cfg.max_epochs - 1,
                   best_metric=best_metric,
                   arch={"base_channels": cfg.base_channels, "patch_size": list(cfg.patch_size)})
    with open(hist_path, "w", encoding="utf-8") as fh:
        json.dump({"tag": tag, "history": history, "completed": True,
                   "best_metric": best_metric}, fh, indent=2)
    return best_path


# ---------------------------------------------------------------------------
# Public entry point - wraps the attempt above with the OOM fallback ladder
# ---------------------------------------------------------------------------


def train_one_fold(cfg_in: Config, tag: str, train_ids: Sequence[str],
                   val_ids: Sequence[str], switches: Dict[str, bool]) -> Path:
    """Train a single (condition, fold) pair and hand back the best checkpoint.

    Safe to call more than once - if the tag already finished, we just return
    its checkpoint instead of redoing the work. If CUDA runs out of memory
    mid-attempt, we drop down a rung on the memory ladder (smaller patch,
    fewer channels, eventually batch size 1) and try the tag again.
    """
    cfg = copy.deepcopy(cfg_in)
    cfg.use_cldice = switches["use_cldice"]
    cfg.use_constraints = switches["use_constraints"]

    ckpt_dir = cfg.checkpoints_dir / tag
    best_path = ckpt_dir / "best.pt"
    if best_path.exists() and _is_completed(ckpt_dir):
        print(f"   [SKIP] {tag} already completed.")
        return best_path

    rungs = ([None] if cfg.smoke_test else cfg.memory_ladder)
    for ri, rung in enumerate(rungs):
        if rung is not None:
            cfg.apply_rung(rung)
        active = (f"patch={cfg.patch_size} base_ch={cfg.base_channels} "
                  f"batch={cfg.batch_size}")
        print(f"   [MEM] active config: {active}"
              + ("" if rung is None else f"  (ladder rung {ri})"))
        try:
            return _train_attempt(cfg, tag, train_ids, val_ids)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and ri < len(rungs) - 1:
                print(f"   [OOM] {tag}: descending memory ladder ...")
                torch.cuda.empty_cache()
                lp = ckpt_dir / "latest.pt"
                if lp.exists():
                    lp.unlink()  # shapes won't match the new rung, so start that one clean
                continue
            raise
    return best_path