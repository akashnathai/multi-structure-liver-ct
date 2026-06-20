"""Command-line entry point.

Subcommands
-----------
  audit        Generate dataset_audit.json from DICOM and print the summary table.      ||  python -m src.run audit
  folds        Build (and print) the tumour-burden-stratified 5-fold assignment.        ||  python -m src.run smoke-test 
  preprocess   Resample/window/normalise/cache every patient to .npy.                   ||  python -m src.run preprocess
  smoke-test   Run the entire pipeline end-to-end on 2 patients / 3 epochs.             ||  python -m src.run run-cv   
  run-cv       Run the full 5-fold cross-validation over the four ablation conditions.  ||  python -m src.run run-cv --conditions full baseline --folds 0 1
  evaluate     (Re)collect test metrics from existing checkpoints.                      ||   python -m src.run run-cv --conditions full baseline --folds 0 1
  figures      Build all tables and figures from collected metrics.                     || python -m src.run figures
  
"""

 
from __future__ import annotations
 
import argparse
import sys
from pathlib import Path
from typing import List, Optional
 
from .config import Config, ABLATION_CONDITIONS
 
 
def _build_cfg(args) -> Config:
    cfg = Config()
    if args.data_root:
        cfg.data_root = Path(args.data_root)
    if args.outputs_dir:
        cfg.outputs_dir = Path(args.outputs_dir)
    if getattr(args, "seed", None) is not None:
        cfg.seed = args.seed
    cfg.ensure_dirs()
    return cfg
 
 
def cmd_audit(cfg: Config, args) -> None:
    from .cross_val import build_audit, summarise_audit
    audit = build_audit(cfg, force=args.force)
    summarise_audit(audit)
 
 
def cmd_folds(cfg: Config, args) -> None:
    from .cross_val import build_audit, build_folds, summarise_folds, fold_split
    audit = build_audit(cfg)
    folds = build_folds(cfg, audit, force=args.force)
    summarise_folds(folds, audit)
    print("\nPer-fold train / internal-val / test split:")
    for f in range(cfg.n_folds):
        tr, va, te = fold_split(folds, f, cfg)
        print(f"  fold {f}: train={tr}\n           val  ={va}\n           test ={te}")
 
 
def cmd_preprocess(cfg: Config, args) -> None:
    from .data import preprocess_all
    cfg.save()
    preprocess_all(cfg, force=args.force)
 
 
def cmd_smoke_test(cfg: Config, args) -> None:
    """Wiring check, nothing more: audit -> preprocess(2) -> train(3ep) -> eval -> figures.
 
    The point isn't to get good metrics, it's to catch a broken pipeline before
    burning hours on a real run."""
    from .data import preprocess_all
    from .cross_val import build_audit, build_folds, summarise_folds, fold_split
    from .train import train_one_fold
    from .evaluate import evaluate_fold, append_per_patient_rows
    from . import figures as F
 
    cfg.apply_smoke_test()
    # Keep every smoke-test artifact in its own subfolder so it can never clobber real results.
    cfg.outputs_dir = cfg.outputs_dir / "_smoke"
    cfg.ensure_dirs()
    cfg.save()
    pids = [1, 2]  # patient 1 has tumour + vessel, both have liver - that's all we need here
    print("=" * 72, "\n[SMOKE] 1/5  audit", "\n" + "=" * 72)
    audit = build_audit(cfg, pids=pids, force=True)
 
    print("=" * 72, "\n[SMOKE] 2/5  preprocess (2 patients)", "\n" + "=" * 72)
    preprocess_all(cfg, pids=pids, force=args.force)
 
    print("=" * 72, "\n[SMOKE] 3/5  folds", "\n" + "=" * 72)
    folds = build_folds(cfg, audit, force=True)
    summarise_folds(folds, audit)
 
    # Tiny manual split - train on P1, val/test on P2. Don't need anything cleverer for a smoke test.
    train_ids = ["01"]
    val_ids = ["02"]
    test_ids = ["02"]
    cond = "full"
    switches = ABLATION_CONDITIONS[cond]
    tag = f"smoke_{cond}"
    print("=" * 72, f"\n[SMOKE] 4/5  train ({tag}) + evaluate", "\n" + "=" * 72)
    ckpt = train_one_fold(cfg, tag, train_ids, val_ids, switches)
    rows = evaluate_fold(cfg, tag, cond, 0, test_ids, ckpt, switches)
    append_per_patient_rows(cfg, rows)
 
    print("=" * 72, "\n[SMOKE] 5/5  figures", "\n" + "=" * 72)
    F.fig_dataset_overview(cfg, audit)
    try:
        F.fig_preprocess_verification(cfg, pids=["01", "02"])
    except Exception as e:
        print(f"[SMOKE] preprocess fig skipped: {e}")
    F.fig_training_curves(cfg, conditions=[])  # tag's nonstandard - just checking this doesn't crash
    print("\n[SMOKE] DONE -- pipeline wired end-to-end. "
          "Inspect outputs/ for cached preprocessed data, checkpoints, and figures.")
 
 
def cmd_run_cv(cfg: Config, args) -> None:
    from .cross_val import run_cross_validation
    if args.max_epochs is not None:
        cfg.max_epochs = args.max_epochs
    if args.val_every is not None:
        cfg.val_every = args.val_every
    if args.patience is not None:
        cfg.early_stop_patience = args.patience
    cfg.save()
    conditions = args.conditions or list(ABLATION_CONDITIONS)
    folds = args.folds if args.folds is not None else list(range(cfg.n_folds))
    print(f"[CV] max_epochs={cfg.max_epochs} val_every={cfg.val_every} "
          f"patience={cfg.early_stop_patience} conditions={conditions} folds={folds}")
    run_cross_validation(cfg, conditions=conditions, folds_to_run=folds)
 
 
def cmd_evaluate(cfg: Config, args) -> None:
    from .cross_val import build_audit, build_folds, fold_split
    from .evaluate import evaluate_fold, append_per_patient_rows
    audit = build_audit(cfg)
    folds = build_folds(cfg, audit)
    conditions = args.conditions or list(ABLATION_CONDITIONS)
    fold_list = args.folds if args.folds is not None else list(range(cfg.n_folds))
    for cond in conditions:
        for f in fold_list:
            _, _, test_ids = fold_split(folds, f, cfg)
            ckpt = cfg.checkpoints_dir / f"{cond}_fold{f}" / "best.pt"
            if not ckpt.exists():
                print(f"  [SKIP] no checkpoint for {cond} fold {f}")
                continue
            rows = evaluate_fold(cfg, f"{cond}_fold{f}", cond, f, test_ids, ckpt,
                                 ABLATION_CONDITIONS[cond])
            append_per_patient_rows(cfg, rows)
 
 
def cmd_figures(cfg: Config, args) -> None:
    from . import figures as F
    F.make_all_figures(cfg)
 
 
def cmd_status(cfg: Config, args) -> None:
    """Dump where the CV run currently stands. Read-only, so it's safe to run
    in another terminal while training is still going."""
    import json
    import csv as _csv
    conds = list(ABLATION_CONDITIONS)
    folds = list(range(cfg.n_folds))
    total = len(conds) * len(folds)
    print("=" * 64)
    print("CROSS-VALIDATION PROGRESS")
    print("=" * 64)
    done = 0
    inprog = 0
    for c in conds:
        for f in folds:
            tag = f"{c}_fold{f}"
            hp = cfg.checkpoints_dir / tag / "history.json"
            if not hp.exists():
                print(f"  {tag:20s} : not started")
                continue
            h = json.loads(hp.read_text(encoding="utf-8"))
            best = h.get("best_metric", float("nan"))
            last = h["history"][-1]["epoch"] if h.get("history") else -1
            if h.get("completed", False):
                done += 1
                print(f"  {tag:20s} : DONE   (best val Dice={best:.4f}, last epoch {last})")
            else:
                inprog += 1
                print(f"  {tag:20s} : training (epoch {last + 1}, best so far={best:.4f})")
    print("-" * 64)
    print(f"  Completed: {done}/{total}   In progress: {inprog}   "
          f"Not started: {total - done - inprog}")
    p = cfg.results_dir / "per_patient_metrics.csv"
    if p.exists():
        rows = list(_csv.DictReader(open(p, encoding="utf-8")))
        pairs = {(r["condition"], r["fold"]) for r in rows}
        print(f"  Test results on disk: {len(pairs)} (condition,fold) pairs, {len(rows)} rows")
    else:
        print("  Test results on disk: none yet (appears after a fold finishes)")
    print("=" * 64)
 
 
def _int_list(values: Optional[List[str]]) -> Optional[List[int]]:
    return [int(v) for v in values] if values else None
 
 
def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="src.run", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default=None, help="Folder containing 3Dircadb1.1 ... .20")
    p.add_argument("--outputs-dir", default=None)
    p.add_argument("--seed", type=int, default=None)
    sub = p.add_subparsers(dest="command", required=True)
 
    sp = sub.add_parser("audit"); sp.add_argument("--force", action="store_true")
    sp = sub.add_parser("folds"); sp.add_argument("--force", action="store_true")
    sp = sub.add_parser("preprocess"); sp.add_argument("--force", action="store_true")
    sp = sub.add_parser("smoke-test"); sp.add_argument("--force", action="store_true")
 
    sp = sub.add_parser("run-cv")
    sp.add_argument("--conditions", nargs="*", choices=list(ABLATION_CONDITIONS), default=None)
    sp.add_argument("--folds", nargs="*", type=int, default=None)
    sp.add_argument("--max-epochs", type=int, default=None, help="override Config.max_epochs")
    sp.add_argument("--val-every", type=int, default=None, help="epochs between validation passes")
    sp.add_argument("--patience", type=int, default=None, help="early-stop patience (val passes)")
 
    sp = sub.add_parser("evaluate")
    sp.add_argument("--conditions", nargs="*", choices=list(ABLATION_CONDITIONS), default=None)
    sp.add_argument("--folds", nargs="*", type=int, default=None)
 
    sub.add_parser("figures")
    sub.add_parser("status")
 
    args = p.parse_args(argv)
    cfg = _build_cfg(args)
 
    dispatch = {
        "audit": cmd_audit, "folds": cmd_folds, "preprocess": cmd_preprocess,
        "smoke-test": cmd_smoke_test, "run-cv": cmd_run_cv,
        "evaluate": cmd_evaluate, "figures": cmd_figures, "status": cmd_status,
    }
    dispatch[args.command](cfg, args)
    return 0
 
 
if __name__ == "__main__":
    sys.exit(main())