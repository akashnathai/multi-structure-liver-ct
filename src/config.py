"""Central configuration.
Array convention used throughout the codebase
---------------------------------------------
Volumes and masks are stored as ``(H, W, D)`` numpy arrays, where ``D`` is the
slice (axial) axis.  Tensors fed to the 3D U-Net are ``(B, C, H, W, D)``.
``patch_size`` is therefore ``(pH, pW, pD)`` = ``(128, 128, 64)`` — 128x128 in
plane, 64 along the slice axis.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


DEFAULT_DATA_ROOT = Path(
    os.environ.get("IRCADB_DATA_ROOT", r"x:\liver_ai\3Dircadb1\3Dircadb1")
)

# The three structures this project segments simultaneously.
STRUCTURES: Tuple[str, ...] = ("liver", "tumour", "vessel")

# Raw IRCADb mask-folder name fragments that compose each training target.
# tumour: any mask whose name contains "tumor"/"tumour" (e.g. livertumor01..07).
# vessel: union of whichever of these are present for the patient.
VESSEL_SOURCE_NAMES: Tuple[str, ...] = ("portalvein", "venoussystem", "venacava", "artery")
TUMOUR_NAME_FRAGMENTS: Tuple[str, ...] = ("tumor", "tumour")
LIVER_NAME: str = "liver"

# Patients known to have NO tumour mask (documented dataset fact; the audit
# verifies this empirically and is the runtime source of truth).
KNOWN_NO_TUMOUR_IDS: Tuple[int, ...] = (11, 14, 20)


@dataclass
class MemoryRung:
    """One rung of the OOM fallback ladder (implementation detail only)."""

    patch_size: Tuple[int, int, int]
    base_channels: int
    batch_size: int

    def describe(self) -> str:
        return (f"patch={self.patch_size} base_channels={self.base_channels} "
                f"batch_size={self.batch_size}")


@dataclass
class Config:
    """All hyperparameters for preprocessing, model, losses, and training."""

    # ---- Paths +---------
    data_root: Path = DEFAULT_DATA_ROOT
    work_dir: Path = Path(r"x:\liver_ai")
    outputs_dir: Path = Path(r"x:\liver_ai\outputs")

    # ---- Reproducibility -------------------------------------------------
    seed: int = 42

    # ---- Preprocessing +-
    target_spacing_mm: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    hu_min: float = -200.0
    hu_max: float = 250.0

    # ---- Model +---------
    base_channels: int = 32          # encoder level-0 width; levels = [1,2,4,8]x
    leaky_slope: float = 0.1
    use_checkpointing: bool = True   # gradient checkpointing in enc/dec blocks

    # ---- Patch sampling +
    patch_size: Tuple[int, int, int] = (128, 128, 64)   # (pH, pW, pD)
    batch_size: int = 2
    patches_per_volume: int = 8      # virtual dataset length multiplier (train)
    foreground_ratio: float = 0.60   # fraction of patches centred in liver bbox
    num_workers: int = 0             # Windows-safe default; raise on Linux

    # ---- Cross-validation ------------------------------------------------
    n_folds: int = 5
    internal_val_fraction: float = 0.2   # held out of each train fold for early stop

    # ---- Training +------
    max_epochs: int = 300
    lr: float = 1e-4
    weight_decay: float = 1e-5
    cosine_T0: int = 50
    cosine_T_mult: int = 2
    grad_clip_norm: float = 1.0
    use_amp: bool = True
    early_stop_patience: int = 50
    log_every: int = 1
    val_every: int = 1               # epochs between internal-validation passes

    # ---- Loss weights +--
    w_liver: float = 1.0
    w_tumour: float = 2.0
    w_vessel: float = 1.0
    focal_gamma: float = 2.0
    focal_alpha: float = 0.75
    cldice_iters: int = 5            # soft-skeleton iterations

    # Constraint losses (the contribution) -- individually toggleable.
    lambda_contain: float = 0.5
    lambda_exclude: float = 0.5
    constraint_warmup_start: int = 10   # epoch where ramp begins
    constraint_warmup_end: int = 30     # epoch where ramp reaches full weight

    # Ablation switches (set per condition by cross_val).
    use_cldice: bool = True          # else vessel uses Dice+BCE
    use_constraints: bool = True     # else containment+exclusion disabled

    # ---- Augmentation (train only) --------------------------------------
    aug_flip_prob: float = 0.5
    aug_rot90_prob: float = 0.5
    aug_noise_sigma_max: float = 0.05
    aug_brightness: float = 0.10
    aug_contrast: float = 0.10

    # ---- Sliding-window inference ---------------------------------------
    sw_overlap: float = 0.5          # test-time overlap (accurate)
    sw_overlap_val: float = 0.25     # validation overlap (coarser/faster, early-stop only)
    sw_sigma_scale: float = 0.125
    sw_batch_size: int = 1

    # ---- Figures +-------
    fig_dpi: int = 200
    fig_format: str = "png"

    # ---- Memory fallback ladder (implementation only; never reported) ----
    memory_ladder: List[MemoryRung] = field(default_factory=lambda: [
        MemoryRung(patch_size=(128, 128, 64), base_channels=32, batch_size=2),
        MemoryRung(patch_size=(96, 96, 96),  base_channels=24, batch_size=1),
        MemoryRung(patch_size=(64, 64, 64),  base_channels=16, batch_size=1),
    ])

    # ---- Smoke-test overrides (set by --smoke-test) ---------------------
    smoke_test: bool = False

    # =====================================================================
    # Derived path helpers
    # =====================================================================
    @property
    def preprocessed_dir(self) -> Path:
        return self.outputs_dir / "preprocessed"

    @property
    def checkpoints_dir(self) -> Path:
        return self.outputs_dir / "checkpoints"

    @property
    def figures_dir(self) -> Path:
        return self.outputs_dir / "figures"

    @property
    def results_dir(self) -> Path:
        return self.outputs_dir / "results"

    @property
    def logs_dir(self) -> Path:
        return self.outputs_dir / "logs"

    @property
    def audit_path(self) -> Path:
        return self.outputs_dir / "dataset_audit.json"

    @property
    def folds_path(self) -> Path:
        return self.outputs_dir / "folds.json"

    @property
    def config_path(self) -> Path:
        return self.outputs_dir / "config.json"

    def all_dirs(self) -> List[Path]:
        return [self.outputs_dir, self.preprocessed_dir, self.checkpoints_dir,
                self.figures_dir, self.results_dir, self.logs_dir]

    def ensure_dirs(self) -> None:
        for d in self.all_dirs():
            d.mkdir(parents=True, exist_ok=True)

    # =====================================================================
    # Memory-ladder application
    # =====================================================================
    def apply_rung(self, rung: MemoryRung) -> None:
        """Mutate config to use a memory-ladder rung (after an OOM)."""
        self.patch_size = rung.patch_size
        self.base_channels = rung.base_channels
        self.batch_size = rung.batch_size

    def apply_smoke_test(self) -> None:
        """Shrink everything so the whole pipeline runs end-to-end in minutes."""
        self.smoke_test = True
        self.patch_size = (64, 64, 32)
        self.base_channels = 8
        self.batch_size = 1
        self.patches_per_volume = 2
        self.max_epochs = 3
        self.early_stop_patience = 999       # never early-stop in a 3-epoch run
        self.constraint_warmup_start = 0
        self.constraint_warmup_end = 1
        self.cldice_iters = 2
        self.num_workers = 0

    # =====================================================================
    # Serialisation
    # =====================================================================
    def to_dict(self) -> Dict:
        d = asdict(self)
        # JSON-friendly conversions
        d["data_root"] = str(self.data_root)
        d["work_dir"] = str(self.work_dir)
        d["outputs_dir"] = str(self.outputs_dir)
        d["memory_ladder"] = [asdict(r) for r in self.memory_ladder]
        return d

    def save(self, path: Optional[Path] = None) -> Path:
        path = Path(path) if path is not None else self.config_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        return path


# Mapping from an ablation condition name to its loss switches.
ABLATION_CONDITIONS: Dict[str, Dict[str, bool]] = {
    "baseline":      {"use_cldice": False, "use_constraints": False},
    "cldice":        {"use_cldice": True,  "use_constraints": False},
    "constraints":   {"use_cldice": False, "use_constraints": True},
    "full":          {"use_cldice": True,  "use_constraints": True},
}

ABLATION_LABELS: Dict[str, str] = {
    "baseline":    "Baseline (Dice+BCE vessel, no constraints)",
    "cldice":      "+clDice (connectivity-aware vessel)",
    "constraints": "+Constraints (containment + exclusion)",
    "full":        "Full (Ours: clDice + constraints)",
}
