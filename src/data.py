

from __future__ import annotations

import io
import json
import random
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pydicom
import SimpleITK as sitk

from .config import (
    Config,
    LIVER_NAME,
    TUMOUR_NAME_FRAGMENTS,
    VESSEL_SOURCE_NAMES,
    STRUCTURES,
)

# +------------------------- #
# Low-level DICOM helpers
# +------------------------- #


def _slice_z(ds: pydicom.Dataset) -> float:
    """Sort key: z of ImagePositionPatient, else SliceLocation, else InstanceNumber."""
    if hasattr(ds, "ImagePositionPatient"):
        return float(ds.ImagePositionPatient[2])
    if hasattr(ds, "SliceLocation"):
        return float(ds.SliceLocation)
    return float(getattr(ds, "InstanceNumber", 0))


def _read_dataset(raw: bytes) -> pydicom.Dataset:
    return pydicom.dcmread(io.BytesIO(raw), force=True)


def _stack_ct(slices: List[pydicom.Dataset]) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """Sort CT slices by z, apply rescale, return (H,W,D) HU volume + (dx,dy,dz)."""
    slices = sorted(slices, key=_slice_z)
    origins = np.array([_slice_z(s) for s in slices])

    ds0 = slices[0]
    if hasattr(ds0, "PixelSpacing"):
        dy, dx = float(ds0.PixelSpacing[0]), float(ds0.PixelSpacing[1])
    else:
        dy, dx = 1.0, 1.0
    if len(slices) > 1:
        dz = float(abs(origins[1] - origins[0]))
        if dz < 1e-6:
            dz = float(getattr(ds0, "SliceThickness", 1.0))
    else:
        dz = float(getattr(ds0, "SliceThickness", 1.0))

    arrays = []
    for ds in slices:
        arr = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        arrays.append(arr * slope + intercept)
    volume = np.stack(arrays, axis=-1)  # (H, W, D)
    return volume, (dx, dy, dz)


def _stack_mask(slices: List[pydicom.Dataset]) -> np.ndarray:
    """Sort mask slices by z, binarise, return (H,W,D) uint8."""
    slices = sorted(slices, key=_slice_z)
    arrays = [(ds.pixel_array > 0).astype(np.uint8) for ds in slices]
    return np.stack(arrays, axis=-1)


# +------------------------- #
# Zip-aware DICOM volume loader (adapted & reused from the notebook)
# +------------------------- #


class DICOMVolumeLoader:
    """Load a 3D-IRCADb patient (zipped or extracted) into numpy arrays.

    Parameters
    ----------
    patient_dir : Path
        Root directory of a single patient (e.g. ``.../3Dircadb1.1``) that
        contains either ``PATIENT_DICOM[.zip]`` and ``MASKS_DICOM[.zip]``.
    verbose : bool
        Print shape/spacing diagnostics after loading.
    """

    _PATIENT_DIR_VARIANTS = ("PATIENT_DICOM", "patient_dicom")
    _MASKS_DIR_VARIANTS = ("MASKS_DICOM", "masks_dicom")

    def __init__(self, patient_dir: Path, verbose: bool = False) -> None:
        self.patient_dir = Path(patient_dir)
        self.verbose = verbose

    # -- source discovery +
    def _find_dir(self, variants: Sequence[str]) -> Optional[Path]:
        for name in variants:
            cand = self.patient_dir / name
            if cand.is_dir():
                return cand
        return None

    def _find_zip(self, variants: Sequence[str]) -> Optional[Path]:
        for name in variants:
            cand = self.patient_dir / f"{name}.zip"
            if cand.is_file():
                return cand
        return None

    # -- readers (folder + zip) -------------------------------------------
    @staticmethod
    def _datasets_from_dir(dicom_dir: Path) -> List[pydicom.Dataset]:
        files = [f for f in sorted(dicom_dir.iterdir())
                 if f.is_file() and not f.name.startswith(".")]
        out = []
        for fp in files:
            try:
                out.append(pydicom.dcmread(str(fp), force=True))
            except Exception:
                pass
        return out

    @staticmethod
    def _datasets_from_zip(zf: zipfile.ZipFile, prefix: str) -> List[pydicom.Dataset]:
        """Read DICOM members directly under *prefix* (one level deep)."""
        out = []
        for info in zf.infolist():
            name = info.filename
            if info.is_dir() or not name.startswith(prefix):
                continue
            rest = name[len(prefix):]
            if "/" in rest.strip("/"):  # deeper than this level -> skip
                continue
            if not rest or rest.startswith("."):
                continue
            try:
                out.append(_read_dataset(zf.read(name)))
            except Exception:
                pass
        return out

    @staticmethod
    def _zip_subdirs(zf: zipfile.ZipFile, prefix: str) -> List[str]:
        subs = set()
        for name in zf.namelist():
            if not name.startswith(prefix):
                continue
            rest = name[len(prefix):].strip("/")
            if not rest:
                continue
            subs.add(rest.split("/")[0])
        return sorted(subs)

    # -- public API +------
    def list_mask_names(self) -> List[str]:
        """Return all raw mask/organ names present for this patient."""
        masks_dir = self._find_dir(self._MASKS_DIR_VARIANTS)
        if masks_dir is not None:
            return sorted(d.name for d in masks_dir.iterdir() if d.is_dir())
        masks_zip = self._find_zip(self._MASKS_DIR_VARIANTS)
        if masks_zip is not None:
            with zipfile.ZipFile(masks_zip) as zf:
                # masks live under "MASKS_DICOM/<organ>/..."
                root = self._zip_subdirs(zf, "")  # find the top folder name
                top = next((r for r in root if r.lower().startswith("masks_dicom")), None)
                prefix = f"{top}/" if top else "MASKS_DICOM/"
                return self._zip_subdirs(zf, prefix)
        return []

    def load(self) -> Dict:
        """Load CT volume + every raw mask present.

        Returns dict with keys: ``volume`` (H,W,D float32 HU), ``spacing``
        (dx,dy,dz mm), ``masks`` (dict[name -> (H,W,D) uint8]),
        ``mask_names`` (list of all raw organ names present).
        """
        # ---- CT volume ----
        patient_dir = self._find_dir(self._PATIENT_DIR_VARIANTS)
        if patient_dir is not None:
            ct_slices = self._datasets_from_dir(patient_dir)
        else:
            patient_zip = self._find_zip(self._PATIENT_DIR_VARIANTS)
            if patient_zip is None:
                raise FileNotFoundError(
                    f"No PATIENT_DICOM folder or .zip under {self.patient_dir}")
            with zipfile.ZipFile(patient_zip) as zf:
                top = next((r for r in self._zip_subdirs(zf, "")
                            if r.lower().startswith("patient_dicom")), "PATIENT_DICOM")
                ct_slices = self._datasets_from_zip(zf, f"{top}/")
        if not ct_slices:
            raise FileNotFoundError(f"No CT DICOM slices found for {self.patient_dir}")
        volume, spacing = _stack_ct(ct_slices)

        # ---- masks ----
        masks: Dict[str, np.ndarray] = {}
        masks_dir = self._find_dir(self._MASKS_DIR_VARIANTS)
        if masks_dir is not None:
            for organ_dir in sorted(d for d in masks_dir.iterdir() if d.is_dir()):
                ds = self._datasets_from_dir(organ_dir)
                if ds:
                    masks[organ_dir.name] = _stack_mask(ds)
        else:
            masks_zip = self._find_zip(self._MASKS_DIR_VARIANTS)
            if masks_zip is not None:
                with zipfile.ZipFile(masks_zip) as zf:
                    top = next((r for r in self._zip_subdirs(zf, "")
                                if r.lower().startswith("masks_dicom")), "MASKS_DICOM")
                    for organ in self._zip_subdirs(zf, f"{top}/"):
                        ds = self._datasets_from_zip(zf, f"{top}/{organ}/")
                        if ds:
                            masks[organ] = _stack_mask(ds)

        if self.verbose:
            print(f"  volume {volume.shape}  spacing {spacing}  "
                  f"HU[{volume.min():.0f},{volume.max():.0f}]  masks={list(masks)}")

        return {"volume": volume, "spacing": spacing, "masks": masks,
                "mask_names": sorted(masks.keys())}


# +------------------------- #
# Target composition (liver / tumour / vessel) from raw masks
# +------------------------- #


def _is_tumour(name: str) -> bool:
    n = name.lower()
    return any(frag in n for frag in TUMOUR_NAME_FRAGMENTS)


def _is_vessel(name: str) -> bool:
    n = name.lower()
    return any(src in n for src in VESSEL_SOURCE_NAMES)


def compose_targets(masks: Dict[str, np.ndarray], shape: Tuple[int, int, int]
                    ) -> Tuple[Dict[str, np.ndarray], Dict[str, List[str]]]:
    """Build the three binary training targets from all raw masks.

    Returns (targets, sources) where ``targets`` maps each structure to a
    (H,W,D) uint8 array (all-zeros if that structure is absent) and ``sources``
    records which raw mask names contributed to each structure.
    """
    targets = {s: np.zeros(shape, dtype=np.uint8) for s in STRUCTURES}
    sources: Dict[str, List[str]] = {s: [] for s in STRUCTURES}

    for name, m in masks.items():
        nl = name.lower()
        if nl == LIVER_NAME or (nl.startswith("liver") and not _is_tumour(nl)
                                and "kyst" not in nl and not _is_vessel(nl)):
            # exact 'liver' (the organ). Guard against livertumor*/liverkyst.
            if nl == LIVER_NAME:
                targets["liver"] = np.maximum(targets["liver"], m)
                sources["liver"].append(name)
            continue
        if _is_tumour(nl):
            targets["tumour"] = np.maximum(targets["tumour"], m)
            sources["tumour"].append(name)
        elif _is_vessel(nl):
            targets["vessel"] = np.maximum(targets["vessel"], m)
            sources["vessel"].append(name)

    return targets, sources


# +------------------------- #
# Resampling / normalisation (reused from the notebook)
# +------------------------- #


def _numpy_to_sitk(volume: np.ndarray, spacing: Tuple[float, float, float]) -> sitk.Image:
    img = sitk.GetImageFromArray(volume.transpose(2, 1, 0).astype(np.float32))
    img.SetSpacing(tuple(float(s) for s in spacing))
    return img


def _sitk_to_numpy(img: sitk.Image) -> np.ndarray:
    return sitk.GetArrayFromImage(img).transpose(1, 2, 0).astype(np.float32)


def resample_volume(volume: np.ndarray,
                    original_spacing: Tuple[float, float, float],
                    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                    interpolator: int = sitk.sitkLinear,
                    default_value: Optional[float] = None) -> np.ndarray:
    """Resample (H,W,D) volume to ``target_spacing`` (mm) preserving extent."""
    img = _numpy_to_sitk(volume, original_spacing)
    orig_size = np.array(img.GetSize(), dtype=float)
    orig_spacing = np.array(img.GetSpacing(), dtype=float)
    new_size = np.maximum(
        (orig_size * orig_spacing / np.array(target_spacing)).round().astype(int), 1
    ).tolist()

    r = sitk.ResampleImageFilter()
    r.SetOutputSpacing(tuple(float(s) for s in target_spacing))
    r.SetSize([int(s) for s in new_size])
    r.SetOutputDirection(img.GetDirection())
    r.SetOutputOrigin(img.GetOrigin())
    r.SetTransform(sitk.Transform())
    r.SetDefaultPixelValue(float(volume.min()) if default_value is None else float(default_value))
    r.SetInterpolator(interpolator)
    return _sitk_to_numpy(r.Execute(img))


def clip_and_normalise(volume: np.ndarray, hu_min: float, hu_max: float) -> np.ndarray:
    """Clip HU to [hu_min, hu_max] then min-max scale to [0, 1]."""
    v = np.clip(volume, hu_min, hu_max).astype(np.float32)
    return (v - hu_min) / (hu_max - hu_min)


# +------------------------- #
# Per-patient preprocessing with caching
# +------------------------- #


def patient_dir_for(cfg: Config, pid: int) -> Path:
    return cfg.data_root / f"3Dircadb1.{pid}"


def preprocess_patient(cfg: Config, pid: int, force: bool = False) -> Dict:
    """Preprocess one patient and cache to ``outputs/preprocessed/<pid>/``.

    Steps: load DICOM -> resample volume (linear) + masks (nearest) to 1 mm ->
    clip HU to liver window and scale to [0,1] -> save .npy + meta.json.
    Idempotent: returns cached meta if present and not forced.
    """
    pid_str = f"{pid:02d}"
    save_dir = cfg.preprocessed_dir / pid_str
    meta_path = save_dir / "meta.json"
    vol_path = save_dir / "volume.npy"

    if not force and meta_path.exists() and vol_path.exists():
        with open(meta_path, encoding="utf-8") as fh:
            return json.load(fh)

    loader = DICOMVolumeLoader(patient_dir_for(cfg, pid), verbose=False)
    data = loader.load()
    volume_raw, spacing = data["volume"], data["spacing"]
    raw_masks = data["masks"]

    targets, sources = compose_targets(raw_masks, volume_raw.shape)

    # Resample CT (linear) and each target (nearest).
    volume_rs = resample_volume(volume_raw, spacing, cfg.target_spacing_mm,
                                interpolator=sitk.sitkLinear)
    masks_rs: Dict[str, np.ndarray] = {}
    for s, m in targets.items():
        m_rs = resample_volume(m.astype(np.float32), spacing, cfg.target_spacing_mm,
                               interpolator=sitk.sitkNearestNeighbor, default_value=0.0)
        masks_rs[s] = (m_rs > 0.5).astype(np.uint8)

    volume_norm = clip_and_normalise(volume_rs, cfg.hu_min, cfg.hu_max)

    save_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(vol_path), volume_norm)
    for s, m in masks_rs.items():
        np.save(str(save_dir / f"mask_{s}.npy"), m)

    present = {s: bool(len(sources[s]) > 0) for s in STRUCTURES}
    nonempty = {s: bool(masks_rs[s].sum() > 0) for s in STRUCTURES}
    meta = {
        "patient_id": pid_str,
        "patient_index": pid,
        "raw_shape": list(volume_raw.shape),
        "resampled_shape": list(volume_norm.shape),
        "raw_spacing_mm": list(spacing),
        "target_spacing_mm": list(cfg.target_spacing_mm),
        "raw_mask_names": data["mask_names"],
        "sources": sources,
        "present": present,
        "nonempty": nonempty,
        "voxels": {s: int(masks_rs[s].sum()) for s in STRUCTURES},
    }
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return meta


def preprocess_all(cfg: Config, pids: Optional[Sequence[int]] = None,
                   force: bool = False) -> List[Dict]:
    pids = list(pids) if pids is not None else list(range(1, 21))
    metas = []
    for pid in pids:
        pdir = patient_dir_for(cfg, pid)
        if not pdir.exists():
            print(f"  [SKIP] patient {pid:02d} not found at {pdir}")
            continue
        meta = preprocess_patient(cfg, pid, force=force)
        metas.append(meta)
        print(f"  [OK] patient {pid:02d}  shape={tuple(meta['resampled_shape'])}  "
              f"present={[s for s in STRUCTURES if meta['present'][s]]}  "
              f"voxels={meta['voxels']}")
    return metas


def load_preprocessed(cfg: Config, pid_str: str) -> Dict:
    """Load a cached preprocessed patient (volume + 3 masks + meta)."""
    d = cfg.preprocessed_dir / pid_str
    volume = np.load(str(d / "volume.npy"), mmap_mode="r")
    masks = {}
    for s in STRUCTURES:
        mp = d / f"mask_{s}.npy"
        masks[s] = np.load(str(mp), mmap_mode="r") if mp.exists() else \
            np.zeros(volume.shape, dtype=np.uint8)
    with open(d / "meta.json", encoding="utf-8") as fh:
        meta = json.load(fh)
    return {"volume": volume, "masks": masks, "meta": meta}


# +------------------------- #
# Patch dataset
# +------------------------- #


def _liver_bbox(liver: np.ndarray) -> Optional[Tuple[slice, slice, slice]]:
    if liver.sum() == 0:
        return None
    idx = np.where(liver > 0)
    return tuple(slice(int(i.min()), int(i.max()) + 1) for i in idx)


def _pad_to(arr: np.ndarray, target: Tuple[int, int, int], value: float = 0) -> np.ndarray:
    pad = [(0, max(t - s, 0)) for s, t in zip(arr.shape, target)]
    if any(p[1] > 0 for p in pad):
        arr = np.pad(arr, pad, mode="constant", constant_values=value)
    return arr


class PatchDataset:
    """Random-patch dataset over a set of preprocessed patients.

    Each item is a dict with ``image`` (1,pH,pW,pD) float32 tensor-ready array,
    ``targets`` (dict structure -> (1,pH,pW,pD) float32), and ``present``
    (dict structure -> 0/1 float) marking which structures are annotated &
    non-empty for that patient (used to skip empty masks in the loss).
    """

    def __init__(self, cfg: Config, pid_strs: Sequence[str], train: bool = True):
        self.cfg = cfg
        self.pids = list(pid_strs)
        self.train = train
        self.patch = cfg.patch_size
        # Lazy-load (mmap) volumes/masks once.
        self._cache: Dict[str, Dict] = {}
        for p in self.pids:
            self._cache[p] = load_preprocessed(cfg, p)

    def __len__(self) -> int:
        return len(self.pids) * self.cfg.patches_per_volume

    # -- patch extraction +
    def _sample_center(self, liver: np.ndarray, shape: Tuple[int, int, int]):
        """Pick a patch-center voxel, oversampling the liver bbox."""
        pH, pW, pD = self.patch
        use_fg = self.train and (random.random() < self.cfg.foreground_ratio)
        bbox = _liver_bbox(np.asarray(liver)) if use_fg else None
        if bbox is not None:
            cy = random.randint(bbox[0].start, bbox[0].stop - 1)
            cx = random.randint(bbox[1].start, bbox[1].stop - 1)
            cz = random.randint(bbox[2].start, bbox[2].stop - 1)
        else:
            cy = random.randint(0, max(shape[0] - 1, 0))
            cx = random.randint(0, max(shape[1] - 1, 0))
            cz = random.randint(0, max(shape[2] - 1, 0))
        # convert center -> start, clamp inside volume
        sy = int(np.clip(cy - pH // 2, 0, max(shape[0] - pH, 0)))
        sx = int(np.clip(cx - pW // 2, 0, max(shape[1] - pW, 0)))
        sz = int(np.clip(cz - pD // 2, 0, max(shape[2] - pD, 0)))
        return sy, sx, sz

    def _crop(self, arr: np.ndarray, start, value=0) -> np.ndarray:
        pH, pW, pD = self.patch
        sy, sx, sz = start
        patch = np.asarray(arr[sy:sy + pH, sx:sx + pW, sz:sz + pD])
        return _pad_to(patch, self.patch, value=value)

    # -- augmentation +----
    def _augment(self, image: np.ndarray, masks: Dict[str, np.ndarray]):
        cfg = self.cfg
        # flips per axis
        for axis in range(3):
            if random.random() < cfg.aug_flip_prob:
                image = np.flip(image, axis)
                masks = {k: np.flip(v, axis) for k, v in masks.items()}
        # 90-deg in-plane rotation (axes 0,1)
        if random.random() < cfg.aug_rot90_prob:
            k = random.choice([1, 2, 3])
            image = np.rot90(image, k, axes=(0, 1))
            masks = {kk: np.rot90(v, k, axes=(0, 1)) for kk, v in masks.items()}
        image = np.ascontiguousarray(image)
        masks = {k: np.ascontiguousarray(v) for k, v in masks.items()}
        # intensity jitter (image only)
        if cfg.aug_noise_sigma_max > 0:
            sigma = random.uniform(0, cfg.aug_noise_sigma_max)
            image = image + np.random.normal(0, sigma, image.shape).astype(np.float32)
        if cfg.aug_brightness > 0:
            image = image + random.uniform(-cfg.aug_brightness, cfg.aug_brightness)
        if cfg.aug_contrast > 0:
            factor = 1.0 + random.uniform(-cfg.aug_contrast, cfg.aug_contrast)
            mean = float(image.mean())
            image = (image - mean) * factor + mean
        image = np.clip(image, 0.0, 1.0).astype(np.float32)
        return image, masks

    def __getitem__(self, idx: int) -> Dict:
        pid = self.pids[idx % len(self.pids)]
        rec = self._cache[pid]
        vol = rec["volume"]
        masks_full = rec["masks"]
        shape = vol.shape

        start = self._sample_center(masks_full["liver"], shape)
        image = self._crop(vol, start, value=0.0).astype(np.float32)
        masks = {s: self._crop(masks_full[s], start, value=0).astype(np.float32)
                 for s in STRUCTURES}

        if self.train:
            image, masks = self._augment(image, masks)

        present = {s: float(rec["meta"]["nonempty"][s]) for s in STRUCTURES}

        return {
            "image": image[None].copy(),                      # (1,pH,pW,pD)
            "targets": {s: masks[s][None].copy() for s in STRUCTURES},
            "present": present,
            "pid": pid,
        }


def collate_patches(batch: List[Dict]) -> Dict:
    """Collate a list of PatchDataset items into batched torch tensors."""
    import torch
    images = torch.from_numpy(np.stack([b["image"] for b in batch])).float()
    targets = {s: torch.from_numpy(np.stack([b["targets"][s] for b in batch])).float()
               for s in STRUCTURES}
    present = {s: torch.tensor([b["present"][s] for b in batch]).float()
               for s in STRUCTURES}
    return {"image": images, "targets": targets, "present": present,
            "pids": [b["pid"] for b in batch]}
