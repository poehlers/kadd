"""DICOM loading and study-level dataset for the v1 knee classifier.

Adapted from the lazy-loading + train-split-only-normalization-stats pattern in
poehlers/DenPIE (den_pie/density/data.py FLIDatasetLazy/FLIDataLoader), which
loads one .npz per __getitem__ and z-score normalizes with statistics computed
once over the training split. Two things are genuinely different here, not
ported as-is:
  - Per-volume DICOM windowing (WindowCenter/WindowWidth -> [0,1]) replaces
    voxel-wise z-score, since that's the clinically meaningful normalization for
    MRI display, not a statistical artifact of cosmological density fields.
  - A study is a *bag* of series (attention-MIL), not a single fixed-shape
    array, so __getitem__ returns a variable-length list of per-series volumes
    instead of one tensor — collate_fn flattens across the batch.

HIGHEST-PRIORITY CORRECTNESS ITEM (see plan): DICOM slices are sorted by
ImagePositionPatient projected onto the slice normal, never by InstanceNumber
alone or by filename/SOPInstanceUID. Getting this wrong doesn't crash anything —
it silently scrambles or flips the volume and the model trains on garbage while
loss curves still look normal. The DICOM allowlist strips to 86 tags and it is
NOT YET CONFIRMED that ImagePositionPatient/ImageOrientationPatient/
WindowCenter/WindowWidth survive it in the bulk training data (only verified
against one sample DICOM, which belongs to the test set) — this must be checked
against real training DICOMs the moment the bulk download lands (see CLAUDE.md /
the plan's "First-run verification" section) before trusting any training run.
"""
import hashlib
import json
import logging
import os
import random
import tempfile
import time

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from kadd.classify.model import PLANES

_PLANE_TO_IDX = {p: i for i, p in enumerate(PLANES)}
_UNKNOWN_PLANE_IDX = len(PLANES)

# Throttled decode-vs-preprocess timing breakdown for load_series_volume,
# added to answer "is it the DICOM format (codec decompression) or the
# preprocessing (windowing/crop/resize) that's slow" with a real measurement
# instead of a guess — this project already got two inferences (GPU-bound,
# then CPU-bound) correctly challenged as unverified this session, so this
# one is measured from the start. Logs only the first few calls PER
# DATALOADER WORKER PROCESS (module-level counter — each worker process gets
# its own copy after fork, so this naturally caps total log volume to
# roughly num_workers x _TIMING_SAMPLE_LIMIT lines instead of one line per
# volume across an entire epoch).
_TIMING_SAMPLE_LIMIT = 5
_timing_samples_logged = 0


def _plane_to_idx(plane: str) -> int:
    return _PLANE_TO_IDX.get(plane, _UNKNOWN_PLANE_IDX)


# Number of shards + hash function must match
# scripts/kaggle_push_precomputed_dataset.py's _shard_for_study exactly (same
# NUM_SHARDS=32, same md5-based hash) — that script is what actually wrote the
# shard_XX/<study>/<series>.npy layout on disk/on the Kaggle-hosted dataset;
# this is a read-side copy, not an independent design, and drifting the two
# apart would make every precomputed file unfindable. Covered by a
# cross-check test against the script's own copy.
PRECOMPUTED_NUM_SHARDS = 32


def _shard_for_study(study_uid: str) -> str:
    digest = hashlib.md5(study_uid.encode()).hexdigest()
    return f"shard_{int(digest, 16) % PRECOMPUTED_NUM_SHARDS:02d}"


def load_precomputed_manifest(precomputed_dir: str) -> set:
    """Returns the set of (study, series) pairs with status "ok" in
    <precomputed_dir>/manifest.jsonl. Missing/malformed lines are skipped
    with a warning (a real precompute run can leave a truncated final line —
    see kaggle_push_precomputed_dataset.py's _read_manifest_lines, same
    tolerance applied here) rather than crashing training over it. Returns
    an empty set (not an error) if precomputed_dir doesn't exist or has no
    manifest yet, so this feature degrades to "no precomputed volumes found,
    fall back to raw DICOM for everything" instead of a hard failure.
    """
    manifest_path = os.path.join(precomputed_dir, "manifest.jsonl")
    ok = set()
    if not os.path.isfile(manifest_path):
        logging.warning(f"No precomputed manifest found at {manifest_path}; "
                         f"all series will be loaded from raw DICOM")
        return ok
    with open(manifest_path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                logging.warning(f"Skipping malformed manifest line {i + 1} in {manifest_path}")
                continue
            if rec.get("status") == "ok":
                ok.add((rec["study"], rec["series"]))
    logging.info(f"Loaded precomputed manifest: {len(ok)} series available at {precomputed_dir}")
    return ok


def _write_cache_file(cache_path: str, arr: np.ndarray) -> None:
    """Atomically writes arr to cache_path (temp file + os.replace, same
    idiom as this project's atomic checkpoint saves) so a concurrent
    DataLoader worker reading cache_path never sees a partially-written
    file. Raises OSError on failure (e.g. ENOSPC) — caller decides whether
    to disable further cache writes; this function itself has no retry/
    disable logic, it's a pure single-attempt write.
    """
    cache_dir = os.path.dirname(cache_path)
    os.makedirs(cache_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=cache_dir, suffix=".npy")
    try:
        os.close(fd)
        np.save(tmp_path, arr)  # suffix is already .npy, so np.save won't append another
        os.replace(tmp_path, cache_path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


# Errors that mean "this file is corrupted or uses an encoding we can't
# decode" — real-world DICOM data quality issues to skip and log, not bugs.
# AttributeError is deliberately excluded: a missing ImagePositionPatient/
# ImageOrientationPatient tag has no safe fallback (see module docstring)
# and must crash loudly rather than be silently skipped.
_RECOVERABLE_DICOM_ERRORS = (
    ValueError, RuntimeError, OSError, FileNotFoundError, NotImplementedError,
    pydicom.errors.InvalidDicomError,
)


def _slice_position(ds: pydicom.Dataset) -> float:
    """Project ImagePositionPatient onto the slice normal (cross product of
    the row/column direction cosines from ImageOrientationPatient). This is
    the physically correct way to order slices — InstanceNumber can be
    missing or wrong, and file/UID order carries no ordering guarantee.
    """
    iop = np.array(ds.ImageOrientationPatient, dtype=float)
    row_cosine, col_cosine = iop[:3], iop[3:]
    normal = np.cross(row_cosine, col_cosine)
    ipp = np.array(ds.ImagePositionPatient, dtype=float)
    return float(np.dot(normal, ipp))


def _determine_side(slices) -> str:
    """Study side ('L'/'R') for laterality normalization, or None if
    undetermined. Prefers the DICOM Laterality tag (Type 2C — legitimately
    absent on roughly half the studies in this corpus, observed to correlate
    with whole vendors rather than being scattered randomly, per a public
    reference solution for this competition). Falls back to the sign of the
    median in-plane image-center x-coordinate in patient space (+x = the
    patient's left, DICOM patient-coordinate convention) when the tag is
    missing or slices disagree on it.
    """
    lat_vals = {str(getattr(ds, 'Laterality', '') or '').strip().upper() for ds in slices}
    lat_vals.discard('')
    if len(lat_vals) == 1:
        v = lat_vals.pop()
        if v in ('L', 'R'):
            return v

    centers_x = []
    for ds in slices:
        try:
            iop = np.array(ds.ImageOrientationPatient, dtype=float)
            row_cosine, col_cosine = iop[:3], iop[3:]
            ipp = np.array(ds.ImagePositionPatient, dtype=float)
            spacing = getattr(ds, 'PixelSpacing', None)
            rows, cols = getattr(ds, 'Rows', None), getattr(ds, 'Columns', None)
            if spacing is None or rows is None or cols is None:
                continue
            row_sp, col_sp = float(spacing[0]), float(spacing[1])
            # Image center = corner + half-extent along each in-plane direction.
            center = ipp + row_cosine * col_sp * (cols / 2) + col_cosine * row_sp * (rows / 2)
            centers_x.append(center[0])
        except (AttributeError, TypeError, ValueError):
            continue
    if not centers_x:
        return None
    median_x = float(np.median(centers_x))
    # A knee scanned near the body midline gives a sign no more reliable than
    # chance — left unresolved (None) rather than guessed, matching the same
    # reasoning applied to the reference solution this was adapted from.
    if abs(median_x) < 5.0:
        return None
    return 'R' if median_x < 0 else 'L'


def _normalize_laterality(volume: torch.Tensor, plane: str, side: str) -> torch.Tensor:
    """Maps every knee onto a canonical left-knee convention, so the 5
    medial/lateral-defined targets (both menisci, both tibiofemoral
    compartments, MCL) don't have their side flipped inconsistently between
    left- and right-knee studies. volume: (1, 1, D, H, W).

    Coronal/axial: medial-lateral lies in-plane, so a horizontal (W-axis)
    flip maps one knee onto the other. Sagittal: medial-lateral is the SLICE
    axis instead (each slice is unchanged by mirroring — what differs is
    which end of the stack is medial), so the slice order is reversed
    instead of the image being flipped.
    """
    if side != 'R':
        return volume
    if plane in ('Coronal', 'Axial'):
        return torch.flip(volume, dims=[-1])
    if plane == 'Sagittal':
        return torch.flip(volume, dims=[2])  # (1, 1, D, H, W) -> D is dim 2
    return volume  # unknown/missing plane: nothing safe to normalize, leave as-is


def _physical_crop_hw(volume: torch.Tensor, pixel_spacing, crop_mm: float) -> torch.Tensor:
    """Center-crops the H/W spatial dims to a fixed real-world extent
    (crop_mm millimeters), converting mm -> pixels via PixelSpacing, BEFORE
    the later resize to a fixed pixel grid. Without this, the same
    millimeter-scale lesion (a meniscal tear is 1-3mm) occupies a different
    pixel footprint on every series depending on the scanner's raw
    resolution/field of view, and a blind resize can shrink it below the
    resample grid's Nyquist limit — destroying the signal before the model
    ever sees it, in a way no amount of downstream capacity can recover.
    volume: (1, 1, D, H, W). Returns volume unchanged on any axis where the
    crop would be no smaller than the image itself (already within the
    target physical extent on that axis) — never pads or upsamples.
    """
    if pixel_spacing is None:
        return volume
    row_mm, col_mm = float(pixel_spacing[0]), float(pixel_spacing[1])
    h, w = volume.shape[-2], volume.shape[-1]
    crop_h = min(h, max(1, round(crop_mm / row_mm))) if row_mm > 0 else h
    crop_w = min(w, max(1, round(crop_mm / col_mm))) if col_mm > 0 else w
    h0 = (h - crop_h) // 2
    w0 = (w - crop_w) // 2
    return volume[..., h0:h0 + crop_h, w0:w0 + crop_w]


def load_series_volume(series_dir: str, target_depth: int, target_size: int,
                        plane: str = None, crop_mm: float = 130.0) -> torch.Tensor:
    """Load one DICOM series into a normalized (1, target_depth, target_size,
    target_size) tensor.

    Raises FileNotFoundError if series_dir has no .dcm files. Raises
    AttributeError (loudly, on purpose — see module docstring) if a slice is
    missing ImagePositionPatient/ImageOrientationPatient, since there is no
    safe fallback for slice ordering.

    plane (Anatomical_Plane from train_series.csv) drives laterality
    normalization (_normalize_laterality) — pass None to skip it (e.g. a
    caller that doesn't have series metadata handy). crop_mm drives the
    physical-scale crop (_physical_crop_hw) applied before the final resize.
    """
    global _timing_samples_logged
    do_time = _timing_samples_logged < _TIMING_SAMPLE_LIMIT

    paths = [os.path.join(series_dir, f) for f in os.listdir(series_dir)
             if f.lower().endswith('.dcm')]
    if not paths:
        raise FileNotFoundError(f"No .dcm files found in {series_dir}")

    t0 = time.time() if do_time else None
    slices = [pydicom.dcmread(p) for p in paths]
    slices.sort(key=_slice_position)
    t1 = time.time() if do_time else None

    # ds.pixel_array is where pydicom actually decompresses pixel data for
    # compressed transfer syntaxes (JPEG Lossless, JPEG 2000, etc.) —
    # dcmread() above only parses headers and keeps the raw (possibly still
    # compressed) bytes. Timed separately from the arithmetic below so
    # "is it the DICOM format/codec, or the windowing/normalization" has an
    # actual answer instead of a guess.
    t_pixel_decode = 0.0
    frames = []
    for ds in slices:
        if do_time:
            t_px0 = time.time()
        arr = ds.pixel_array.astype(np.float32)
        if do_time:
            t_pixel_decode += time.time() - t_px0

        slope = float(getattr(ds, 'RescaleSlope', 1))
        intercept = float(getattr(ds, 'RescaleIntercept', 0))
        arr = arr * slope + intercept

        if getattr(ds, 'PhotometricInterpretation', 'MONOCHROME2') == 'MONOCHROME1':
            arr = arr.max() - arr

        wc = getattr(ds, 'WindowCenter', None)
        ww = getattr(ds, 'WindowWidth', None)
        if wc is not None and ww is not None:
            wc = float(wc[0] if hasattr(wc, '__iter__') else wc)
            ww = float(ww[0] if hasattr(ww, '__iter__') else ww)
            lo, hi = wc - ww / 2, wc + ww / 2
        else:
            lo, hi = np.percentile(arr, [1, 99])
        hi = max(hi, lo + 1e-6)
        arr = np.clip(arr, lo, hi)
        arr = (arr - lo) / (hi - lo)

        frames.append(torch.from_numpy(arr))
    t2 = time.time() if do_time else None

    volume = torch.stack(frames, dim=0).unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)

    # Laterality normalization, before any cropping (flip/reversal are
    # symmetric operations that commute with the crops below either way, but
    # doing it first keeps every downstream step working in one consistent
    # canonical-left-knee frame). side=None (undetermined) is a no-op.
    if plane is not None:
        side = _determine_side(slices)
        volume = _normalize_laterality(volume, plane, side)

    # Crop to the bounding box of above-threshold signal before resizing, so a
    # blind downsize from a large raw matrix (e.g. 960x960) doesn't blur away
    # fine detail (meniscal tears / cartilage lesions are a few mm).
    mask = volume[0, 0] > 0.02
    if mask.any():
        d_idx = mask.any(dim=(1, 2)).nonzero().squeeze(-1)
        h_idx = mask.any(dim=(0, 2)).nonzero().squeeze(-1)
        w_idx = mask.any(dim=(0, 1)).nonzero().squeeze(-1)
        volume = volume[:, :, d_idx.min():d_idx.max() + 1,
                         h_idx.min():h_idx.max() + 1, w_idx.min():w_idx.max() + 1]

    # Physical-scale crop, on top of the content-bbox crop above: the bbox
    # crop trims gross empty background but doesn't guarantee a consistent
    # mm/pixel scale across scanners with different fields of view — this
    # crop does, using the series' own PixelSpacing (constant across a
    # series, read once from the first slice).
    pixel_spacing = getattr(slices[0], 'PixelSpacing', None)
    volume = _physical_crop_hw(volume, pixel_spacing, crop_mm)

    volume = F.interpolate(
        volume, size=(target_depth, target_size, target_size),
        mode='trilinear', align_corners=False,
    )
    if do_time:
        t3 = time.time()
        _timing_samples_logged += 1
        read_s = t1 - t0
        pixel_decode_s = t_pixel_decode
        preprocess_s = (t2 - t1) - t_pixel_decode  # rescale/window/monochrome arithmetic only
        crop_resize_s = t3 - t2
        total_s = t3 - t0
        logging.info(
            f"  [load_series_volume timing] n_slices={len(slices)} total={total_s:.3f}s  "
            f"file_read={read_s:.3f}s ({100 * read_s / total_s:.0f}%)  "
            f"pixel_decode={pixel_decode_s:.3f}s ({100 * pixel_decode_s / total_s:.0f}%)  "
            f"preprocess={preprocess_s:.3f}s ({100 * preprocess_s / total_s:.0f}%)  "
            f"crop_resize={crop_resize_s:.3f}s ({100 * crop_resize_s / total_s:.0f}%)"
        )
    return volume.squeeze(0)  # (1, target_depth, target_size, target_size)


class KneeDataset(Dataset):
    """Lazy, one entry per study. Returns every series for that study (capped
    at max_series, randomly subsampled above the cap — max observed locally
    was 14 series/study) rather than picking a single "representative" plane,
    since attention-MIL pooling handles the variable-length bag.
    """

    def __init__(self, study_ids, study_to_series: dict, labels: dict,
                 data_path: str, target_depth: int, target_size: int,
                 max_series: int = 8, precomputed_dir: str = None,
                 precomputed_ok: set = None, local_cache_dir: str = None,
                 crop_mm: float = 130.0):
        self.study_ids = list(study_ids)
        self.study_to_series = study_to_series
        self.labels = labels
        self.data_path = data_path
        self.target_depth = target_depth
        self.target_size = target_size
        self.max_series = max_series
        # Physical-scale crop target (mm) for the raw-DICOM path only — see
        # load_series_volume's _physical_crop_hw. Precomputed volumes (below)
        # were already cropped/normalized at precompute time with whatever
        # crop_mm that run used; this value has no effect on a cache hit.
        self.crop_mm = crop_mm
        # Precomputed .npy volumes (kaggle_kernel_precompute/precompute_kernel.py
        # output) let __getitem__ skip DICOM decode entirely for series that
        # have one — see CLAUDE.md sec 7's precompute pipeline. Only usable at
        # exactly the depth/size precompute_kernel.py was run at (32/224);
        # falls back to raw DICOM (existing behavior) for anything else,
        # including a shape mismatch or a series precompute never covered.
        self.precomputed_dir = precomputed_dir
        self.precomputed_ok = precomputed_ok or set()
        self._precomputed_shape_warned = False
        # Local write-through cache: precomputed_dir is a REMOTE Kaggle
        # dataset mount, re-read from scratch every single epoch (confirmed —
        # there is no caching anywhere else in this pipeline, see CLAUDE.md
        # sec 7). local_cache_dir, when set, points at genuinely local
        # scratch (e.g. Kaggle's /kaggle/tmp) — the first time a series is
        # read, it's copied there; every later read (including in later
        # epochs) checks the local copy first. A DataLoader worker is a
        # separate PROCESS (not a thread), so this cache is naturally shared
        # across all workers via the filesystem — no per-worker duplication
        # the way an in-memory cache would have. No eviction: since
        # sub_epoch_size is typically unset (every study touched every epoch,
        # uniformly, not a skewed access pattern), whatever gets cached
        # first stays useful for the rest of the run — an LRU would add
        # complexity for no benefit here. Sized to whatever local_cache_dir's
        # disk can hold: _cache_disabled flips true (per-process) the first
        # time a write hits OSError (e.g. ENOSPC, "disk full") — the cache
        # then serves whatever's already cached but stops trying to grow.
        self.local_cache_dir = local_cache_dir
        self._cache_disabled = False

    def __len__(self):
        return len(self.study_ids)

    def __getitem__(self, i):
        study_uid = self.study_ids[i]
        series_list = self.study_to_series[study_uid]
        if len(series_list) > self.max_series:
            series_list = random.sample(series_list, self.max_series)

        volumes, planes, fluid, fatsup = [], [], [], []
        for s in series_list:
            series_uid = s['SeriesInstanceUID']
            vol = None

            if self.precomputed_dir and (study_uid, series_uid) in self.precomputed_ok:
                cache_path = None
                if self.local_cache_dir:
                    cache_path = os.path.join(self.local_cache_dir, study_uid, f"{series_uid}.npy")
                try:
                    if cache_path and os.path.isfile(cache_path):
                        arr = np.load(cache_path)
                    else:
                        npy_path = os.path.join(self.precomputed_dir, _shard_for_study(study_uid),
                                                 study_uid, f"{series_uid}.npy")
                        arr = np.load(npy_path)
                        if cache_path and not self._cache_disabled:
                            try:
                                _write_cache_file(cache_path, arr)
                            except OSError as e:
                                self._cache_disabled = True
                                logging.warning(
                                    f"Local cache write failed ({e}) — disabling further cache "
                                    f"writes in this worker process (likely disk full; already-"
                                    f"cached series keep being served from cache as normal)")
                    expected_shape = (1, self.target_depth, self.target_size, self.target_size)
                    if arr.shape == expected_shape:
                        vol = torch.from_numpy(arr.astype(np.float32))
                    elif not self._precomputed_shape_warned:
                        logging.warning(
                            f"Precomputed volume shape {arr.shape} != expected {expected_shape} "
                            f"(target_depth/target_size config mismatch with the precompute run) — "
                            f"falling back to raw DICOM for this and any further mismatches")
                        self._precomputed_shape_warned = True
                except (OSError, ValueError) as e:
                    logging.warning(f"Failed to load precomputed volume for {study_uid}/{series_uid}: "
                                     f"{e}; falling back to raw DICOM")

            if vol is None:
                series_dir = os.path.join(self.data_path, study_uid, series_uid)
                try:
                    vol = load_series_volume(series_dir, self.target_depth, self.target_size,
                                              plane=s.get('Anatomical_Plane'), crop_mm=self.crop_mm)
                except _RECOVERABLE_DICOM_ERRORS as e:
                    # Real-world DICOM corruption/unsupported-encoding, not a bug:
                    # e.g. pydicom's "number of bytes of pixel data is less than
                    # expected" on a truncated file, seen scaling from the 58 gold
                    # studies to the full 4,349-study silver set. AttributeError
                    # (missing slice-ordering tags) is deliberately NOT caught
                    # here — see module docstring, that must stay loud.
                    logging.warning(f"Skipping unreadable series "
                                     f"{series_uid} in study {study_uid}: {e}")
                    continue

            volumes.append(vol)
            planes.append(_plane_to_idx(s.get('Anatomical_Plane')))
            fluid.append(float(s.get('Fluid_Sensitive', 0)))
            fatsup.append(float(s.get('Fat_Suppression', 0)))

        if not volumes:
            logging.warning(f"Study {study_uid}: all series failed to load, skipping study")
            return None

        label = torch.tensor(self.labels[study_uid], dtype=torch.float32)
        return {
            'volumes': torch.stack(volumes, dim=0),           # (n_series, 1, D, H, W)
            'plane_idx': torch.tensor(planes, dtype=torch.long),
            'fluid_sensitive': torch.tensor(fluid, dtype=torch.float32),
            'fat_suppression': torch.tensor(fatsup, dtype=torch.float32),
            'label': label,
            'study_uid': study_uid,
        }


def collate_fn(batch):
    """Flattens every series across the batch into one tensor, plus a
    study_index mapping each series back to its study's position in the
    batch. The model runs its shared encoder once on the flattened tensor,
    then pools per-study via study_index.

    Returns None if every study in this batch failed to load (all series
    unreadable) — callers (train_epoch/val_epoch) must skip a None batch.
    """
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    study_index = []
    for b_idx, item in enumerate(batch):
        study_index.extend([b_idx] * item['volumes'].shape[0])

    return {
        'volumes': torch.cat([b['volumes'] for b in batch], dim=0),
        'plane_idx': torch.cat([b['plane_idx'] for b in batch], dim=0),
        'fluid_sensitive': torch.cat([b['fluid_sensitive'] for b in batch], dim=0),
        'fat_suppression': torch.cat([b['fat_suppression'] for b in batch], dim=0),
        'study_index': torch.tensor(study_index, dtype=torch.long),
        'n_studies': len(batch),
        'labels': torch.stack([b['label'] for b in batch], dim=0),
        'study_uids': [b['study_uid'] for b in batch],
    }


class KneeDataLoader:
    """Parses train.csv + train_series.csv, builds the study->series index,
    and produces a train/val split.

    Two modes, chosen by `gold_only`:
      - gold_only=True (v1 default): train only on the 58 directly-labeled
        studies, stratified train/val split within them (see plan for why).
      - gold_only=False + use_silver_labels=True: train on the LLM-derived
        silver labels for the ~4,349 report-only studies
        (data/report_labels_llm.csv, see CLAUDE.md §8), and hold out ALL 58
        gold studies as a clean, never-trained-on validation set. This is
        the design called out in CLAUDE.md §3/§8: the gold labels are the
        only directly-supervised signal, so they're kept for evaluation
        rather than folded into noisy weak-label training.
    """

    def __init__(self, params: dict):
        self.cfg = params['classify']['data']
        self.data = self.load()

    def load(self) -> dict:
        cfg = self.cfg
        label_columns = cfg['label_columns']
        gold_only = cfg.get('gold_only', True)
        use_silver_labels = cfg.get('use_silver_labels', False)
        val_split = cfg.get('val_split', 0.2)
        max_series = cfg.get('max_series', 8)
        target_depth = cfg.get('target_depth', 32)
        target_size = cfg.get('target_size', 224)
        crop_mm = cfg.get('crop_mm', 130.0)
        precomputed_dir = cfg.get('precomputed_dir')
        precomputed_ok = load_precomputed_manifest(precomputed_dir) if precomputed_dir else set()
        local_cache_dir = cfg.get('local_cache_dir')
        if local_cache_dir:
            logging.info(f"local_cache_dir={local_cache_dir}: precomputed volumes will be "
                         f"write-through cached here on first read (see KneeDataset docstring)")

        labels_df = pd.read_csv(cfg['labels_csv'])
        series_df = pd.read_csv(cfg['series_csv'])

        gold_mask = labels_df[label_columns].notna().all(axis=1)
        gold_df = labels_df[gold_mask].reset_index(drop=True)
        logging.info(f"{len(gold_df)} directly-labeled (gold) studies found")

        labels = {
            row['StudyInstanceUID']: row[label_columns].astype(float).tolist()
            for _, row in gold_df.iterrows()
        }

        if gold_only or not use_silver_labels:
            study_ids = gold_df['StudyInstanceUID'].tolist()
            study_to_series = self._index_series(study_ids, series_df)
            study_ids = [s for s in study_ids if s in study_to_series]
            train_ids, val_ids = self._stratified_split(study_ids, labels, val_split)
        else:
            silver_csv = cfg.get('silver_labels_csv', 'data/report_labels_llm.csv')
            silver_df = pd.read_csv(silver_csv)
            gold_ids = set(gold_df['StudyInstanceUID'])
            silver_df = silver_df[~silver_df['StudyInstanceUID'].isin(gold_ids)]
            logging.info(f"use_silver_labels=True: {len(silver_df)} report-derived "
                         f"(silver) studies loaded from {silver_csv}")
            for _, row in silver_df.iterrows():
                labels[row['StudyInstanceUID']] = row[label_columns].astype(float).tolist()

            all_ids = list(silver_df['StudyInstanceUID']) + list(gold_df['StudyInstanceUID'])
            study_to_series = self._index_series(all_ids, series_df)

            train_ids = [s for s in silver_df['StudyInstanceUID'] if s in study_to_series]
            val_ids = [s for s in gold_df['StudyInstanceUID'] if s in study_to_series]
            logging.info("Gold studies held out entirely as a clean validation set "
                         "(never trained on) — see CLAUDE.md sec 3/8")

        logging.info(f"Data split: train={len(train_ids)}, val={len(val_ids)}")

        common = dict(data_path=cfg['data_path'], target_depth=target_depth,
                      target_size=target_size, max_series=max_series,
                      precomputed_dir=precomputed_dir, precomputed_ok=precomputed_ok,
                      local_cache_dir=local_cache_dir, crop_mm=crop_mm)
        train_ds = KneeDataset(train_ids, study_to_series, labels, **common)
        val_ds = KneeDataset(val_ids, study_to_series, labels, **common)

        return {
            'train_ds': train_ds,
            'val_ds': val_ds,
            'label_columns': label_columns,
        }

    @staticmethod
    def _index_series(study_ids, series_df) -> dict:
        study_to_series = {}
        fallback_count = 0
        for study_uid in study_ids:
            rows = series_df[series_df['StudyInstanceUID'] == study_uid]
            recs = rows.to_dict('records')
            if not recs:
                fallback_count += 1
                continue
            study_to_series[study_uid] = recs
        if fallback_count:
            logging.warning(f"{fallback_count} labeled studies have no entry in "
                             f"train_series.csv and will be skipped")
        return study_to_series

    @staticmethod
    def _stratified_split(study_ids, labels, val_split):
        # At N~58 a plain random split has high variance, so bucket by number
        # of positive labels per study and split within each bucket. This is
        # a coarse proxy stratification (not per-label), documented as such —
        # true multi-label stratification is unnecessary complexity at this
        # data size.
        n_pos = {s: sum(labels[s]) for s in study_ids}
        buckets = {}
        for s in study_ids:
            buckets.setdefault(n_pos[s], []).append(s)
        rng = random.Random(0)
        train_ids, val_ids = [], []
        for bucket_ids in buckets.values():
            bucket_ids = sorted(bucket_ids)
            rng.shuffle(bucket_ids)
            n_val = max(1, round(len(bucket_ids) * val_split)) if len(bucket_ids) > 1 else 0
            val_ids.extend(bucket_ids[:n_val])
            train_ids.extend(bucket_ids[n_val:])
        return train_ids, val_ids
