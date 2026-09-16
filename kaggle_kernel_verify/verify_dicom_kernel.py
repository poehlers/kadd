"""Standalone diagnostic kernel: checks whether the DICOM tags the whole
classification pipeline depends on for slice ordering actually survive the
competition's 86-tag metadata allowlist on REAL TRAINING data.

Never checked before despite two successful training runs — CLAUDE.md's
"First-run verification" checklist flagged this as unconfirmed since the
original plan was written; the one local sample DICOM this project had was
from the public *test* set, not training data. Two failure modes this
catches that a training run's loss curve would NOT catch (loss looks normal
either way — this is why it's a separate check, not "just watch training"):
  1. A required tag (ImagePositionPatient/ImageOrientationPatient/
     WindowCenter/WindowWidth) is missing on real training DICOMs, even
     though it was present on the one test-set sample checked earlier.
  2. The tags are present but kadd/classify/data.py's geometry-based sort
     (_slice_position) produces an order that's wrong or inconsistent with
     the series' own declared slice spacing — this would silently scramble
     training volumes without any visible symptom.

Deliberately NO GPU, NO model, NO torch training — pure pydicom + numpy +
pandas reading real DICOM headers/pixel geometry. Cheap and fast (doesn't
touch the weekly GPU quota, doesn't compete with the 1-concurrent-GPU-
session limit), so it can run ahead of / independently from the real
training kernel.

Module structure: everything Kaggle-specific (mount discovery, sys.path
surgery, pip install) is confined to _bootstrap() and the __main__ guard
below, so check_series()/main() have no import-time side effects and can be
unit-tested locally against synthetic pydicom datasets (see
tests/test_smoke.py's test_check_series_* cases) without needing /kaggle/input
to exist. Mount discovery itself is duplicated from train_kernel.py rather
than imported from kadd, since finding the kadd mount is what has to happen
BEFORE kadd is importable.
"""
import json
import logging
import os
import sys
import time

import numpy as np
import pandas as pd
import pydicom

logging.basicConfig(level=logging.INFO, format='%(message)s')

N_STUDIES = 5              # how many distinct studies to sample series from
MAX_SERIES_PER_STUDY = 3   # cap series checked per study, so this stays fast
SPACING_CV_WARN = 0.15     # coefficient of variation (std/mean) above this -> WARN on spacing uniformity
REQUIRED_TAGS = ["ImagePositionPatient", "ImageOrientationPatient",
                  "WindowCenter", "WindowWidth", "PixelSpacing"]
SPACING_TAGS = ["SpacingBetweenSlices", "SliceThickness"]  # either may carry declared inter-slice spacing


def check_series(series_dir, study_uid, series_uid):
    """Returns a dict report for one series. Never raises — a real DICOM
    problem here is exactly what this diagnostic exists to surface, not
    something that should crash the whole run.
    """
    from kadd.classify.data import _slice_position

    report = {"study": study_uid, "series": series_uid, "n_slices": 0,
              "missing_tags": [], "status": "FAIL", "notes": []}

    paths = [os.path.join(series_dir, f) for f in os.listdir(series_dir)
             if f.lower().endswith(".dcm")]
    if len(paths) < 2:
        report["notes"].append(f"only {len(paths)} .dcm file(s) found, need >=2 to check spacing")
        return report

    slices = [pydicom.dcmread(p) for p in paths]
    report["n_slices"] = len(slices)

    missing = set()
    for ds in slices:
        for tag in REQUIRED_TAGS:
            if not hasattr(ds, tag):
                missing.add(tag)
    report["missing_tags"] = sorted(missing)
    if missing:
        report["status"] = "FAIL"
        report["notes"].append(f"required tag(s) missing on at least one slice: {sorted(missing)}")
        return report

    # Geometry-based sort (the actual production sort key).
    try:
        positions = [(_slice_position(ds), ds) for ds in slices]
    except Exception as e:
        report["status"] = "FAIL"
        report["notes"].append(f"_slice_position() raised: {e}")
        return report
    positions.sort(key=lambda t: t[0])
    sorted_pos = np.array([p for p, _ in positions])
    sorted_slices = [ds for _, ds in positions]

    deltas = np.diff(sorted_pos)
    if len(deltas) == 0 or np.any(deltas == 0):
        report["status"] = "FAIL"
        report["notes"].append("duplicate/tied slice positions after sort — geometry can't order these slices")
        return report

    signs = np.sign(deltas)
    if not np.all(signs == signs[0]):
        report["status"] = "FAIL"
        report["notes"].append("slice spacing is not monotonic (sign flips in position deltas) "
                                "— sorted order is internally inconsistent")
        return report

    abs_deltas = np.abs(deltas)
    mean_spacing, std_spacing = float(abs_deltas.mean()), float(abs_deltas.std())
    cv = std_spacing / mean_spacing if mean_spacing else float("inf")
    report["measured_spacing_mm"] = {"mean": round(mean_spacing, 4), "std": round(std_spacing, 4), "cv": round(cv, 4)}

    declared = None
    for tag in SPACING_TAGS:
        v = getattr(sorted_slices[0], tag, None)
        if v is not None:
            declared = float(v)
            break
    report["declared_spacing_mm"] = declared

    status = "PASS"
    if cv > SPACING_CV_WARN:
        status = "WARN"
        report["notes"].append(f"spacing coefficient of variation {cv:.3f} exceeds {SPACING_CV_WARN} "
                                f"— slices may not be evenly spaced (could be legitimate, e.g. a gapped protocol)")
    if declared is not None and abs(mean_spacing - declared) / declared > 0.25:
        status = "WARN"
        report["notes"].append(f"measured mean spacing {mean_spacing:.3f}mm is >25% off the declared "
                                f"{declared:.3f}mm — geometry sort may disagree with the scanner's own spacing")

    # Corroboration check: does InstanceNumber (when present) agree with the
    # geometry-based order, up to a global reversal? Not authoritative on its
    # own (InstanceNumber can be wrong) — but a strong disagreement is worth
    # knowing about either way.
    has_instance_number = all(hasattr(ds, "InstanceNumber") for ds in slices)
    if has_instance_number:
        geom_order = [id(ds) for ds in sorted_slices]
        inst_order_asc = [id(ds) for ds in sorted(slices, key=lambda d: int(d.InstanceNumber))]
        inst_order_desc = list(reversed(inst_order_asc))
        agrees = geom_order == inst_order_asc or geom_order == inst_order_desc
        report["instance_number_agrees_with_geometry"] = agrees
        if not agrees:
            status = "WARN"
            report["notes"].append("InstanceNumber-based order disagrees with geometry-based order "
                                    "(even allowing a global reversal) — worth a manual look")
    else:
        report["instance_number_agrees_with_geometry"] = None

    report["status"] = status
    return report


def main(competition_data_root):
    series_csv = os.path.join(competition_data_root, "train_series.csv")
    train_series_dir = os.path.join(competition_data_root, "train_series")
    df = pd.read_csv(series_csv)

    reports = []
    checked_studies = 0
    for study_uid, group in df.groupby("StudyInstanceUID"):
        if checked_studies >= N_STUDIES:
            break
        study_dir = os.path.join(train_series_dir, study_uid)
        if not os.path.isdir(study_dir):
            continue
        n_this_study = 0
        for _, row in group.iterrows():
            if n_this_study >= MAX_SERIES_PER_STUDY:
                break
            series_uid = row["SeriesInstanceUID"]
            series_dir = os.path.join(study_dir, series_uid)
            if not os.path.isdir(series_dir):
                continue
            print(f"Checking study={study_uid} series={series_uid} "
                  f"(plane={row.get('Anatomical_Plane', '?')}) ...")
            t0 = time.time()
            report = check_series(series_dir, study_uid, series_uid)
            elapsed = time.time() - t0
            report["seconds"] = round(elapsed, 3)
            per_slice = elapsed / report["n_slices"] if report["n_slices"] else float("nan")
            print(f"  -> {report['status']}  n_slices={report['n_slices']}  "
                  f"{elapsed:.2f}s/3d-image ({per_slice:.3f}s/slice)  "
                  f"missing_tags={report['missing_tags']}  notes={report['notes']}")
            reports.append(report)
            n_this_study += 1
        if n_this_study > 0:
            checked_studies += 1

    n_pass = sum(r["status"] == "PASS" for r in reports)
    n_warn = sum(r["status"] == "WARN" for r in reports)
    n_fail = sum(r["status"] == "FAIL" for r in reports)
    seconds_per_image = [r["seconds"] for r in reports if "seconds" in r]
    avg_seconds_per_image = float(np.mean(seconds_per_image)) if seconds_per_image else float("nan")
    summary = {"n_series_checked": len(reports), "n_studies_checked": checked_studies,
               "n_pass": n_pass, "n_warn": n_warn, "n_fail": n_fail,
               "avg_seconds_per_3d_image": round(avg_seconds_per_image, 3), "reports": reports}

    print("-----------------------------------------------")
    print(f"Checked {len(reports)} series across {checked_studies} studies: "
          f"{n_pass} PASS, {n_warn} WARN, {n_fail} FAIL  "
          f"(avg {avg_seconds_per_image:.2f}s/3d-image, this diagnostic's load+check time only — "
          f"not the same as real training throughput, which also does windowing/crop/resize/model forward)")
    if n_fail:
        print("FAIL means a required tag was missing or the geometry sort was internally "
              "inconsistent — this would need fixing in kadd/classify/data.py before trusting "
              "any real training run's slice ordering.")
    print("-----------------------------------------------")

    os.makedirs("output", exist_ok=True)
    with open("output/dicom_verification_report.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("Full report written to output/dicom_verification_report.json")


# ------------------------------------------------------------------ #
#  Kaggle-specific bootstrap — only runs when executed as a script, so
#  importing this module locally (e.g. for tests) never touches /kaggle/input.
# ------------------------------------------------------------------ #

def _listdir_safe(path):
    return os.listdir(path) if os.path.isdir(path) else "MISSING"


def _walk_bounded(root, max_depth):
    if not os.path.isdir(root):
        return
    root_depth = root.rstrip(os.sep).count(os.sep)
    for dirpath, dirnames, filenames in os.walk(root):
        depth = dirpath.rstrip(os.sep).count(os.sep) - root_depth
        if depth >= max_depth:
            dirnames[:] = []
        yield dirpath, dirnames, filenames


def _find_kadd_mount(search_roots, max_depth=6):
    for root in search_roots:
        for dirpath, _dirnames, filenames in _walk_bounded(root, max_depth):
            if os.path.basename(dirpath) == "kadd" and "__init__.py" in filenames:
                return os.path.dirname(dirpath)
    return None


def _find_competition_root(search_roots, max_depth=4):
    for root in search_roots:
        for dirpath, _dirnames, filenames in _walk_bounded(root, max_depth):
            if "train.csv" in filenames and "train_series.csv" in filenames:
                return dirpath
    return None


def _bootstrap():
    input_root = "/kaggle/input"
    print(f"{input_root} top-level: {_listdir_safe(input_root)}")

    dataset_roots = [os.path.join(input_root, "datasets"), input_root]
    competition_roots = [os.path.join(input_root, "competitions"), input_root]

    kadd_mount = _find_kadd_mount(dataset_roots)
    if kadd_mount is None:
        raise RuntimeError(f"Could not find a mounted kadd/ package under {dataset_roots}. "
                            f"/kaggle/input top-level: {_listdir_safe(input_root)}")
    print(f"Found kadd package at: {kadd_mount}")
    sys.path.insert(0, kadd_mount)

    competition_data_root = _find_competition_root(competition_roots)
    if competition_data_root is None:
        raise RuntimeError(f"Could not find competition data under {competition_roots}. "
                            f"/kaggle/input top-level: {_listdir_safe(input_root)}")
    print(f"Found competition data at: {competition_data_root}")

    os.system(f"{sys.executable} -m pip install -q pydicom")
    return competition_data_root


if __name__ == "__main__":
    _competition_data_root = _bootstrap()
    main(_competition_data_root)
