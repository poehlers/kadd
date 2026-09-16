#!/usr/bin/env python3
"""Push kadd/ + config_files/ to Kaggle as a private dataset, so
kaggle_kernel/train_kernel.py can `import kadd` on Kaggle's infrastructure
(a kernel can't see this local checkout directly).

The bulk DICOM tree (train_series/ etc.) is NOT pushed here and never
should be — it's already on Kaggle via the competition_sources mount in
kernel-metadata.json. This dataset carries only code + small label files,
notably the locally-computed report-derived silver labels
(data/report_labels_llm.csv, see CLAUDE.md sec 8), which have no other way
to reach the kernel since they're generated locally, not part of the
competition's own data.

Uses the `kaggle` CLI via subprocess (not kagglehub — kagglehub is
download-only and doesn't support pushing).

Usage:
    python scripts/kaggle_push_dataset.py
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_ID = "pieteroehlers/kadd-package"
INCLUDE = ["kadd", "config_files", "setup.py", "requirements.txt"]
# Individual files (not whole dirs) copied preserving their relative path —
# used for small locally-generated artifacts that live inside data/, which
# we do NOT want to copy wholesale (it also holds the multi-GB DICOM
# download/cache that must never leave this machine via this dataset).
EXTRA_FILES = [
    "data/report_labels_llm.csv", "data/report_labels_llm_calibration.json",
    # train.csv/train_series.csv (5.7MB/3.5MB) ride along here so the
    # training kernel no longer needs to attach the full competition dataset
    # (~535GB of raw DICOM, see CLAUDE.md's compression-estimate investigation)
    # just to read two small CSVs — now that the precompute pipeline covers
    # every series, raw DICOM is only ever a fallback path, not a hard
    # requirement. See kaggle_kernel/train_kernel.py's competition-mount
    # handling (made optional) and kernel-metadata.json (competition_sources
    # dropped).
    "data/train.csv", "data/train_series.csv",
    # A specific run's checkpoint, staged here ONLY when actually resuming a
    # kernel-killed/quota-exhausted run (see kadd/classify/train.py's
    # load_checkpoint_for_resume, added 2026-09-03). Not present most of the
    # time — the WARNING-and-skip below is expected then, not an error.
    "data/resume_checkpoint.pth",
    # Separate checkpoint file for swin3d/medium (added 2026-09-12) — medium
    # and light have incompatible architectures (different stage-3 depth, so
    # different parameter shapes, not just a naming mismatch), so each tier
    # needs its own resume checkpoint rather than sharing
    # data/resume_checkpoint.pth and clobbering whichever tier isn't
    # currently training. Referenced by classify_swin3d_medium.yml's
    # resume_from.
    "data/resume_checkpoint_medium.pth",
]


def main():
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / "kadd-package"
        stage.mkdir()

        for item in INCLUDE:
            src = REPO_ROOT / item
            if not src.exists():
                print(f"WARNING: {src} not found, skipping", file=sys.stderr)
                continue
            dst = stage / item
            if src.is_dir():
                shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            else:
                shutil.copy(src, dst)

        for item in EXTRA_FILES:
            src = REPO_ROOT / item
            if not src.exists():
                print(f"WARNING: {src} not found, skipping", file=sys.stderr)
                continue
            dst = stage / item
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dst)

        shutil.copy(REPO_ROOT / "kaggle_kernel" / "dataset-metadata.json",
                    stage / "dataset-metadata.json")

        # `datasets version` fails if the dataset doesn't exist yet (the
        # first push); `datasets create` fails if it already does. Try
        # version first since that's the common case after the first push.
        # -d/--delete-old-versions: kadd/ gets pushed repeatedly during
        # iteration, so drop prior versions rather than accumulating storage.
        version_cmd = ["kaggle", "datasets", "version", "-p", str(stage),
                        "-m", "Update kadd package", "-r", "zip", "-d"]
        result = subprocess.run(version_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("`datasets version` failed (expected on first push) — "
                  "trying `datasets create` instead:")
            print(result.stdout, result.stderr)
            subprocess.run(["kaggle", "datasets", "create", "-p", str(stage), "-r", "zip"],
                            check=True)
        else:
            print(result.stdout)

    print(f"\nDataset: https://www.kaggle.com/datasets/{DATASET_ID}")


if __name__ == "__main__":
    main()
