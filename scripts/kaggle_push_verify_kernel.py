#!/usr/bin/env python3
"""Push the DICOM-verification diagnostic kernel to Kaggle and check/pull its
status. See kaggle_kernel_verify/verify_dicom_kernel.py for what it checks
(whether slice-ordering tags survive the 86-tag allowlist on real training
DICOMs, and whether the geometry-based sort is internally consistent).

Deliberately CPU-only (enable_gpu: false in kernel-metadata.json) — this is
pure pydicom/numpy/pandas header inspection, no model, no GPU quota spent,
and it doesn't compete with the 1-concurrent-GPU-session limit that applies
to scripts/kaggle_push_kernel.py's training runs.

Prerequisite: run scripts/kaggle_push_dataset.py at least once first (this
kernel imports kadd.classify.data._slice_position from the same
pieteroehlers/kadd-package dataset the training kernel uses).

Usage:
    python scripts/kaggle_push_verify_kernel.py            # push + check status
    python scripts/kaggle_push_verify_kernel.py --follow    # push + stream live logs
    python scripts/kaggle_push_verify_kernel.py --status    # just check status, don't push
    python scripts/kaggle_push_verify_kernel.py --logs      # pull output files after status
"""
import argparse
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
KERNEL_DIR = REPO_ROOT / "kaggle_kernel_verify"
KERNEL_ID = "pieteroehlers/kadd-verify-dicom"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true",
                     help="Only check kernel run status, don't push a new version")
    ap.add_argument("--logs", action="store_true",
                     help="Pull kernel output files after checking status")
    ap.add_argument("--follow", action="store_true",
                     help="After pushing, stream live logs instead of just printing status")
    args = ap.parse_args()

    if not args.status:
        subprocess.run(["kaggle", "kernels", "push", "-p", str(KERNEL_DIR)], check=True)
        print(f"\nPushed. Track progress at: https://www.kaggle.com/code/{KERNEL_ID}")

    if args.follow:
        subprocess.run(["kaggle", "kernels", "logs", "-f", KERNEL_ID])
    else:
        subprocess.run(["kaggle", "kernels", "status", KERNEL_ID])

    if args.logs:
        out_dir = REPO_ROOT / "output" / "kaggle_verify_output"
        out_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(["kaggle", "kernels", "output", KERNEL_ID, "-p", str(out_dir)])
        print(f"Kernel output pulled to {out_dir}")


if __name__ == "__main__":
    main()
