#!/usr/bin/env python3
"""Push the volume-precompute kernel to Kaggle and check/pull its status.

See kaggle_kernel_precompute/precompute_kernel.py for what it does: decodes
+ windows + crops + resizes every training series' DICOM ONCE and saves the
final tensor, so future training runs read one small file instead of
re-doing that work (up to 160 raw DICOM file reads per series) every epoch.

CPU-only (enable_gpu: false in kernel-metadata.json) — doesn't touch the
weekly GPU quota or the 1-concurrent-GPU-session limit, so it can run
independently of / alongside the training kernel's own quota usage.

Resumable across separate pushes (this project's real track record this
session — repeated OOM-kills, cancellations — makes a single-run completion
unlikely for a multi-hour job). The loop:
    1. python scripts/kaggle_push_precompute_kernel.py --follow
    2. python scripts/kaggle_push_precomputed_dataset.py   (pulls this run's
       new output, merges into data/precomputed/, pushes as its own
       dataset, and wires it into this kernel's dataset_sources so step 1
       skips already-done series next time)
    3. repeat step 1/2 until the manifest shows full coverage

Usage:
    python scripts/kaggle_push_precompute_kernel.py              # push + check status
    python scripts/kaggle_push_precompute_kernel.py --follow      # push + stream live logs
    python scripts/kaggle_push_precompute_kernel.py --status      # just check status, don't push
    python scripts/kaggle_push_precompute_kernel.py --logs        # pull output files after status
"""
import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
KERNEL_DIR = REPO_ROOT / "kaggle_kernel_precompute"
KERNEL_ID = "pieteroehlers/kadd-precompute-volumes"
DEFAULT_TIMEOUT_SECONDS = 6 * 3600  # matches precompute_kernel.py's own 5.5h soft self-stop, with margin

_EXPECTED_MIN_DONE_RE = re.compile(r'^EXPECTED_MIN_DONE = \d+$', re.MULTILINE)


def _stage_kernel_with_expected_count():
    """Copy kaggle_kernel_precompute/ to a temp dir with EXPECTED_MIN_DONE
    patched to the local data/precomputed/manifest.jsonl's current line
    count — the true floor for what a genuinely up-to-date mount should show
    on the NEXT kernel run, letting it detect (and wait out) a stale mount
    instead of just checking "does a manifest exist" (see
    precompute_kernel.py's _find_precomputed_mount_with_retry docstring for
    the real bug this fixes — kernels v13/v15 both silently resumed from a
    stale, older push). Leaves the tracked precompute_kernel.py untouched,
    same staging pattern as scripts/kaggle_push_kernel.py's --type/--size.
    Fails loudly if the EXPECTED_MIN_DONE line doesn't match the expected
    pattern exactly once, rather than silently pushing an unpatched kernel.
    """
    manifest_path = REPO_ROOT / "data" / "precomputed" / "manifest.jsonl"
    if not manifest_path.is_file():
        print("No local data/precomputed/manifest.jsonl found — pushing with "
              "EXPECTED_MIN_DONE=0 (no freshness check, matches a first-ever run).")
        return KERNEL_DIR

    with open(manifest_path) as f:
        expected_min_done = sum(1 for line in f if line.strip())

    stage = Path(tempfile.mkdtemp()) / "kaggle_kernel_precompute"
    shutil.copytree(KERNEL_DIR, stage, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    kernel_path = stage / "precompute_kernel.py"
    source = kernel_path.read_text()
    new_line = f"EXPECTED_MIN_DONE = {expected_min_done}"
    patched, n = _EXPECTED_MIN_DONE_RE.subn(new_line, source)
    if n != 1:
        raise RuntimeError(
            f"Expected exactly one EXPECTED_MIN_DONE line matching the known pattern in "
            f"precompute_kernel.py, found {n} — refusing to push an unpatched/ambiguous kernel. "
            f"Check precompute_kernel.py hasn't changed shape since this script was written.")
    kernel_path.write_text(patched)
    print(f"Staged kernel with EXPECTED_MIN_DONE={expected_min_done} "
          f"(from local {manifest_path.relative_to(REPO_ROOT)})")
    return stage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true",
                     help="Only check kernel run status, don't push a new version")
    ap.add_argument("--logs", action="store_true",
                     help="Pull kernel output files after checking status")
    ap.add_argument("--follow", action="store_true",
                     help="After pushing, stream live logs instead of just printing status")
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS,
                     help=f"Max kernel run-time in seconds (default: {DEFAULT_TIMEOUT_SECONDS}s / 6h)")
    args = ap.parse_args()

    if not args.status:
        push_dir = _stage_kernel_with_expected_count()
        subprocess.run(["kaggle", "kernels", "push", "-p", str(push_dir),
                         "-t", str(args.timeout)], check=True)
        print(f"\nPushed. Track progress at: https://www.kaggle.com/code/{KERNEL_ID}")

    if args.follow:
        subprocess.run(["kaggle", "kernels", "logs", "-f", KERNEL_ID])
    else:
        subprocess.run(["kaggle", "kernels", "status", KERNEL_ID])

    if args.logs:
        out_dir = REPO_ROOT / "output" / "kaggle_precompute_output"
        out_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(["kaggle", "kernels", "output", KERNEL_ID, "-p", str(out_dir)])
        print(f"Kernel output pulled to {out_dir}")


if __name__ == "__main__":
    main()
