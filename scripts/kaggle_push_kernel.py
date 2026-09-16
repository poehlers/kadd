#!/usr/bin/env python3
"""Push the kadd training kernel to Kaggle and check/pull its status.

Runs training remotely on Kaggle's free GPU quota — this is a development/
training kernel (internet enabled), NOT the final no-internet competition-
submission kernel (see kaggle_kernel/train_kernel.py docstring).

Prerequisite: run scripts/kaggle_push_dataset.py at least once first, so the
kernel's attached `pieteroehlers/kadd-package` dataset source exists.

Explicitly requests the NvidiaTeslaT4 accelerator rather than relying on the
`enable_gpu: true` default in kernel-metadata.json. Per the Kaggle CLI docs
(github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md), a bare `enable_gpu`
with no accelerator specified defaults to NvidiaTeslaP100 — which is broken
against Kaggle's current default PyTorch image (a cu128 build with no
Pascal/sm_60 kernels: torch.cuda.is_available() returns True but the first
CUDA op throws cudaErrorNoKernelImageForDevice). This would otherwise
silently break the very first real training run.

`kaggle kernels output` does NOT block until the run finishes — it just
grabs whatever's available from the latest run right now. Use --follow to
stream logs live instead of guessing when to pull output.

--type/--size selects which config_files/classify_<type>_<size>.yml the
KERNEL runs (train_kernel.py has no argv on Kaggle, so this can't be a
runtime flag there — instead this script patches a STAGED copy of
train_kernel.py's CONFIG_PATH line before pushing, leaving the tracked file
in kaggle_kernel/ untouched). This is the Kaggle-side equivalent of
`python -m kadd classify-train --type/--size` for local runs (see
kadd/util/config_registry.py) — same naming convention, same resolution
logic, different reason it can't just be a CLI arg on the kernel itself.
Remember: run scripts/kaggle_push_dataset.py first whenever config_files/
changed, since the kernel reads the config from the pushed dataset mount,
not from this push.

Usage:
    python scripts/kaggle_push_kernel.py              # push + check status
    python scripts/kaggle_push_kernel.py --follow      # push + stream live logs
    python scripts/kaggle_push_kernel.py --status      # just check status, don't push
    python scripts/kaggle_push_kernel.py --logs        # pull output files after status
    python scripts/kaggle_push_kernel.py --type cnn3d --size xlight --follow
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

KERNEL_DIR = REPO_ROOT / "kaggle_kernel"
KERNEL_ID = "pieteroehlers/kadd-classify-train"
ACCELERATOR = "NvidiaTeslaT4"
# Safety cap so a runaway run can't quietly burn the whole weekly GPU quota —
# but must stay under Kaggle's real platform session limit or it does nothing
# useful. RAISED 6h->10h, 2026-08-22: the previous 6h value was an unverified
# guess, and it actually fired for real on kernel v35 (the meta_fusion='early'
# cnn3d ablation) — killed with exit code 137 at epoch 54/100, still
# improving (macro_auc climbing, val_loss trending down, LR only annealed
# twice of a patience=20 budget), not because it converged. Confirmed via
# web search that Kaggle's actual GPU/CPU notebook session limit is 12h (
# raised from 9h — kaggle.com/discussions/product-feedback/302908), so 6h was
# needlessly conservative.
# RAISED AGAIN 10h->11.5h, 2026-09-03: the exact same failure mode hit again —
# kernel v54 (swin3d/light, full-dataset epochs) was killed by this timeout at
# epoch 11, still improving every epoch (macro_auc 0.51->0.68), not because it
# converged. 10h had the same problem 6h did: needlessly conservative against
# the confirmed 12h ceiling. 11.5h leaves 30min margin rather than cutting
# flush against it.
DEFAULT_TIMEOUT_SECONDS = int(11.5 * 3600)

_CONFIG_PATH_RE = re.compile(r'^CONFIG_PATH = f"\{KADD_MOUNT\}/config_files/[^"]+"$', re.MULTILINE)
_EXPECTED_PRECOMPUTED_DONE_RE = re.compile(r'^EXPECTED_PRECOMPUTED_DONE = \d+$', re.MULTILINE)


def _stage_kernel(type_=None, size=None):
    """Copy kaggle_kernel/ to a temp dir, patched with:
      - CONFIG_PATH (only if type_/size are given) -> the resolved
        config_files/classify_<type>_<size>.yml.
      - EXPECTED_PRECOMPUTED_DONE (always) -> the local
        data/precomputed/manifest.jsonl's current line count, i.e. the true
        floor for how many series a genuinely up-to-date kadd-precomputed-
        volumes mount should show. Without this, kernel v59 (2026-09-07)
        silently accepted a stale/still-syncing mount and burned real GPU
        quota training on zero real data — see train_kernel.py's
        _find_precomputed_mount_with_retry docstring for the full story.

    Leaves the tracked kaggle_kernel/train_kernel.py untouched. Fails loudly
    if either patch target doesn't match its expected pattern exactly once,
    rather than silently pushing an unpatched/ambiguous kernel.
    """
    stage = Path(tempfile.mkdtemp()) / "kaggle_kernel"
    shutil.copytree(KERNEL_DIR, stage, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    train_kernel_path = stage / "train_kernel.py"
    source = train_kernel_path.read_text()

    if type_ and size:
        from kadd.util.config_registry import resolve_config

        config_path = resolve_config(type_, size, str(REPO_ROOT / "config_files"))
        rel_path = Path(config_path).relative_to(REPO_ROOT).as_posix()
        new_config_line = f'CONFIG_PATH = f"{{KADD_MOUNT}}/{rel_path}"'
        source, n = _CONFIG_PATH_RE.subn(new_config_line, source)
        if n != 1:
            raise RuntimeError(
                f"Expected exactly one CONFIG_PATH line matching the known pattern in "
                f"train_kernel.py, found {n} — refusing to push an unpatched/ambiguous kernel. "
                f"Check train_kernel.py hasn't changed shape since this script was written."
            )
        print(f"Staged kernel with CONFIG_PATH -> {rel_path}")

    # Must count only status=="ok" lines, matching train_kernel.py's
    # _count_manifest_ok exactly — NOT total line count. A real manifest can
    # (and does: confirmed 2 of 24,371 here) contain permanent "error"
    # entries for genuinely corrupted DICOM series that will never become
    # "ok" no matter how long you wait. Counting total lines here produced
    # an unsatisfiable expected_min_done (24371 vs the true ceiling of
    # 24369 "ok") that made kernel v60 falsely detect its own correct, fresh
    # mount as stale — caught live in that run's own retry-warning log
    # before it could raise, fixed in v61's staging rather than guessed at.
    manifest_path = REPO_ROOT / "data" / "precomputed" / "manifest.jsonl"
    expected_min_done = 0
    if manifest_path.is_file():
        with open(manifest_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    if json.loads(line).get("status") == "ok":
                        expected_min_done += 1
                except json.JSONDecodeError:
                    continue
    new_expected_line = f"EXPECTED_PRECOMPUTED_DONE = {expected_min_done}"
    source, n = _EXPECTED_PRECOMPUTED_DONE_RE.subn(new_expected_line, source)
    if n != 1:
        raise RuntimeError(
            f"Expected exactly one EXPECTED_PRECOMPUTED_DONE line matching the known pattern "
            f"in train_kernel.py, found {n} — refusing to push an unpatched/ambiguous kernel."
        )
    print(f"Staged kernel with EXPECTED_PRECOMPUTED_DONE={expected_min_done} "
          f"(from local {manifest_path.relative_to(REPO_ROOT) if manifest_path.is_file() else 'no local manifest found'})")

    train_kernel_path.write_text(source)
    return stage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true",
                     help="Only check kernel run status, don't push a new version")
    ap.add_argument("--logs", action="store_true",
                     help="Pull kernel output files after checking status")
    ap.add_argument("--follow", action="store_true",
                     help="After pushing, stream live logs (kaggle kernels logs -f) instead of just printing status")
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS,
                     help=f"Max kernel run-time in seconds (default: {DEFAULT_TIMEOUT_SECONDS}s / 6h)")
    ap.add_argument("--type", choices=["swin3d", "cnn3d"],
                     help="Encoder type — used with --size to select which config the kernel runs")
    ap.add_argument("--size", choices=["xlight", "light", "medium", "large"],
                     help="Model size tier — used with --type (see above)")
    args = ap.parse_args()

    if bool(args.type) != bool(args.size):
        ap.error("--type and --size must be given together")

    if not args.status:
        push_dir = _stage_kernel(args.type, args.size)

        subprocess.run(["kaggle", "quota"])
        subprocess.run([
            "kaggle", "kernels", "push", "-p", str(push_dir),
            "--accelerator", ACCELERATOR,
            "-t", str(args.timeout),
        ], check=True)
        print(f"\nPushed. Track progress at: https://www.kaggle.com/code/{KERNEL_ID}")

    if args.follow:
        subprocess.run(["kaggle", "kernels", "logs", "-f", KERNEL_ID])
    else:
        subprocess.run(["kaggle", "kernels", "status", KERNEL_ID])

    if args.logs:
        out_dir = REPO_ROOT / "output" / "kaggle_kernel_output"
        out_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(["kaggle", "kernels", "output", KERNEL_ID, "-p", str(out_dir)])
        print(f"Kernel output pulled to {out_dir}")


if __name__ == "__main__":
    main()
