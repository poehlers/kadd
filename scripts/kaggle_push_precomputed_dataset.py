#!/usr/bin/env python3
"""Pull the precompute kernel's latest output, merge it into a growing local
accumulation directory, and push the whole accumulated set as the
pieteroehlers/kadd-precomputed-volumes Kaggle dataset.

This is the local-side half of the multi-session resume loop (see
kaggle_kernel_precompute/precompute_kernel.py's docstring for the full
picture and why this is resumable-across-runs, not just within one):

    1. python scripts/kaggle_push_precompute_kernel.py --follow
    2. python scripts/kaggle_push_precomputed_dataset.py   <- this script
    3. repeat until the manifest shows full coverage

Why accumulate locally rather than push incrementally: `kaggle datasets
version` uploads whatever's in the staging directory as the COMPLETE new
version (there's no "append only new files" mode via this API) — so the
staging directory must contain everything accumulated so far, not just this
run's delta. Real tradeoff, stated plainly: this does mean data/precomputed/
grows on local disk toward the full precomputed set's eventual size
(rough estimate: 35-75GB for all ~4,349 studies at float16 — see
CLAUDE.md sec 9), unlike the rest of this project's data/ handling, which
deliberately avoids ever holding the bulk raw DICOM data locally. That
tradeoff was accepted explicitly because the alternative — no local staging
— would mean either (a) re-uploading the SAME already-done volumes from a
Kaggle-side location Kaggle doesn't actually expose a copy-only-input-to-
dataset primitive for, or (b) never being able to resume across separate
kernel pushes at all, which this project's own reliability track record
this session (repeated OOM-kills/cancellations) makes a real risk for a
job this long.

After a successful push, also wires the new dataset into
kaggle_kernel_precompute/kernel-metadata.json's dataset_sources (if not
already present) — idempotent, safe to run every time — so the NEXT
precompute kernel run can see this run's progress and skip it.

Usage:
    python scripts/kaggle_push_precomputed_dataset.py
"""
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
KERNEL_ID = "pieteroehlers/kadd-precompute-volumes"
DATASET_ID = "pieteroehlers/kadd-precomputed-volumes"
LOCAL_ACCUMULATION_DIR = REPO_ROOT / "data" / "precomputed"
MANIFEST_FILENAME = "manifest.jsonl"

# Scratch dir for the pulled kernel output, staged on T7 (via REPO_ROOT) rather
# than the OS default temp dir. tempfile.mkdtemp() with no `dir=` lands under
# /var/folders/.../T on the internal boot volume on macOS regardless of where
# the repo/data live — a real incident, not a hypothetical: repeated ~15-20GB
# pulls there filled the internal disk to 0 bytes free mid-session, hard-
# blocking every Bash command (even the harness's own tiny per-command output
# file couldn't be created). Cleaned up after every pull in main()'s finally
# block, since nothing was cleaning these up before, which is how they piled
# up in the first place.
PULL_SCRATCH_DIR = REPO_ROOT / "data" / "_pull_scratch"

# Sharding — real fix for a real failed push, not a nice-to-have. Checked the
# installed `kaggle` CLI's own source (kaggle/api/kaggle_api_extended.py,
# _upload_file_or_folder): `-r zip` builds ONE archive PER TOP-LEVEL
# DIRECTORY in the staged path and uploads each as its own file — that's
# also how Kaggle knows to auto-extract it server-side, so a single zip
# built by hand (not going through this per-directory mechanism) would just
# sit in the mount as an opaque, unextracted file — not a usable fix.
# precompute_kernel.py writes one directory per STUDY, and a real push with
# ~1,200 study directories staged got through dozens of slow per-directory
# archive+upload cycles before failing outright. Bucketing studies into a
# small, fixed number of shard directories keeps Kaggle's own (correct,
# auto-extracting) per-directory zip mechanism, just with ~32 directories to
# archive instead of thousands. A study's shard is a stable hash of its UID
# (not Python's built-in hash(), which is randomized per-process for
# strings) so the same study always lands in the same shard across separate
# merge runs.
NUM_SHARDS = 32


def _shard_for_study(study_uid: str) -> str:
    digest = hashlib.md5(study_uid.encode()).hexdigest()
    return f"shard_{int(digest, 16) % NUM_SHARDS:02d}"


def _pull_latest_kernel_output():
    if PULL_SCRATCH_DIR.exists():
        shutil.rmtree(PULL_SCRATCH_DIR)
    tmp_dir = PULL_SCRATCH_DIR / "precompute_pull"
    subprocess.run(["kaggle", "kernels", "output", KERNEL_ID, "-p", str(tmp_dir)], check=True)
    pulled_root = tmp_dir / "output" / "precomputed"
    if not pulled_root.is_dir():
        raise RuntimeError(f"Expected {pulled_root} in the pulled kernel output but it's missing — "
                            f"has the precompute kernel actually run and produced output yet? "
                            f"Pulled contents: {list(tmp_dir.rglob('*'))[:20]}")
    return pulled_root


def _read_manifest_lines(path):
    """Yields parsed records, skipping (and warning about) any malformed
    line rather than crashing — a real precompute run that hits its output-
    byte budget or gets killed mid-write can leave a truncated final line
    (e.g. the process died mid-flush when /kaggle/working ran out of space),
    and that's exactly the kind of run this merge step needs to survive."""
    if not path.exists():
        return
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                print(f"WARNING: skipping malformed manifest line {i + 1} in {path} "
                      f"(likely a truncated write from an interrupted run): {line[:80]!r}",
                      file=sys.stderr)


def _merge_manifest(pulled_manifest_path, local_manifest_path):
    seen = {}
    for rec in _read_manifest_lines(local_manifest_path):
        seen[(rec["study"], rec["series"])] = rec

    n_new = 0
    for rec in _read_manifest_lines(pulled_manifest_path):
        key = (rec["study"], rec["series"])
        if key not in seen:
            n_new += 1
        seen[key] = rec

    local_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(local_manifest_path, "w") as f:
        for rec in seen.values():
            f.write(json.dumps(rec) + "\n")
    return len(seen), n_new


def _merge_npy_files(pulled_root, local_root):
    """Copies new .npy files from a pulled kernel run (flat <study>/<series>.npy,
    written that way by precompute_kernel.py — fine for a plain `kaggle
    kernels output` pull, which has no per-directory-zip behavior) into the
    LOCAL accumulation dir, bucketed into shard_XX/<study>/<series>.npy so
    the eventual dataset push stays under NUM_SHARDS top-level directories.
    See NUM_SHARDS' comment for why this sharding exists.
    """
    n_copied = 0
    for src in pulled_root.rglob("*.npy"):
        rel = src.relative_to(pulled_root)
        study_uid = rel.parts[0]
        dst = local_root / _shard_for_study(study_uid) / rel
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        n_copied += 1
    return n_copied


def _push_dataset():
    dataset_metadata_src = REPO_ROOT / "kaggle_kernel_precompute" / "dataset-metadata.json"
    shutil.copy(dataset_metadata_src, LOCAL_ACCUMULATION_DIR / "dataset-metadata.json")

    version_cmd = ["kaggle", "datasets", "version", "-p", str(LOCAL_ACCUMULATION_DIR),
                   "-m", "Update precomputed volumes", "-r", "zip", "-d"]
    result = subprocess.run(version_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("`datasets version` failed (expected on first push) — trying `datasets create` instead:")
        print(result.stdout, result.stderr)
        subprocess.run(["kaggle", "datasets", "create", "-p", str(LOCAL_ACCUMULATION_DIR), "-r", "zip"],
                        check=True)
    else:
        print(result.stdout)


def _wire_dataset_into_kernel_sources():
    kernel_metadata_path = REPO_ROOT / "kaggle_kernel_precompute" / "kernel-metadata.json"
    with open(kernel_metadata_path) as f:
        metadata = json.load(f)
    if DATASET_ID in metadata.get("dataset_sources", []):
        return False
    metadata["dataset_sources"].append(DATASET_ID)
    with open(kernel_metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")
    return True


def main():
    try:
        print(f"Pulling latest output from kernel {KERNEL_ID}...")
        pulled_root = _pull_latest_kernel_output()

        n_copied = _merge_npy_files(pulled_root, LOCAL_ACCUMULATION_DIR)
        n_total, n_new_manifest = _merge_manifest(
            pulled_root / MANIFEST_FILENAME, LOCAL_ACCUMULATION_DIR / MANIFEST_FILENAME)
        print(f"Merged {n_copied} new .npy files, {n_new_manifest} new manifest entries "
              f"({n_total} total accumulated series).")

        if n_copied == 0 and n_new_manifest == 0:
            print("Nothing new to push — skipping dataset push.")
            return

        print(f"Pushing accumulated {LOCAL_ACCUMULATION_DIR} as {DATASET_ID}...")
        _push_dataset()
        print(f"\nDataset: https://www.kaggle.com/datasets/{DATASET_ID}")

        if _wire_dataset_into_kernel_sources():
            print(f"Added {DATASET_ID} to kaggle_kernel_precompute/kernel-metadata.json's "
                  f"dataset_sources — next precompute kernel run will see this progress and skip it.")
    finally:
        if PULL_SCRATCH_DIR.exists():
            shutil.rmtree(PULL_SCRATCH_DIR)


if __name__ == "__main__":
    main()
