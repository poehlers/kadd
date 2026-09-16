"""Offline, one-time precompute of every training series' preprocessed
volume tensor — decode + window + crop + resize ONCE, save to disk, reuse
forever instead of redoing this on every single training epoch.

Why this exists: kadd/classify/data.py's load_series_volume was measured
(via real Kaggle timing, see CLAUDE.md sec 9) to spend ~63% of its time on
raw file I/O (some series have up to 160 separate slice files — each a
separate open/read against Kaggle's mounted competition storage), ~22% on
crop+resize, ~11% on windowing arithmetic, and only ~4% on actual DICOM
codec decompression. None of that is cached anywhere — every epoch re-pays
the full cost for every study it touches. This kernel pays that cost once
per series, ever, and writes the final (1, 32, 224, 224) float16 tensor —
exactly what load_series_volume would have returned — to a single small
.npy file. Training then reads one small file instead of opening up to 160
raw DICOM files and redoing decode+crop+resize from scratch.

TARGET_DEPTH/TARGET_SIZE are fixed at 32/224 — confirmed (2026-08-15) to be
identical across every one of this project's 10 config files, so one
precomputed set serves every current model size/type without needing to be
redone per config.

Deliberately CPU-only (see kernel-metadata.json — enable_gpu: false), same
reasoning as verify_dicom_kernel.py: this is pure decode/preprocessing, no
model involved, so it doesn't touch the weekly GPU quota or the
1-concurrent-GPU-session limit.

RESUMABLE ACROSS SEPARATE KERNEL RUNS, not just within one — this project's
own track record this session (repeated OOM-kills, cancellations) makes it
likely a multi-hour precompute job needs more than one run to finish.
Mechanism: a JSONL manifest (one line per series, flushed immediately after
each write, same pattern as scripts/extract_report_labels.py's checkpoint)
records what's done. If a prior version of the `kadd-precomputed-volumes`
dataset is attached (via kernel-metadata.json's dataset_sources, added once
that dataset exists), this run loads ITS manifest first and skips anything
already done — only computing what's genuinely new. See
scripts/kaggle_push_precomputed_dataset.py for the local-side half of this
loop (pulls this kernel's new output, merges into a growing local
accumulation directory, re-pushes the whole accumulated set as the next
dataset version).

Module structure: everything Kaggle-specific (mount discovery, sys.path
surgery, pip install) is confined to _bootstrap() and the __main__ guard
below, matching kaggle_kernel_verify/verify_dicom_kernel.py's pattern — so
main()/_load_prior_manifest() have no import-time side effects and can be
unit-tested locally against synthetic data without needing /kaggle/input to
exist (see tests/test_smoke.py's test_precompute_* cases).
"""
import json
import logging
import os
import time

logging.basicConfig(level=logging.INFO, format='%(message)s')

# Set by scripts/kaggle_push_precompute_kernel.py's staging step, patched to
# the local data/precomputed/manifest.jsonl's line count at push time (i.e.
# "how many series SHOULD be visible once the mount has caught up to the
# latest push"). 0 is the safe default for a manual/unpatched push — means
# "accept whatever the mount shows immediately, no freshness check." See
# _find_precomputed_mount_with_retry's docstring for why this exists: the
# mount can be non-empty but reflect an OLDER push than the one just made,
# and a bare "does a manifest exist" check can't tell the difference.
EXPECTED_MIN_DONE = 0

TARGET_DEPTH = 32
TARGET_SIZE = 224
# Must match kadd/classify/data.py's load_series_volume default exactly — a
# precomputed volume that used a different crop_mm than the current config
# expects is silently wrong (no shape mismatch to catch it, since crop_mm
# only changes what physical extent survived, not the final tensor shape).
CROP_MM = 130.0
OUTPUT_ROOT = "output/precomputed"
MANIFEST_FILENAME = "manifest.jsonl"
PROGRESS_EVERY = 25
# Soft self-stop, well under Kaggle's hard kernel timeout (see
# scripts/kaggle_push_precompute_kernel.py's -t/--timeout, default 6h) — so
# the script exits cleanly with a consistent manifest/output instead of
# being SIGKILL'd mid-write partway through a series.
MAX_RUNTIME_SECONDS = 5.5 * 3600

# /kaggle/working (this kernel's cwd — see kernel-metadata.json) is hard-
# capped at 20GB (confirmed via Kaggle's own product-feedback pages, not
# assumed — a real run hit this and errored out mid-write before this check
# existed). Tracked as cumulative bytes actually written this run, not via
# shutil.disk_usage/free-space queries — the 20GB figure is a Kaggle-side
# quota on this specific mount, not necessarily what the container's own
# disk-usage reporting reflects, so counting our own output is the only
# fully reliable way to know how close we are to it. 15GB soft budget
# leaves ~5GB margin for logs, pip-installed packages, and other overhead
# sharing the same quota.
MAX_OUTPUT_BYTES = 15 * 1024 ** 3


def _load_prior_manifest(mount_dir):
    """Returns {(study, series): status} from a prior run's manifest, or {}
    if there's no prior mount (expected on the very first run, before the
    kadd-precomputed-volumes dataset exists)."""
    if mount_dir is None:
        return {}
    manifest_path = os.path.join(mount_dir, MANIFEST_FILENAME)
    if not os.path.isfile(manifest_path):
        return {}
    done = {}
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            done[(rec["study"], rec["series"])] = rec["status"]
    return done


def main(competition_data_root, precomputed_mount):
    from kadd.classify.data import load_series_volume, _RECOVERABLE_DICOM_ERRORS
    import numpy as np
    import pandas as pd

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    manifest_path = os.path.join(OUTPUT_ROOT, MANIFEST_FILENAME)

    prior_done = _load_prior_manifest(precomputed_mount)
    print(f"{len(prior_done)} series already done in a prior run, will be skipped.")

    series_csv = os.path.join(competition_data_root, "train_series.csv")
    train_series_dir = os.path.join(competition_data_root, "train_series")
    df = pd.read_csv(series_csv)
    total = len(df)

    start_time = time.time()
    n_done_this_run = n_skipped = n_errors = 0
    bytes_written = 0

    with open(manifest_path, "a") as manifest_f:
        for i, row in df.iterrows():
            study_uid = row["StudyInstanceUID"]
            series_uid = row["SeriesInstanceUID"]

            if (study_uid, series_uid) in prior_done:
                n_skipped += 1
                continue

            elapsed = time.time() - start_time
            if elapsed > MAX_RUNTIME_SECONDS:
                print(f"Hit soft runtime budget ({MAX_RUNTIME_SECONDS:.0f}s) at row {i}/{total} "
                      f"— stopping cleanly so this run's output is consistent. Re-run to continue.")
                break
            if bytes_written > MAX_OUTPUT_BYTES:
                print(f"Hit soft output-size budget ({MAX_OUTPUT_BYTES / 1e9:.1f}GB) at row {i}/{total} "
                      f"— stopping cleanly before /kaggle/working's 20GB hard cap. Re-run to continue "
                      f"(push this run's output as a dataset first so the next run can skip what's done).")
                break

            series_dir = os.path.join(train_series_dir, study_uid, series_uid)
            out_dir = os.path.join(OUTPUT_ROOT, study_uid)
            out_path = os.path.join(out_dir, f"{series_uid}.npy")

            record = {"study": study_uid, "series": series_uid}
            try:
                volume = load_series_volume(series_dir, TARGET_DEPTH, TARGET_SIZE,
                                             plane=row.get("Anatomical_Plane"), crop_mm=CROP_MM)
                os.makedirs(out_dir, exist_ok=True)
                arr = volume.numpy().astype(np.float16)
                np.save(out_path, arr)
                bytes_written += arr.nbytes
                record["status"] = "ok"
                n_done_this_run += 1
            except _RECOVERABLE_DICOM_ERRORS as e:
                # Same class of real, expected DICOM corruption already
                # handled in KneeDataset.__getitem__ — recorded so future
                # runs don't retry a hopeless series forever, not silently
                # dropped.
                record["status"] = "error"
                record["error"] = str(e)
                n_errors += 1
            except FileNotFoundError as e:
                record["status"] = "error"
                record["error"] = f"series directory not found: {e}"
                n_errors += 1

            manifest_f.write(json.dumps(record) + "\n")
            manifest_f.flush()

            n_total_processed = n_done_this_run + n_errors
            if n_total_processed % PROGRESS_EVERY == 0 and n_total_processed > 0:
                rate = n_total_processed / (time.time() - start_time)
                remaining = total - n_skipped - n_total_processed
                eta_s = remaining / rate if rate > 0 else float("inf")
                print(f"  [{n_skipped + n_total_processed}/{total}] "
                      f"{n_done_this_run} ok, {n_errors} errors this run, "
                      f"{rate:.2f} series/s, ETA ~{eta_s / 60:.0f} min, "
                      f"{bytes_written / 1e9:.2f}GB written this run")

    print("-----------------------------------------------")
    print(f"This run: {n_done_this_run} computed, {n_errors} errors, "
          f"{n_skipped} skipped (already done in a prior run), "
          f"{bytes_written / 1e9:.2f}GB written")
    print(f"Manifest: {manifest_path}")
    print("-----------------------------------------------")


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


def _find_precomputed_mount(search_roots, max_depth=4):
    """Optional — returns None on the first-ever run, before this dataset
    exists. Identified by the presence of MANIFEST_FILENAME at its root
    (not just any .npy, to avoid accidentally matching an unrelated
    dataset)."""
    for root in search_roots:
        for dirpath, _dirnames, filenames in _walk_bounded(root, max_depth):
            if MANIFEST_FILENAME in filenames:
                return dirpath
    return None


def _precomputed_dataset_dir_visible(search_roots, max_depth=4):
    """True if a directory literally named 'kadd-precomputed-volumes' is
    visible anywhere under search_roots — regardless of whether its content
    (manifest.jsonl) has finished syncing yet. Used to distinguish a real
    first-ever run (no such directory at all — don't waste time waiting) from
    a mount-attachment lag (the directory is attached but Kaggle hasn't
    finished syncing its files into the container yet — worth a retry)."""
    for root in search_roots:
        for dirpath, dirnames, _filenames in _walk_bounded(root, max_depth):
            if "kadd-precomputed-volumes" in dirnames or \
               os.path.basename(dirpath) == "kadd-precomputed-volumes":
                return True
    return False


def _find_precomputed_mount_with_retry(search_roots, retries=40, delay_seconds=30,
                                        expected_min_done=0):
    """Guards against TWO real, distinct failure modes hit in this project
    (2026-09-04/07):

    1. Kernel v11: the mount directory was visible in /kaggle/input's
       top-level listing, but its contents (manifest.jsonl) hadn't finished
       syncing yet — _find_precomputed_mount silently returned None (dir
       present but empty), and the run redid already-done series from
       scratch. Guarded by _precomputed_dataset_dir_visible + retrying while
       no manifest is found at all.

    2. Kernel v13/v15 (found AFTER the above fix already shipped, so it's a
       genuinely different bug, not a leftover): the mount was non-empty and
       _find_precomputed_mount returned immediately — but the manifest it
       found reflected an OLDER push, not the LATEST one (e.g. one run saw
       15,050 done when the actual latest push had 20,066). A bare "does a
       manifest exist" check can't tell a stale-but-present manifest from a
       current one. Real propagation lag between a dataset finishing
       processing and a NEWLY-SCHEDULED kernel's container actually mounting
       that latest revision turned out to be much longer and more variable
       than assumed — external waits of 90s and even 5 minutes both proved
       insufficient on different occasions, while on other occasions extra
       incidental delay (time spent investigating/debugging between runs)
       was what actually made the next run succeed, not the explicit wait
       itself. So this can't be fixed by picking a bigger fixed sleep with
       any real confidence — it needs an actual freshness check.

    expected_min_done (see EXPECTED_MIN_DONE / kaggle_push_precompute_kernel.py's
    staging step): the number of series the LOCAL accumulation had at push
    time, i.e. the true floor for what a genuinely up-to-date mount should
    show. If the mounted manifest's real count is below this, it's
    confirmed stale — keep retrying instead of accepting it. 0 (the default
    for an unpatched/manual push) disables this check entirely, falling back
    to "accept the first manifest found," matching the pre-fix behavior.

    retries*delay_seconds now defaults to 20 minutes (was ~90s) — the real
    lag observed was on that order, not seconds; a CPU-only kernel's 5.5h
    self-stop budget comfortably absorbs this. Proceeds anyway (with the
    stale mount) after exhausting retries rather than waiting forever —
    duplicate compute is wasteful but not incorrect, worth bounding.
    """
    if not _precomputed_dataset_dir_visible(search_roots):
        return None  # genuinely no prior dataset — first run, don't wait around for nothing
    for attempt in range(retries):
        mount = _find_precomputed_mount(search_roots)
        if mount is not None:
            n_done = len(_load_prior_manifest(mount))
            if n_done >= expected_min_done:
                return mount
            print(f"Found a mount at {mount} but it only shows {n_done} done "
                  f"(expected at least {expected_min_done} from the latest push) — stale, "
                  f"not the current revision yet (attempt {attempt + 1}/{retries}), "
                  f"retrying in {delay_seconds}s...")
        else:
            print(f"kadd-precomputed-volumes directory is visible but manifest.jsonl not found yet "
                  f"(attempt {attempt + 1}/{retries}) — likely still syncing, retrying in {delay_seconds}s...")
        time.sleep(delay_seconds)
    # Exhausted retries — proceed with whatever's there (possibly still stale) rather than
    # waiting indefinitely; report clearly so a stale-resume is visible in the log, not silent.
    mount = _find_precomputed_mount(search_roots)
    n_done = len(_load_prior_manifest(mount)) if mount else 0
    print(f"Gave up waiting after {retries} retries ({retries * delay_seconds}s) — proceeding with "
          f"{'no mount' if mount is None else f'{mount} ({n_done} done, possibly still stale)'}.")
    return mount


def _bootstrap():
    import sys

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

    precomputed_mount = _find_precomputed_mount_with_retry(dataset_roots, expected_min_done=EXPECTED_MIN_DONE)
    if precomputed_mount is None:
        print("No prior kadd-precomputed-volumes mount found — starting fresh (expected on the first run).")
    else:
        print(f"Found prior precomputed volumes at: {precomputed_mount} — will skip already-done series.")

    os.system(f"{sys.executable} -m pip install -q pydicom")
    return competition_data_root, precomputed_mount


if __name__ == "__main__":
    _competition_data_root, _precomputed_mount = _bootstrap()
    main(_competition_data_root, _precomputed_mount)
