"""Runs remotely on Kaggle's infrastructure via `kaggle kernels push`
(scripts/kaggle_push_kernel.py) — a training/dev kernel using Kaggle's free
GPU quota, NOT the final no-internet competition-submission kernel.
enable_internet is on here (needed for `pip install pydicom psutil`, not
guaranteed preinstalled on Kaggle's base image); the real submission kernel
would need internet off and these bundled some other way (e.g. a wheel-file
dataset).

`kadd/` and `config_files/` reach the kernel via the `pieteroehlers/kadd-
package` dataset attached in kernel-metadata.json (pushed by
scripts/kaggle_push_dataset.py). Precomputed volumes reach it via the
`pieteroehlers/kadd-precomputed-volumes` dataset (also in dataset_sources).
`classify.model.pretrained=True` Kinetics-400 checkpoints reach it via the
`pieteroehlers/kadd-pretrained-weights` dataset (pushed by
scripts/kaggle_push_pretrained_dataset.py, rarely — these files never
change) — see _populate_torch_hub_cache below: this pre-populates torch's
hub cache BEFORE the model is built so torchvision never attempts a network
download, which matters beyond just this dev kernel: the real no-internet
submission kernel would otherwise fail outright on pretrained=True.

`competition_sources` is deliberately EMPTY (dropped 2026-08-19) — attaching
the full ~535GB raw-DICOM competition dataset was the dominant cost of this
kernel's session-startup mount time (confirmed: a cancelled run's kernel log
was completely empty, meaning it never got past Kaggle's own dataset-attach
phase before our code even started). Now that the precompute pipeline covers
every series, raw DICOM is only ever a fallback path — train.csv/
train_series.csv ride along in kadd-package instead (see EXTRA_FILES in
scripts/kaggle_push_dataset.py), so nothing here actually needs the
competition mount. Re-add a slug to competition_sources only if a run
genuinely needs raw-DICOM fallback (e.g. a precompute gap for a specific
series) and is willing to pay that mount cost again.

MOUNT LAYOUT — history, don't re-hardcode a guess again:
  - Attempt 1 assumed a flat /kaggle/input/<slug>/ layout (the classic
    Kaggle Notebooks convention). Failed: ModuleNotFoundError, even though
    the dataset was independently confirmed correct via the `kaggle` CLI
    (`kaggle datasets files`, `kaggle kernels pull`).
  - Attempt 2 added a diagnostic printing /kaggle/input's contents before
    failing. Result: `/kaggle/input contents: ['competitions', 'datasets']`
    — this runtime nests mounts by source type, not flat by slug.
  - This version (attempt 3) doesn't hardcode a specific nested path either
    (e.g. /kaggle/input/datasets/pieteroehlers/kadd-package) since that's
    still an inferred guess, not confirmed. Instead it SEARCHES for the
    mounted kadd/ package and the competition CSVs, bounded in depth so it
    can't accidentally recurse into the (huge, thousands-of-directories)
    train_series/ DICOM tree while looking. Whatever it finds gets printed,
    so a future failure here is self-explanatory from the log alone.
"""
import json
import logging
import os
import shutil
import socket
import sys
import time


def _listdir_safe(path):
    return os.listdir(path) if os.path.isdir(path) else "MISSING"


def _walk_bounded(root, max_depth):
    """os.walk that never descends more than max_depth levels below root —
    keeps the search from wandering into train_series/'s huge DICOM tree.
    """
    if not os.path.isdir(root):
        return
    root_depth = root.rstrip(os.sep).count(os.sep)
    for dirpath, dirnames, filenames in os.walk(root):
        depth = dirpath.rstrip(os.sep).count(os.sep) - root_depth
        if depth >= max_depth:
            dirnames[:] = []  # prune: don't descend further from here
        yield dirpath, dirnames, filenames


def _find_kadd_mount(search_roots, max_depth=6):
    for root in search_roots:
        for dirpath, _dirnames, filenames in _walk_bounded(root, max_depth):
            if os.path.basename(dirpath) == "kadd" and "__init__.py" in filenames:
                return os.path.dirname(dirpath)
    return None


def _find_competition_root(search_roots, max_depth=4):
    """Requires an actual train_series/ SUBDIRECTORY (the bulk DICOM tree),
    not just the two CSVs by name — matching on filenames alone produces a
    false positive against kadd-package's own bundled copies of train.csv/
    train_series.csv (see scripts/kaggle_push_dataset.py EXTRA_FILES) once
    the search falls back to walking all of /kaggle/input (which it does
    now that competition_sources is usually empty, see CONFIG_PATH section
    below). Confirmed as a real bug 2026-08-19: without this check,
    data_path resolved into kadd-package/data/train_series (never exists
    there), and every raw-DICOM-fallback attempt failed with ENOENT."""
    for root in search_roots:
        for dirpath, dirnames, filenames in _walk_bounded(root, max_depth):
            if ("train.csv" in filenames and "train_series.csv" in filenames
                    and "train_series" in dirnames):
                return dirpath
    return None


def _find_precomputed_mount(search_roots, max_depth=3):
    """Finds the kadd-precomputed-volumes dataset mount (top level holds
    manifest.jsonl + shard_XX/ dirs — see scripts/kaggle_push_precomputed_dataset.py).
    Shallow max_depth on purpose: this must find the TOP of the mount, not
    wander into a shard_XX/<study>/ subtree looking for it."""
    for root in search_roots:
        for dirpath, _dirnames, filenames in _walk_bounded(root, max_depth):
            if "manifest.jsonl" in filenames:
                return dirpath
    return None


# Set by scripts/kaggle_push_kernel.py's staging step, patched to the local
# data/precomputed/manifest.jsonl's line count at push time — the true floor
# for how many "ok" series a genuinely up-to-date kadd-precomputed-volumes
# mount should show. 0 is the safe default for a manual/unpatched push (no
# freshness check, accept whatever's mounted). See
# _find_precomputed_mount_with_retry's docstring for the real bug this fixes.
EXPECTED_PRECOMPUTED_DONE = 0


def _precomputed_dataset_dir_visible(search_roots, max_depth=4):
    """True if a directory literally named 'kadd-precomputed-volumes' is
    visible anywhere under search_roots, regardless of whether its content
    has finished syncing. Same pattern as
    kaggle_kernel_precompute/precompute_kernel.py's function of the same
    name — distinguishes "genuinely not attached" from "attached but still
    syncing"."""
    for root in search_roots:
        for dirpath, dirnames, _filenames in _walk_bounded(root, max_depth):
            if "kadd-precomputed-volumes" in dirnames or \
               os.path.basename(dirpath) == "kadd-precomputed-volumes":
                return True
    return False


def _count_manifest_ok(mount_dir):
    """Counts status=="ok" entries in <mount_dir>/manifest.jsonl — mirrors
    exactly what kadd/classify/data.py's load_precomputed_manifest will
    later count as usable, so this freshness check measures the thing that
    actually matters for training, not just "does a file exist"."""
    if mount_dir is None:
        return 0
    manifest_path = os.path.join(mount_dir, "manifest.jsonl")
    if not os.path.isfile(manifest_path):
        return 0
    n_ok = 0
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("status") == "ok":
                n_ok += 1
    return n_ok


def _find_precomputed_mount_with_retry(search_roots, retries=10, delay_seconds=30,
                                        expected_min_done=0):
    """Guards against the exact bug hit on kernel v59 (2026-09-07): the
    training run found A kadd-precomputed-volumes mount immediately and
    accepted it, but the manifest it read back reflected a stale/still-
    syncing snapshot — verified directly against the local manifest that
    every series the run failed on was actually marked "status": "ok" in
    what had genuinely been pushed. Since competition_sources (raw DICOM
    fallback) is deliberately not attached (see this file's header), a
    stale/empty precomputed mount doesn't degrade gracefully here — it means
    EVERY series fails to load, so the run silently burns real GPU quota
    (confirmed: 0.33h consumed) producing zero real training. This is the
    same class of mount-propagation-lag bug already root-caused and fixed
    for the (CPU-only, cheap-to-retry) precompute kernel — see
    kaggle_kernel_precompute/precompute_kernel.py's function of the same
    name — ported here.

    Deliberately SHORT retry budget compared to the precompute kernel's 20
    minutes: this kernel holds a billed GPU session the whole time it
    waits, so retries*delay_seconds defaults to 5 minutes, not 20. If the
    mount still isn't fresh after that, the caller raises rather than
    silently proceeding — unlike the precompute kernel, "proceed anyway
    with a stale mount" is not an acceptable fallback here (it would mean
    hours of GPU time spent training on nothing, not just some duplicate
    CPU work).
    """
    if not _precomputed_dataset_dir_visible(search_roots):
        return None  # genuinely not attached — not a lag issue, let the caller's existing warning handle it
    for attempt in range(retries):
        mount = _find_precomputed_mount(search_roots)
        if mount is not None:
            n_done = _count_manifest_ok(mount)
            if n_done >= expected_min_done:
                print(f"Found precomputed-volumes mount at {mount} with {n_done} "
                      f"series confirmed fresh (expected >= {expected_min_done}).")
                return mount
            print(f"Found a precomputed-volumes mount at {mount} but it only shows {n_done} "
                  f"'ok' series (expected at least {expected_min_done} from the latest push) — "
                  f"stale/still-syncing (attempt {attempt + 1}/{retries}), retrying in "
                  f"{delay_seconds}s...")
        else:
            print(f"kadd-precomputed-volumes directory is visible but manifest.jsonl not found "
                  f"yet (attempt {attempt + 1}/{retries}) — likely still syncing, retrying in "
                  f"{delay_seconds}s...")
        time.sleep(delay_seconds)
    mount = _find_precomputed_mount(search_roots)
    n_done = _count_manifest_ok(mount)
    raise RuntimeError(
        f"kadd-precomputed-volumes mount never showed a fresh manifest after "
        f"{retries * delay_seconds}s ({'no mount found' if mount is None else f'{mount} stuck at {n_done} done, expected >= {expected_min_done}'}). "
        f"Refusing to proceed and silently train on a stale/incomplete precomputed cache — "
        f"raw DICOM fallback is unavailable (competition_sources is deliberately unattached, "
        f"see this file's header), so a stale mount means literally every series fails to load. "
        f"Wait longer after the last kaggle_push_precomputed_dataset.py push before re-pushing "
        f"this kernel, or check `kaggle datasets status pieteroehlers/kadd-precomputed-volumes`."
    )


def _find_pretrained_mount(search_roots, max_depth=3):
    """Finds the kadd-pretrained-weights dataset mount (top level holds the
    swin3d_*.pth Kinetics-400 checkpoint files directly — see
    scripts/kaggle_push_pretrained_dataset.py)."""
    for root in search_roots:
        for dirpath, _dirnames, filenames in _walk_bounded(root, max_depth):
            if any(f.startswith("swin3d_") and f.endswith(".pth") for f in filenames):
                return dirpath
    return None


def _populate_torch_hub_cache(pretrained_mount):
    """Copies bundled swin3d_*.pth files into torch's hub checkpoint cache
    BEFORE the model is built, so torchvision's weights=... loading finds a
    cache hit and never attempts a network download. This is what makes
    classify.model.pretrained=True work identically regardless of
    enable_internet — needed for the eventual no-internet submission kernel
    (see file header), not just this dev kernel. Filenames must match
    torchvision's own checkpoint URLs exactly (see
    scripts/kaggle_push_pretrained_dataset.py EXPECTED_FILES) for the cache
    hit to register; torch.hub.load_state_dict_from_url matches on basename."""
    import torch
    checkpoints_dir = os.path.join(torch.hub.get_dir(), "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    copied = []
    for fname in os.listdir(pretrained_mount):
        if fname.endswith(".pth"):
            dst = os.path.join(checkpoints_dir, fname)
            if not os.path.isfile(dst):
                shutil.copy(os.path.join(pretrained_mount, fname), dst)
            copied.append(fname)
    return copied


INPUT_ROOT = "/kaggle/input"
print(f"{INPUT_ROOT} top-level: {_listdir_safe(INPUT_ROOT)}")
for _sub in ("datasets", "competitions", "models"):
    _p = os.path.join(INPUT_ROOT, _sub)
    if os.path.isdir(_p):
        print(f"{_p} top-level: {_listdir_safe(_p)}")

# Prefer the categorized subdirs Kaggle actually uses (per the attempt-2 log),
# fall back to a bounded search of all of /kaggle/input in case that
# categorization isn't always present.
_dataset_roots = [os.path.join(INPUT_ROOT, "datasets"), INPUT_ROOT]
_competition_roots = [os.path.join(INPUT_ROOT, "competitions"), INPUT_ROOT]

KADD_MOUNT = _find_kadd_mount(_dataset_roots)
if KADD_MOUNT is None:
    raise RuntimeError(
        "Could not find a mounted kadd/ package (a directory named 'kadd' "
        f"containing __init__.py) under {_dataset_roots}. "
        f"/kaggle/input top-level: {_listdir_safe(INPUT_ROOT)}. "
        "Check that scripts/kaggle_push_dataset.py ran successfully and that "
        "kernel-metadata.json's dataset_sources references the right dataset id."
    )
print(f"Found kadd package at: {KADD_MOUNT}")
sys.path.insert(0, KADD_MOUNT)

COMPETITION_DATA_ROOT = _find_competition_root(_competition_roots)
if COMPETITION_DATA_ROOT is None:
    # No longer fatal — see kernel-metadata.json, competition_sources was
    # deliberately dropped (attaching the full ~535GB raw-DICOM competition
    # dataset was the dominant cost of this kernel's slow session-startup
    # mount time, confirmed 2026-08-19: an empty kernel log at cancel time
    # showed the run was still stuck in Kaggle's dataset-attach phase before
    # any of our code had even executed). train.csv/train_series.csv now
    # ride along in kadd-package instead (see scripts/kaggle_push_dataset.py
    # EXTRA_FILES), so they don't require the competition mount at all. Raw
    # DICOM (data_path below) becomes genuinely unavailable without this
    # mount — fine as long as the precompute pipeline actually covers every
    # series a run touches; KneeDataset's DICOM fallback already degrades
    # gracefully (skips the series, logs a warning) rather than crashing if
    # it's ever hit with no mount present.
    print("No competition data mount found (competition_sources not attached — "
          "expected, see kernel-metadata.json). Raw DICOM fallback will be "
          "unavailable; relying entirely on precomputed volumes + the CSVs "
          "bundled in kadd-package.")
else:
    print(f"Found competition data at: {COMPETITION_DATA_ROOT}")
    print(f"  contents: {_listdir_safe(COMPETITION_DATA_ROOT)}")

os.system(f"{sys.executable} -m pip install -q pydicom psutil")

import torch  # noqa: E402

from kadd.classify.data import KneeDataLoader  # noqa: E402
from kadd.classify.model import build_model  # noqa: E402
from kadd.classify.train import ClassifierTraining  # noqa: E402
from kadd.util.logger import init_logger, separator  # noqa: E402
from kadd.util.parse import parse  # noqa: E402

CONFIG_PATH = f"{KADD_MOUNT}/config_files/classify_cnn3d_light_probe.yml"

params = parse(CONFIG_PATH)

# train.csv/train_series.csv: prefer the copies bundled in kadd-package
# (scripts/kaggle_push_dataset.py EXTRA_FILES) over the competition mount —
# avoids ever needing the full ~535GB competition dataset attached just to
# read two small CSVs. Falls back to the competition mount if the bundled
# copies aren't there for some reason (e.g. an older kadd-package push) and
# that mount happens to be attached.
_kadd_labels_csv = f"{KADD_MOUNT}/data/train.csv"
_kadd_series_csv = f"{KADD_MOUNT}/data/train_series.csv"
if os.path.isfile(_kadd_labels_csv) and os.path.isfile(_kadd_series_csv):
    print(f"Using train.csv/train_series.csv bundled in kadd-package at {KADD_MOUNT}/data")
    params['classify']['data']['labels_csv'] = _kadd_labels_csv
    params['classify']['data']['series_csv'] = _kadd_series_csv
elif COMPETITION_DATA_ROOT is not None:
    print("train.csv/train_series.csv not found in kadd-package — falling back to the "
          "competition mount for them.")
    params['classify']['data']['labels_csv'] = f"{COMPETITION_DATA_ROOT}/train.csv"
    params['classify']['data']['series_csv'] = f"{COMPETITION_DATA_ROOT}/train_series.csv"
else:
    raise RuntimeError(
        "Neither kadd-package's bundled train.csv/train_series.csv nor a competition "
        "mount are available — nothing to train on. Re-push kadd-package "
        "(scripts/kaggle_push_dataset.py) or attach competition_sources.")

# Raw DICOM tree: only available if the competition mount is attached (now
# optional, see above). Points at a path that simply won't exist if not —
# KneeDataset's DICOM fallback already handles a missing series_dir
# gracefully (FileNotFoundError is a recoverable/skip-this-series case).
params['classify']['data']['data_path'] = (
    f"{COMPETITION_DATA_ROOT}/train_series" if COMPETITION_DATA_ROOT is not None
    else "/kaggle/input/__no_raw_dicom_mount_attached__"
)

if params['classify']['data'].get('use_silver_labels'):
    params['classify']['data']['silver_labels_csv'] = f"{KADD_MOUNT}/data/report_labels_llm.csv"

# resume_from (added 2026-09-03, see kadd/classify/train.py's
# load_checkpoint_for_resume): config holds a relative path exactly like
# silver_labels_csv above; rewrite it to the actual kadd-package mount path.
# Warn-and-clear rather than raise if the file isn't there (e.g. the
# resume_checkpoint.pth wasn't staged for this push) — a config that
# requests resume but finds nothing should fall back to training from
# scratch, not crash the whole run.
_resume_from = params['classify']['train'].get('resume_from')
if _resume_from:
    _resume_path = f"{KADD_MOUNT}/{_resume_from}"
    if os.path.isfile(_resume_path):
        print(f"Found resume checkpoint at: {_resume_path}")
        params['classify']['train']['resume_from'] = _resume_path
    else:
        print(f"WARNING: resume_from={_resume_from!r} requested but not found at "
              f"{_resume_path} — training from scratch instead.")
        params['classify']['train']['resume_from'] = None

# If this config wants precomputed volumes (CLAUDE.md sec 7 precompute
# pipeline), find the kadd-precomputed-volumes mount and point at it instead
# of the config's local `data/precomputed` path. Missing mount isn't fatal —
# KneeDataset just falls back to raw DICOM for every series in that case
# (see load_precomputed_manifest's own no-manifest-found handling), so this
# only warns rather than raising, in case the dataset isn't attached yet.
if params['classify']['data'].get('precomputed_dir'):
    precomputed_mount = _find_precomputed_mount_with_retry(
        _dataset_roots, expected_min_done=EXPECTED_PRECOMPUTED_DONE)
    if precomputed_mount is None:
        print("WARNING: config requests precomputed_dir but no kadd-precomputed-volumes "
              "mount was found (checked for a manifest.jsonl under "
              f"{_dataset_roots}) — falling back to raw DICOM decode for every series. "
              "Check kernel-metadata.json's dataset_sources.")
        params['classify']['data']['precomputed_dir'] = None
    else:
        print(f"Found precomputed-volumes mount at: {precomputed_mount}")
        params['classify']['data']['precomputed_dir'] = precomputed_mount

    # Local write-through cache (kadd/classify/data.py's KneeDataset) — TRIED
    # and REVERTED, 2026-08-20. Two real runs on real Kaggle hardware:
    #   - kernel v27: assumed /kaggle/tmp exists (an old, never-directly-
    #     confirmed community-thread reference) — WRONG, log showed
    #     "/kaggle/tmp not present", cache silently disabled itself, zero
    #     effect (data loading unchanged from the no-cache baseline).
    #   - kernel v28: fixed to probe candidates + confirm real writability
    #     (not just os.path.isdir), found /tmp genuinely writable — but
    #     epoch 1 data-wait jumped to 561.6s (84% of step time, up from the
    #     ~216-243s/57-69% no-cache baseline) and epoch 2 only improved to
    #     314.5s, still meaningfully worse than baseline. /tmp reported a
    #     suspicious 8.6TB total size — implausible for a container's real
    #     local scratch disk, strongly suggesting /tmp on this Kaggle kernel
    #     type is itself network-backed, not local, so every cache write was
    #     pure added cost (a second network write) with no compensating
    #     faster-read benefit.
    # Conclusion: no confirmed-genuinely-local, confirmed-fast scratch path
    # exists on this Kaggle kernel type. ENABLE_LOCAL_CACHE stays False —
    # the probing/write-through code below (and in KneeDataset) is left
    # intact, not deleted, in case a genuinely local path is ever found
    # (e.g. a different Kaggle accelerator/instance type), but do not
    # re-enable this without first directly benchmarking read/write
    # throughput against the candidate path — a written vs. skipped
    # comparison, not another size/existence check, since that's exactly
    # what failed to predict this outcome twice.
    ENABLE_LOCAL_CACHE = False

    _CACHE_ROOT_CANDIDATES = ["/kaggle/tmp", "/tmp"]

    def _find_writable_cache_root(candidates):
        for root in candidates:
            if not os.path.isdir(root):
                print(f"  candidate {root}: not present")
                continue
            test_path = os.path.join(root, ".kadd_write_test")
            try:
                with open(test_path, "w") as f:
                    f.write("x")
                os.remove(test_path)
            except OSError as e:
                print(f"  candidate {root}: present but not writable ({e})")
                continue
            return root
        return None

    if not ENABLE_LOCAL_CACHE:
        print("Local data cache disabled (ENABLE_LOCAL_CACHE=False) — reverted 2026-08-20 "
              "after real evidence it made data loading slower, not faster (see code comment).")
    else:
        print("Probing for a writable local scratch dir for the data cache...")
        _cache_root = _find_writable_cache_root(_CACHE_ROOT_CANDIDATES)
        if _cache_root is None:
            print(f"WARNING: no writable local scratch dir found among {_CACHE_ROOT_CANDIDATES} "
                  f"— local caching disabled, every epoch will keep re-reading precomputed "
                  f"volumes from the remote mount")
        else:
            try:
                _usage = shutil.disk_usage(_cache_root)
                print(f"Using {_cache_root} — disk usage: {_usage.free / 1e9:.1f}GB free / "
                      f"{_usage.total / 1e9:.1f}GB total")
            except OSError as e:
                print(f"WARNING: could not stat {_cache_root} ({e}) — proceeding anyway")
            params['classify']['data']['local_cache_dir'] = f"{_cache_root}/kadd_cache"
            print(f"Local cache enabled at {params['classify']['data']['local_cache_dir']}")

# Kaggle kernels run with /kaggle/working as cwd; this lands in its output.
run_name = f"output/{params['name']}"
os.makedirs(run_name, exist_ok=True)
params['name'] = run_name

init_logger(fn=run_name, verbose=True)
logging.info(f'{socket.gethostname()}: kadd classify-train starting on Kaggle kernel')
separator()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Device: {device}")

if params['classify']['model'].get('pretrained'):
    pretrained_mount = _find_pretrained_mount(_dataset_roots)
    if pretrained_mount is None:
        print("WARNING: classify.model.pretrained=True but no kadd-pretrained-weights "
              f"mount was found under {_dataset_roots} — torchvision will attempt to "
              "download Kinetics-400 weights from the network instead (fine only if "
              "enable_internet=true; would fail on a no-internet kernel). Check "
              "kernel-metadata.json's dataset_sources and "
              "scripts/kaggle_push_pretrained_dataset.py.")
    else:
        _copied = _populate_torch_hub_cache(pretrained_mount)
        print(f"Found pretrained-weights mount at: {pretrained_mount} — pre-populated "
              f"torch hub cache with: {_copied} (torchvision will use these instead of "
              "downloading)")

data_loader = KneeDataLoader(params)
model = build_model(params)
ClassifierTraining(params, model, data_loader.data, device).main()
