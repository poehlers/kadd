#!/usr/bin/env python3
"""Push the Swin3D Kinetics-400 pretrained checkpoints to Kaggle as their own
small, rarely-changing dataset, so training kernels (and eventually the
no-internet submission kernel) never need to download them from
download.pytorch.org.

Why this exists: the training kernel currently has enable_internet=true, so
`classify.model.pretrained: true` works today via torchvision's normal
download-on-first-use into ~/.cache/torch/hub/checkpoints/. But the actual
competition-submission notebook has internet disabled during scoring
(CLAUDE.md sec 1), so that download would simply fail there. Pre-bundling the
exact checkpoint files torchvision expects, and having train_kernel.py copy
them into torch's hub cache before the model is built (see
_populate_torch_hub_cache in train_kernel.py), makes pretrained weights work
identically whether or not internet is available — fixed now, not deferred
until the submission kernel is actually built.

These files never change (they're a fixed, versioned release from
pytorch.org, not project code), so this is pushed once and rarely touched —
same pattern as kaggle_push_precomputed_dataset.py, not
kaggle_push_dataset.py's push-on-every-iteration pattern.

Usage:
    python scripts/kaggle_push_pretrained_dataset.py
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_ID = "pieteroehlers/kadd-pretrained-weights"
WEIGHTS_DIR = REPO_ROOT / "data" / "pretrained_weights"

# Exact filenames torchvision's Swin3D_*_Weights.KINETICS400_V1 URLs use —
# must match exactly, since train_kernel.py places them under this same name
# in torch's hub cache dir for a cache-hit to register.
EXPECTED_FILES = [
    "swin3d_t-7615ae03.pth",  # size=light
    "swin3d_s-da41c237.pth",  # size=medium
    "swin3d_b_1k-24f7c7c6.pth",  # size=large
]


def main():
    missing = [f for f in EXPECTED_FILES if not (WEIGHTS_DIR / f).exists()]
    if missing:
        print(f"ERROR: missing checkpoint file(s) in {WEIGHTS_DIR}: {missing}", file=sys.stderr)
        print("Download them first, e.g.:", file=sys.stderr)
        print("  curl -L -o data/pretrained_weights/<name> "
              "https://download.pytorch.org/models/<name>", file=sys.stderr)
        sys.exit(1)

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / "kadd-pretrained-weights"
        stage.mkdir()

        for fname in EXPECTED_FILES:
            shutil.copy(WEIGHTS_DIR / fname, stage / fname)

        metadata = {
            "title": "kadd pretrained weights",
            "id": DATASET_ID,
            "licenses": [{"name": "CC0-1.0"}],
        }
        import json
        (stage / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2))

        version_cmd = ["kaggle", "datasets", "version", "-p", str(stage),
                        "-m", "Update pretrained weights", "-r", "zip", "-d"]
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
