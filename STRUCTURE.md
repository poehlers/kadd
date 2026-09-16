# KADD: Project Structure & Usage Guide

## Overview

**KADD** trains an attention-MIL Swin3D classifier for the RSNA Knee
Abnormality Detection Kaggle competition: 12 binary knee-abnormality findings
per MRI study, scored by macro-averaged ROC-AUC. The model has two
components:

- **Per-series encoder** — a 3D Swin Transformer (`swin3d_t/s/b`, via
  `torchvision.models.video`) that embeds one MRI series (a 3D volume) into a
  fixed-size summary vector, with acquisition metadata (plane,
  fluid-sensitivity, fat-suppression) mixed in.
- **Attention-MIL pooling** — a gated-attention layer (Ilse et al. 2018-style)
  that combines every series in a study into one pooled representation, since
  no single plane serves all 12 labels well and studies have a variable
  number of series (3–14 locally).

Training is single-stage (encoder + pooling + classification head trained
end-to-end together) — unlike density-field SBI pipelines that split into
encoder/flow stages, this is direct multi-label classification with
`BCEWithLogitsLoss`.

This project's training harness (config-YAML-driven, `output/<name>_<ts>/`
run directories, `python -m <pkg> <verb> <config>` CLI) is adapted from
[poehlers/DenPIE](https://github.com/poehlers/DenPIE); see `CLAUDE.md` for
what was ported as-is vs. deliberately changed and why (mainly: no periodic
augmentation, DICOM windowing instead of z-score normalization, and
attention-MIL over all series instead of single-field input).

---

## Directory layout

```
KADD/
├── kadd/                              # Python package
│   ├── __main__.py                     # CLI entry point
│   ├── classify/
│   │   ├── model.py                     # Swin3DSeriesEncoder, SeriesMetaEmbed,
│   │   │                                 #   GatedAttentionMIL, KneeClassifier, build_model()
│   │   ├── data.py                      # DICOM loading (slice ordering, windowing,
│   │   │                                 #   crop/resize/resample), KneeDataset, collate_fn
│   │   └── train.py                     # ClassifierTraining — single-stage training loop
│   └── util/
│       ├── logger.py                    # Logging setup
│       └── parse.py                     # YAML parsing, output-dir helpers
│
├── config_files/
│   └── classify_swin3d_v1.yml          # v1 baseline config
│
├── kaggle_kernel/                      # Remote-training-on-Kaggle assets
│   ├── dataset-metadata.json            # metadata for the pushed kadd/ code dataset
│   ├── kernel-metadata.json             # metadata for the pushed training kernel
│   └── train_kernel.py                  # runs on Kaggle's infra (GPU, attaches the
│                                         #   code dataset + competition data)
│
├── scripts/
│   ├── download_data.py                # local competition data download (kagglehub)
│   ├── kaggle_push_dataset.py           # push kadd/+config_files/ as a Kaggle dataset
│   ├── kaggle_push_kernel.py            # push + poll the remote training kernel
│   └── smoke_test.sh                    # wraps tests/test_smoke.py
│
├── tests/
│   └── test_smoke.py                   # model shape test (always runs) + DICOM
│                                         #   pipeline tests (skip cleanly if no local data)
│
├── data/                                # gitignored — CSVs + bulk DICOM tree
├── output/                              # gitignored — training run artifacts
├── docs/                                # reserved; empty for now (see note below)
├── setup.py / requirements.txt
├── CLAUDE.md                            # running project notes: competition rules/data
│                                         #   details, decisions and why, current status
├── README.md                            # quick start
└── STRUCTURE.md                         # this file
```

`docs/` is intentionally empty right now rather than holding a Sphinx site
like DenPIE's — that's real infrastructure (conf.py, a GitHub Actions publish
workflow, a hosted site) that's premature before there's a completed training
run or a second contributor to write for. `README.md` + `STRUCTURE.md` are
the docs for now; revisit if/when this project needs a published site.

---

## CLI

```
python -m kadd classify-train <config.yml> [--verbose]
python -m kadd classify-train --type swin3d --size xlight [--verbose]
```

The `--type`/`--size` form autodetects the matching config under
`config_files/` (`classify_<type>_<size>.yml` —
`kadd/util/config_registry.py`) instead of typing the path — only works for
configs following that naming convention. The legacy `classify_swin3d_v1.yml`
/`v2.yml` predate it and still need an explicit path (`--config-dir` overrides
the search directory if needed).

| Subcommand        | What it does                                          |
|--------------------|--------------------------------------------------------|
| `classify-train`  | Train the attention-MIL classifier (swin3d ViT or cnn3d, 4 size tiers) |

Each run creates `output/<name>_<YYYYMMDD_HHMMSS>/` (from the config's `name`
field) plus an `output/<name>_latest` symlink, containing:

```
output/<name>_<timestamp>/
├── log                                # training log
├── classify_swin3d_v1.yml             # copy of the config used
├── loss/
│   ├── loss.pdf / loss.npy             # train/val BCE loss curves
│   └── macro_auc.pdf / macro_auc.npy   # validation macro-AUC per epoch
└── models/
    ├── best.pth                        # checkpoint at best macro-AUC
    ├── final.pth                       # checkpoint at end of training
    └── epoch_N.pth                     # periodic checkpoints (every 10 epochs)
```

---

## Config reference (`classify` mode)

```yaml
name: classify_swin3d_light

classify:
  model:
    type: swin3d             # swin3d (ViT, torchvision Swin3D-based) | cnn3d (ported from DenPIE)
    size: light               # xlight | light | medium | large — see kadd/classify/model.py
                               #   SWIN3D_XLIGHT_KWARGS/SWIN3D_CANONICAL_SIZES (swin3d) or
                               #   CNN3D_CHANNELS (cnn3d) for what each tier actually is
    pretrained: false         # Kinetics-400 weights, swin3d only, light/medium/large only —
                               #   only reuses the deeper blocks, see model.py; xlight has no
                               #   matching weights and errors if pretrained: true is set
    summary_dim: 256          # per-series embedding size
    meta_dim: 16              # plane/sequence metadata embedding size
    use_checkpointing: true   # gradient checkpointing (memory vs. speed)
    proj_dropout: 0.1         # swin3d only

  data:
    data_path: data/train_series/     # <StudyInstanceUID>/<SeriesInstanceUID>/*.dcm tree
    labels_csv: data/train.csv
    series_csv: data/train_series.csv
    label_columns: [...]              # the 12 target column names, in submission order
    gold_only: false                  # true: train only on the 58 directly-labeled studies (v1)
    use_silver_labels: true           # false+use_silver_labels: train on report-derived labels
    silver_labels_csv: data/report_labels_llm.csv  #   (gold held out entirely as val — sec 8)
    max_series: 8                     # cap on series/study fed to attention-MIL pooling
    target_depth: 32                  # slices resampled to this count per series
    target_size: 224                  # H=W resize per slice
    val_split: 0.2                    # unused when use_silver_labels: true
    num_workers: 4

  train:
    epochs: 100
    batch_size: 2                     # small: multi-series 3D volumes are memory-heavy
    lr: 1.0e-5
    optimizer: AdamW                  # AdamW | Adam
    weight_decay: 0.01
    scheduler: ReduceLROnPlateau      # ReduceLROnPlateau | StepLR | CosineAnnealingLR
    scheduler_params: {factor: 0.3, patience: 10, min_lr: 1.0e-7}
    gradient_accumulation_steps: 4
    grad_clip: 1.0
    patience: 20                      # early-stopping patience, in epochs
```

**8 ready-made size-tier configs** live in `config_files/`:
`classify_{swin3d,cnn3d}_{xlight,light,medium,large}.yml` — same data setup
(silver labels, gold held out as val) across all 8, `batch_size`/`max_series`/
`gradient_accumulation_steps` scaled per tier. Only the swin3d `light` tier
(equivalent to `classify_swin3d_v2.yml`) has been confirmed OOM-free on real
2x-Tesla-T4 hardware — the other 7 are reasoned starting points; watch each
config's own comments plus the per-epoch GPU memory log
(`kadd/classify/train.py`'s `_log_gpu_memory`) on its first real run. `v1`/`v2`
are kept around unchanged for reproducibility of the runs already logged in
CLAUDE.md.

**Note on `macro_auc`**: at very small data sizes (e.g. `gold_only: true`,
~58 labeled studies), macro-AUC is a pipeline-correctness signal, not a
leaderboard-meaningful number — some labels are near-certain to have only one
class present in a given validation split, and the training loop skips those
from the macro average rather than crashing (see `train.py`'s `val_epoch`
docstring). Don't read a high or low value here as a claim about real model
quality until the training set grows beyond the 58 gold-labeled studies.

---

## Remote training on Kaggle kernels

Kaggle kernels run in an isolated cloud container that can't see this local
checkout, so `kadd/` + `config_files/` are pushed as a private Kaggle
**dataset** that the training kernel attaches and imports from.

```bash
# 1. Push the code as a dataset (re-run this after any kadd/ or config change)
python scripts/kaggle_push_dataset.py

# 2. Push + run the training kernel on Kaggle's GPU (prints your weekly
#    GPU/TPU quota first via `kaggle quota`)
python scripts/kaggle_push_kernel.py

# 2b. ...or push and stream live logs instead of just checking status once
python scripts/kaggle_push_kernel.py --follow

# 3. Check on it later / pull results once it finishes
python scripts/kaggle_push_kernel.py --status
python scripts/kaggle_push_kernel.py --status --logs
```

`kaggle kernels output` (what `--logs` calls) does **not** block until the
run finishes — it just grabs whatever's available from the latest run right
now. Use `--follow` to stream logs live instead of guessing when to check.

The push explicitly requests the `NvidiaTeslaT4` accelerator rather than
relying on `kernel-metadata.json`'s bare `enable_gpu: true` — a bare
`enable_gpu` with no accelerator specified defaults to `NvidiaTeslaP100`,
which is broken against Kaggle's current default PyTorch image
(`cudaErrorNoKernelImageForDevice` on the first real CUDA op, despite
`torch.cuda.is_available()` reporting `True`). If you ever edit
`scripts/kaggle_push_kernel.py`, keep the `--accelerator NvidiaTeslaT4` flag.

This is a **development/training kernel, not the final competition-submission
kernel** — `kaggle_kernel/kernel-metadata.json` has `enable_internet: true`
(needed for `pip install pydicom`, which isn't preinstalled on Kaggle's base
image). A real submission notebook must run with internet disabled per the
competition rules, which means `pydicom` would need to be bundled a different
way (e.g. as a wheel-file dataset) — not built yet, since v1 hasn't gotten to
a submission-worthy model.

`kaggle_kernel/train_kernel.py` overrides the config's local `data/` paths to
Kaggle's mounted competition data
(`/kaggle/input/rsna-knee-abnormality-detection/...`). That mounted path
layout is **unconfirmed until the kernel actually runs once** — the script
logs `os.listdir()` of the mount root on startup specifically so a first run
tells you whether the sub-paths need adjusting.

---

## Local data layout expected by `config_files/classify_swin3d_v1.yml`

```
data/
├── train.csv
├── train_series.csv
├── test.csv
├── test_series.csv
├── sample_submission.csv
└── train_series/
    └── <StudyInstanceUID>/
        └── <SeriesInstanceUID>/
            └── <SOPInstanceUID>.dcm
```

Populated by `python scripts/download_data.py` (needs a Kaggle account that
has joined the competition, plus an API credential — see the script's
docstring, or `CLAUDE.md` §7 for the full manual-steps checklist).
