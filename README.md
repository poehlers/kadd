# KADD — RSNA Knee Abnormality Detection

An attention-MIL Swin3D classification baseline for the
[RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
Kaggle competition: predicting 12 clinically important knee-abnormality
findings (ACL, MCL, meniscus tears, three compartments of osteoarthritis,
effusion, synovitis, Baker's cyst, contusion, fracture) per knee-MRI study,
scored by macro-averaged ROC-AUC.

See [STRUCTURE.md](STRUCTURE.md) for the full directory layout, CLI usage,
and configuration reference. See [CLAUDE.md](CLAUDE.md) for the running
project notes (competition rules/data details, decisions made and why,
current status) — that file is the source of truth for *why* things are
built the way they are; this README and STRUCTURE.md are the stable
"how to use it" reference.

## Quick start

```bash
# 1. Activate the environment (already provisioned)
source kaddvenv/bin/activate

# 2. Get the competition data (needs a Kaggle account that has joined the
#    competition, plus an API credential — see STRUCTURE.md)
python scripts/download_data.py

# 3. Sanity-check the pipeline (passes without data; DICOM checks activate
#    once step 2 has run)
bash scripts/smoke_test.sh

# 4. Train locally
python -m kadd classify-train config_files/classify_swin3d_v1.yml --verbose
```

To train remotely on Kaggle's free GPU quota instead of locally, see the
"Remote training on Kaggle kernels" section of [STRUCTURE.md](STRUCTURE.md).

## Approach, in one paragraph

Each study is a bag of ~5.5 MRI series across different planes (sagittal,
coronal, axial) and sequence types — no single plane shows evidence for all
12 findings. A shared Swin3D (3D Vision Transformer) encoder embeds every
series in a study; a gated attention-MIL layer pools them into one per-study
prediction, letting the model learn which series carries evidence for which
finding rather than that being hand-picked. v1 trains only on the 58
directly-labeled studies (of 4,407 total) as a "does the pipeline work
end-to-end" baseline — see CLAUDE.md for what's explicitly out of scope
(report-derived weak labels, anomaly localization, Optuna HPO) and why.
