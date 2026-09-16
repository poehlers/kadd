#!/usr/bin/env bash
# Smoke-test the kadd v1 classification pipeline.
#
#   bash scripts/smoke_test.sh
#
# DICOM-dependent checks skip cleanly if data/ has no local DICOM files yet
# (see tests/test_smoke.py) — this is expected before scripts/download_data.py
# has been run.
set -euo pipefail
cd "$(dirname "$0")/.."

exec python -m pytest tests/test_smoke.py -v "$@"
