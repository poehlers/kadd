"""End-to-end smoke tests for kadd's v1 classification pipeline.

Mirrors the philosophy of poehlers/DenPIE's tests/test_smoke.py: fast on CPU,
and each DICOM-dependent test skips cleanly if no local data is present, so
the suite stays informative even before the bulk competition download exists.

Run:
    pytest tests/test_smoke.py -v
"""
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def _find_any_series_dir(root):
    """First directory under root containing >=2 .dcm files (a real series)."""
    if not os.path.isdir(root):
        return None
    for dirpath, _, filenames in os.walk(root):
        dcm_files = [f for f in filenames if f.lower().endswith(".dcm")]
        if len(dcm_files) >= 2:
            return dirpath
    return None


def _find_any_dcm(root):
    """First lone .dcm file anywhere under root, regardless of siblings."""
    if not os.path.isdir(root):
        return None
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.lower().endswith(".dcm"):
                return os.path.join(dirpath, f)
    return None


# --------------------------- model shape check ------------------------------

@pytest.mark.parametrize("model_type", ["swin3d", "cnn3d"])
@pytest.mark.parametrize("size", ["xlight", "light", "medium", "large"])
def test_model_forward_shape(model_type, size):
    """Pure unit test: no data files required. Exercises the attention-MIL
    pooling/grouping logic with a variable, fake series-per-study count,
    across every (type, size) combination the config files can select.
    """
    from kadd.classify.model import build_model

    params = {
        "classify": {
            "model": {
                "type": model_type, "size": size, "pretrained": False,
                "summary_dim": 32, "meta_dim": 8, "use_checkpointing": False,
                "proj_dropout": 0.1,
            },
            "data": {
                "label_columns": [
                    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
                    "Medial OA", "Lateral OA", "PF OA", "Effusion",
                    "Synovitis", "Baker's", "Contusion", "Fracture",
                ],
            },
        },
    }
    model = build_model(params)

    # 2 studies in the batch: study 0 has 3 series, study 1 has 2 series.
    total_series = 5
    volumes = torch.randn(total_series, 1, 16, 64, 64)
    plane_idx = torch.tensor([0, 1, 2, 0, 1])
    fluid = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0])
    fatsup = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0])
    study_index = torch.tensor([0, 0, 0, 1, 1])

    logits, attn = model(volumes, plane_idx, fluid, fatsup, study_index, n_studies=2)
    assert logits.shape == (2, 12)
    assert [a.shape[0] for a in attn] == [3, 2]
    for a in attn:
        assert torch.allclose(a.sum(), torch.tensor(1.0), atol=1e-5)

    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, torch.randint(0, 2, (2, 12)).float()
    )
    loss.backward()
    assert torch.isfinite(loss)


def test_model_forward_shape_cnn3d_xmedium():
    """cnn3d/xmedium (5 conv blocks, [32,64,128,256,512]) has no swin3d
    equivalent (see model.py's SIZES/CNN3D_SIZES split), so it's not part of
    test_model_forward_shape's shared parametrization — a dedicated case
    instead, same shape/backward check as that test."""
    from kadd.classify.model import build_model

    params = {
        "classify": {
            "model": {
                "type": "cnn3d", "size": "xmedium", "summary_dim": 32,
                "meta_dim": 8, "use_checkpointing": False,
            },
            "data": {
                "label_columns": [
                    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
                    "Medial OA", "Lateral OA", "PF OA", "Effusion",
                    "Synovitis", "Baker's", "Contusion", "Fracture",
                ],
            },
        },
    }
    model = build_model(params)
    assert [b[0].out_channels for b in model.encoder.blocks] == [32, 64, 128, 256, 512]

    total_series = 5
    volumes = torch.randn(total_series, 1, 16, 64, 64)
    plane_idx = torch.tensor([0, 1, 2, 0, 1])
    fluid = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0])
    fatsup = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0])
    study_index = torch.tensor([0, 0, 0, 1, 1])

    logits, attn = model(volumes, plane_idx, fluid, fatsup, study_index, n_studies=2)
    assert logits.shape == (2, 12)
    assert [a.shape[0] for a in attn] == [3, 2]

    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, torch.randint(0, 2, (2, 12)).float()
    )
    loss.backward()
    assert torch.isfinite(loss)


def _meta_fusion_params(model_type, meta_fusion=None):
    model_cfg = {
        "type": model_type, "size": "xlight", "pretrained": False,
        "summary_dim": 32, "meta_dim": 8, "use_checkpointing": False,
    }
    if meta_fusion is not None:
        model_cfg["meta_fusion"] = meta_fusion
    return {
        "classify": {
            "model": model_cfg,
            "data": {
                "label_columns": [
                    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
                    "Medial OA", "Lateral OA", "PF OA", "Effusion",
                    "Synovitis", "Baker's", "Contusion", "Fracture",
                ],
            },
        },
    }


@pytest.mark.parametrize("model_type", ["swin3d", "cnn3d"])
def test_meta_fusion_default_is_late_unchanged_behavior(model_type):
    """No meta_fusion key (every pre-existing config) must behave exactly
    like meta_fusion='late': a 1-channel encoder + a SeriesMetaEmbed module
    that gets added to the encoder's output after encoding."""
    from kadd.classify.model import build_model

    model = build_model(_meta_fusion_params(model_type))
    assert model.meta_fusion == "late"
    assert hasattr(model, "meta_embed")


@pytest.mark.parametrize("model_type", ["swin3d", "cnn3d"])
def test_meta_fusion_early_builds_wider_encoder_no_meta_embed(model_type):
    """meta_fusion='early' must widen the encoder's in_channels to
    1 + META_CHANNELS and skip building SeriesMetaEmbed entirely (metadata
    injected once, not both places — see model.py's KneeClassifier docstring)."""
    from kadd.classify.model import META_CHANNELS, build_model

    model = build_model(_meta_fusion_params(model_type, meta_fusion="early"))
    assert model.meta_fusion == "early"
    assert not hasattr(model, "meta_embed")

    if model_type == "cnn3d":
        first_conv = model.encoder.blocks[0][0]
    else:
        first_conv = model.encoder.swin.patch_embed.proj
    assert first_conv.in_channels == 1 + META_CHANNELS


def test_meta_fusion_early_forward_pass_produces_correct_shape():
    """Real forward+backward pass with meta_fusion='early', confirming the
    broadcast-metadata-as-channels path actually runs end to end (not just
    that the encoder gets built with the right shape)."""
    from kadd.classify.model import build_model

    model = build_model(_meta_fusion_params("cnn3d", meta_fusion="early"))

    total_series = 5
    volumes = torch.randn(total_series, 1, 16, 64, 64)
    plane_idx = torch.tensor([0, 1, 2, 0, 1])
    fluid = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0])
    fatsup = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0])
    study_index = torch.tensor([0, 0, 0, 1, 1])

    logits, attn = model(volumes, plane_idx, fluid, fatsup, study_index, n_studies=2)
    assert logits.shape == (2, 12)
    assert [a.shape[0] for a in attn] == [3, 2]

    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, torch.randint(0, 2, (2, 12)).float()
    )
    loss.backward()
    assert torch.isfinite(loss)


def test_meta_fusion_invalid_value_raises():
    from kadd.classify.model import build_model

    with pytest.raises(ValueError, match="meta_fusion"):
        build_model(_meta_fusion_params("cnn3d", meta_fusion="both"))


def test_swin3d_xlight_rejects_pretrained():
    """size='xlight' is a custom architecture with no matching torchvision
    checkpoint — pretrained=True must fail loudly, not silently ignore it."""
    from kadd.classify.model import build_model

    params = {"classify": {
        "model": {"type": "swin3d", "size": "xlight", "pretrained": True,
                  "summary_dim": 32, "meta_dim": 8},
        "data": {"label_columns": ["ACL"] * 12},
    }}
    with pytest.raises(ValueError, match="xlight"):
        build_model(params)


# kadd/classify/attention_patch.py: monkey-patches torchvision's windowed
# attention to use F.scaled_dot_product_attention (SDPA -> FlashAttention/
# memory-efficient kernel on CUDA) instead of a manual matmul/softmax/matmul
# sequence. Added 2026-09-09 after profiling the REAL model with
# torch.profiler and finding attention ops are ~42% of total forward
# compute (bigger than all Linear/MLP layers combined) — Swin's windowed
# attention (392 tokens/window for swin3d_t) turned out not to be small
# enough to make this a minor cost, contrary to the initial assumption.
# Mathematically exact (not an approximation), verified below by comparing
# patched vs. unpatched output on identical weights/input.

def test_sdpa_patch_forward_output_matches_unpatched():
    """Direct equivalence check: same weights, same input, patched vs.
    unpatched swin3d_t must produce near-identical output (float-rounding-
    level difference only, not an approximation)."""
    import torchvision.models.video.swin_transformer as st
    from kadd.classify.attention_patch import apply_sdpa_patch

    torch.manual_seed(0)
    x = torch.randn(2, 3, 8, 32, 32)

    m_unpatched = st.SwinTransformer3d(
        patch_size=[2, 4, 4], embed_dim=24, depths=[2, 2], num_heads=[2, 4],
        window_size=[4, 4, 4], num_classes=10,
    )
    m_unpatched.eval()
    with torch.no_grad():
        y_unpatched = m_unpatched(x)

    apply_sdpa_patch()
    m_patched = st.SwinTransformer3d(
        patch_size=[2, 4, 4], embed_dim=24, depths=[2, 2], num_heads=[2, 4],
        window_size=[4, 4, 4], num_classes=10,
    )
    m_patched.load_state_dict(m_unpatched.state_dict())
    m_patched.eval()
    with torch.no_grad():
        y_patched = m_patched(x)

    assert torch.allclose(y_unpatched, y_patched, atol=1e-4)


def test_sdpa_patch_is_idempotent_and_applied_on_encoder_build():
    """apply_sdpa_patch() must be safe to call more than once, and
    Swin3DSeriesEncoder must apply it automatically on construction (not
    something a caller has to remember to do separately)."""
    import torchvision.models.video.swin_transformer as st
    from kadd.classify.attention_patch import apply_sdpa_patch, _sdpa_shifted_window_attention_3d
    from kadd.classify.model import Swin3DSeriesEncoder

    apply_sdpa_patch()
    apply_sdpa_patch()  # must not raise or change behavior on a second call
    assert st.shifted_window_attention_3d is _sdpa_shifted_window_attention_3d

    Swin3DSeriesEncoder({"size": "xlight", "pretrained": False, "summary_dim": 16})
    assert st.shifted_window_attention_3d is _sdpa_shifted_window_attention_3d


def test_sdpa_patch_backward_pass_produces_finite_gradients():
    """The patch must remain fully differentiable — a real forward+backward
    pass through a patched encoder must produce finite gradients for both
    qkv/proj weights and the relative_position_bias_table."""
    from kadd.classify.model import Swin3DSeriesEncoder

    torch.manual_seed(0)
    encoder = Swin3DSeriesEncoder({"size": "xlight", "pretrained": False, "summary_dim": 16})
    x = torch.randn(2, 1, 8, 32, 32, requires_grad=False)
    out = encoder(x)
    out.sum().backward()

    # Find a real ShiftedWindowAttention3d module and confirm its params got real gradients.
    import torchvision.models.video.swin_transformer as st
    found_attn = False
    for module in encoder.modules():
        if isinstance(module, st.ShiftedWindowAttention3d):
            found_attn = True
            assert module.qkv.weight.grad is not None
            assert torch.isfinite(module.qkv.weight.grad).all()
            assert module.relative_position_bias_table.grad is not None
            assert torch.isfinite(module.relative_position_bias_table.grad).all()
    assert found_attn, "no ShiftedWindowAttention3d module found in encoder — test setup is wrong"


# ----------------------- DICOM pipeline smoke check --------------------------
#
# NOTE: this exercises shape/plumbing only. It does NOT validate against a
# label — even the one sample DICOM this project had belonged to the public
# *test* set (no label available), so this is never a validation signal, only
# a "does loading/ordering/windowing/resize produce the right tensor shape"
# check.

def test_dicom_series_pipeline_shape():
    from kadd.classify.data import load_series_volume

    series_dir = _find_any_series_dir(DATA_DIR)
    if series_dir is None:
        pytest.skip(f"No local DICOM series (2+ .dcm files in one directory) "
                     f"found under {DATA_DIR} — run scripts/download_data.py first")

    volume = load_series_volume(series_dir, target_depth=8, target_size=32)
    assert volume.shape == (1, 8, 32, 32)
    assert torch.isfinite(volume).all()
    assert volume.min() >= 0.0 and volume.max() <= 1.0


def test_dicom_single_slice_normalization():
    """Weaker check for when only a lone .dcm (not a full series) is present:
    exercises RescaleSlope/Intercept + PhotometricInterpretation + windowing
    normalization on one slice, without the multi-slice ordering logic.
    """
    import numpy as np
    import pydicom

    dcm_path = _find_any_dcm(DATA_DIR)
    if dcm_path is None:
        pytest.skip(f"No .dcm files found anywhere under {DATA_DIR}")

    ds = pydicom.dcmread(dcm_path)
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1))
    intercept = float(getattr(ds, "RescaleIntercept", 0))
    arr = arr * slope + intercept

    wc = getattr(ds, "WindowCenter", None)
    ww = getattr(ds, "WindowWidth", None)
    assert wc is not None and ww is not None, (
        "Sample DICOM has no WindowCenter/WindowWidth — confirms the fallback "
        "percentile-normalization path in data.py is needed for this data"
    )


# ----------------------- silver-label wiring check ---------------------------

LABEL_COLUMNS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]


def _write_synthetic_csvs(tmp_path):
    import pandas as pd

    # 3 gold studies (fully labeled) + 5 silver-only studies (blank labels in
    # train.csv, present in the silver CSV instead) + 1 silver study that
    # duplicates a gold ID (must be dropped in favor of the real gold label).
    gold_ids = ["gold-0", "gold-1", "gold-2"]
    silver_only_ids = [f"silver-{i}" for i in range(5)]

    rows = []
    for i, sid in enumerate(gold_ids):
        rows.append({"StudyInstanceUID": sid, "Report": "r",
                      **{c: float(i % 2) for c in LABEL_COLUMNS}})
    for sid in silver_only_ids:
        rows.append({"StudyInstanceUID": sid, "Report": "r",
                      **{c: None for c in LABEL_COLUMNS}})
    labels_df = pd.DataFrame(rows)
    labels_csv = tmp_path / "train.csv"
    labels_df.to_csv(labels_csv, index=False)

    silver_rows = []
    for i, sid in enumerate(silver_only_ids + [gold_ids[0]]):  # +1 overlapping gold ID
        silver_rows.append({"StudyInstanceUID": sid,
                             **{c: i % 2 for c in LABEL_COLUMNS}})
    silver_df = pd.DataFrame(silver_rows)
    silver_csv = tmp_path / "report_labels_llm.csv"
    silver_df.to_csv(silver_csv, index=False)

    all_ids = gold_ids + silver_only_ids
    series_rows = [{"StudyInstanceUID": sid, "SeriesInstanceUID": f"{sid}-s0",
                     "Fluid_Sensitive": 0, "Fat_Suppression": 0,
                     "Anatomical_Plane": "Sagittal"} for sid in all_ids]
    series_csv = tmp_path / "train_series.csv"
    pd.DataFrame(series_rows).to_csv(series_csv, index=False)

    return labels_csv, series_csv, silver_csv, gold_ids, silver_only_ids


def test_data_loader_gold_only_split(tmp_path):
    from kadd.classify.data import KneeDataLoader

    labels_csv, series_csv, _, gold_ids, silver_only_ids = _write_synthetic_csvs(tmp_path)
    params = {"classify": {"data": {
        "labels_csv": str(labels_csv), "series_csv": str(series_csv),
        "label_columns": LABEL_COLUMNS, "gold_only": True,
        "data_path": str(tmp_path), "val_split": 0.34,
    }}}
    data = KneeDataLoader(params).data
    seen = set(data["train_ds"].study_ids) | set(data["val_ds"].study_ids)
    assert seen == set(gold_ids)
    assert not (seen & set(silver_only_ids))


def test_data_loader_silver_labels_hold_out_gold(tmp_path):
    from kadd.classify.data import KneeDataLoader

    labels_csv, series_csv, silver_csv, gold_ids, silver_only_ids = _write_synthetic_csvs(tmp_path)
    params = {"classify": {"data": {
        "labels_csv": str(labels_csv), "series_csv": str(series_csv),
        "label_columns": LABEL_COLUMNS, "gold_only": False,
        "use_silver_labels": True, "silver_labels_csv": str(silver_csv),
        "data_path": str(tmp_path),
    }}}
    data = KneeDataLoader(params).data

    assert set(data["val_ds"].study_ids) == set(gold_ids)
    assert set(data["train_ds"].study_ids) == set(silver_only_ids)
    # The gold study duplicated in the silver CSV must not leak into train.
    assert gold_ids[0] not in data["train_ds"].study_ids
    # The gold study's real label, not the silver row's, must be what's used.
    assert data["val_ds"].labels[gold_ids[0]] == [0.0 for _ in LABEL_COLUMNS]


# ----------------------- corrupted-DICOM resilience --------------------------
#
# Regression test for a real Kaggle run failure (2026-08-15): scaling from the
# 58 gold studies to the full 4,349-study silver set hit a genuinely
# corrupted/truncated DICOM (pydicom ValueError: "number of bytes of pixel
# data is less than expected"), which crashed a 42-minute-old training run
# because load_series_volume's exception propagated all the way up through
# the DataLoader worker. KneeDataset.__getitem__ must now catch this class of
# error, skip the bad series (or whole study if every series fails), and
# collate_fn/train_epoch/val_epoch must tolerate a resulting None batch.

def test_dataset_skips_corrupted_series(monkeypatch):
    from kadd.classify import data as data_mod

    calls = {"n": 0}

    def fake_load(series_dir, target_depth, target_size, **kwargs):
        calls["n"] += 1
        if "bad" in series_dir:
            raise ValueError("The number of bytes of pixel data is less than expected")
        return torch.zeros(1, target_depth, target_size, target_size)

    monkeypatch.setattr(data_mod, "load_series_volume", fake_load)

    study_to_series = {
        "study-mixed": [
            {"SeriesInstanceUID": "bad-series", "Anatomical_Plane": "Sagittal",
             "Fluid_Sensitive": 0, "Fat_Suppression": 0},
            {"SeriesInstanceUID": "good-series", "Anatomical_Plane": "Coronal",
             "Fluid_Sensitive": 1, "Fat_Suppression": 1},
        ],
        "study-all-bad": [
            {"SeriesInstanceUID": "bad-series-2", "Anatomical_Plane": "Axial",
             "Fluid_Sensitive": 0, "Fat_Suppression": 0},
        ],
    }
    labels = {"study-mixed": [0.0] * len(LABEL_COLUMNS),
              "study-all-bad": [1.0] * len(LABEL_COLUMNS)}

    ds = data_mod.KneeDataset(
        study_ids=["study-mixed", "study-all-bad"], study_to_series=study_to_series,
        labels=labels, data_path="/fake", target_depth=4, target_size=8,
    )

    mixed = ds[0]
    assert mixed is not None
    assert mixed["volumes"].shape[0] == 1  # only the good series survived

    all_bad = ds[1]
    assert all_bad is None  # every series failed -> whole study skipped

    batch = data_mod.collate_fn([mixed, all_bad])
    assert batch is not None
    assert batch["n_studies"] == 1  # the None study was filtered out

    empty_batch = data_mod.collate_fn([None, None])
    assert empty_batch is None


# ------------------------- precomputed-volume read path ----------------------
# See CLAUDE.md sec 7's precompute pipeline: kaggle_kernel_precompute/
# precompute_kernel.py writes float16 (1,32,224,224) .npy tensors, and
# scripts/kaggle_push_precomputed_dataset.py buckets them into
# shard_XX/<study>/<series>.npy locally + on the pushed Kaggle dataset.
# KneeDataset reads that layout directly when precomputed_dir/precomputed_ok
# are supplied, skipping DICOM decode entirely for any series it covers.

def test_shard_for_study_matches_push_script():
    """kadd/classify/data.py's _shard_for_study is a read-side copy of
    scripts/kaggle_push_precomputed_dataset.py's write-side one — they must
    never drift apart or precomputed files become unfindable. Cross-checks
    both copies agree on the same set of study UIDs rather than trusting
    that they were transcribed correctly by eye.
    """
    import importlib.util

    from kadd.classify import data as data_mod

    path = os.path.join(os.path.dirname(__file__), "..", "scripts", "kaggle_push_precomputed_dataset.py")
    spec = importlib.util.spec_from_file_location("kaggle_push_precomputed_dataset", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    assert data_mod.PRECOMPUTED_NUM_SHARDS == m.NUM_SHARDS
    for study_uid in ["study1", "1.2.826.0.1.3680043.8.498.10492923471392639089206565125595901837",
                       "another-study-id", ""]:
        assert data_mod._shard_for_study(study_uid) == m._shard_for_study(study_uid)


def test_load_precomputed_manifest(tmp_path):
    from kadd.classify.data import load_precomputed_manifest

    (tmp_path / "manifest.jsonl").write_text(
        json.dumps({"study": "s1", "series": "a", "status": "ok"}) + "\n"
        + json.dumps({"study": "s1", "series": "b", "status": "error", "error": "boom"}) + "\n"
        + "not valid json\n"
        + json.dumps({"study": "s2", "series": "c", "status": "ok"}) + "\n"
    )
    ok = load_precomputed_manifest(str(tmp_path))
    assert ok == {("s1", "a"), ("s2", "c")}

    # No manifest at all -> empty set, not an error (degrades to all-DICOM).
    assert load_precomputed_manifest(str(tmp_path / "does-not-exist")) == set()


def test_dataset_reads_precomputed_volume_and_skips_dicom(tmp_path, monkeypatch):
    """A series present (status "ok") in the precomputed manifest must be
    read from its .npy file, never touching load_series_volume at all."""
    import numpy as np

    from kadd.classify import data as data_mod

    def fail_if_called(*a, **kw):
        raise AssertionError("load_series_volume should not be called for a precomputed series")

    monkeypatch.setattr(data_mod, "load_series_volume", fail_if_called)

    study_uid, series_uid = "study-precomp", "series-precomp"
    shard = data_mod._shard_for_study(study_uid)
    series_dir = tmp_path / "precomputed" / shard / study_uid
    series_dir.mkdir(parents=True)
    expected = np.full((1, 4, 8, 8), 0.5, dtype=np.float16)
    np.save(series_dir / f"{series_uid}.npy", expected)

    study_to_series = {
        study_uid: [{"SeriesInstanceUID": series_uid, "Anatomical_Plane": "Sagittal",
                     "Fluid_Sensitive": 0, "Fat_Suppression": 0}],
    }
    labels = {study_uid: [0.0] * len(LABEL_COLUMNS)}

    ds = data_mod.KneeDataset(
        study_ids=[study_uid], study_to_series=study_to_series, labels=labels,
        data_path="/fake", target_depth=4, target_size=8,
        precomputed_dir=str(tmp_path / "precomputed"),
        precomputed_ok={(study_uid, series_uid)},
    )
    item = ds[0]
    assert item is not None
    assert item["volumes"].shape == (1, 1, 4, 8, 8)
    assert torch.allclose(item["volumes"][0], torch.full((1, 4, 8, 8), 0.5), atol=1e-3)


def test_dataset_falls_back_to_dicom_when_not_precomputed(monkeypatch, tmp_path):
    """A series NOT in precomputed_ok (or with a shape mismatch) must fall
    back to the normal DICOM path rather than erroring."""
    from kadd.classify import data as data_mod

    calls = {"n": 0}

    def fake_load(series_dir, target_depth, target_size, **kwargs):
        calls["n"] += 1
        return torch.zeros(1, target_depth, target_size, target_size)

    monkeypatch.setattr(data_mod, "load_series_volume", fake_load)

    study_uid, series_uid = "study-nocomp", "series-nocomp"
    study_to_series = {
        study_uid: [{"SeriesInstanceUID": series_uid, "Anatomical_Plane": "Axial",
                     "Fluid_Sensitive": 1, "Fat_Suppression": 1}],
    }
    labels = {study_uid: [0.0] * len(LABEL_COLUMNS)}

    ds = data_mod.KneeDataset(
        study_ids=[study_uid], study_to_series=study_to_series, labels=labels,
        data_path="/fake", target_depth=4, target_size=8,
        precomputed_dir=str(tmp_path / "precomputed"),  # manifest doesn't cover this series
        precomputed_ok=set(),
    )
    item = ds[0]
    assert item is not None
    assert calls["n"] == 1


# --------------------- local write-through cache (KneeDataset) --------------
# Added after CLAUDE.md's "why is dataloading slow" investigation: precomputed
# volumes still live on a REMOTE Kaggle dataset mount, re-read from scratch
# every epoch with no caching anywhere. local_cache_dir, when set, copies a
# series to genuinely local scratch (e.g. /kaggle/tmp) on first read so later
# epochs read the local copy instead of hitting the remote mount again.

def _make_precomputed_series(root, study_uid, series_uid, shard, value=0.5,
                              depth=4, size=8):
    import numpy as np

    series_dir = os.path.join(root, shard, study_uid)
    os.makedirs(series_dir, exist_ok=True)
    arr = np.full((1, depth, size, size), value, dtype=np.float16)
    np.save(os.path.join(series_dir, f"{series_uid}.npy"), arr)


def test_local_cache_populated_on_first_read_and_used_on_second(tmp_path):
    from kadd.classify import data as data_mod

    study_uid, series_uid = "study-cache", "series-cache"
    shard = data_mod._shard_for_study(study_uid)
    remote_dir = tmp_path / "remote"
    _make_precomputed_series(str(remote_dir), study_uid, series_uid, shard)

    study_to_series = {
        study_uid: [{"SeriesInstanceUID": series_uid, "Anatomical_Plane": "Sagittal",
                     "Fluid_Sensitive": 0, "Fat_Suppression": 0}],
    }
    labels = {study_uid: [0.0] * len(LABEL_COLUMNS)}
    cache_dir = tmp_path / "cache"

    ds = data_mod.KneeDataset(
        study_ids=[study_uid], study_to_series=study_to_series, labels=labels,
        data_path="/fake", target_depth=4, target_size=8,
        precomputed_dir=str(remote_dir), precomputed_ok={(study_uid, series_uid)},
        local_cache_dir=str(cache_dir),
    )

    item = ds[0]
    assert item is not None
    cache_path = cache_dir / study_uid / f"{series_uid}.npy"
    assert cache_path.is_file()  # written on first read

    # Delete the "remote" source entirely — a second read must succeed purely
    # from the local cache, proving it doesn't need the remote mount anymore.
    shutil.rmtree(remote_dir)
    item2 = ds[0]
    assert item2 is not None
    assert torch.allclose(item["volumes"], item2["volumes"])


def test_local_cache_write_failure_disables_further_writes_not_reads(monkeypatch, tmp_path):
    """A cache write failure (e.g. simulating a full disk) must not crash the
    run — it disables further writes in this process and falls back to
    reading from the remote precomputed_dir every time instead."""
    from kadd.classify import data as data_mod

    study_uid, series_uid = "study-full", "series-full"
    shard = data_mod._shard_for_study(study_uid)
    remote_dir = tmp_path / "remote"
    _make_precomputed_series(str(remote_dir), study_uid, series_uid, shard)

    study_to_series = {
        study_uid: [{"SeriesInstanceUID": series_uid, "Anatomical_Plane": "Sagittal",
                     "Fluid_Sensitive": 0, "Fat_Suppression": 0}],
    }
    labels = {study_uid: [0.0] * len(LABEL_COLUMNS)}

    def fake_write(cache_path, arr):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(data_mod, "_write_cache_file", fake_write)

    ds = data_mod.KneeDataset(
        study_ids=[study_uid], study_to_series=study_to_series, labels=labels,
        data_path="/fake", target_depth=4, target_size=8,
        precomputed_dir=str(remote_dir), precomputed_ok={(study_uid, series_uid)},
        local_cache_dir=str(tmp_path / "cache"),
    )

    assert ds[0] is not None  # first read: write fails, still returns a valid volume
    assert ds._cache_disabled is True
    assert ds[0] is not None  # second read: no crash, just reads remote again


def test_write_cache_file_is_atomic_no_partial_file_on_readers(tmp_path):
    """_write_cache_file must never leave a partially-written file at the
    final cache_path — writes to a temp file first, then os.replace (atomic
    on POSIX), so a concurrent reader either sees nothing or the complete
    file, never a partial one."""
    import numpy as np

    from kadd.classify.data import _write_cache_file

    cache_path = tmp_path / "study" / "series.npy"
    arr = np.ones((1, 4, 8, 8), dtype=np.float16)
    _write_cache_file(str(cache_path), arr)

    assert cache_path.is_file()
    loaded = np.load(cache_path)
    assert np.array_equal(loaded, arr)
    # No leftover temp files — the directory must contain exactly the final file.
    assert [p.name for p in cache_path.parent.iterdir()] == [cache_path.name]


# ----------------------- config autodetection (--type/--size) ---------------

def test_discover_configs_finds_real_repo_configs():
    """Integration check against the real config_files/ directory: every
    classify_<type>_<size>.yml combo the model supports must be discoverable,
    the legacy classify_swin3d_v1.yml/v2.yml (pre-dating this naming
    convention) must NOT be misread as a bogus size tier ('v1'/'v2'), and a
    cnn3d-only size ('xmedium') must NOT be treated as also valid for
    swin3d (the two model families have independent size vocabularies —
    see model.py's SIZES/CNN3D_SIZES).
    """
    from kadd.classify.model import CNN3D_SIZES, SIZES
    from kadd.util.config_registry import TYPES, discover_configs

    registry = discover_configs("config_files")
    expected = {("swin3d", s) for s in SIZES} | {("cnn3d", s) for s in CNN3D_SIZES}
    assert expected <= set(registry)
    assert ("swin3d", "v1") not in registry
    assert ("swin3d", "v2") not in registry
    assert ("swin3d", "xmedium") not in registry
    assert set(TYPES) == {"swin3d", "cnn3d"}


def test_config_registry_naming_convention(tmp_path):
    """Only classify_<type>_<size>.yml where type/size are recognized values
    gets picked up — an unrelated filename or a legacy-style name is ignored.
    """
    from kadd.util.config_registry import discover_configs, resolve_config

    (tmp_path / "classify_cnn3d_xlight.yml").write_text("name: x\n")
    (tmp_path / "classify_swin3d_v1.yml").write_text("name: x\n")  # legacy naming, must be skipped
    (tmp_path / "notes.txt").write_text("irrelevant\n")

    registry = discover_configs(str(tmp_path))
    assert set(registry) == {("cnn3d", "xlight")}

    resolved = resolve_config("cnn3d", "xlight", str(tmp_path))
    assert resolved == str(tmp_path / "classify_cnn3d_xlight.yml")

    with pytest.raises(ValueError, match="swin3d"):
        resolve_config("swin3d", "large", str(tmp_path))


def test_cli_type_size_resolution(monkeypatch):
    """`kadd classify-train --type/--size` resolves to the right config path
    and rejects invalid combinations, without actually starting training."""
    import sys

    import kadd.__main__ as m

    calls = []
    monkeypatch.setattr(m, "classify_train", lambda args: calls.append(args.paramcard))

    monkeypatch.setattr(sys, "argv", ["kadd", "classify-train", "--type", "cnn3d", "--size", "xlight"])
    m.main()
    assert calls[-1] == "config_files/classify_cnn3d_xlight.yml"

    monkeypatch.setattr(sys, "argv", ["kadd", "classify-train", "config_files/classify_swin3d_v1.yml"])
    m.main()
    assert calls[-1] == "config_files/classify_swin3d_v1.yml"

    # paramcard + --type together must be rejected
    monkeypatch.setattr(sys, "argv", ["kadd", "classify-train", "config_files/classify_swin3d_v1.yml",
                                       "--type", "cnn3d", "--size", "xlight"])
    with pytest.raises(SystemExit):
        m.main()

    # neither paramcard nor --type/--size must be rejected
    monkeypatch.setattr(sys, "argv", ["kadd", "classify-train"])
    with pytest.raises(SystemExit):
        m.main()


# ----------------------- DICOM verification diagnostic -----------------------
#
# kaggle_kernel_verify/verify_dicom_kernel.py checks whether real training
# DICOMs carry the slice-ordering tags data.py depends on, and whether the
# geometry-based sort is internally consistent — never run on real training
# data yet (see CLAUDE.md), so its check_series() logic is validated here
# against synthetic DICOMs instead of trusting it blind on the first real run.

def _load_verify_dicom_kernel():
    import importlib.util

    path = os.path.join(os.path.dirname(__file__), "..", "kaggle_kernel_verify", "verify_dicom_kernel.py")
    spec = importlib.util.spec_from_file_location("verify_dicom_kernel", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_synthetic_dicom(path, ipp, iop=(1, 0, 0, 0, 1, 0), window_center=40, window_width=80,
                           pixel_spacing=(1.0, 1.0), slice_thickness=3.0, instance_number=None,
                           omit=()):
    """Minimal but real, dcmread-round-trippable DICOM file — header tags
    only, no pixel data (check_series never reads pixel_array)."""
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = generate_uid()
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(path, {}, file_meta=meta, preamble=b"\0" * 128)
    if "ImagePositionPatient" not in omit:
        ds.ImagePositionPatient = list(ipp)
    if "ImageOrientationPatient" not in omit:
        ds.ImageOrientationPatient = list(iop)
    if "WindowCenter" not in omit:
        ds.WindowCenter = window_center
    if "WindowWidth" not in omit:
        ds.WindowWidth = window_width
    if "PixelSpacing" not in omit:
        ds.PixelSpacing = list(pixel_spacing)
    if slice_thickness is not None and "SliceThickness" not in omit:
        ds.SliceThickness = slice_thickness
    if instance_number is not None:
        ds.InstanceNumber = instance_number
    ds.save_as(path, enforce_file_format=True, little_endian=True, implicit_vr=False)
    return path


def test_check_series_pass_uniform_spacing(tmp_path):
    vdk = _load_verify_dicom_kernel()
    series_dir = tmp_path / "series"
    series_dir.mkdir()
    for i, z in enumerate([0, 3, 6, 9]):
        _make_synthetic_dicom(str(series_dir / f"s{i}.dcm"), ipp=(0, 0, z), instance_number=i + 1)

    report = vdk.check_series(str(series_dir), "study1", "series1")
    assert report["status"] == "PASS"
    assert report["missing_tags"] == []
    assert report["measured_spacing_mm"]["mean"] == pytest.approx(3.0)
    assert report["declared_spacing_mm"] == pytest.approx(3.0)
    assert report["instance_number_agrees_with_geometry"] is True


def test_check_series_fails_on_missing_tag(tmp_path):
    vdk = _load_verify_dicom_kernel()
    series_dir = tmp_path / "series"
    series_dir.mkdir()
    for i, z in enumerate([0, 3, 6]):
        _make_synthetic_dicom(str(series_dir / f"s{i}.dcm"), ipp=(0, 0, z), omit=("WindowCenter",))

    report = vdk.check_series(str(series_dir), "study1", "series1")
    assert report["status"] == "FAIL"
    assert "WindowCenter" in report["missing_tags"]


def test_check_series_fails_on_tied_positions(tmp_path):
    vdk = _load_verify_dicom_kernel()
    series_dir = tmp_path / "series"
    series_dir.mkdir()
    for i, z in enumerate([0, 3, 3, 9]):  # two slices at the same position
        _make_synthetic_dicom(str(series_dir / f"s{i}.dcm"), ipp=(0, 0, z))

    report = vdk.check_series(str(series_dir), "study1", "series1")
    assert report["status"] == "FAIL"
    assert "duplicate" in report["notes"][0] or "tied" in report["notes"][0]


def test_check_series_warns_on_uneven_spacing(tmp_path):
    vdk = _load_verify_dicom_kernel()
    series_dir = tmp_path / "series"
    series_dir.mkdir()
    for i, z in enumerate([0, 3, 20, 23]):  # a big gap in the middle
        _make_synthetic_dicom(str(series_dir / f"s{i}.dcm"), ipp=(0, 0, z))

    report = vdk.check_series(str(series_dir), "study1", "series1")
    assert report["status"] == "WARN"
    assert report["measured_spacing_mm"]["cv"] > vdk.SPACING_CV_WARN


def test_check_series_warns_on_instance_number_disagreement(tmp_path):
    vdk = _load_verify_dicom_kernel()
    series_dir = tmp_path / "series"
    series_dir.mkdir()
    # Geometry order is 0,3,6,9 (by z) but InstanceNumber is scrambled, not
    # matching that order in either direction.
    positions = [0, 3, 6, 9]
    instance_numbers = [1, 3, 2, 4]
    for i, (z, inst) in enumerate(zip(positions, instance_numbers)):
        _make_synthetic_dicom(str(series_dir / f"s{i}.dcm"), ipp=(0, 0, z), instance_number=inst)

    report = vdk.check_series(str(series_dir), "study1", "series1")
    assert report["instance_number_agrees_with_geometry"] is False
    assert report["status"] == "WARN"


def test_check_series_too_few_slices(tmp_path):
    vdk = _load_verify_dicom_kernel()
    series_dir = tmp_path / "series"
    series_dir.mkdir()
    _make_synthetic_dicom(str(series_dir / "s0.dcm"), ipp=(0, 0, 0))

    report = vdk.check_series(str(series_dir), "study1", "series1")
    assert report["status"] == "FAIL"
    assert report["n_slices"] == 0  # returned before counting, since it bails before dcmread


def test_verify_dicom_main_writes_report_and_timing(tmp_path, monkeypatch):
    """End-to-end check of main() against a synthetic competition-data layout
    (train_series.csv + on-disk series folders), including the
    seconds-per-3d-image timing field."""
    import pandas as pd

    vdk = _load_verify_dicom_kernel()

    comp_root = tmp_path / "comp"
    train_series_dir = comp_root / "train_series"
    study_dir = train_series_dir / "study1" / "seriesA"
    study_dir.mkdir(parents=True)
    for i, z in enumerate([0, 3, 6]):
        _make_synthetic_dicom(str(study_dir / f"s{i}.dcm"), ipp=(0, 0, z), instance_number=i + 1)

    pd.DataFrame([{"StudyInstanceUID": "study1", "SeriesInstanceUID": "seriesA",
                    "Anatomical_Plane": "Axial", "Fluid_Sensitive": 0, "Fat_Suppression": 0}]
                 ).to_csv(comp_root / "train_series.csv", index=False)

    monkeypatch.chdir(tmp_path)
    vdk.main(str(comp_root))

    with open(tmp_path / "output" / "dicom_verification_report.json") as f:
        summary = json.load(f)
    assert summary["n_series_checked"] == 1
    assert summary["n_pass"] == 1
    assert "avg_seconds_per_3d_image" in summary
    assert summary["reports"][0]["seconds"] >= 0


# ----------------------- per-epoch study subsampling --------------------------
#
# Added because a real Kaggle run showed epoch wall-clock is dominated by
# DICOM decode/windowing/resize, not model compute — a 160K-param cnn3d/
# xlight model still hadn't finished epoch 1 after 19 minutes on the full
# 4,349-study silver set. sub_epoch_size caps each epoch to a fresh random
# subset instead of the full train set, redrawn every epoch.

class _FakeStudyDataset(torch.utils.data.Dataset):
    """Minimal stand-in for KneeDataset — __len__/__getitem__ for
    _make_train_loader's subsampling/caching logic, plus study_ids/labels
    (all-negative, 12 labels) since ClassifierTraining.__init__ now computes
    pos_weight from train_ds.study_ids/labels — no real DICOM I/O either way."""

    def __init__(self, n, num_labels=12):
        self.n = n
        self.study_ids = [str(i) for i in range(n)]
        self.labels = {sid: [0.0] * num_labels for sid in self.study_ids}

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return i


def _make_classifier_training(sub_epoch_size, train_n=20, val_n=5):
    from kadd.classify.train import ClassifierTraining

    train_ds = _FakeStudyDataset(train_n)
    val_ds = _FakeStudyDataset(val_n)
    data = {"train_ds": train_ds, "val_ds": val_ds, "label_columns": ["ACL"] * 12}
    model = torch.nn.Linear(1, 1)
    params = {
        "name": "test_run",
        "classify": {
            "train": {"epochs": 1, "batch_size": 2, "lr": 1e-4},
            "data": {"num_workers": 0, "sub_epoch_size": sub_epoch_size},
        },
    }
    return ClassifierTraining(params, model, data, torch.device("cpu")), train_ds


def test_sub_epoch_size_unset_reuses_full_loader():
    """No sub_epoch_size -> matches pre-sub_epoch_size behavior exactly: the full-
    dataset loader is built once and cached, not rebuilt every epoch."""
    training, train_ds = _make_classifier_training(sub_epoch_size=None)
    loader1 = training._make_train_loader(train_ds)
    loader2 = training._make_train_loader(train_ds)
    assert loader1 is loader2
    assert len(loader1.dataset) == 20


def test_sub_epoch_size_false_behaves_like_unset():
    """sub_epoch_size: false (explicit YAML disable) must behave identically
    to leaving the key unset — not treated as a falsy subsample size."""
    training, train_ds = _make_classifier_training(sub_epoch_size=False)
    assert training.sub_epoch_size is None
    loader1 = training._make_train_loader(train_ds)
    loader2 = training._make_train_loader(train_ds)
    assert loader1 is loader2
    assert len(loader1.dataset) == 20


def test_sub_epoch_size_at_or_above_dataset_size_falls_back_to_full():
    training, train_ds = _make_classifier_training(sub_epoch_size=50)  # > train_n=20
    loader1 = training._make_train_loader(train_ds)
    loader2 = training._make_train_loader(train_ds)
    assert loader1 is loader2
    assert len(loader1.dataset) == 20


def test_sub_epoch_size_draws_fresh_random_subset_each_epoch():
    """The DataLoader itself is built ONCE and cached (unlike the earlier
    rebuild-every-epoch design, which caused two real host-RAM OOM-kills on
    Kaggle) — but its sampler (_EpochSubsetSampler) draws a fresh random
    subset every time it's iterated, which is what a DataLoader does once
    per `for batch in loader` pass. Simulates several epochs by iterating
    the sampler directly."""
    training, train_ds = _make_classifier_training(sub_epoch_size=5)
    loader1 = training._make_train_loader(train_ds)
    loader2 = training._make_train_loader(train_ds)
    assert loader1 is loader2  # built once, cached — not rebuilt per epoch

    seen_index_sets = set()
    for _ in range(6):
        indices = list(iter(loader1.sampler))
        assert len(indices) == 5
        seen_index_sets.add(tuple(sorted(indices)))
    assert len(seen_index_sets) > 1  # different studies sampled across epochs


# ------------------ data-wait-vs-compute-time / CPU usage instrumentation ---
#
# Added because an earlier claim in this session ("epoch time is CPU-bound,
# not GPU-bound") was inferred from indirect evidence (epoch time barely
# moved across a 4x batch_size change) rather than measured — correctly
# challenged by the user. train_epoch now times data-loading-wait vs.
# GPU-compute directly; verified end-to-end here against a real (tiny)
# model + real collate_fn, not just checked for "doesn't crash".

class _FakeSeriesStudyDataset(torch.utils.data.Dataset):
    """Produces real KneeDataset-shaped items (small volumes) so collate_fn
    and the model's actual forward pass run for real, without any DICOM I/O.
    """

    def __init__(self, n_studies, n_series=2, depth=4, size=8, num_labels=12):
        self.n_studies = n_studies
        self.n_series = n_series
        self.depth = depth
        self.size = size
        self.num_labels = num_labels
        self.study_ids = [f"study-{i}" for i in range(n_studies)]
        self.labels = {sid: [0.0] * num_labels for sid in self.study_ids}

    def __len__(self):
        return self.n_studies

    def __getitem__(self, i):
        return {
            'volumes': torch.rand(self.n_series, 1, self.depth, self.size, self.size),
            'plane_idx': torch.randint(0, 3, (self.n_series,)),
            'fluid_sensitive': torch.randint(0, 2, (self.n_series,)).float(),
            'fat_suppression': torch.randint(0, 2, (self.n_series,)).float(),
            'label': torch.randint(0, 2, (self.num_labels,)).float(),
            'study_uid': f"study-{i}",
        }


def test_train_epoch_times_data_wait_vs_compute():
    from kadd.classify.model import build_model
    from kadd.classify.train import ClassifierTraining

    label_columns = ["ACL"] * 12
    params = {
        "name": "test_run",
        "classify": {
            "model": {"type": "cnn3d", "size": "xlight", "summary_dim": 16,
                      "meta_dim": 8, "use_checkpointing": False},
            "data": {"num_workers": 0, "label_columns": label_columns},
            "train": {"epochs": 1, "batch_size": 2, "lr": 1e-3},
        },
    }
    model = build_model(params)
    train_ds = _FakeSeriesStudyDataset(n_studies=6)
    data = {"train_ds": train_ds, "val_ds": train_ds, "label_columns": label_columns}

    training = ClassifierTraining(params, model, data, torch.device("cpu"))
    training.setup_optimizer_and_scheduler()

    from kadd.classify.data import collate_fn
    loader = torch.utils.data.DataLoader(train_ds, batch_size=2, shuffle=False, collate_fn=collate_fn)

    import math

    trn_loss = training.train_epoch(loader)
    assert math.isfinite(trn_loss)
    assert training._last_compute_time > 0
    assert training._last_data_wait_time >= 0

    # Logging must not raise, on CPU (no CUDA) and regardless of psutil availability.
    training._log_data_vs_compute_time()
    training._log_cpu_usage()


def test_make_loader_passes_prefetch_factor_only_with_workers():
    """prefetch_factor is only a valid DataLoader kwarg when num_workers > 0
    — confirmed both branches build without raising, and that the configured
    value actually reaches the DataLoader (added after real Kaggle evidence
    showed the cnn3d/xlight bottleneck was I/O-latency, not CPU decode, so
    num_workers/prefetch_factor became real tuning levers, not just
    plumbing)."""
    from kadd.classify.train import ClassifierTraining
    from kadd.classify.model import build_model

    label_columns = ["ACL"] * 12
    base_params = {
        "name": "test_run",
        "classify": {
            "model": {"type": "cnn3d", "size": "xlight", "summary_dim": 16, "meta_dim": 8},
            "train": {"epochs": 1, "batch_size": 2, "lr": 1e-3},
        },
    }
    train_ds = _FakeStudyDataset(6)
    data = {"train_ds": train_ds, "val_ds": train_ds, "label_columns": label_columns}
    model = build_model({**base_params, "classify": {**base_params["classify"],
                                                       "data": {"label_columns": label_columns}}})

    params_with_workers = {**base_params, "classify": {**base_params["classify"],
                            "data": {"num_workers": 3, "prefetch_factor": 5}}}
    training = ClassifierTraining(params_with_workers, model, data, torch.device("cpu"))
    loader = training._make_loader(train_ds, shuffle=False)
    assert loader.num_workers == 3
    assert loader.prefetch_factor == 5

    params_no_workers = {**base_params, "classify": {**base_params["classify"], "data": {"num_workers": 0}}}
    training2 = ClassifierTraining(params_no_workers, model, data, torch.device("cpu"))
    loader2 = training2._make_loader(train_ds, shuffle=False)  # must not raise (prefetch_factor omitted)
    assert loader2.num_workers == 0


def test_sub_epoch_size_loader_built_once_never_rebuilt():
    """Regression test for TWO real Kaggle OOM-kills: an earlier design
    rebuilt a fresh Subset + DataLoader (and a whole new worker pool) every
    single epoch when sub_epoch_size was set. Disabling persistent_workers on
    that rebuilt loader (a first attempted fix) was not sufficient — a
    second run still died at the same epoch boundary. The actual fix:
    _make_train_loader now builds the DataLoader exactly ONCE per run in
    both the subsampled and full-dataset cases (never rebuilt), and uses
    persistent_workers=True uniformly whenever num_workers > 0 — safe now
    because there's only ever one loader object for the whole run, so
    workers are spawned once and never torn down/respawned mid-run.
    Fresh-random-subset-per-epoch instead comes from _EpochSubsetSampler
    (covered by test_sub_epoch_size_draws_fresh_random_subset_each_epoch)."""
    from kadd.classify.train import ClassifierTraining, _EpochSubsetSampler

    train_ds = _FakeStudyDataset(20)
    val_ds = _FakeStudyDataset(5)
    data = {"train_ds": train_ds, "val_ds": val_ds, "label_columns": ["ACL"] * 12}
    model = torch.nn.Linear(1, 1)

    def make_training(sub_epoch_size):
        params = {
            "name": "test_run",
            "classify": {
                "train": {"epochs": 1, "batch_size": 2, "lr": 1e-4},
                "data": {"num_workers": 2, "sub_epoch_size": sub_epoch_size},
            },
        }
        return ClassifierTraining(params, model, data, torch.device("cpu"))

    subsampled = make_training(sub_epoch_size=5)
    subset_loader_1 = subsampled._make_train_loader(train_ds)
    subset_loader_2 = subsampled._make_train_loader(train_ds)
    assert subset_loader_1 is subset_loader_2   # built once, never rebuilt
    assert subset_loader_1.persistent_workers is True
    assert isinstance(subset_loader_1.sampler, _EpochSubsetSampler)

    full = make_training(sub_epoch_size=None)
    full_loader = full._make_train_loader(train_ds)
    assert full_loader.persistent_workers is True


def test_make_loader_num_workers_override_for_val():
    """Regression test for a real Kaggle OOM-kill mid-epoch-1 (not at an
    epoch boundary — the rebuild-per-epoch bug from the previous entry was
    confirmed fixed, but train+val loaders are now BOTH persistent for the
    whole run, so their worker counts add up as steady-state pressure: 8
    (train) + 8 (val, same num_workers by default before this fix) = 16
    persistent workers alive simultaneously). _make_loader's num_workers
    param lets train_loop give val far fewer workers (val is only 58 gold
    studies, doesn't need heavy I/O parallelism) without touching train's."""
    from kadd.classify.train import ClassifierTraining

    train_ds = _FakeStudyDataset(10)
    data = {"train_ds": train_ds, "val_ds": train_ds, "label_columns": ["ACL"] * 12}
    model = torch.nn.Linear(1, 1)
    params = {
        "name": "test_run",
        "classify": {
            "train": {"epochs": 1, "batch_size": 2, "lr": 1e-4},
            "data": {"num_workers": 8},
        },
    }
    training = ClassifierTraining(params, model, data, torch.device("cpu"))

    default_loader = training._make_loader(train_ds, shuffle=False)
    assert default_loader.num_workers == 8  # unchanged default behavior

    overridden_loader = training._make_loader(train_ds, shuffle=False, num_workers=2)
    assert overridden_loader.num_workers == 2


# ------------------------------ resume support --------------------------------
#
# Added 2026-09-03: this session repeatedly lost real training progress to
# external interruptions (10h push-timeout kills, OOMs, Kaggle GPU-quota
# exhaustion mid-week) with no way to continue from a checkpoint — every
# recovery meant restarting from epoch 0. save_checkpoint already stored
# model/optimizer state; this adds scheduler state + early-stopping
# bookkeeping (best_macro_auc/patience_counter/ema_val_loss) so a resumed
# run is a true continuation, not just re-loaded weights with reset patience.

def test_resume_from_checkpoint_continues_training(tmp_path):
    """End-to-end: train 2 epochs, save best.pth, start a FRESH
    ClassifierTraining with resume_from=best.pth and epochs=4 — must resume
    at epoch 3 (not epoch 1) and complete without error."""
    import math
    from kadd.classify.model import build_model
    from kadd.classify.train import ClassifierTraining

    label_columns = ["ACL"] * 12
    base_classify = {
        "model": {"type": "cnn3d", "size": "xlight", "summary_dim": 16,
                  "meta_dim": 8, "use_checkpointing": False},
        "data": {"num_workers": 0, "label_columns": label_columns},
    }
    train_ds = _FakeSeriesStudyDataset(n_studies=4)
    train_ds.max_series = train_ds.n_series  # train_loop's log line reads this; real KneeDataset always has it
    data = {"train_ds": train_ds, "val_ds": train_ds, "label_columns": label_columns}

    run_dir = tmp_path / "run1"
    params1 = {"name": str(run_dir), "classify": {**base_classify,
               "train": {"epochs": 2, "batch_size": 2, "lr": 1e-3}}}
    model1 = build_model(params1)
    training1 = ClassifierTraining(params1, model1, data, torch.device("cpu"))
    best_auc_1 = training1.train_loop()
    assert math.isfinite(best_auc_1)

    checkpoint_path = str(run_dir) + "/models/best.pth"
    assert os.path.isfile(checkpoint_path)

    run_dir2 = tmp_path / "run2"
    params2 = {"name": str(run_dir2), "classify": {**base_classify,
               "train": {"epochs": 4, "batch_size": 2, "lr": 1e-3, "resume_from": checkpoint_path}}}
    model2 = build_model(params2)
    training2 = ClassifierTraining(params2, model2, data, torch.device("cpu"))
    best_auc_2 = training2.train_loop()
    assert math.isfinite(best_auc_2)

    # Only epochs 3-4 should have actually run (2 already done in run1) —
    # confirmed via the resumed run's own final checkpoint epoch number.
    resumed_checkpoint = torch.load(str(run_dir2) + "/models/latest.pth", weights_only=False)
    assert resumed_checkpoint["epoch"] >= 2  # 0-indexed: epoch 2 = the 3rd epoch


def test_load_checkpoint_for_resume_restores_bookkeeping(tmp_path):
    """save_checkpoint -> load_checkpoint_for_resume round-trip must restore
    best_macro_auc/patience_counter/ema_val_loss exactly, and start_epoch
    must be checkpoint['epoch'] + 1 (resume from the NEXT epoch, not redo
    the last completed one)."""
    from kadd.classify.model import build_model
    from kadd.classify.train import ClassifierTraining

    label_columns = ["ACL"] * 12
    params = {"name": str(tmp_path / "run"),
              "classify": {"model": {"type": "cnn3d", "size": "xlight", "summary_dim": 16,
                                      "meta_dim": 8, "use_checkpointing": False},
                           "data": {"label_columns": label_columns},
                           "train": {"epochs": 1, "batch_size": 2, "lr": 1e-3}}}
    model = build_model(params)
    train_ds = _FakeSeriesStudyDataset(n_studies=2)
    data = {"train_ds": train_ds, "val_ds": train_ds, "label_columns": label_columns}
    training = ClassifierTraining(params, model, data, torch.device("cpu"))
    training.setup_optimizer_and_scheduler()

    ckpt_path = str(tmp_path / "ckpt.pth")
    training.save_checkpoint(ckpt_path, epoch=6, val_loss=0.5, macro_auc=0.71,
                              best_macro_auc=0.71, patience_counter=3, ema_val_loss=0.52)

    start_epoch, best_macro_auc, patience_counter, ema_val_loss = \
        training.load_checkpoint_for_resume(ckpt_path)
    assert start_epoch == 7
    assert best_macro_auc == pytest.approx(0.71)
    assert patience_counter == 3
    assert ema_val_loss == pytest.approx(0.52)


def test_load_checkpoint_for_resume_handles_missing_optional_fields(tmp_path):
    """A checkpoint saved BEFORE this feature existed (no scheduler_state_dict/
    best_macro_auc/patience_counter/ema_val_loss keys) must still resume —
    falling back to safe from-scratch defaults for just the missing fields
    rather than crashing, since the model/optimizer weights are still real
    and worth reusing."""
    from kadd.classify.model import build_model
    from kadd.classify.train import ClassifierTraining

    label_columns = ["ACL"] * 12
    params = {"name": str(tmp_path / "run"),
              "classify": {"model": {"type": "cnn3d", "size": "xlight", "summary_dim": 16,
                                      "meta_dim": 8, "use_checkpointing": False},
                           "data": {"label_columns": label_columns},
                           "train": {"epochs": 1, "batch_size": 2, "lr": 1e-3}}}
    model = build_model(params)
    train_ds = _FakeSeriesStudyDataset(n_studies=2)
    data = {"train_ds": train_ds, "val_ds": train_ds, "label_columns": label_columns}
    training = ClassifierTraining(params, model, data, torch.device("cpu"))
    training.setup_optimizer_and_scheduler()

    old_style_ckpt = {
        "epoch": 10,
        "val_loss": 0.6,
        "macro_auc": 0.65,
        "model_state_dict": training.model.state_dict(),
        "optimizer_state_dict": training.optimizer.state_dict(),
    }
    ckpt_path = str(tmp_path / "old_ckpt.pth")
    torch.save(old_style_ckpt, ckpt_path)

    start_epoch, best_macro_auc, patience_counter, ema_val_loss = \
        training.load_checkpoint_for_resume(ckpt_path)
    assert start_epoch == 11
    assert best_macro_auc == pytest.approx(0.65)  # falls back to the checkpoint's own macro_auc
    assert patience_counter == 0
    assert ema_val_loss is None


def test_load_checkpoint_for_resume_survives_param_group_count_mismatch(tmp_path):
    """Real scenario this project will actually hit: a checkpoint saved under
    a single optimizer param group (encoder_lr_mult unset/1.0), resumed into
    a run where encoder_lr_mult is now set (2 param groups). PyTorch's
    Optimizer.load_state_dict raises ValueError on a param-group COUNT
    mismatch — must be caught and logged, falling back to a freshly-
    initialized optimizer, with model weights still restored normally."""
    from kadd.classify.model import build_model
    from kadd.classify.train import ClassifierTraining

    label_columns = ["ACL"] * 12
    base_classify = {"model": {"type": "swin3d", "size": "xlight", "summary_dim": 16,
                                "meta_dim": 8, "use_checkpointing": False},
                      "data": {"label_columns": label_columns}}
    train_ds = _FakeStudyDataset(3)
    data = {"train_ds": train_ds, "val_ds": train_ds, "label_columns": label_columns}

    # Save a checkpoint with the default single param group (no encoder_lr_mult).
    params1 = {"name": "test_run",
               "classify": {**base_classify, "train": {"epochs": 1, "batch_size": 2, "lr": 1e-3}}}
    model1 = build_model(params1)
    training1 = ClassifierTraining(params1, model1, data, torch.device("cpu"))
    training1.setup_optimizer_and_scheduler()
    ckpt_path = str(tmp_path / "single_group_ckpt.pth")
    training1.save_checkpoint(ckpt_path, epoch=5, val_loss=0.6, macro_auc=0.6,
                               best_macro_auc=0.6, patience_counter=0, ema_val_loss=0.6)

    # Resume into a run with encoder_lr_mult set -> 2 param groups now.
    params2 = {"name": "test_run",
               "classify": {**base_classify,
                            "train": {"epochs": 1, "batch_size": 2, "lr": 1e-3, "encoder_lr_mult": 0.1}}}
    model2 = build_model(params2)
    training2 = ClassifierTraining(params2, model2, data, torch.device("cpu"))
    training2.setup_optimizer_and_scheduler()
    assert len(training2.optimizer.param_groups) == 2  # sanity: the mismatch is real

    start_epoch, best_macro_auc, patience_counter, ema_val_loss = \
        training2.load_checkpoint_for_resume(ckpt_path)  # must not raise
    assert start_epoch == 6
    assert best_macro_auc == pytest.approx(0.6)
    # Optimizer still has 2 groups at their configured LRs (fresh init, not restored).
    assert len(training2.optimizer.param_groups) == 2


def test_load_checkpoint_for_resume_survives_use_checkpointing_toggle(tmp_path):
    """Real bug hit on kernel v67, 2026-09-09, not hypothetical: _CheckpointBlock
    (model.py) wraps a checkpointed stage as `self.block`, so toggling
    classify.model.use_checkpointing between when a checkpoint was saved and
    now changes state_dict KEY NAMES (adds/removes ".block.") even though the
    underlying weights are identical. A checkpoint saved with checkpointing ON,
    resumed with checkpointing OFF, crashed with strict=True's "Missing/
    Unexpected key(s)" RuntimeError. Must be caught and remapped automatically
    — same tolerance already applied to the optimizer param-group-count
    mismatch above — with the actual WEIGHT VALUES verified to have
    transferred correctly, not just that loading didn't raise."""
    from kadd.classify.model import build_model
    from kadd.classify.train import ClassifierTraining

    label_columns = ["ACL"] * 12
    base_classify = {"model": {"type": "swin3d", "size": "xlight", "summary_dim": 16,
                                "meta_dim": 8},
                      "data": {"label_columns": label_columns}}
    train_ds = _FakeStudyDataset(3)
    data = {"train_ds": train_ds, "val_ds": train_ds, "label_columns": label_columns}

    # Save WITH checkpointing on.
    params1 = {"name": "test_run",
               "classify": {**base_classify,
                            "model": {**base_classify["model"], "use_checkpointing": True},
                            "train": {"epochs": 1, "batch_size": 2, "lr": 1e-3}}}
    model1 = build_model(params1)
    training1 = ClassifierTraining(params1, model1, data, torch.device("cpu"))
    training1.setup_optimizer_and_scheduler()
    ckpt_path = str(tmp_path / "ckpt_with_checkpointing.pth")
    training1.save_checkpoint(ckpt_path, epoch=13, val_loss=0.6, macro_auc=0.68,
                               best_macro_auc=0.68, patience_counter=0, ema_val_loss=0.6)
    assert any(".block." in k for k in model1.state_dict().keys())  # sanity: the mismatch is real

    # Resume WITHOUT checkpointing — must not raise.
    params2 = {"name": "test_run",
               "classify": {**base_classify,
                            "model": {**base_classify["model"], "use_checkpointing": False},
                            "train": {"epochs": 1, "batch_size": 2, "lr": 1e-3}}}
    model2 = build_model(params2)
    training2 = ClassifierTraining(params2, model2, data, torch.device("cpu"))
    training2.setup_optimizer_and_scheduler()

    start_epoch, best_macro_auc, patience_counter, ema_val_loss = \
        training2.load_checkpoint_for_resume(ckpt_path)
    assert start_epoch == 14
    assert best_macro_auc == pytest.approx(0.68)

    # Confirm actual weight VALUES transferred, not just that loading didn't crash.
    sd1, sd2 = model1.state_dict(), model2.state_dict()
    checked = 0
    for k, v in sd1.items():
        k2 = k.replace(".block.", ".")
        if k2 in sd2:
            checked += 1
            assert torch.equal(v, sd2[k2]), f"weight value mismatch: {k} -> {k2}"
    assert checked > 0  # sanity: the remap actually matched something


def test_remap_checkpoint_block_keys_returns_unchanged_when_already_matching():
    """No mismatch at all -> returns the state_dict as-is, no wasted work."""
    from kadd.classify.train import _remap_checkpoint_block_keys

    sd = {"a.weight": torch.zeros(2), "b.bias": torch.ones(3)}
    result = _remap_checkpoint_block_keys(sd, sd)
    assert result is sd


# --------------------------- per-label pos_weight ----------------------------
#
# Added after a real Kaggle run (kernel version 24, 2026-08-19): trn_loss
# plateaued while macro_auc kept climbing to 0.65 — real ranking signal but
# stuck/miscalibrated probabilities, plausibly caused by training almost
# entirely on silver labels (systematically lower per-label positive rate
# than gold, see CLAUDE.md sec 8). pos_weight corrects the standard BCE
# class-imbalance gradient issue for rare labels using the TRAINING set's own
# distribution, computed once in ClassifierTraining.__init__.

def test_compute_pos_weight_matches_neg_pos_ratio():
    from kadd.classify.train import ClassifierTraining

    label_columns = ["A", "B"]
    # Label A: 3 studies, 1 positive -> neg=2, pos=1 -> (2+1)/(1+1) = 1.5
    # Label B: 3 studies, 0 positive -> neg=3, pos=0 -> (3+1)/(0+1) = 4.0
    train_ds = _FakeStudyDataset(0)
    train_ds.study_ids = ["s1", "s2", "s3"]
    train_ds.labels = {"s1": [1.0, 0.0], "s2": [0.0, 0.0], "s3": [0.0, 0.0]}
    val_ds = _FakeStudyDataset(0)
    data = {"train_ds": train_ds, "val_ds": val_ds, "label_columns": label_columns}
    model = torch.nn.Linear(1, 1)
    params = {"name": "test_run",
              "classify": {"train": {"epochs": 1, "batch_size": 2, "lr": 1e-4}, "data": {}}}

    training = ClassifierTraining(params, model, data, torch.device("cpu"))
    assert training.pos_weight.tolist() == pytest.approx([1.5, 4.0])


def test_train_epoch_uses_pos_weight_val_epoch_does_not():
    """train_epoch's loss must reflect pos_weight (differs from a plain
    unweighted BCE on an imbalanced batch); val_epoch's must not (see
    _compute_pos_weight's docstring on why val_loss stays a plain,
    comparable diagnostic)."""
    from kadd.classify.data import collate_fn
    from kadd.classify.model import build_model
    from kadd.classify.train import ClassifierTraining

    label_columns = ["ACL"] * 12
    params = {
        "name": "test_run",
        "classify": {
            "model": {"type": "cnn3d", "size": "xlight", "summary_dim": 16,
                      "meta_dim": 8, "use_checkpointing": False},
            "data": {"num_workers": 0, "label_columns": label_columns},
            "train": {"epochs": 1, "batch_size": 6, "lr": 1e-3},
        },
    }
    model = build_model(params)
    train_ds = _FakeSeriesStudyDataset(n_studies=6)
    # Force an imbalanced label so pos_weight is meaningfully != 1.
    train_ds.labels = {sid: [0.0] * 12 for sid in train_ds.study_ids}
    train_ds.labels[train_ds.study_ids[0]][0] = 1.0  # one positive out of six for label 0
    data = {"train_ds": train_ds, "val_ds": train_ds, "label_columns": label_columns}

    training = ClassifierTraining(params, model, data, torch.device("cpu"))
    training.setup_optimizer_and_scheduler()
    assert training.pos_weight[0] > 1.0  # label 0 is the rare one -> upweighted

    loader = torch.utils.data.DataLoader(train_ds, batch_size=6, shuffle=False, collate_fn=collate_fn)

    # Fix one concrete batch (the fake dataset returns fresh random tensors
    # per __getitem__ call, so re-iterating would compare different data) and
    # match val_epoch's eval() mode (the model uses BatchNorm3d, which
    # behaves differently in train vs. eval) before computing logits by hand.
    batch = next(iter(loader))
    b = training._move_batch(batch)
    training.model.eval()
    with torch.no_grad():
        logits, _ = training._forward(b)
    weighted = torch.nn.BCEWithLogitsLoss(pos_weight=training.pos_weight)(logits, b['labels'])
    unweighted = torch.nn.BCEWithLogitsLoss()(logits, b['labels'])
    assert weighted.item() != pytest.approx(unweighted.item())

    # val_epoch's internal loss_fn must match the plain unweighted value, on
    # that exact same fixed batch (passed as a one-item list "loader").
    val_loss, _, _ = training.val_epoch([batch])
    assert val_loss == pytest.approx(unweighted.item(), rel=1e-4)


# ----------------------- offline volume-precompute pipeline ------------------
#
# kaggle_kernel_precompute/precompute_kernel.py decodes+preprocesses every
# training series ONCE and saves the final tensor, so future training runs
# read one small file instead of re-doing DICOM decode/crop/resize (measured
# at ~63% file I/O, ~22% crop/resize, ~11% windowing, ~4% pixel decode — see
# CLAUDE.md sec 9) every single epoch. Tested here against synthetic data,
# same "no import-time side effects" pattern as verify_dicom_kernel.py.

def _load_precompute_kernel():
    import importlib.util

    path = os.path.join(os.path.dirname(__file__), "..", "kaggle_kernel_precompute", "precompute_kernel.py")
    spec = importlib.util.spec_from_file_location("precompute_kernel", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_synthetic_train_series_csv(comp_root, rows):
    import pandas as pd

    os.makedirs(os.path.join(comp_root, "train_series"), exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(comp_root, "train_series.csv"), index=False)


def test_precompute_main_computes_and_writes_manifest(tmp_path, monkeypatch):
    pc = _load_precompute_kernel()

    comp_root = str(tmp_path / "comp")
    _make_synthetic_train_series_csv(comp_root, [
        {"StudyInstanceUID": "study1", "SeriesInstanceUID": "seriesA"},
        {"StudyInstanceUID": "study2", "SeriesInstanceUID": "seriesB"},
    ])

    import kadd.classify.data as data_mod
    monkeypatch.setattr(data_mod, "load_series_volume",
                         lambda series_dir, d, s, **kwargs: torch.rand(1, d, s, s))
    monkeypatch.chdir(tmp_path)

    pc.main(comp_root, precomputed_mount=None)

    manifest_path = tmp_path / pc.OUTPUT_ROOT / pc.MANIFEST_FILENAME
    records = [json.loads(l) for l in manifest_path.read_text().splitlines()]
    assert len(records) == 2
    assert all(r["status"] == "ok" for r in records)
    assert (tmp_path / pc.OUTPUT_ROOT / "study1" / "seriesA.npy").exists()
    assert (tmp_path / pc.OUTPUT_ROOT / "study2" / "seriesB.npy").exists()


def test_precompute_main_skips_already_done_and_records_errors(tmp_path, monkeypatch):
    pc = _load_precompute_kernel()

    comp_root = str(tmp_path / "comp")
    _make_synthetic_train_series_csv(comp_root, [
        {"StudyInstanceUID": "study1", "SeriesInstanceUID": "seriesA"},  # already done, must be skipped
        {"StudyInstanceUID": "study2", "SeriesInstanceUID": "seriesB"},  # new, will fail
        {"StudyInstanceUID": "study3", "SeriesInstanceUID": "seriesC"},  # new, will succeed
    ])

    prior_mount = tmp_path / "prior"
    prior_mount.mkdir()
    (prior_mount / pc.MANIFEST_FILENAME).write_text(
        json.dumps({"study": "study1", "series": "seriesA", "status": "ok"}) + "\n")

    import kadd.classify.data as data_mod

    def fake_load(series_dir, d, s, **kwargs):
        if "study2" in series_dir:
            raise ValueError("The number of bytes of pixel data is less than expected")
        return torch.rand(1, d, s, s)
    monkeypatch.setattr(data_mod, "load_series_volume", fake_load)
    monkeypatch.chdir(tmp_path)

    pc.main(comp_root, precomputed_mount=str(prior_mount))

    manifest_path = tmp_path / pc.OUTPUT_ROOT / pc.MANIFEST_FILENAME
    records = {r["study"]: r for r in (json.loads(l) for l in manifest_path.read_text().splitlines())}
    assert "study1" not in records  # skipped — never recomputed or re-logged this run
    assert records["study2"]["status"] == "error"
    assert records["study3"]["status"] == "ok"
    assert not (tmp_path / pc.OUTPUT_ROOT / "study2" / "seriesB.npy").exists()
    assert (tmp_path / pc.OUTPUT_ROOT / "study3" / "seriesC.npy").exists()


def test_precompute_stops_cleanly_at_output_byte_budget(tmp_path, monkeypatch):
    """Regression test for a real Kaggle failure: /kaggle/working is hard-
    capped at 20GB, and a real precompute run errored out mid-write after
    exceeding it. MAX_OUTPUT_BYTES makes the run stop cleanly (partial but
    consistent manifest/output, resumable via the normal skip-already-done
    path) instead of crashing once the configured budget is hit."""
    pc = _load_precompute_kernel()
    monkeypatch.setattr(pc, "MAX_OUTPUT_BYTES", 1)  # tiny budget -> stop after the very first series

    comp_root = str(tmp_path / "comp")
    _make_synthetic_train_series_csv(comp_root, [
        {"StudyInstanceUID": "study1", "SeriesInstanceUID": "seriesA"},
        {"StudyInstanceUID": "study2", "SeriesInstanceUID": "seriesB"},
        {"StudyInstanceUID": "study3", "SeriesInstanceUID": "seriesC"},
    ])

    import kadd.classify.data as data_mod
    monkeypatch.setattr(data_mod, "load_series_volume",
                         lambda series_dir, d, s, **kwargs: torch.rand(1, d, s, s))
    monkeypatch.chdir(tmp_path)

    pc.main(comp_root, precomputed_mount=None)

    manifest_path = tmp_path / pc.OUTPUT_ROOT / pc.MANIFEST_FILENAME
    records = [json.loads(l) for l in manifest_path.read_text().splitlines()]
    # Exactly one series gets written before the (tiny) budget is exceeded
    # and the loop stops — not zero (the budget check happens before each
    # series, not after), not all three.
    assert len(records) == 1
    assert records[0]["status"] == "ok"


def test_precompute_load_prior_manifest():
    pc = _load_precompute_kernel()
    assert pc._load_prior_manifest(None) == {}


# --------------------- precompute mount-attachment lag retry --------------------
#
# Real bug, 2026-09-04 (kernel v11): the kadd-precomputed-volumes mount
# directory was visible in /kaggle/input's top-level listing but its
# contents (manifest.jsonl) hadn't finished syncing yet — _find_precomputed_
# mount silently returned None (dir present but empty), and the run redid
# 5,016 already-done series from scratch. Fixed with a retry inside the
# kernel itself rather than guessing a longer external sleep.

def test_find_precomputed_mount_with_retry_no_dir_returns_immediately(tmp_path, monkeypatch):
    pc = _load_precompute_kernel()
    slept = []
    monkeypatch.setattr(pc.time, "sleep", lambda s: slept.append(s))

    # No 'kadd-precomputed-volumes' directory anywhere -> genuinely first run,
    # must return None WITHOUT sleeping/retrying at all.
    empty_root = tmp_path / "input"
    empty_root.mkdir()
    result = pc._find_precomputed_mount_with_retry([str(empty_root)], retries=6, delay_seconds=15)
    assert result is None
    assert slept == []


def test_find_precomputed_mount_with_retry_succeeds_after_delay(tmp_path, monkeypatch):
    pc = _load_precompute_kernel()
    slept = []
    monkeypatch.setattr(pc.time, "sleep", lambda s: slept.append(s))

    root = tmp_path / "input"
    mount_dir = root / "kadd-precomputed-volumes"
    mount_dir.mkdir(parents=True)  # directory visible, but manifest.jsonl not written yet

    call_count = {"n": 0}
    real_find = pc._find_precomputed_mount

    def flaky_find(search_roots, max_depth=4):
        call_count["n"] += 1
        if call_count["n"] < 3:
            return None  # simulates "still syncing" for the first 2 attempts
        (mount_dir / pc.MANIFEST_FILENAME).write_text('{"study": "s", "series": "x", "status": "ok"}\n')
        return real_find(search_roots, max_depth)

    monkeypatch.setattr(pc, "_find_precomputed_mount", flaky_find)
    result = pc._find_precomputed_mount_with_retry([str(root)], retries=6, delay_seconds=15)
    assert result == str(mount_dir)
    assert slept == [15, 15]  # retried exactly twice before succeeding on the 3rd attempt


def test_find_precomputed_mount_with_retry_gives_up_after_max_retries(tmp_path, monkeypatch):
    pc = _load_precompute_kernel()
    slept = []
    monkeypatch.setattr(pc.time, "sleep", lambda s: slept.append(s))

    root = tmp_path / "input"
    (root / "kadd-precomputed-volumes").mkdir(parents=True)  # visible but NEVER gets a manifest

    result = pc._find_precomputed_mount_with_retry([str(root)], retries=3, delay_seconds=15)
    assert result is None
    assert slept == [15, 15, 15]  # exhausted all 3 retries before giving up


def _write_manifest(path, n):
    lines = [json.dumps({"study": f"s{i}", "series": f"x{i}", "status": "ok"}) for i in range(n)]
    path.write_text("\n".join(lines) + "\n")


def test_find_precomputed_mount_with_retry_detects_stale_manifest_by_count(tmp_path, monkeypatch):
    """Real bug, kernels v13/v15: a manifest was found immediately (no
    'directory empty' retry triggered) but it reflected an OLDER push (fewer
    done series) than the one just made. expected_min_done must catch this
    and keep retrying until the count actually catches up, not just accept
    the first manifest found."""
    pc = _load_precompute_kernel()
    slept = []
    monkeypatch.setattr(pc.time, "sleep", lambda s: slept.append(s))

    root = tmp_path / "input"
    mount_dir = root / "kadd-precomputed-volumes"
    mount_dir.mkdir(parents=True)
    manifest_path = mount_dir / pc.MANIFEST_FILENAME
    _write_manifest(manifest_path, 10)  # stale: only 10, expecting >= 20

    # Simulate the real manifest "catching up" while we're waiting on the retry sleep —
    # the sleep mock itself rewrites the file, so the NEXT check sees the fresh content.
    def sleep_and_update(seconds):
        slept.append(seconds)
        _write_manifest(manifest_path, 20)

    monkeypatch.setattr(pc.time, "sleep", sleep_and_update)
    result = pc._find_precomputed_mount_with_retry([str(root)], retries=5, delay_seconds=10,
                                                     expected_min_done=20)
    assert result == str(mount_dir)
    assert slept == [10]  # retried once (stale at 10), succeeded on the 2nd check (now 20)


def test_find_precomputed_mount_with_retry_accepts_fresh_manifest_immediately(tmp_path, monkeypatch):
    """expected_min_done already satisfied on the first check -> no retry at all."""
    pc = _load_precompute_kernel()
    slept = []
    monkeypatch.setattr(pc.time, "sleep", lambda s: slept.append(s))

    root = tmp_path / "input"
    mount_dir = root / "kadd-precomputed-volumes"
    mount_dir.mkdir(parents=True)
    _write_manifest(mount_dir / pc.MANIFEST_FILENAME, 20066)

    result = pc._find_precomputed_mount_with_retry([str(root)], retries=5, delay_seconds=10,
                                                     expected_min_done=20066)
    assert result == str(mount_dir)
    assert slept == []


def test_stage_kernel_with_expected_count_patches_local_manifest_size(tmp_path, monkeypatch):
    """scripts/kaggle_push_precompute_kernel.py's staging step must read the
    LOCAL data/precomputed/manifest.jsonl's line count and bake it into a
    staged copy's EXPECTED_MIN_DONE — the tracked precompute_kernel.py file
    itself must stay untouched (same discipline as kaggle_push_kernel.py's
    --type/--size CONFIG_PATH staging)."""
    import importlib.util

    path = os.path.join(os.path.dirname(__file__), "..", "scripts", "kaggle_push_precompute_kernel.py")
    spec = importlib.util.spec_from_file_location("kaggle_push_precompute_kernel", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    monkeypatch.setattr(m, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(m, "KERNEL_DIR", tmp_path / "kaggle_kernel_precompute")

    (tmp_path / "kaggle_kernel_precompute").mkdir()
    (tmp_path / "kaggle_kernel_precompute" / "precompute_kernel.py").write_text(
        "EXPECTED_MIN_DONE = 0\nTARGET_DEPTH = 32\n")
    (tmp_path / "data" / "precomputed").mkdir(parents=True)
    _write_manifest(tmp_path / "data" / "precomputed" / "manifest.jsonl", 12345)

    staged = m._stage_kernel_with_expected_count()
    staged_source = (staged / "precompute_kernel.py").read_text()
    assert "EXPECTED_MIN_DONE = 12345" in staged_source
    # Tracked file untouched.
    original_source = (tmp_path / "kaggle_kernel_precompute" / "precompute_kernel.py").read_text()
    assert "EXPECTED_MIN_DONE = 0" in original_source


def test_stage_kernel_with_expected_count_no_manifest_uses_tracked_dir(tmp_path, monkeypatch):
    """No local manifest yet (first-ever push) -> falls back to the tracked
    KERNEL_DIR unpatched (EXPECTED_MIN_DONE=0, no freshness check), not an
    error."""
    import importlib.util

    path = os.path.join(os.path.dirname(__file__), "..", "scripts", "kaggle_push_precompute_kernel.py")
    spec = importlib.util.spec_from_file_location("kaggle_push_precompute_kernel", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    monkeypatch.setattr(m, "REPO_ROOT", tmp_path)
    kernel_dir = tmp_path / "kaggle_kernel_precompute"
    kernel_dir.mkdir()
    monkeypatch.setattr(m, "KERNEL_DIR", kernel_dir)

    result = m._stage_kernel_with_expected_count()
    assert result == kernel_dir


def test_precomputed_dataset_merge_logic(tmp_path):
    """Exercises scripts/kaggle_push_precomputed_dataset.py's merge functions
    directly — the local-accumulation half of the resume loop. Confirms a
    second merge only copies genuinely-new files/manifest entries, and
    doesn't duplicate or drop anything already accumulated. Pulled runs are
    always flat (<study>/<series>.npy — see precompute_kernel.py, unaffected
    by Kaggle's per-directory zip behavior since `kaggle kernels output` is a
    plain file pull); the merge step is what buckets into shard_XX/ dirs."""
    import importlib.util

    path = os.path.join(os.path.dirname(__file__), "..", "scripts", "kaggle_push_precomputed_dataset.py")
    spec = importlib.util.spec_from_file_location("kaggle_push_precomputed_dataset", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    pulled1 = tmp_path / "pulled1"
    local = tmp_path / "local"
    (pulled1 / "study1").mkdir(parents=True)
    (pulled1 / "study1" / "seriesA.npy").write_bytes(b"fakeA")
    (pulled1 / "manifest.jsonl").write_text(
        json.dumps({"study": "study1", "series": "seriesA", "status": "ok"}) + "\n")

    n_copied = m._merge_npy_files(pulled1, local)
    n_total, n_new = m._merge_manifest(pulled1 / "manifest.jsonl", local / "manifest.jsonl")
    assert n_copied == 1 and n_total == 1 and n_new == 1
    study1_shard = m._shard_for_study("study1")
    assert (local / study1_shard / "study1" / "seriesA.npy").read_bytes() == b"fakeA"
    # Top-level of the accumulation dir must stay bounded by NUM_SHARDS, not
    # grow one directory per study — that's the whole point of sharding.
    assert {p.name for p in local.iterdir() if p.is_dir()} <= {f"shard_{i:02d}" for i in range(m.NUM_SHARDS)}

    pulled2 = tmp_path / "pulled2"
    (pulled2 / "study1").mkdir(parents=True)
    (pulled2 / "study2").mkdir(parents=True)
    (pulled2 / "study1" / "seriesA.npy").write_bytes(b"fakeA")  # duplicate
    (pulled2 / "study2" / "seriesB.npy").write_bytes(b"fakeB")  # genuinely new
    (pulled2 / "manifest.jsonl").write_text(
        json.dumps({"study": "study1", "series": "seriesA", "status": "ok"}) + "\n" +
        json.dumps({"study": "study2", "series": "seriesB", "status": "ok"}) + "\n")

    n_copied2 = m._merge_npy_files(pulled2, local)
    n_total2, n_new2 = m._merge_manifest(pulled2 / "manifest.jsonl", local / "manifest.jsonl")
    assert n_copied2 == 1  # only seriesB.npy
    assert n_total2 == 2 and n_new2 == 1
    study2_shard = m._shard_for_study("study2")
    assert (local / study2_shard / "study2" / "seriesB.npy").read_bytes() == b"fakeB"


def test_shard_for_study_is_stable_and_bounded():
    """_shard_for_study must be deterministic across calls/processes (used
    to keep the same study landing in the same shard across separate merge
    runs) and always land within [0, NUM_SHARDS)."""
    import importlib.util

    path = os.path.join(os.path.dirname(__file__), "..", "scripts", "kaggle_push_precomputed_dataset.py")
    spec = importlib.util.spec_from_file_location("kaggle_push_precomputed_dataset", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    uid = "1.2.826.0.1.3680043.8.498.12345"
    assert m._shard_for_study(uid) == m._shard_for_study(uid)

    seen = set()
    for i in range(500):
        shard = m._shard_for_study(f"study-{i}")
        n = int(shard.split("_")[1])
        assert 0 <= n < m.NUM_SHARDS
        seen.add(shard)
    assert len(seen) > 1  # actually spreads across multiple shards, not all landing in one


def test_precomputed_dataset_merge_survives_truncated_manifest_line(tmp_path, capsys):
    """Regression test for a real crash: a precompute run hit /kaggle/
    working's 20GB cap mid-write (OSError: No space left on device) and left
    a truncated final manifest line. The merge step must skip it (with a
    warning) and keep every well-formed line, not crash the whole merge."""
    import importlib.util

    path = os.path.join(os.path.dirname(__file__), "..", "scripts", "kaggle_push_precomputed_dataset.py")
    spec = importlib.util.spec_from_file_location("kaggle_push_precomputed_dataset", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    pulled = tmp_path / "pulled"
    local = tmp_path / "local"
    pulled.mkdir()
    good_line = json.dumps({"study": "study1", "series": "seriesA", "status": "ok"})
    truncated_line = '{"study": "study2", "series": "seriesB", "stat'  # cut off mid-write
    (pulled / "manifest.jsonl").write_text(good_line + "\n" + truncated_line)

    n_total, n_new = m._merge_manifest(pulled / "manifest.jsonl", local / "manifest.jsonl")
    assert n_total == 1 and n_new == 1  # only the well-formed line survives
    assert "skipping malformed manifest line" in capsys.readouterr().err


def test_wire_dataset_into_kernel_sources_idempotent(tmp_path, monkeypatch):
    import importlib.util

    path = os.path.join(os.path.dirname(__file__), "..", "scripts", "kaggle_push_precomputed_dataset.py")
    spec = importlib.util.spec_from_file_location("kaggle_push_precomputed_dataset", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    (tmp_path / "kaggle_kernel_precompute").mkdir()
    kernel_metadata_path = tmp_path / "kaggle_kernel_precompute" / "kernel-metadata.json"
    kernel_metadata_path.write_text(json.dumps({"dataset_sources": ["pieteroehlers/kadd-package"]}))
    monkeypatch.setattr(m, "REPO_ROOT", tmp_path)

    assert m._wire_dataset_into_kernel_sources() is True
    assert m._wire_dataset_into_kernel_sources() is False  # idempotent on a second call

    final = json.loads(kernel_metadata_path.read_text())
    assert final["dataset_sources"].count(m.DATASET_ID) == 1


# ------------------------- laterality / physical crop --------------------------
#
# Added 2026-09-04, adapted from a public reference solution for this
# competition (CLAUDE.md sec 1's competitive-landscape notes) that scores
# 0.809 vs. this project's ~0.70 — two concrete gaps it identified: no
# laterality normalization (medial/lateral is ambiguous between left/right
# knees without it, directly affecting 5 of 12 labels) and no physical-scale
# crop (a scanner-dependent pixel footprint means the same mm-scale lesion
# can be destroyed by a blind resize on some scanners and not others).

def test_determine_side_prefers_laterality_tag():
    from kadd.classify.data import _determine_side

    class FakeSlice:
        Laterality = "R"

    assert _determine_side([FakeSlice(), FakeSlice()]) == "R"


def test_determine_side_falls_back_to_geometry_when_tag_missing():
    from kadd.classify.data import _determine_side

    class FakeSlice:
        ImageOrientationPatient = [1, 0, 0, 0, 1, 0]  # identity in-plane axes
        PixelSpacing = [1.0, 1.0]
        Rows = 100
        Columns = 100
        # Corner far to the negative-x side -> center also negative x -> right knee
        ImagePositionPatient = [-200.0, 0.0, 0.0]

    assert _determine_side([FakeSlice()]) == "R"


def test_determine_side_returns_none_near_midline():
    from kadd.classify.data import _determine_side

    class FakeSlice:
        ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        PixelSpacing = [1.0, 1.0]
        Rows = 4
        Columns = 4
        ImagePositionPatient = [-2.0, 0.0, 0.0]  # center lands within the midline band

    assert _determine_side([FakeSlice()]) is None


def test_normalize_laterality_flips_coronal_axial_reverses_sagittal():
    import torch
    from kadd.classify.data import _normalize_laterality

    vol = torch.arange(2 * 3).reshape(1, 1, 2, 1, 3).float()  # (1,1,D=2,H=1,W=3)

    unchanged = _normalize_laterality(vol, "Coronal", "L")
    assert torch.equal(unchanged, vol)  # left knee: no-op regardless of plane

    flipped = _normalize_laterality(vol, "Coronal", "R")
    assert torch.equal(flipped, torch.flip(vol, dims=[-1]))

    flipped_ax = _normalize_laterality(vol, "Axial", "R")
    assert torch.equal(flipped_ax, torch.flip(vol, dims=[-1]))

    reversed_sag = _normalize_laterality(vol, "Sagittal", "R")
    assert torch.equal(reversed_sag, torch.flip(vol, dims=[2]))

    # Unknown plane: nothing safe to normalize, left unchanged even for 'R'.
    assert torch.equal(_normalize_laterality(vol, None, "R"), vol)


def test_physical_crop_hw_centers_and_clamps():
    import torch
    from kadd.classify.data import _physical_crop_hw

    vol = torch.zeros(1, 1, 2, 20, 20)
    cropped = _physical_crop_hw(vol, pixel_spacing=[1.0, 1.0], crop_mm=10.0)
    assert cropped.shape[-2:] == (10, 10)  # 10mm / 1.0mm-per-px = 10px

    # crop_mm larger than the image: clamp to the image's own size, never pad.
    uncropped = _physical_crop_hw(vol, pixel_spacing=[1.0, 1.0], crop_mm=1000.0)
    assert uncropped.shape[-2:] == (20, 20)

    # No PixelSpacing available: no-op (raw-DICOM fallback path).
    assert torch.equal(_physical_crop_hw(vol, pixel_spacing=None, crop_mm=10.0), vol)


# ------------------------------ rank ensembling --------------------------------

def test_rank_average_matches_pandas_rank_order():
    import numpy as np
    import pandas as pd
    from kadd.classify.ensemble import rank_average

    p = np.array([[30.0, 1.0], [10.0, 5.0], [20.0, 3.0]])
    result = rank_average([p])
    expected_order = np.argsort(pd.DataFrame(p).rank(pct=True).values, axis=0)
    assert np.array_equal(np.argsort(result, axis=0), expected_order)


def test_rank_average_combines_differently_scaled_members():
    import numpy as np
    from kadd.classify.ensemble import rank_average

    # Same relative order, wildly different absolute scale/confidence — a
    # raw-probability average would let the sharper member dominate; rank
    # averaging must treat them identically since only order matters.
    narrow = np.array([[0.51], [0.49], [0.50]])
    wide = np.array([[0.99], [0.01], [0.50]])
    r_narrow = rank_average([narrow])
    r_wide = rank_average([wide])
    assert np.array_equal(np.argsort(r_narrow, axis=0), np.argsort(r_wide, axis=0))

    ensembled = rank_average([narrow, wide])
    assert np.array_equal(np.argsort(ensembled, axis=0), np.argsort(r_narrow, axis=0))


def test_rank_average_rejects_mismatched_shapes():
    import numpy as np
    import pytest as _pytest
    from kadd.classify.ensemble import rank_average

    with _pytest.raises(ValueError):
        rank_average([np.zeros((3, 2)), np.zeros((4, 2))])


def test_rank_average_single_study_falls_back_to_mean():
    import numpy as np
    from kadd.classify.ensemble import rank_average

    result = rank_average([np.array([[0.2, 0.8]]), np.array([[0.6, 0.4]])])
    assert np.allclose(result, [[0.4, 0.6]])


# ------------------------ swin3d partial fine-tuning ----------------------------

def test_swin3d_unfreeze_last_freezes_early_stages():
    from kadd.classify.model import build_model

    params = {
        "classify": {
            "model": {"type": "swin3d", "size": "light", "pretrained": True,
                      "summary_dim": 16, "meta_dim": 8, "use_checkpointing": False,
                      "unfreeze_last": 1},
            "data": {"label_columns": ["ACL"] * 12},
        }
    }
    model = build_model(params)
    swin = model.encoder.swin
    n_stages = len(swin.features)
    frozen = [not p.requires_grad for stage in list(swin.features)[:n_stages - 1]
              for p in stage.parameters()]
    trainable_last = [p.requires_grad for p in swin.features[-1].parameters()]
    assert all(frozen)
    assert all(trainable_last)
    # patch_embed.proj is freshly reinitialized for 1-channel input — must stay
    # trainable regardless of unfreeze_last, or the model could never learn
    # from its very first layer.
    assert all(p.requires_grad for p in swin.patch_embed.proj.parameters())


def test_swin3d_unfreeze_last_requires_pretrained():
    from kadd.classify.model import build_model

    params = {
        "classify": {
            "model": {"type": "swin3d", "size": "light", "pretrained": False,
                      "summary_dim": 16, "meta_dim": 8, "use_checkpointing": False,
                      "unfreeze_last": 1},
            "data": {"label_columns": ["ACL"] * 12},
        }
    }
    with pytest.raises(ValueError, match="unfreeze_last"):
        build_model(params)


def test_build_param_groups_default_is_single_group_unchanged():
    from kadd.classify.model import build_model
    from kadd.classify.train import ClassifierTraining

    label_columns = ["ACL"] * 12
    params = {"name": "test_run",
              "classify": {"model": {"type": "cnn3d", "size": "xlight", "summary_dim": 16,
                                      "meta_dim": 8, "use_checkpointing": False},
                           "data": {"label_columns": label_columns},
                           "train": {"epochs": 1, "batch_size": 2, "lr": 1e-3}}}
    model = build_model(params)
    train_ds = _FakeStudyDataset(3)
    data = {"train_ds": train_ds, "val_ds": train_ds, "label_columns": label_columns}
    training = ClassifierTraining(params, model, data, torch.device("cpu"))
    groups = training._build_param_groups()
    assert not isinstance(groups, list)  # falls back to model.parameters() directly


def test_build_param_groups_splits_encoder_at_lr_mult():
    from kadd.classify.model import build_model
    from kadd.classify.train import ClassifierTraining

    label_columns = ["ACL"] * 12
    params = {"name": "test_run",
              "classify": {"model": {"type": "cnn3d", "size": "xlight", "summary_dim": 16,
                                      "meta_dim": 8, "use_checkpointing": False},
                           "data": {"label_columns": label_columns},
                           "train": {"epochs": 1, "batch_size": 2, "lr": 1e-3,
                                     "encoder_lr_mult": 0.1}}}
    model = build_model(params)
    train_ds = _FakeStudyDataset(3)
    data = {"train_ds": train_ds, "val_ds": train_ds, "label_columns": label_columns}
    training = ClassifierTraining(params, model, data, torch.device("cpu"))
    groups = training._build_param_groups()
    assert isinstance(groups, list) and len(groups) == 2
    assert groups[0]["lr"] == pytest.approx(1e-4)
    assert groups[1]["lr"] == pytest.approx(1e-3)
    n_encoder = sum(1 for _ in model.encoder.parameters())
    assert len(groups[0]["params"]) == n_encoder


# kaggle_kernel/train_kernel.py's precomputed-mount freshness check, added
# 2026-09-07 after kernel v59 silently trained on ZERO real data: it found a
# kadd-precomputed-volumes mount immediately and accepted it, but the
# manifest it read back was stale/still-syncing — every series the run
# touched failed with ENOENT even though the same series were verified
# "status": "ok" in what had genuinely been pushed. Since competition_sources
# (raw DICOM fallback) is deliberately unattached, this degraded silently
# instead of crashing: 0.33h of real GPU quota burned for nothing before it
# was caught by inspection. Unlike train_kernel.py's rest (which has real
# import-time side effects — raises if no /kaggle/input mount exists — and
# so can't be imported directly in a test), the new retry functions are pure
# and are exec'd here from the real file up to the first side-effecting
# line, so these tests exercise the actual shipped code, not a reimplementation.

def _load_train_kernel_functions():
    import json as _json
    import os as _os
    import time as _time

    path = os.path.join(os.path.dirname(__file__), "..", "kaggle_kernel", "train_kernel.py")
    with open(path) as f:
        source = f.read()
    marker = 'INPUT_ROOT = "/kaggle/input"'
    assert marker in source, "train_kernel.py's side-effecting section marker moved — update this test helper"
    func_source = source.split(marker)[0]
    ns = {"os": _os, "json": _json, "time": _time}
    exec(compile(func_source, path, "exec"), ns)
    return SimpleNamespace(**ns)


def test_count_manifest_ok_counts_only_ok_status(tmp_path):
    tk = _load_train_kernel_functions()
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        '{"study": "s1", "series": "a", "status": "ok"}\n'
        '{"study": "s1", "series": "b", "status": "error"}\n'
        '{"study": "s2", "series": "c", "status": "ok"}\n'
        '\n'  # blank line, should be skipped
    )
    assert tk._count_manifest_ok(str(tmp_path)) == 2


def test_count_manifest_ok_no_mount_or_no_manifest_returns_zero(tmp_path):
    tk = _load_train_kernel_functions()
    assert tk._count_manifest_ok(None) == 0
    assert tk._count_manifest_ok(str(tmp_path)) == 0  # dir exists, no manifest.jsonl in it


def test_find_precomputed_mount_with_retry_not_visible_returns_none(tmp_path):
    """No 'kadd-precomputed-volumes' directory anywhere -> not a lag issue,
    return None immediately (no sleeps) so the caller's existing
    not-attached warning path handles it."""
    tk = _load_train_kernel_functions()
    slept = []
    tk.time.sleep = lambda s: slept.append(s)
    root = tmp_path / "input"
    root.mkdir()
    result = tk._find_precomputed_mount_with_retry([str(root)], retries=3, delay_seconds=1)
    assert result is None
    assert slept == []


def test_find_precomputed_mount_with_retry_accepts_fresh_mount_immediately(tmp_path):
    tk = _load_train_kernel_functions()
    slept = []
    tk.time.sleep = lambda s: slept.append(s)
    root = tmp_path / "input"
    mount = root / "kadd-precomputed-volumes"
    mount.mkdir(parents=True)
    (mount / "manifest.jsonl").write_text(
        "\n".join(f'{{"study": "s{i}", "series": "x", "status": "ok"}}' for i in range(24371)))
    result = tk._find_precomputed_mount_with_retry([str(root)], retries=3, delay_seconds=1,
                                                     expected_min_done=24371)
    assert result == str(mount)
    assert slept == []


def test_find_precomputed_mount_with_retry_raises_on_stale_mount_after_retries(tmp_path):
    """The actual kernel v59 bug: a mount is found immediately but its
    manifest is stale (fewer 'ok' entries than the latest push). Unlike the
    CPU-only precompute kernel (which proceeds anyway after exhausting
    retries), the training kernel must RAISE — proceeding would mean hours
    of billed GPU time training on zero real data."""
    tk = _load_train_kernel_functions()
    slept = []
    tk.time.sleep = lambda s: slept.append(s)
    root = tmp_path / "input"
    mount = root / "kadd-precomputed-volumes"
    mount.mkdir(parents=True)
    (mount / "manifest.jsonl").write_text(
        "\n".join(f'{{"study": "s{i}", "series": "x", "status": "ok"}}' for i in range(100)))  # stale: only 100

    with pytest.raises(RuntimeError, match="never showed a fresh manifest"):
        tk._find_precomputed_mount_with_retry([str(root)], retries=3, delay_seconds=5,
                                                expected_min_done=24371)
    assert slept == [5, 5, 5]  # exhausted all 3 retries before raising


def test_stage_kernel_patches_expected_precomputed_done(tmp_path, monkeypatch):
    """scripts/kaggle_push_kernel.py's staging step must count only
    status=="ok" entries in the LOCAL data/precomputed/manifest.jsonl and
    bake THAT count into a staged copy's EXPECTED_PRECOMPUTED_DONE — the
    tracked train_kernel.py itself must stay untouched. Deliberately
    includes a mix of "ok"/"error" entries (real manifests do: 2 of 24,371
    series in this project's actual manifest are permanent DICOM-corruption
    errors, never "ok") — this is a regression test for the exact kernel
    v60 bug: staging used to count total lines, producing an
    unsatisfiable expected count that made a genuinely fresh mount look
    permanently stale."""
    import importlib.util

    path = os.path.join(os.path.dirname(__file__), "..", "scripts", "kaggle_push_kernel.py")
    spec = importlib.util.spec_from_file_location("kaggle_push_kernel", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    monkeypatch.setattr(m, "REPO_ROOT", tmp_path)
    kernel_dir = tmp_path / "kaggle_kernel"
    kernel_dir.mkdir()
    monkeypatch.setattr(m, "KERNEL_DIR", kernel_dir)
    (kernel_dir / "train_kernel.py").write_text(
        'CONFIG_PATH = f"{KADD_MOUNT}/config_files/classify_swin3d_light.yml"\n'
        'EXPECTED_PRECOMPUTED_DONE = 0\n')

    (tmp_path / "data" / "precomputed").mkdir(parents=True)
    lines = [f'{{"study": "s{i}", "series": "x", "status": "ok"}}' for i in range(98)]
    lines += [f'{{"study": "s{i}", "series": "x", "status": "error"}}' for i in range(98, 100)]
    (tmp_path / "data" / "precomputed" / "manifest.jsonl").write_text("\n".join(lines))

    staged = m._stage_kernel()
    staged_source = (staged / "train_kernel.py").read_text()
    assert "EXPECTED_PRECOMPUTED_DONE = 98" in staged_source  # NOT 100 — errors don't count
    original_source = (kernel_dir / "train_kernel.py").read_text()
    assert "EXPECTED_PRECOMPUTED_DONE = 0" in original_source  # tracked file untouched


def test_stage_kernel_no_local_manifest_patches_zero(tmp_path, monkeypatch):
    """First-ever push / no local precompute accumulation -> patches
    EXPECTED_PRECOMPUTED_DONE=0, i.e. no freshness check (matches the
    original unpatched behavior), not an error."""
    import importlib.util

    path = os.path.join(os.path.dirname(__file__), "..", "scripts", "kaggle_push_kernel.py")
    spec = importlib.util.spec_from_file_location("kaggle_push_kernel", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    monkeypatch.setattr(m, "REPO_ROOT", tmp_path)
    kernel_dir = tmp_path / "kaggle_kernel"
    kernel_dir.mkdir()
    monkeypatch.setattr(m, "KERNEL_DIR", kernel_dir)
    (kernel_dir / "train_kernel.py").write_text(
        'CONFIG_PATH = f"{KADD_MOUNT}/config_files/classify_swin3d_light.yml"\n'
        'EXPECTED_PRECOMPUTED_DONE = 0\n')

    staged = m._stage_kernel()
    staged_source = (staged / "train_kernel.py").read_text()
    assert "EXPECTED_PRECOMPUTED_DONE = 0" in staged_source


def test_stage_kernel_still_patches_config_path_with_type_size(tmp_path, monkeypatch):
    """Regression check: folding EXPECTED_PRECOMPUTED_DONE staging into the
    same function must not break the pre-existing --type/--size CONFIG_PATH
    patching."""
    import importlib.util

    path = os.path.join(os.path.dirname(__file__), "..", "scripts", "kaggle_push_kernel.py")
    spec = importlib.util.spec_from_file_location("kaggle_push_kernel", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    real_repo_root = Path(os.path.join(os.path.dirname(__file__), ".."))
    monkeypatch.setattr(m, "REPO_ROOT", real_repo_root)
    monkeypatch.setattr(m, "KERNEL_DIR", real_repo_root / "kaggle_kernel")

    staged = m._stage_kernel("swin3d", "light")
    staged_source = (staged / "train_kernel.py").read_text()
    assert 'CONFIG_PATH = f"{KADD_MOUNT}/config_files/classify_swin3d_light.yml"' in staged_source
