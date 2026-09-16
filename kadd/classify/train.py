"""Single-stage training loop for the v1 knee classifier.

Adapted from poehlers/DenPIE's DensityTraining (den_pie/density/train.py):
config-driven optimizer/scheduler choice, gradient accumulation + clipping,
EMA-smoothed val-loss tracking, atomic checkpoint saves, best + latest
checkpoints, loss-curve plotting. DenPIE's loop is 3-stage (encoder
pretrain / flow / both) because it pairs an encoder with a normalizing flow
for SBI; this is direct end-to-end classification, so there's one stage, one
model, BCEWithLogitsLoss instead of MSE/flow-loss, and macro-AUC is tracked
as the primary metric (with the degeneracy handling documented below).

Uses automatic mixed precision (torch.amp autocast + GradScaler) — added
after the first real Kaggle T4 run hit `torch.OutOfMemoryError` in
`loss.backward()` (14.56GiB GPU, ~12GiB already in use by the time the
backward pass needed another 2.64GiB). AMP roughly halves activation memory
for the Swin3D forward/backward at the cost of a small numerical-precision
tradeoff (standard, not a correctness concern) — cheaper and less invasive
than cutting batch_size/max_series further, which would reduce how much of
each study's series the model sees per step on top of an already-tiny
58-study dataset. GradScaler's `enabled=False` on CPU/MPS makes all of this a
no-op, so this doesn't change local (non-CUDA) behavior at all.

Multi-GPU support (added 2026-08-14, when a Kaggle log showed "GPU 0 has a
total capacity of 14.56 GiB" — the "GPU 0" naming implies more than one GPU
index may exist on that kernel, i.e. Kaggle's T4 accelerator may already be
provisioning 2 T4s, with the second one sitting completely idle since the
code only ever used device 0). Deliberately NOT a plain `nn.DataParallel`
wrap: our batch is series flattened across ALL studies in the batch with
attention-MIL pooling happening *inside* forward(), keyed by a study_index
relative to the whole batch. DataParallel's automatic dim-0 tensor split
would cut a single study's series across GPUs at an arbitrary boundary, so
each replica would attention-pool over only *part* of that study's evidence
and silently produce a wrong prediction — this is a correctness bug, not a
performance detail, so it can't be delegated to the default wrapper.
Instead: `_shard_batch_by_study` splits by STUDY boundary (never splitting
one study's series across devices), remaps each shard's study_index to be
locally 0-based, and `_forward_multi_gpu` uses the same low-level primitives
DataParallel itself uses (`replicate` + `parallel_apply` + concatenation)
but with this study-aware scatter instead of the default one.
`parallel_apply`'s worker threads pick up the *calling* thread's autocast
state automatically (confirmed by reading torch's parallel_apply source
directly, not assumed) — so calling it from inside the existing
`torch.amp.autocast(...)` block is enough, no extra plumbing needed.
Falls back to the single-GPU path automatically when only one (or zero)
CUDA device is visible — `torch.cuda.device_count()` is logged explicitly
at startup so the next real run confirms how many GPUs Kaggle actually gave
us; this couldn't be tested locally (no multi-GPU CUDA hardware available
here) so correctness rests on careful reasoning about study boundaries and
verified torch primitive behavior, not an end-to-end local test — treat the
next Kaggle run's log as the real verification.
"""
import logging
import os
import random
import re
import shutil
import tempfile
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel as parallel
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from kadd.classify.data import collate_fn
from kadd.util.logger import separator


def _even_chunks(n, k):
    """Split range(n) into k contiguous, as-even-as-possible (start, end) chunks."""
    base, rem = divmod(n, k)
    chunks = []
    start = 0
    for i in range(k):
        size = base + (1 if i < rem else 0)
        chunks.append((start, start + size))
        start += size
    return chunks


class _EpochSubsetSampler(torch.utils.data.Sampler):
    """Draws a fresh random subset of `num_samples` indices every time it's
    iterated — a PyTorch DataLoader calls sampler.__iter__() fresh on every
    `for batch in loader` pass, so wrapping this in a DataLoader built ONCE
    gives a different random subset each epoch without ever rebuilding the
    DataLoader (or its worker processes). This replaces an earlier design
    that rebuilt a fresh Subset + DataLoader every epoch — which caused two
    separate real host-RAM OOM-kills on Kaggle at epoch boundaries (spawning
    a whole new worker pool every epoch, sometimes overlapping with the
    previous epoch's workers still tearing down). Sampling without
    replacement within an epoch (random.sample); no state carried between
    epochs, so a study can repeat on consecutive epochs by chance.
    """

    def __init__(self, data_source_len: int, num_samples: int):
        self.data_source_len = data_source_len
        self.num_samples = num_samples

    def __iter__(self):
        return iter(random.sample(range(self.data_source_len), self.num_samples))

    def __len__(self):
        return self.num_samples


_BLOCK_ADD_RE = re.compile(r'(features\.\d+\.\d+)\.(?!block\.)')
_BLOCK_STRIP_RE = re.compile(r'(features\.\d+\.\d+)\.block\.')


def _remap_checkpoint_block_keys(saved_sd, current_sd):
    """See load_checkpoint_for_resume's docstring for why this exists:
    _CheckpointBlock (model.py) wraps a checkpointed swin stage as
    `self.block`, so classify.model.use_checkpointing changes parameter
    NAMES (adds/removes a ".block." segment after "features.<i>.<j>") even
    though the actual weights are identical. Only nn.Sequential stages
    (real block-groups) ever get this treatment — patch-merging layers
    (e.g. "features.1.reduction.weight") have no numeric block index after
    the stage index, so the regex naturally leaves them untouched.

    Tries stripping ".block." (checkpoint saved WITH checkpointing, loading
    into a model WITHOUT) and adding it (the reverse) — whichever direction
    produces a key set that exactly matches current_sd's real parameter
    names is used. Falls back to returning saved_sd unchanged if neither
    matches, so the caller's retry raises the original, real error instead
    of masking some other, unrelated mismatch as a bogus "fixed" load.
    """
    current_keys = set(current_sd.keys())
    if set(saved_sd.keys()) == current_keys:
        return saved_sd

    stripped = {_BLOCK_STRIP_RE.sub(r'\1.', k): v for k, v in saved_sd.items()}
    if set(stripped.keys()) == current_keys:
        return stripped

    added = {_BLOCK_ADD_RE.sub(r'\1.block.', k): v for k, v in saved_sd.items()}
    if set(added.keys()) == current_keys:
        return added

    return saved_sd


class ClassifierTraining:
    def __init__(self, params: dict, model: nn.Module, data: dict, device: torch.device):
        self.params = params
        self.cfg = params['classify']['train']
        self.model = model.to(device)
        self.data = data
        self.device = device
        self.save_dir = params['name'] + '/'
        self.label_columns = data['label_columns']
        self.pos_weight = self._compute_pos_weight(data['train_ds'], device)

        self.epochs = self.cfg['epochs']
        self.batch_size = self.cfg['batch_size']
        self.lr = self.cfg['lr']
        self.grad_accum_steps = self.cfg.get('gradient_accumulation_steps', 1)
        self.grad_clip = self.cfg.get('grad_clip', None)
        self.patience = self.cfg.get('patience', None)

        # Per-epoch study subsampling: with a large train set (e.g. the
        # ~4,349-study silver-label set), re-decoding + windowing + resizing
        # every series of every study every epoch dominates wall-clock time
        # far more than model compute — confirmed on a real Kaggle run where
        # a 160K-param cnn3d/xlight model still hadn't finished epoch 1 after
        # 19 minutes. sub_epoch_size caps each epoch to a fresh random subset
        # of this many studies (redrawn every epoch, so different studies are
        # seen over successive epochs) instead of the full train set every
        # time — directly bounds epoch wall-clock regardless of dataset size.
        # Renamed from epoch_size (2026-08-21): "epoch_size" read as if it
        # were the size of an epoch itself, when it's actually a sub-sample
        # taken FROM each epoch — sub_epoch_size names that relationship
        # correctly. False or unset (default) preserves the original
        # behavior: full train set, every epoch — this is NOT the same
        # tradeoff for every model. It was validated as free/beneficial for
        # cnn3d/light specifically (precomputed volumes made full-dataset
        # epochs I/O-cheap there), but a from-scratch 28M-param swin3d
        # encoder is compute-bound, not I/O-bound — full-dataset epochs cost
        # real GPU time regardless of precompute, so this knob still matters
        # per-model. See CLAUDE.md sec 7 for the real measurements behind
        # this on both encoder families.
        self.sub_epoch_size = self.params['classify']['data'].get('sub_epoch_size', None)
        if self.sub_epoch_size is False:
            self.sub_epoch_size = None
        if self.sub_epoch_size is not None:
            logging.info(f"sub_epoch_size={self.sub_epoch_size}: each epoch trains on a fresh "
                         f"random subset of this many studies, not the full train set")
        self._trn_loader = None  # built once, cached for the whole run — see _make_train_loader
        self._last_data_wait_time = 0.0   # set by train_epoch, see _log_data_vs_compute_time
        self._last_compute_time = 0.0

        try:
            import psutil
            psutil.cpu_percent(interval=None)  # prime — first real reading comes at the next call
            self._psutil = psutil
        except ImportError:
            self._psutil = None

        self.amp_enabled = device.type == 'cuda'
        self.scaler = torch.amp.GradScaler(device.type, enabled=self.amp_enabled)

        self.device_ids = list(range(torch.cuda.device_count())) if device.type == 'cuda' else []
        self.multi_gpu = len(self.device_ids) > 1
        if device.type == 'cuda':
            names = [torch.cuda.get_device_name(i) for i in self.device_ids]
            logging.info(f"CUDA devices visible: {len(self.device_ids)} {names}")
        if self.multi_gpu:
            logging.info(f"Multi-GPU study-sharded training enabled across devices {self.device_ids} "
                         f"(requires batch_size >= number of devices to actually use more than one — "
                         f"see config_files/classify_swin3d_v1.yml)")

    def _compute_pos_weight(self, train_ds, device) -> torch.Tensor:
        """Per-label pos_weight (train-set neg/pos ratio) for BCEWithLogitsLoss.

        Motivation, from a real run (kernel version 24, 2026-08-19): trn_loss
        plateaued hard by epoch ~10 while macro_auc kept climbing to 0.65 —
        real ranking signal, but stuck, badly-calibrated absolute
        probabilities. The likely cause: this config trains almost entirely
        on silver (report-derived) labels, which have a systematically LOWER
        per-label positive rate than gold on every one of the 12 labels (see
        CLAUDE.md sec 8's calibration table — e.g. Fracture 4.9% silver vs
        31.0% gold, Synovitis 16.6% vs 46.6%) because gold is a small,
        curated, pathology-enriched subset (58 hand-picked studies, never
        all-negative) while silver reflects the true, much more "normal"-
        leaning population (12.9% of silver studies are entirely negative).
        Standard unweighted BCE lets the abundant negative class dominate
        the gradient for rare labels like Fracture, so the model
        under-predicts positive probability relative to what gold's much
        higher prevalence needs — exactly the stuck-loss/improving-AUC
        pattern observed.

        pos_weight > 1 for a label upweights its positive-class loss
        contribution by that label's train-set negative:positive ratio,
        the standard fix for class imbalance (see torch.nn.BCEWithLogitsLoss
        docs). Computed from train_ds's OWN study_ids/labels (i.e. whatever
        is actually being trained on — silver-only, gold-only, or both,
        depending on config) so this adapts automatically rather than
        hardcoding an assumption about which mode is active.

        Applied to train_epoch's loss only, NOT val_epoch's — val_loss
        should stay a plain, comparable BCE number reflecting calibration
        against gold as-is; re-weighting it too would just be applying the
        same training-time correction twice and make the number harder to
        interpret as a diagnostic.
        """
        labels = torch.tensor([train_ds.labels[s] for s in train_ds.study_ids], dtype=torch.float32)
        n_pos = labels.sum(dim=0)
        n_neg = labels.shape[0] - n_pos
        pos_weight = (n_neg + 1.0) / (n_pos + 1.0)  # +1 smoothing avoids div-by-zero for an all-negative label
        logging.info("Per-label pos_weight (train-set neg:pos ratio): " + ", ".join(
            f"{c}={w:.2f}" for c, w in zip(self.label_columns, pos_weight.tolist())))
        return pos_weight.to(device)

    def _log_gpu_memory(self):
        """Per-GPU peak-memory-since-last-reset, logged once per epoch so
        actual utilization (not guessed) drives batch_size/max_series tuning
        across the model size tiers (see config_files/ and CLAUDE.md sec 9's
        'change batch sizes to maximize GPU usage' request) — added instead
        of leaving that tuning as a blind guess, since GPU memory can't be
        measured on this (non-CUDA) machine.
        """
        if self.device.type != 'cuda':
            return
        for i in self.device_ids:
            peak = torch.cuda.max_memory_allocated(i) / 1e9
            reserved = torch.cuda.memory_reserved(i) / 1e9
            total = torch.cuda.get_device_properties(i).total_memory / 1e9
            logging.info(f"  GPU {i} ({torch.cuda.get_device_name(i)}): "
                         f"peak_allocated={peak:.2f}GB reserved={reserved:.2f}GB "
                         f"/ {total:.2f}GB total ({100 * peak / total:.0f}% of capacity)")
            torch.cuda.reset_peak_memory_stats(i)

    def _log_data_vs_compute_time(self):
        """Direct evidence for the "is this data-loading-bound or GPU-
        compute-bound" question, instead of inferring it from indirect
        signals (e.g. "epoch time didn't change when batch_size did") —
        that inference was made once in this project and got correctly
        challenged as unverified. train_epoch times two things separately:
        time spent blocked waiting for the DataLoader to yield the next
        batch (data_wait_time) vs. time spent in forward/backward/optimizer
        step (compute_time, includes a loss.item() sync so CUDA work is
        actually waited on, not just kernel-launched). With num_workers>0
        these overlap (workers prefetch while the GPU computes the current
        step), so a low data_wait_time doesn't just mean "loading is fast in
        isolation" — it means loading comfortably keeps up with compute. A
        high data_wait_time means the opposite: the DataLoader can't produce
        batches as fast as the GPU consumes them, i.e. genuinely data-bound.
        """
        total = self._last_data_wait_time + self._last_compute_time
        if total <= 0:
            return
        pct_data = 100 * self._last_data_wait_time / total
        logging.info(f"  Data loading: {self._last_data_wait_time:.1f}s  "
                     f"GPU compute: {self._last_compute_time:.1f}s  "
                     f"({pct_data:.0f}% of step time spent waiting on data)")

    def _log_cpu_usage(self):
        """CPU utilization since the last call (psutil.cpu_percent with
        interval=None compares against the previous call, which — called
        once per epoch — naturally gives 'average CPU load over roughly
        this epoch' with no added blocking delay). Read alongside
        _log_data_vs_compute_time: high CPU + high data-wait-time together
        is the signature of num_workers being maxed out and unable to keep
        up (more workers might help); high data-wait-time with LOW CPU
        usage would instead point at I/O (disk/network mount) being the
        limit, not CPU decode — a different fix. Requires psutil, which
        isn't guaranteed on Kaggle's base image (added to the same pip
        install line as pydicom in train_kernel.py); skips cleanly if
        unavailable rather than failing the run over a diagnostic.
        """
        if self._psutil is None:
            logging.info("  CPU: psutil not installed, skipping")
            return
        percent = self._psutil.cpu_percent(interval=None)
        num_workers = self.params['classify']['data'].get('num_workers', 4)
        logging.info(f"  CPU: {percent:.0f}% avg utilization across "
                     f"{self._psutil.cpu_count()} logical cores since last epoch "
                     f"(num_workers={num_workers})")

    def _log_host_memory(self):
        """Host (not GPU) RAM — added after a real run got OOM-killed by the
        OS (bare "Killed" in the log, no Python traceback) at an epoch
        boundary while num_workers=16/prefetch_factor=4 held many decoded 3D
        volumes in worker prefetch queues simultaneously. _log_gpu_memory
        was already being watched; this was the blind spot — DataLoader
        worker memory lives on the host, not the GPU, and nothing was
        logging it. Same psutil dependency/fallback as _log_cpu_usage.
        """
        if self._psutil is None:
            logging.info("  Host RAM: psutil not installed, skipping")
            return
        vm = self._psutil.virtual_memory()
        logging.info(f"  Host RAM: {vm.used / 1e9:.2f}GB used / {vm.total / 1e9:.2f}GB total "
                     f"({vm.percent:.0f}%), {vm.available / 1e9:.2f}GB available")

    def _make_loader(self, ds, shuffle=False, sampler=None, num_workers=None):
        data_cfg = self.params['classify']['data']
        if num_workers is None:
            num_workers = data_cfg.get('num_workers', 4)
        persistent_workers = num_workers > 0
        # prefetch_factor only accepted by DataLoader when num_workers > 0.
        # Worth raising above PyTorch's default (2) specifically when the
        # bottleneck is I/O latency rather than CPU decode work — confirmed
        # on a real Kaggle run via _log_data_vs_compute_time +
        # _log_cpu_usage: 96% of step time spent waiting on data, but CPU
        # only 44% utilized despite num_workers already equal to the
        # kernel's core count. That combination means workers are blocked on
        # file-read I/O, not pegged on decode compute — more workers (even
        # oversubscribing physical cores) and a deeper prefetch queue let
        # more reads be in flight concurrently instead of raising CPU
        # compute, which is what actually helps here. See CLAUDE.md sec 9
        # and config_files/classify_cnn3d_xlight.yml's comment for the log.
        #
        # BUT: oversubscribing num_workers/prefetch_factor pushes cost onto
        # HOST RAM (each worker's prefetch queue holds real decoded 3D
        # volumes), which was not being watched — a real run OOM-killed
        # (bare "Killed" in the log, the OS killer's signature, not a Python
        # exception) at an epoch boundary. See _log_host_memory.
        extra = {}
        if num_workers > 0:
            extra['prefetch_factor'] = data_cfg.get('prefetch_factor', 2)
        if sampler is not None:
            shuffle = False  # DataLoader forbids shuffle + sampler together
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle, sampler=sampler,
                           num_workers=num_workers, pin_memory=True,
                           persistent_workers=persistent_workers,
                           collate_fn=collate_fn, drop_last=False, **extra)

    def _make_train_loader(self, train_ds):
        """Built ONCE and cached for the whole run — including when
        sub_epoch_size subsampling is active. Earlier versions rebuilt a
        fresh Subset + DataLoader every single epoch, which meant spawning
        and tearing down a whole pool of worker processes every epoch; even
        after disabling persistent_workers on that rebuilt loader (an
        attempted fix for a real host-RAM OOM-kill at an epoch boundary),
        a second real run OOM-killed at the exact same point. Rebuilding
        the loader every epoch was the actual problem, not just a
        persistent_workers detail — this eliminates it entirely instead of
        making the rebuild "safer".

        Fresh-random-subset-per-epoch now comes from _EpochSubsetSampler
        instead: its __iter__ is called fresh by the DataLoader every time
        `for batch in loader` runs, so a NEW random draw happens every
        epoch without ever rebuilding the DataLoader or its workers. When
        sub_epoch_size is unset/False (or >= the full train set), behaves
        exactly like the original pre-sub_epoch_size design (plain
        shuffle=True, no sampler) — a single persistent-workers loader for
        the whole run.
        """
        if self._trn_loader is not None:
            return self._trn_loader
        if self.sub_epoch_size is None or self.sub_epoch_size >= len(train_ds):
            self._trn_loader = self._make_loader(train_ds, shuffle=True)
        else:
            sampler = _EpochSubsetSampler(len(train_ds), self.sub_epoch_size)
            self._trn_loader = self._make_loader(train_ds, sampler=sampler)
        return self._trn_loader

    # ------------------------------------------------------------------ #
    #  Optimizer / scheduler
    # ------------------------------------------------------------------ #

    def setup_optimizer_and_scheduler(self):
        weight_decay = self.cfg.get('weight_decay', 0.0)
        optimizer_name = self.cfg.get('optimizer', 'AdamW')
        param_groups = self._build_param_groups()

        if optimizer_name == 'AdamW':
            self.optimizer = optim.AdamW(param_groups, lr=self.lr, weight_decay=weight_decay)
        elif optimizer_name == 'Adam':
            self.optimizer = optim.Adam(param_groups, lr=self.lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        scheduler_type = self.cfg.get('scheduler', 'ReduceLROnPlateau')
        scheduler_params = self.cfg.get('scheduler_params', {})
        if scheduler_type == 'ReduceLROnPlateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, **scheduler_params)
        elif scheduler_type == 'StepLR':
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, **scheduler_params)
        elif scheduler_type == 'CosineAnnealingLR':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, **scheduler_params)
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_type}")

        logging.info(f"Optimizer: {optimizer_name}, lr={self.lr}, scheduler={scheduler_type}")

    def _build_param_groups(self):
        """Splits model.encoder params into their own optimizer group at
        lr * encoder_lr_mult, everything else (meta_embed, attention pool,
        classifier head) at the full configured lr. Defaults to
        encoder_lr_mult=1.0 (a single implicit group, identical to
        self.model.parameters() — no behavior change for any config that
        doesn't set this).

        Motivation: a pretrained encoder starts from a good solution and
        needs only to be nudged, while the head is randomly initialized and
        has everything to learn — one shared lr either undertrains the head
        or destroys the pretrained encoder in the first few hundred steps
        (this is the standard differential-LR fine-tuning pattern; see the
        classify_swin3d_light.yml header for where this was motivated from —
        a public reference solution for this competition explicitly calling
        out the same asymmetry). Applies uniformly regardless of encoder
        type (swin3d or cnn3d) — harmless/no-op for cnn3d configs, which
        simply don't set encoder_lr_mult since cnn3d has no pretrained
        weights to protect in the first place.
        """
        encoder_lr_mult = self.cfg.get('encoder_lr_mult', 1.0)
        if encoder_lr_mult == 1.0:
            return self.model.parameters()

        encoder_params = list(self.model.encoder.parameters())
        encoder_ids = {id(p) for p in encoder_params}
        other_params = [p for p in self.model.parameters() if id(p) not in encoder_ids]
        logging.info(f"Differential LR: encoder params at lr*{encoder_lr_mult}="
                     f"{self.lr * encoder_lr_mult:.2e}, other params at lr={self.lr:.2e}")
        return [
            {'params': encoder_params, 'lr': self.lr * encoder_lr_mult},
            {'params': other_params, 'lr': self.lr},
        ]

    def _move_batch(self, batch):
        return {
            'volumes': batch['volumes'].to(self.device, non_blocking=True),
            'plane_idx': batch['plane_idx'].to(self.device, non_blocking=True),
            'fluid_sensitive': batch['fluid_sensitive'].to(self.device, non_blocking=True),
            'fat_suppression': batch['fat_suppression'].to(self.device, non_blocking=True),
            'study_index': batch['study_index'].to(self.device, non_blocking=True),
            'n_studies': batch['n_studies'],
            'labels': batch['labels'].to(self.device, non_blocking=True),
        }

    # ------------------------------------------------------------------ #
    #  Multi-GPU: study-boundary-aware sharding (see module docstring for
    #  why this can't be a plain nn.DataParallel wrap)
    # ------------------------------------------------------------------ #

    def _shard_batch_by_study(self, b, n_devices):
        """Split a flattened multi-study batch into up to n_devices shards,
        each holding a contiguous range of whole studies (never splitting one
        study's series across shards) with a freshly 0-based study_index.

        Returns (shards: list[dict]) in the same order as the original
        studies, so concatenating each shard's output logits along dim 0
        reproduces the original per-study order.
        """
        n_studies = b['n_studies']
        n_devices = max(1, min(n_devices, n_studies))
        shards = []
        for start, end in _even_chunks(n_studies, n_devices):
            study_ids = torch.arange(start, end, device=b['study_index'].device)
            mask = torch.isin(b['study_index'], study_ids)
            shards.append({
                'volumes': b['volumes'][mask],
                'plane_idx': b['plane_idx'][mask],
                'fluid_sensitive': b['fluid_sensitive'][mask],
                'fat_suppression': b['fat_suppression'][mask],
                'study_index': b['study_index'][mask] - start,  # contiguous chunk -> simple offset remap
                'n_studies': end - start,
            })
        return shards

    def _forward_multi_gpu(self, b):
        n_devices = max(1, min(len(self.device_ids), b['n_studies']))
        device_ids = self.device_ids[:n_devices]
        shards = self._shard_batch_by_study(b, n_devices)

        # Fresh replicate() every call, matching nn.DataParallel's own
        # behavior — parameters change after every optimizer.step(), and
        # replicate() builds the differentiable Broadcast graph that lets
        # gradients accumulate back to self.model's original parameters.
        replicas = parallel.replicate(self.model, device_ids)

        inputs = []
        for shard, dev in zip(shards, device_ids):
            inputs.append((
                shard['volumes'].to(dev, non_blocking=True),
                shard['plane_idx'].to(dev, non_blocking=True),
                shard['fluid_sensitive'].to(dev, non_blocking=True),
                shard['fat_suppression'].to(dev, non_blocking=True),
                shard['study_index'].to(dev, non_blocking=True),
                shard['n_studies'],
            ))
        # parallel_apply's worker threads inherit the calling thread's
        # autocast-enabled state automatically (verified against torch's own
        # source) — must be called from inside the existing autocast block.
        outputs = parallel.parallel_apply(replicas, inputs, devices=device_ids)

        primary = self.device_ids[0]
        logits = torch.cat([logits.to(primary) for logits, _ in outputs], dim=0)
        attn_weights = sum((attn for _, attn in outputs), [])
        return logits, attn_weights

    def _forward(self, b):
        if self.multi_gpu:
            return self._forward_multi_gpu(b)
        return self.model(b['volumes'], b['plane_idx'], b['fluid_sensitive'],
                           b['fat_suppression'], b['study_index'], b['n_studies'])

    # ------------------------------------------------------------------ #
    #  Epochs
    # ------------------------------------------------------------------ #

    def train_epoch(self, loader):
        self.model.train()
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        total_loss = 0.0
        n_batches = 0
        self.optimizer.zero_grad()

        # See _log_data_vs_compute_time's docstring for what this measures
        # and why (direct evidence for the data-bound-vs-compute-bound
        # question, replacing an earlier unverified inference).
        data_wait_time = 0.0
        compute_time = 0.0
        t_prev = time.time()

        for step, batch in enumerate(loader):
            t_got_batch = time.time()
            data_wait_time += t_got_batch - t_prev

            if batch is None:
                # Every study in this batch failed to load (see
                # data.py collate_fn / KneeDataset.__getitem__) — real DICOM
                # corruption, not a bug. Skip the step rather than crash a
                # multi-hour run over a handful of bad files.
                t_prev = time.time()
                continue
            b = self._move_batch(batch)
            with torch.amp.autocast(self.device.type, enabled=self.amp_enabled):
                logits, _ = self._forward(b)
                loss = loss_fn(logits, b['labels']) / self.grad_accum_steps
            self.scaler.scale(loss).backward()

            if (step + 1) % self.grad_accum_steps == 0:
                if self.grad_clip:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.grad_accum_steps  # .item() syncs CUDA, so compute_time below is real wall-clock, not just kernel-launch time
            n_batches += 1

            t_prev = time.time()
            compute_time += t_prev - t_got_batch

        self._last_data_wait_time = data_wait_time
        self._last_compute_time = compute_time
        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def val_epoch(self, loader):
        """Returns (val_loss, macro_auc, per_label_auc).

        Macro-AUC at this data size (N~58) is degenerate by construction:
        sklearn.roc_auc_score errors/NaNs when a label has only one class
        present in the eval split (near-certain for rare labels like Fracture
        or Baker's over a ~12-study val fold). Those labels are skipped from
        the macro average and logged, rather than crashing the run. This
        metric is a PIPELINE-CORRECTNESS signal at this data size, not a
        leaderboard-meaningful number — see CLAUDE.md / the plan.
        """
        self.model.eval()
        loss_fn = nn.BCEWithLogitsLoss()
        total_loss = 0.0
        n_batches = 0
        all_logits, all_labels = [], []

        for batch in loader:
            if batch is None:
                continue
            b = self._move_batch(batch)
            with torch.amp.autocast(self.device.type, enabled=self.amp_enabled):
                logits, _ = self._forward(b)
                loss = loss_fn(logits, b['labels'])
            total_loss += loss.item()
            n_batches += 1
            all_logits.append(logits.float().cpu())
            all_labels.append(b['labels'].cpu())

        val_loss = total_loss / max(n_batches, 1)
        if not all_logits:
            return val_loss, float('nan'), {}

        probs = torch.sigmoid(torch.cat(all_logits, dim=0)).numpy()
        targets = torch.cat(all_labels, dim=0).numpy()

        per_label_auc = {}
        skipped = []
        for i, name in enumerate(self.label_columns):
            y = targets[:, i]
            if len(np.unique(y)) < 2:
                skipped.append(name)
                continue
            per_label_auc[name] = roc_auc_score(y, probs[:, i])

        if skipped:
            logging.info(f"  macro-AUC: skipped {len(skipped)} single-class label(s) "
                         f"in this val split: {skipped}")

        macro_auc = float(np.mean(list(per_label_auc.values()))) if per_label_auc else float('nan')
        return val_loss, macro_auc, per_label_auc

    # ------------------------------------------------------------------ #
    #  Checkpointing / plotting
    # ------------------------------------------------------------------ #

    def save_checkpoint(self, path, epoch, val_loss, macro_auc,
                         best_macro_auc=None, patience_counter=None, ema_val_loss=None):
        state = {
            'epoch': epoch,
            'val_loss': val_loss,
            'macro_auc': macro_auc,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            # Added 2026-09-03 for resume support (see load_checkpoint_for_resume) —
            # without these, a resumed run would restart early-stopping/LR-scheduler
            # bookkeeping from scratch, silently giving the model MORE patience than
            # the config actually specifies (harmless-ish but not a true resume).
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_macro_auc': best_macro_auc,
            'patience_counter': patience_counter,
            'ema_val_loss': ema_val_loss,
        }
        tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path))
        try:
            torch.save(state, tmp_path)
            shutil.move(tmp_path, path)
        except Exception:
            os.close(tmp_fd)
            raise

    def load_checkpoint_for_resume(self, path):
        """Load a checkpoint saved by save_checkpoint and restore full training
        state (model, optimizer, scheduler, early-stopping bookkeeping). Returns
        (start_epoch, best_macro_auc, patience_counter, ema_val_loss) — start_epoch
        is checkpoint['epoch'] + 1 (the checkpoint records the last COMPLETED
        epoch, so training resumes from the next one). Missing optional keys
        (checkpoints saved before this feature existed) fall back to safe
        from-scratch defaults for just that field rather than failing the whole
        resume — an older best.pth still has real model/optimizer weights worth
        reusing even without full bookkeeping.
        """
        state = torch.load(path, map_location=self.device, weights_only=False)
        try:
            self.model.load_state_dict(state['model_state_dict'])
        except RuntimeError as e:
            # Real bug hit 2026-09-09, not hypothetical: _CheckpointBlock (model.py)
            # wraps a checkpointed stage as `self.block`, which adds/removes a
            # ".block." segment in that stage's parameter names depending on
            # classify.model.use_checkpointing. Toggling that setting between when a
            # checkpoint was saved and now therefore breaks a strict load_state_dict
            # even though the underlying weights are semantically identical (same
            # params, same shapes, just nested one level differently) — kernel v67
            # crashed on exactly this switching use_checkpointing true->false while
            # resuming. Try remapping .block. in/out before giving up.
            remapped = _remap_checkpoint_block_keys(state['model_state_dict'], self.model.state_dict())
            self.model.load_state_dict(remapped)
            logging.info(f"Checkpoint {path} used a different use_checkpointing setting than the "
                         f"current config — remapped '.block.' in state_dict keys automatically "
                         f"({e.__class__.__name__} on the first attempt).")
        # Optimizer/scheduler state can genuinely mismatch a checkpoint from before —
        # e.g. encoder_lr_mult changes the optimizer's param-group COUNT (1 group ->
        # encoder+other), which torch.optim.Optimizer.load_state_dict rejects outright
        # (ValueError: "loaded state dict has a different number of parameter groups").
        # This is a real, likely-to-recur scenario in a project where configs change
        # between runs, not a hypothetical — caught and logged rather than failing the
        # whole resume, same tolerance already applied to individually-missing keys
        # below: the model weights are still real and worth reusing even when the
        # optimizer's momentum/variance state and the scheduler's own plateau-tracking
        # state can't carry over cleanly.
        try:
            self.optimizer.load_state_dict(state['optimizer_state_dict'])
        except ValueError as e:
            logging.warning(f"Could not restore optimizer state from {path} ({e}) — "
                             f"resuming with a freshly-initialized optimizer instead "
                             f"(model weights still restored normally).")
        if state.get('scheduler_state_dict') is not None:
            try:
                self.scheduler.load_state_dict(state['scheduler_state_dict'])
            except (ValueError, KeyError) as e:
                logging.warning(f"Could not restore scheduler state from {path} ({e}) — "
                                 f"resuming with a freshly-initialized scheduler instead.")
        best_macro_auc = state.get('best_macro_auc')
        if best_macro_auc is None:
            best_macro_auc = state.get('macro_auc', -float('inf'))
        patience_counter = state.get('patience_counter') or 0
        ema_val_loss = state.get('ema_val_loss')
        start_epoch = state['epoch'] + 1
        logging.info(f"Resumed from checkpoint {path}: start_epoch={start_epoch}, "
                     f"best_macro_auc={best_macro_auc:.4f}, patience_counter={patience_counter}")
        return start_epoch, best_macro_auc, patience_counter, ema_val_loss

    def plot_metrics(self, trn_loss, val_loss, val_auc):
        loss_dir = self.save_dir + 'loss/'
        os.makedirs(loss_dir, exist_ok=True)

        plt.figure()
        plt.plot(trn_loss, label='Training loss')
        plt.plot(val_loss, label='Validation loss')
        plt.xlabel('Epoch')
        plt.ylabel('BCE loss')
        plt.legend()
        plt.savefig(loss_dir + 'loss.pdf')
        plt.close()

        plt.figure()
        plt.plot(val_auc, label='Validation macro-AUC (pipeline check only — see CLAUDE.md)')
        plt.xlabel('Epoch')
        plt.ylabel('Macro-AUC')
        plt.legend()
        plt.savefig(loss_dir + 'macro_auc.pdf')
        plt.close()

        np.save(loss_dir + 'loss.npy', [trn_loss, val_loss])
        np.save(loss_dir + 'macro_auc.npy', val_auc)

    # ------------------------------------------------------------------ #
    #  Main loop
    # ------------------------------------------------------------------ #

    def train_loop(self):
        model_dir = self.save_dir + 'models/'
        os.makedirs(model_dir, exist_ok=True)

        self.setup_optimizer_and_scheduler()

        trn_loader = self._make_train_loader(self.data['train_ds'])
        # val_num_workers defaults much lower than train's num_workers — val
        # is small (58 gold studies, no sub_epoch_size cap) and runs once per
        # epoch, so it doesn't need heavy I/O parallelism, but with the new
        # build-once-per-run loaders (see _make_train_loader), BOTH loaders'
        # workers stay alive persistently for the entire run now, not just
        # transiently. A real Kaggle run got OOM-killed mid-epoch-1 (not at
        # an epoch boundary — confirming the rebuild-per-epoch bug was fixed,
        # but revealing steady-state pressure instead) with train+val workers
        # both at num_workers=8 (16 total persistent workers the whole run).
        # Giving val far fewer workers cuts that steady-state total
        # substantially without reducing train's I/O parallelism at all.
        val_num_workers = self.params['classify']['data'].get('val_num_workers', 2)
        val_loader = self._make_loader(self.data['val_ds'], shuffle=False, num_workers=val_num_workers)

        trn_losses, val_losses, val_aucs = [], [], []
        best_macro_auc = -float('inf')
        ema_val_loss = None
        ema_alpha = 0.1
        patience_counter = 0
        start_epoch = 0

        resume_from = self.cfg.get('resume_from')
        if resume_from:
            start_epoch, best_macro_auc, patience_counter, ema_val_loss = \
                self.load_checkpoint_for_resume(resume_from)

        separator()
        epoch_size_note = (f", sub_epoch_size={self.sub_epoch_size}" if self.sub_epoch_size is not None else "")
        # Read max_series off the actual built dataset, not the raw config dict — guaranteed to
        # match what KneeDataset.__getitem__ is really capping at, not just what was requested.
        max_series_note = f", max_series={self.data['train_ds'].max_series}"
        resume_note = f", resumed from epoch {start_epoch}" if resume_from else ""
        logging.info(f"Training for {self.epochs} epochs "
                     f"(train={len(self.data['train_ds'])}, val={len(self.data['val_ds'])}"
                     f"{epoch_size_note}{max_series_note}{resume_note})")
        separator()

        start_time = time.time()
        epoch = start_epoch
        val_loss, macro_auc = float('nan'), float('nan')
        for epoch in range(start_epoch, self.epochs):
            trn_loss = self.train_epoch(trn_loader)
            val_loss, macro_auc, _ = self.val_epoch(val_loader)

            trn_losses.append(trn_loss)
            val_losses.append(val_loss)
            val_aucs.append(macro_auc)

            ema_val_loss = val_loss if ema_val_loss is None else (
                ema_alpha * val_loss + (1 - ema_alpha) * ema_val_loss)

            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

            # With differential LR (_build_param_groups), param_groups[0] is the encoder
            # group (a lower LR) — reporting only that would misleadingly read as the
            # whole model's LR having dropped. Report every group when there's more than
            # one; single-group configs (the common case) keep the original plain format.
            if len(self.optimizer.param_groups) > 1:
                lr_note = ", ".join(f"{g.get('lr'):.2e}" for g in self.optimizer.param_groups)
                lr_note = f"lr=[{lr_note}] (encoder, other)"
            else:
                lr_note = f"lr={self.optimizer.param_groups[0]['lr']:.2e}"
            logging.info(f"Epoch {epoch + 1}/{self.epochs}  trn_loss={trn_loss:.6f}  "
                         f"val_loss={val_loss:.6f}  ema_val={ema_val_loss:.6f}  "
                         f"macro_auc={macro_auc:.4f}  {lr_note}")
            self._log_gpu_memory()
            self._log_data_vs_compute_time()
            self._log_cpu_usage()
            self._log_host_memory()

            if not np.isnan(macro_auc) and macro_auc > best_macro_auc:
                best_macro_auc = macro_auc
                patience_counter = 0
                self.save_checkpoint(model_dir + 'best.pth', epoch, val_loss, macro_auc,
                                      best_macro_auc=best_macro_auc, patience_counter=patience_counter,
                                      ema_val_loss=ema_val_loss)
                logging.info(f"  -> new best macro-AUC: {macro_auc:.4f}")
            else:
                patience_counter += 1

            # 'latest.pth' overwritten every epoch (not just every 10th) — the
            # periodic epoch_N.pth snapshots this replaced (2026-08-22) meant
            # a run killed externally mid-epoch-range (e.g. kernel v35's
            # 6-hour timeout kill at epoch 54, see CLAUDE.md) only had
            # epoch_50.pth to recover from, losing real progress. Only 2
            # checkpoint files exist now (best.pth, latest.pth), so this is
            # also less disk pressure toward /kaggle/working's 20GB cap.
            self.save_checkpoint(model_dir + 'latest.pth', epoch, val_loss, macro_auc,
                                  best_macro_auc=best_macro_auc, patience_counter=patience_counter,
                                  ema_val_loss=ema_val_loss)
            self.plot_metrics(trn_losses, val_losses, val_aucs)

            if self.patience and patience_counter >= self.patience:
                logging.info(f"Early stopping at epoch {epoch + 1} (patience={self.patience})")
                break

        elapsed = time.time() - start_time
        h, m, s = int(elapsed // 3600), int(elapsed % 3600 // 60), int(elapsed % 60)
        logging.info(f"Total training time: {h}h {m}m {s}s")
        separator()
        logging.info(f"Training complete. Best macro-AUC: {best_macro_auc:.4f} "
                     "(pipeline-correctness signal at this data size, not leaderboard-meaningful)")
        return best_macro_auc

    def main(self):
        return self.train_loop()
