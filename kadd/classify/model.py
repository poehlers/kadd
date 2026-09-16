"""Classification model: per-series 3D encoder (Swin3D ViT or CNN3D), shared
across every series in a study, pooled via gated attention-MIL, then a linear
classification head over the 12 knee-abnormality labels.

Adapted from the encoder patterns in poehlers/DenPIE (den_pie/density/encoder.py
Swin3DEncoder + ConvNet3DDensity), which wrap torchvision.models.video.swin3d_t /
a plain 3D CNN for 3D cosmological density fields. Key deviations, since this is
clinical MRI, not periodic fields:
  - No periodic rot90/flip augmentation (knee anatomy isn't periodic; flips would
    corrupt left/right laterality and superior/inferior semantics).
  - Every series in a study is encoded (not one field per sample) and pooled via
    gated attention-MIL, since no single plane serves all 12 labels well.
  - Series acquisition metadata (plane, fluid-sensitivity, fat-suppression) is
    embedded and added to each series' representation before pooling.

Both encoder types come in 4 size tiers (`size: xlight|light|medium|large`,
`classify.model.size`), so capacity/GPU-memory tradeoffs can be dialed the same
way regardless of which encoder family is in use — see SWIN3D_XLIGHT_KWARGS /
SWIN3D_CANONICAL_SIZES and CNN3D_CHANNELS below for what each tier actually is.
"""
import logging

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

# Single source of truth for plane vocabulary — data.py maps Anatomical_Plane
# strings to these indices (last index reserved for unknown/missing).
PLANES = ['Sagittal', 'Coronal', 'Axial']

SIZES = ('xlight', 'light', 'medium', 'large')
# ^ swin3d's size vocabulary — constrained by torchvision's canned
# Swin3D_T/S/B configs (+ the custom 'xlight'), so it can't grow a tier
# swin3d has no matching architecture for. CNN3D_SIZES (defined after
# CNN3D_CHANNELS below, since cnn3d is fully parametric — any channel-list
# tier is buildable) is the separate, independent vocabulary for cnn3d. Kept
# as two tuples rather than one shared one specifically because cnn3d grew a
# tier ('xmedium') swin3d has no equivalent of — see CNN3D_CHANNELS.

# Extra channels needed for meta_fusion='early' (see KneeClassifier): one-hot
# plane (incl. unknown, matching _UNKNOWN_PLANE_IDX in data.py) + fluid_sensitive
# + fat_suppression, each broadcast as a spatially-constant channel.
META_CHANNELS = len(PLANES) + 1 + 2


def _broadcast_meta_channels(volumes, plane_idx, fluid_sensitive, fat_suppression):
    """meta_fusion='early': broadcasts each series' own metadata as extra
    spatially-constant channels (same scalar value at every voxel),
    concatenated onto the 1 real image channel before the encoder sees it.

    No cross-series registration is needed — this never stacks DIFFERENT
    series together (which would require them to share a coordinate system,
    which they don't: different series can have different resolution,
    orientation, slice count). Each series only ever gets ITS OWN metadata
    broadcast onto ITS OWN coordinate grid, whatever that grid is, so a
    constant-valued channel trivially "aligns" with anything.
    """
    n = volumes.shape[0]
    plane_onehot = torch.nn.functional.one_hot(
        plane_idx, num_classes=len(PLANES) + 1
    ).float()  # (n, len(PLANES)+1)
    meta = torch.cat([
        plane_onehot,
        fluid_sensitive.float().unsqueeze(-1),
        fat_suppression.float().unsqueeze(-1),
    ], dim=-1)  # (n, META_CHANNELS)
    meta = meta.view(n, -1, 1, 1, 1).expand(-1, -1, *volumes.shape[2:])
    return torch.cat([volumes, meta], dim=1)


def build_model(params: dict) -> nn.Module:
    """Factory: dispatches on params['classify']['model']['type']
    ('swin3d' or 'cnn3d'); each type further dispatches on
    params['classify']['model']['size'] (xlight/light/medium/large).
    """
    cfg = dict(params['classify']['model'])
    cfg['num_labels'] = len(params['classify']['data']['label_columns'])
    model_type = cfg['type']
    if model_type not in ('swin3d', 'cnn3d'):
        raise ValueError(f"Unknown model type: {model_type} (expected 'swin3d' or 'cnn3d')")
    return KneeClassifier(cfg)


# swin3d_t/s/b are torchvision's only canned configs, and the only ones with
# Kinetics-400 pretrained weights available — used as-is for light/medium/large.
# 'xlight' has no torchvision equivalent (nothing smaller than "tiny" ships),
# so it's built directly from torchvision's SwinTransformer3d base class with a
# custom config (same patch_size/window_size as the canonical configs, so the
# input-shape mechanics already validated for t/s/b carry over unchanged;
# just narrower embed_dim and shallower stage depths). No pretrained weights
# exist for this custom shape.
SWIN3D_CANONICAL_SIZES = {'light': 'swin3d_t', 'medium': 'swin3d_s', 'large': 'swin3d_b'}
SWIN3D_XLIGHT_KWARGS = dict(
    patch_size=[2, 4, 4], embed_dim=48, depths=[2, 2, 2, 2],
    num_heads=[2, 4, 8, 16], window_size=[8, 7, 7], stochastic_depth_prob=0.1,
)


class Swin3DSeriesEncoder(nn.Module):
    """Per-series 3D Swin Transformer (ViT) encoder, in 4 size tiers.

    Replaces patch_embed.proj for single-channel (grayscale MRI) input,
    replaces head with a projection to summary_dim, optional gradient
    checkpointing.

    pretrained=True loads Kinetics-400 video-classification weights (only
    available for size in {light, medium, large} — swin3d_t/s/b). Since
    patch_embed.proj is always replaced for in_channels=1, this only reuses the
    deeper transformer blocks, not the input stem — a genuinely stem-reusing
    ablation (replicate the windowed slice to 3 channels, keep the original
    stem + Kinetics normalization stats) is documented as future work, not
    built here.
    """

    def __init__(self, cfg: dict, in_channels: int = 1):
        super().__init__()
        from kadd.classify.attention_patch import apply_sdpa_patch
        apply_sdpa_patch()

        size = cfg.get('size', 'light')
        pretrained = cfg.get('pretrained', False)
        summary_dim = cfg['summary_dim']
        use_checkpointing = cfg.get('use_checkpointing', True)
        proj_dropout = cfg.get('proj_dropout', 0.0)
        unfreeze_last = cfg.get('unfreeze_last')

        if size == 'xlight':
            if pretrained:
                raise ValueError(
                    "classify.model.pretrained=True has no matching weights for "
                    "size='xlight' (a custom architecture smaller than any "
                    "torchvision Swin3D checkpoint) — set pretrained: false"
                )
            from torchvision.models.video import SwinTransformer3d
            swin = SwinTransformer3d(**SWIN3D_XLIGHT_KWARGS, num_classes=400)
        elif size in SWIN3D_CANONICAL_SIZES:
            from torchvision.models.video import (
                swin3d_t, swin3d_s, swin3d_b,
                Swin3D_T_Weights, Swin3D_S_Weights, Swin3D_B_Weights,
            )
            builders = {
                'light': (swin3d_t, Swin3D_T_Weights.KINETICS400_V1),
                'medium': (swin3d_s, Swin3D_S_Weights.KINETICS400_V1),
                'large': (swin3d_b, Swin3D_B_Weights.KINETICS400_V1),
            }
            build_fn, weights_enum = builders[size]
            swin = build_fn(weights=weights_enum if pretrained else None)
        else:
            raise ValueError(f"Unknown swin3d size: {size} (expected one of {SIZES})")

        old_conv = swin.patch_embed.proj
        swin.patch_embed.proj = nn.Conv3d(
            in_channels=in_channels,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        logging.info(f"Swin3DSeriesEncoder: size={size} pretrained={pretrained}, "
                     f"patch_embed.proj reinitialized for {in_channels}-channel input")

        original_head = swin.head
        swin.head = nn.Identity()
        swin_output_dim = original_head.in_features
        logging.info(f"Swin3DSeriesEncoder: backbone output dim = {swin_output_dim}")

        if use_checkpointing:
            for stage in swin.features:
                if isinstance(stage, nn.Sequential):
                    for i in range(len(stage)):
                        stage[i] = _CheckpointBlock(stage[i])
            logging.info("Swin3DSeriesEncoder: gradient checkpointing enabled")

        if unfreeze_last is not None:
            if not pretrained:
                raise ValueError(
                    "classify.model.unfreeze_last only makes sense with pretrained=True — "
                    "there are no pretrained early-block features to protect on a "
                    "from-scratch backbone (freezing random-init weights would just prevent "
                    "the model from ever learning basic low-level features)."
                )
            # Stage-level granularity, not individual-transformer-block: swin.features is
            # an alternating sequence of block-stages (nn.Sequential of transformer
            # blocks) and patch-merging layers between them. Freezing/unfreezing whole
            # top-level entries — rather than digging into individual blocks inside each
            # Sequential — keeps this simple while still matching the real intent ("only
            # the last blocks move, early ones are generic edge/texture filters that
            # there's not enough supervision here to safely improve"). unfreeze_last
            # counts trailing top-level entries of swin.features to keep trainable.
            n_stages = len(swin.features)
            cutoff = max(0, n_stages - unfreeze_last)
            for i, stage in enumerate(swin.features):
                requires_grad = i >= cutoff
                for p in stage.parameters():
                    p.requires_grad = requires_grad
            logging.info(f"Swin3DSeriesEncoder: unfreeze_last={unfreeze_last} — "
                         f"{cutoff}/{n_stages} feature stages frozen, "
                         f"{n_stages - cutoff}/{n_stages} trainable")
        # patch_embed.proj and swin.head are ALWAYS trainable regardless of unfreeze_last:
        # patch_embed.proj is freshly reinitialized above for in_channels=1 (random
        # weights — freezing it would mean the model never learns from the very first
        # layer), and swin.head is already replaced with nn.Identity() (no params).

        self.swin = swin
        self.projection = nn.Linear(swin_output_dim, summary_dim)
        self.proj_dropout = nn.Dropout(p=proj_dropout)

        n_params = sum(p.numel() for p in self.parameters())
        logging.info(f"Swin3DSeriesEncoder: summary_dim={summary_dim}, params={n_params:,}")

    def forward(self, x):
        features = self.swin(x)
        return self.proj_dropout(self.projection(features))


class _CheckpointBlock(nn.Module):
    """Wraps a block for gradient checkpointing (ported from DenPIE)."""

    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x):
        return checkpoint(self.block, x, use_reentrant=False)


# DenPIE's ConvNet3DDensity (den_pie/density/encoder.py) is a fixed
# 1->32->64->128->256 config — that's exactly the 'light' tier here. The
# other tiers scale it by depth (fewer/more conv blocks) and width (channels
# per block).
CNN3D_CHANNELS = {
    'xlight': [16, 32, 64],
    'light': [32, 64, 128, 256],       # DenPIE's original default
    'medium': [64, 128, 256, 512],
    'xmedium': [32, 64, 128, 256, 512],  # 2026-08-27, cnn3d-only (no swin3d equivalent — see
                                          #   SIZES above): light's exact channel progression
                                          #   (starts at 32, doubles each block) extended one
                                          #   block deeper (5 blocks instead of 4), so it reaches
                                          #   medium's final width (512) via a lower, gentler
                                          #   starting point rather than medium's wider-from-the-
                                          #   first-block design. A genuinely different capacity/
                                          #   compute tradeoff from both light and medium, not a
                                          #   value between them on a single axis.
    'large': [64, 128, 256, 512, 1024],
}

# Independent from swin3d's SIZES (see comment there) — cnn3d is fully
# parametric (any channel list is buildable), so its size vocabulary is
# just "whatever CNN3D_CHANNELS defines," not constrained to match swin3d's.
CNN3D_SIZES = tuple(CNN3D_CHANNELS.keys())

# Union of both families' sizes, for contexts that need one vocabulary
# before knowing which type was chosen (e.g. the CLI's --size choices,
# where --type is a separate flag) — real per-(type, size) validity is
# still checked downstream (kadd.util.config_registry.resolve_config,
# Swin3DSeriesEncoder/CNN3DSeriesEncoder's own ValueErrors).
ALL_SIZES = tuple(sorted(set(SIZES) | set(CNN3D_SIZES)))


class CNN3DSeriesEncoder(nn.Module):
    """Per-series plain 3D CNN encoder, in 4 size tiers.

    Ported from DenPIE's ConvNet3DDensity (den_pie/density/encoder.py):
    conv blocks (kernel=3, stride=2, BatchNorm3d, ReLU) doubling channels
    each block -> global average pool -> linear projection to summary_dim.
    No pretrained-weights option — DenPIE's version doesn't have one either
    (a plain CNN over cosmological/MRI-scale volumes has no equivalent of
    Kinetics-400 video-classification pretraining to borrow from).
    """

    def __init__(self, cfg: dict, in_channels: int = 1):
        super().__init__()
        size = cfg.get('size', 'light')
        if size not in CNN3D_CHANNELS:
            raise ValueError(f"Unknown cnn3d size: {size} (expected one of {CNN3D_SIZES})")
        channels = CNN3D_CHANNELS[size]
        summary_dim = cfg['summary_dim']
        self.use_checkpointing = cfg.get('use_checkpointing', False)

        chans = [in_channels] + channels
        self.blocks = nn.ModuleList([
            self._conv_block(chans[i], chans[i + 1]) for i in range(len(channels))
        ])
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.projection = nn.Linear(channels[-1], summary_dim)

        n_params = sum(p.numel() for p in self.parameters())
        logging.info(f"CNN3DSeriesEncoder: size={size}, channels={channels}, "
                     f"summary_dim={summary_dim}, params={n_params:,}")

    @staticmethod
    def _conv_block(in_ch, out_ch):
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        for block in self.blocks:
            if self.use_checkpointing and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        x = self.global_pool(x).flatten(1)
        return self.projection(x)


class SeriesMetaEmbed(nn.Module):
    """Embeds per-series acquisition metadata (Anatomical_Plane,
    Fluid_Sensitive, Fat_Suppression) and adds it to that series' encoder
    embedding before pooling — gives the model explicit plane/sequence
    context instead of forcing it to infer plane purely from pixel content,
    which also mitigates mixing-plane position-embedding confounds.
    """

    def __init__(self, meta_dim: int, summary_dim: int):
        super().__init__()
        self.plane_embed = nn.Embedding(len(PLANES) + 1, meta_dim)  # +1 = unknown
        self.mlp = nn.Sequential(
            nn.Linear(meta_dim + 2, meta_dim),
            nn.ReLU(inplace=True),
            nn.Linear(meta_dim, summary_dim),
        )

    def forward(self, plane_idx, fluid_sensitive, fat_suppression):
        p = self.plane_embed(plane_idx)
        feats = torch.cat(
            [p, fluid_sensitive.unsqueeze(-1), fat_suppression.unsqueeze(-1)], dim=-1
        )
        return self.mlp(feats)


class GatedAttentionMIL(nn.Module):
    """Gated attention-MIL pooling (Ilse et al. 2018-style) over a study's
    series embeddings: a_i = softmax_i(w^T (tanh(V h_i) * sigmoid(U h_i))),
    pooled = sum_i a_i h_i.
    """

    def __init__(self, summary_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.V = nn.Linear(summary_dim, hidden_dim)
        self.U = nn.Linear(summary_dim, hidden_dim)
        self.w = nn.Linear(hidden_dim, 1)

    def forward(self, h, study_index, n_studies):
        """
        h: (total_series_in_batch, summary_dim)
        study_index: (total_series_in_batch,) long — which study each row belongs to
        n_studies: number of studies in this batch

        Returns (pooled: (n_studies, summary_dim), attn_weights: list[Tensor]).
        """
        gate = torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))
        scores = self.w(gate).squeeze(-1)

        pooled = h.new_zeros(n_studies, h.shape[-1])
        attn_weights = []
        for s in range(n_studies):
            mask = study_index == s
            a = torch.softmax(scores[mask], dim=0)
            pooled[s] = (a.unsqueeze(-1) * h[mask]).sum(dim=0)
            attn_weights.append(a.detach())
        return pooled, attn_weights


_ENCODER_BUILDERS = {
    'swin3d': Swin3DSeriesEncoder,
    'cnn3d': CNN3DSeriesEncoder,
}


class KneeClassifier(nn.Module):
    """Full model: shared per-series encoder (swin3d ViT or cnn3d, both in 4
    size tiers) + metadata conditioning + gated attention-MIL pooling +
    linear classification head.

    classify.model.meta_fusion controls WHEN plane/fluid-sensitivity/fat-
    suppression metadata reaches the model — an ablation, not just a config
    knob (2026-08-21):
      - 'late' (default, original design): the encoder sees only the raw
        1-channel image; metadata is embedded (SeriesMetaEmbed) and ADDED to
        the encoder's output embedding, after encoding, right before
        attention-MIL pooling. The encoder itself never sees plane/sequence
        type — it must either infer it implicitly from pixel content or not
        condition on it until pooling.
      - 'early': metadata is broadcast as extra spatially-constant channels
        (see _broadcast_meta_channels) and concatenated onto the image
        channel BEFORE the encoder — every layer, from the first
        conv/patch-embed onward, can condition on plane/sequence type
        directly. SeriesMetaEmbed is not used in this mode (metadata is
        injected once, not both places, for a clean ablation).
    """

    def __init__(self, cfg: dict):
        super().__init__()
        summary_dim = cfg['summary_dim']
        meta_dim = cfg.get('meta_dim', 16)
        num_labels = cfg['num_labels']
        encoder_type = cfg['type']
        self.meta_fusion = cfg.get('meta_fusion', 'late')

        if encoder_type not in _ENCODER_BUILDERS:
            raise ValueError(f"Unknown model type: {encoder_type} (expected 'swin3d' or 'cnn3d')")
        if self.meta_fusion not in ('late', 'early'):
            raise ValueError(f"Unknown meta_fusion: {self.meta_fusion} (expected 'late' or 'early')")

        encoder_in_channels = 1 + META_CHANNELS if self.meta_fusion == 'early' else 1
        self.encoder = _ENCODER_BUILDERS[encoder_type](cfg, in_channels=encoder_in_channels)
        if self.meta_fusion == 'late':
            self.meta_embed = SeriesMetaEmbed(meta_dim, summary_dim)
        self.pool = GatedAttentionMIL(summary_dim)
        self.head = nn.Linear(summary_dim, num_labels)

        n_params = sum(p.numel() for p in self.parameters())
        logging.info(f"KneeClassifier: summary_dim={summary_dim}, num_labels={num_labels}, "
                     f"meta_fusion={self.meta_fusion}, total_params={n_params:,}")

    def forward(self, volumes, plane_idx, fluid_sensitive, fat_suppression,
                study_index, n_studies):
        """
        volumes: (total_series_in_batch, 1, D, H, W)
        plane_idx, fluid_sensitive, fat_suppression, study_index: (total_series_in_batch,)
        n_studies: number of studies in this batch

        Returns (logits: (n_studies, num_labels), attn_weights: list[Tensor]).
        """
        if self.meta_fusion == 'early':
            volumes = _broadcast_meta_channels(volumes, plane_idx, fluid_sensitive, fat_suppression)
            h = self.encoder(volumes)
        else:
            h = self.encoder(volumes)
            h = h + self.meta_embed(plane_idx, fluid_sensitive, fat_suppression)
        pooled, attn_weights = self.pool(h, study_index, n_studies)
        logits = self.head(pooled)
        return logits, attn_weights
