"""Rank-based ensembling of per-label predictions from multiple models.

Why ranks, not raw probabilities: this competition's metric (macro-averaged
ROC-AUC, CLAUDE.md sec 1) only depends on the ORDER of a model's scores
within a label, never their absolute scale or calibration — AUC is
invariant under any strictly increasing transform of one model's outputs.
Averaging raw sigmoid probabilities lets whichever member happens to be
most confident (sharpest sigmoid, best-calibrated, or just more
overconfident) dominate the combination even though the metric never
rewards that confidence directly. Averaging each member's per-label RANKS
instead combines exactly the information the metric reads and nothing
more — confirmed as the approach used by the reference public solution for
this competition scoring 0.809 (CLAUDE.md sec 1's competitive-landscape
notes), not invented here.

Not yet wired into a real ensemble script — CLAUDE.md's roadmap has the
actual cnn3d+swin3d ensemble as a later step once both have real trained
checkpoints. This is the reusable piece that step will need.
"""
import numpy as np


def rank_average(predictions: list) -> np.ndarray:
    """predictions: list of (n_studies, n_labels) arrays, one per ensemble
    member, all aligned to the SAME study order and label order. Returns one
    (n_studies, n_labels) array of averaged fractional ranks in [0, 1] —
    itself a valid submission-shaped probability-like score (higher = more
    likely positive), even though it's not a calibrated probability.

    Each member's scores are ranked independently, per label column, before
    averaging — so a member with a narrow, poorly-calibrated score range
    contributes exactly as much to the final ordering as one with a wide,
    well-calibrated range.
    """
    if not predictions:
        raise ValueError("rank_average requires at least one prediction array")
    shape = predictions[0].shape
    for p in predictions:
        if p.shape != shape:
            raise ValueError(f"All prediction arrays must share one shape — "
                              f"got {shape} and {p.shape}")

    n_studies = shape[0]
    if n_studies <= 1:
        # A single study's rank within itself is degenerate (always the same
        # value regardless of score) — fractional-rank averaging carries no
        # real ordering information at n=1, so return each member's raw mean
        # instead of a meaningless constant.
        return np.mean(predictions, axis=0)

    ranked = []
    for p in predictions:
        # argsort-of-argsort gives each row's rank (0-indexed); scaled to
        # [0, 1] fractional ranks (pandas' rank(pct=True) equivalent) so
        # ensembling members whose n_studies could someday differ still
        # combine on a comparable scale.
        order = np.argsort(p, axis=0)
        ranks = np.empty_like(order, dtype=np.float64)
        rows = np.arange(n_studies)[:, None]
        cols = np.arange(shape[1])[None, :]
        ranks[order, cols] = rows
        ranked.append(ranks / (n_studies - 1))

    return np.mean(ranked, axis=0)
