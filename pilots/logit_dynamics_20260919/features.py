"""Paper section 3.1 and Appendix A.2 features, with explicit deterministic ties."""
from __future__ import annotations

import numpy as np
from .protocol import require_slurm

DYNAMICS = ("top1_switch_rate", "topk_weighted_jaccard", "unique_topk_count",
            "top1_mode_frequency", "top1_entropy", "top1_unique_count", "top1_commitment_depth")


def feature_names(layers=12, k=5):
    names = []
    for depth in [*range(13 - layers, 13), "classifier"]:
        names.append(f"depth{depth}_final_predicted_class_logit")
        names.extend(f"depth{depth}_competitor_rank{rank}_logit" for rank in range(1, k + 1))
    return names + list(DYNAMICS)


def build_features(head_logits, final_logits, *, layers=12, k=5):
    require_slurm()
    heads = np.asarray(head_logits, dtype=np.float64)
    final = np.asarray(final_logits, dtype=np.float64)
    if (heads.ndim != 3 or heads.shape[1:] != (12, 100) or final.shape != (len(heads), 100)
            or not 1 <= layers <= 12 or not 1 <= k < 100
            or not np.isfinite(heads).all() or not np.isfinite(final).all()):
        raise ValueError("Expected finite [N,12,100] heads and [N,100] final logits")
    sequence = np.concatenate((heads[:, -layers:], final[:, None]), axis=1)
    count, depths, classes = sequence.shape
    final_pred = np.argmax(final, axis=-1)
    ranks = np.argsort(-sequence, axis=-1, kind="stable")
    top1, topk = ranks[:, :, 0], ranks[:, :, :k]
    predicted = np.take_along_axis(sequence, np.broadcast_to(final_pred[:, None, None], (count, depths, 1)), -1)
    competitors = sequence.copy()
    np.put_along_axis(competitors, np.broadcast_to(final_pred[:, None, None], (count, depths, 1)), -np.inf, -1)
    competitor_ranks = np.argsort(-competitors, axis=-1, kind="stable")[:, :, :k]
    numeric = np.concatenate((predicted, np.take_along_axis(sequence, competitor_ranks, -1)), axis=-1).reshape(count, -1)
    top_values = np.take_along_axis(sequence, topk, -1)
    top_weights = np.exp(top_values - top_values.max(axis=-1, keepdims=True))
    top_weights /= top_weights.sum(axis=-1, keepdims=True)
    weights = np.zeros_like(sequence)
    np.put_along_axis(weights, topk, top_weights, axis=-1)
    membership = np.zeros(sequence.shape, dtype=bool)
    np.put_along_axis(membership, topk, True, axis=-1)
    intersection = np.minimum(weights[:, :-1], weights[:, 1:]).sum(axis=-1)
    jaccard = (intersection / (2.0 - intersection)).mean(axis=1)
    top1_counts = (top1[:, :, None] == np.arange(classes)[None, None]).sum(axis=1)
    probabilities = top1_counts / depths
    entropy = -(probabilities * np.log(np.maximum(probabilities, np.finfo(np.float64).tiny))).sum(axis=1)
    # Last non-final top1 at zero-based j means normalized commitment (j+1)/L.
    last_mismatch_plus_one = np.where(top1 != final_pred[:, None], np.arange(depths)[None] + 1, 0).max(axis=1)
    dynamics = np.stack((
        (top1[:, :-1] != top1[:, 1:]).mean(axis=1), jaccard,
        membership.any(axis=1).sum(axis=1), probabilities.max(axis=1), entropy,
        (top1_counts > 0).sum(axis=1), last_mismatch_plus_one / layers,
    ), axis=1)
    result = np.concatenate((numeric, dynamics), axis=1).astype(np.float32)
    if not np.isfinite(result).all():
        raise FloatingPointError("Nonfinite LogitDynamics features")
    return result


def fit_normalizer(features):
    require_slurm()
    values = np.asarray(features, dtype=np.float64)
    mean, scale = values.mean(axis=0), values.std(axis=0, ddof=0)
    scale[scale == 0] = 1.0
    return {"mean": mean.tolist(), "scale": scale.tolist(), "fit_role": "probe_train",
            "records": len(values), "ddof": 0}


def normalize(features, normalizer):
    require_slurm()
    if normalizer["fit_role"] != "probe_train":
        raise ValueError("Normalizer must be trained exclusively on probe_train")
    values = (np.asarray(features, dtype=np.float64) - normalizer["mean"]) / normalizer["scale"]
    if not np.isfinite(values).all():
        raise FloatingPointError("Nonfinite standardized features")
    return values.astype(np.float32)
