"""Leakage-resistant selection, alignment, complementarity, and grouped uncertainty tools."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np

from .evaluate import detector_metrics


P_NORMS = (0.5, 1, 2, 3, 4, 8, 16, np.inf)


def pnorm_error_score(logits: np.ndarray, p: float, temperature: float | None = None,
                      epsilon: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(logits.astype(np.float64), ord=p, axis=1, keepdims=True)
    normalized = logits / np.maximum(norm, epsilon)
    if temperature is None:
        return -normalized.max(1)
    shifted = normalized / temperature
    shifted -= shifted.max(1, keepdims=True)
    probability = np.exp(shifted)
    probability /= probability.sum(1, keepdims=True)
    return 1 - probability.max(1)


def select_pnorm(validation_logits: np.ndarray, validation_y: np.ndarray,
                 temperatures: Sequence[float] | None = None) -> Dict[str, float]:
    """Select only from validation inputs; higher AUROC, then lower AURC."""
    candidates = [(p, None) for p in P_NORMS] if temperatures is None else [
        (p, t) for p in P_NORMS for t in temperatures]
    ranked = []
    for p, temperature in candidates:
        score = pnorm_error_score(validation_logits, p, temperature)
        metrics = detector_metrics(validation_y, score)
        ranked.append((metrics["auroc"], -metrics["aurc"], -float(p),
                       -float(temperature or 0), p, temperature, metrics))
    *_, p, temperature, metrics = max(ranked)
    return {"p": float(p), "temperature": None if temperature is None else float(temperature),
            "validation_auroc": metrics["auroc"], "validation_aurc": metrics["aurc"]}


def grouped_split(groups: Sequence[str], train_fraction: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    groups = np.asarray(groups)
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique)
    cut = min(max(int(len(shuffled) * train_fraction), 1), len(shuffled) - 1)
    left = np.isin(groups, shuffled[:cut])
    return np.flatnonzero(left), np.flatnonzero(~left)


def paired_group_bootstrap(y: np.ndarray, candidate: np.ndarray, reference: np.ndarray,
                           groups: Sequence[str], repetitions: int = 2000,
                           seed: int = 20260830) -> Dict[str, object]:
    """Paired resampling unit is the base photograph, never an individual row."""
    from sklearn.metrics import roc_auc_score

    y, candidate, reference, groups = map(np.asarray, (y, candidate, reference, groups))
    unique = np.unique(groups)
    positions = {g: np.flatnonzero(groups == g) for g in unique}
    rng = np.random.default_rng(seed)
    auroc, aurc = [], []

    def aurc_only(labels, scores):
        # Avoid detector_metrics here: the bootstrap needs only AURC, while
        # detector_metrics also recomputes AUROC and AUPRC on every replicate.
        order = np.argsort(-scores, kind="stable")
        retained_errors = labels[order][::-1]
        return float((np.cumsum(retained_errors) /
                      np.arange(1, len(retained_errors) + 1)).mean())
    for _ in range(repetitions):
        sampled = rng.choice(unique, len(unique), replace=True)
        idx = np.concatenate([positions[g] for g in sampled])
        if np.unique(y[idx]).size < 2:
            continue
        auroc.append(roc_auc_score(y[idx], candidate[idx]) - roc_auc_score(y[idx], reference[idx]))
        aurc.append(aurc_only(y[idx], candidate[idx]) -
                    aurc_only(y[idx], reference[idx]))
    def summarize(values):
        values = np.asarray(values)
        return {"mean": float(values.mean()), "ci95": np.percentile(values, [2.5, 97.5]).tolist(),
                "fraction_gt_zero": float((values > 0).mean()), "repetitions": int(len(values))}
    return {"auroc_delta": summarize(auroc), "aurc_delta": summarize(aurc),
            "resampling_unit": "base_image_group"}


def complementarity(y: np.ndarray, score: np.ndarray, msp_score: np.ndarray,
                    budget: float = 0.5) -> Dict[str, float]:
    from scipy.stats import spearmanr

    y, score, msp_score = map(np.asarray, (y, score, msp_score))
    count = int(np.floor(len(y) * budget))
    flagged = np.argsort(score, kind="stable")[-count:]
    msp_flagged = np.argsort(msp_score, kind="stable")[-count:]
    a, b = np.zeros(len(y), bool), np.zeros(len(y), bool)
    a[flagged], b[msp_flagged] = True, True
    blind = (y == 1) & ~b
    return {"errors_caught": int((a & (y == 1)).sum()),
            "errors_caught_msp_misses": int((a & ~b & (y == 1)).sum()),
            "errors_msp_catches_detector_misses": int((b & ~a & (y == 1)).sum()),
            "flag_overlap": int((a & b).sum()),
            "msp_blind_recovery": float((a & blind).sum() / max(blind.sum(), 1)),
            "spearman_with_msp": float(spearmanr(score, msp_score).statistic)}


REGISTRY_ID_FIELDS = ("y", "image_id", "source_id", "severity", "store_index", "plan_hash")


def verify_score_registry(paths: Sequence[Path]) -> Dict[str, object]:
    if not paths:
        raise ValueError("no score files")
    reference = np.load(paths[0], allow_pickle=False)
    for field in REGISTRY_ID_FIELDS:
        if field not in reference:
            raise ValueError(f"{paths[0]} lacks {field}")
    for path in paths[1:]:
        candidate = np.load(path, allow_pickle=False)
        for field in REGISTRY_ID_FIELDS:
            if field not in candidate or not np.array_equal(reference[field], candidate[field]):
                raise ValueError(f"score registry misalignment in {field}: {path}")
    digest = hashlib.sha256(reference["store_index"].tobytes()).hexdigest()
    return {"records": int(len(reference["y"])), "files": list(map(str, paths)),
            "store_index_sha256": digest, "aligned": True}
