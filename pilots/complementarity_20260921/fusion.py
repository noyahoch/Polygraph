"""Frozen two-score fusion; numerical work is restricted to Slurm."""
from __future__ import annotations

import time
import warnings
from pathlib import Path

from .common import atomic_json, atomic_npz, load_campaign, lock, read, require_slurm, sha, verify_receipt

SEEDS = (7, 17, 27)
PAIRING = {7: 17, 17: 27, 27: 7}
ARMS = ("DG", "DS", "DDprime")
RECIPE = {"C": 1.0, "l1_ratio": 0.0, "solver": "lbfgs", "max_iter": 1000,
          "tol": 1e-8, "fit_intercept": True, "class_weight": None,
          "warm_start": False, "dtype": "float64", "normalization": "fit-only population mean/std"}


class FitFailure(RuntimeError):
    """A retained failure, never a reason to redraw or change the recipe."""


def numerical():
    require_slurm()
    import numpy as np
    return np


def check_gate(root):
    gate = verify_receipt(root, Path(root) / "evaluation_gate.json")
    if gate.get("campaign_sha256") != sha(Path(root) / "campaign.json"):
        raise RuntimeError("Assessment gate belongs to a different campaign")
    return gate


def parent_scores(root):
    np = numerical()
    contract = load_campaign(root)
    path = Path(contract["parent_ld_root"]) / "evaluation" / "scores.npz"
    expected = contract["parent_hashes"].get(str(path))
    if expected is None or sha(path) != expected:
        raise RuntimeError("Parent score archive is not bound to the new campaign")
    with np.load(path, allow_pickle=False) as saved:
        metadata = {key: saved[key].copy() for key in contract["metadata_keys"]}
        keys, matrix = saved["score_keys"].tolist(), saved["scores"].copy()
    if tuple(contract["seeds"]) != SEEDS or matrix.shape != (7200, 14) or len(set(keys)) != 14:
        raise RuntimeError("Unexpected immutable parent score schema")
    if not np.isfinite(matrix).all() or any(v.dtype != np.int64 or v.shape != (7200,) for v in metadata.values()):
        raise RuntimeError("Malformed parent scores or metadata")
    if len(np.unique(metadata["record_id"])) != 7200 or not np.array_equal(metadata["y"], metadata["pred"] != metadata["label"]):
        raise RuntimeError("Parent record/error identity changed")
    columns = []
    input_keys = []
    for seed in SEEDS:
        for method, original in (("D", "LogitDynamics"), ("G", "G_mean"), ("S", "S_mean")):
            values = matrix[:, keys.index(f"{original}/seed{seed}")]
            if method in ("G", "S") and (np.any(values < 0) or np.any(values > 1)):
                raise RuntimeError("G/S inputs must remain their original mean probability scores")
            input_keys.append(f"{method}/seed{seed}")
            columns.append(values)
    return contract, metadata, input_keys, np.asarray(columns, dtype=np.float64).T


def select_pool(metadata, scores, photos):
    np = numerical()
    ids = np.asarray(photos, dtype=np.int64)
    if len(ids) != 400 or len(np.unique(ids)) != 400:
        raise RuntimeError("Fusion pool must contain exactly 400 unique source photographs")
    mask = np.isin(metadata["image_id"], ids)
    selected = {key: values[mask] for key, values in metadata.items()}
    groups, counts = np.unique(selected["image_id"], return_counts=True)
    if int(mask.sum()) != 3600 or not np.array_equal(groups, np.sort(ids)) or not np.all(counts == 9):
        raise RuntimeError("Fusion pool lost source photographs or their nine views")
    return selected, scores[mask]


def design(scores, keys, seed, arm):
    np = numerical()
    second = {"DG": f"G/seed{seed}", "DS": f"S/seed{seed}",
              "DDprime": f"D/seed{PAIRING[seed]}"}[arm]
    return np.asarray(scores[:, [keys.index(f"D/seed{seed}"), keys.index(second)]], dtype=np.float64)


def fit_combiner(x, y, weights=None):
    np = numerical()
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.int64)
    w = np.ones(len(y), dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    if (x.shape != (len(y), 2) or w.shape != y.shape or not np.isfinite(x).all()
            or not np.isfinite(w).all() or np.any(w < 0) or not np.isin(y, [0, 1]).all()
            or np.any(w != np.floor(w)) or float(w.sum()) != 3600.0):
        raise FitFailure("Invalid two-score input or unnormalized photograph multiplicity")
    if any(float(w[y == label].sum()) == 0 for label in (0, 1)):
        raise FitFailure("Combiner fit has only one positive-weight outcome class")
    scaler = StandardScaler().fit(x, sample_weight=w)
    z = scaler.transform(x)
    model = LogisticRegression(C=1.0, l1_ratio=0.0, solver="lbfgs", max_iter=1000,
                               tol=1e-8, fit_intercept=True, class_weight=None, warm_start=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(z, y, sample_weight=w)
    failed = [str(item.message) for item in caught if issubclass(item.category, ConvergenceWarning)]
    if failed or np.any(model.n_iter_ >= 1000):
        raise FitFailure("Fixed combiner optimizer did not converge: " + "; ".join(failed))
    if (not np.array_equal(model.classes_, [0, 1]) or not np.isfinite(model.coef_).all()
            or not np.isfinite(model.intercept_).all()):
        raise FitFailure("Invalid fitted combiner parameters")
    state = {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
             "variance": scaler.var_.tolist(), "coef": model.coef_[0].tolist(),
             "intercept": float(model.intercept_[0]), "n_iter": int(model.n_iter_[0]),
             "fit_row_weight_sum": float(w.sum()), "converged": True}
    return state


def predict_combiner(state, x):
    np = numerical()
    x = np.asarray(x, dtype=np.float64)
    result = ((x - np.asarray(state["mean"])) / np.asarray(state["scale"])) @ np.asarray(state["coef"]) + state["intercept"]
    if not np.isfinite(result).all():
        raise FitFailure("Nonfinite combiner decision score")
    return result


def preserve_unsealed(root, paths, stage):
    """Retain partial publication artifacts before retrying a tiny stage."""
    root = Path(root)
    existing = [Path(path) for path in paths if Path(path).exists()]
    if not existing:
        return
    directory = root / "interrupted" / stage / str(time.time_ns())
    directory.mkdir(parents=True, exist_ok=False)
    saved = {}
    for path in existing:
        digest = sha(path)
        target = directory / path.name
        path.replace(target)
        saved[str(path.relative_to(root))] = {"preserved_as": str(target.relative_to(root)), "sha256": digest}
    atomic_json(directory / "receipt.json", {"complete": True, "reason": "Stage files existed without a completion seal",
                "campaign_sha256": sha(root / "campaign.json"), "preserved": saved})


def fit(root):
    np = numerical()
    root = Path(root)
    contract, metadata, keys, scores = parent_scores(root)
    verify_receipt(root, root / "cache" / "complete.json")
    verify_receipt(root, root / "statistics" / "draw_manifest.json")
    if set(contract["fusion_fit_photo_ids"]) & set(contract["fusion_assessment_photo_ids"]):
        raise RuntimeError("Fusion fit and assessment source overlap")
    train, inputs = select_pool(metadata, scores, contract["fusion_fit_photo_ids"])
    directory = root / "fusion"
    with lock(directory, "fit"):
        if (directory / "freeze.json").exists():
            return verify_receipt(root, directory / "freeze.json")
        preserve_unsealed(root, [directory / "models.json", directory / "fit.npz"], "fusion-fit")
        began = time.monotonic()
        models = {}
        for seed in SEEDS:
            for arm in ARMS:
                models[f"{arm}/seed{seed}"] = fit_combiner(design(inputs, keys, seed, arm), train["y"])
        import sklearn
        atomic_json(directory / "models.json", {"complete": True, "recipe": RECIPE, "models": models,
                    "pairing": PAIRING, "sklearn_version": sklearn.__version__,
                    "campaign_sha256": sha(root / "campaign.json"), "elapsed_seconds": time.monotonic() - began})
        atomic_npz(directory / "fit.npz", **train, input_score_keys=np.asarray(keys), input_scores=inputs)
        files = {str(path.relative_to(root)): sha(path) for path in (directory / "models.json", directory / "fit.npz")}
        receipt = {"complete": True, "campaign_sha256": sha(root / "campaign.json"),
                   "models": 9, "fit_photographs": 400, "fit_records": 3600, "files": files,
                   "assessment_metrics_observed": False}
        atomic_json(directory / "freeze.json", receipt)
        return receipt


def predict(root):
    np = numerical()
    root = Path(root)
    gate = check_gate(root)
    verify_receipt(root, root / "fusion" / "freeze.json")
    contract, metadata, keys, scores = parent_scores(root)
    assessment, inputs = select_pool(metadata, scores, contract["fusion_assessment_photo_ids"])
    models = read(root / "fusion" / "models.json")["models"]
    directory = root / "predictions"
    with lock(directory, "fusion"):
        receipt_path = directory / "fusion_complete.json"
        if receipt_path.exists():
            return verify_receipt(root, receipt_path)
        preserve_unsealed(root, [directory / "fusion.npz"], "fusion-predict")
        output, names = [], []
        for seed in SEEDS:
            for method in ("D", "G", "S"):
                name = f"{method}/seed{seed}"
                output.append(inputs[:, keys.index(name)])
                names.append(name)
            for arm in ARMS:
                name = f"{arm}/seed{seed}"
                output.append(predict_combiner(models[name], design(inputs, keys, seed, arm)))
                names.append(name)
        path = directory / "fusion.npz"
        atomic_npz(path, **assessment, input_score_keys=np.asarray(keys), input_scores=inputs,
                   score_keys=np.asarray(names), scores=np.asarray(output, dtype=np.float64).T)
        receipt = {"complete": True, "gate_sha256": sha(root / "evaluation_gate.json"),
                   "campaign_sha256": gate["campaign_sha256"], "assessment_photographs": 400,
                   "assessment_records": 3600, "files": {str(path.relative_to(root)): sha(path)}}
        atomic_json(receipt_path, receipt)
        return receipt
