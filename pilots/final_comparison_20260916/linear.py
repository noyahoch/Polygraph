"""One base-training-only 100-logit linear probe and three gated static scores."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from . import protocol
from .heads import (METADATA, SCOPE, _align_metadata, _array_hashes,
                    _before_new_readouts, _context, _fit_serialized, _lock,
                    _match, _validate_role_arrays, _validate_solver, _verify_files,
                    _write_json, _write_npz, head_scores)

NAME = "logits_linear_100"
STATIC_METHODS = ("L", "MSP", "entropy")
FEATURES = [f"classifier_logit_{index}" for index in range(100)]
COLUMNS = list(range(100))
NORMALIZATION = {
    "input": "all 100 ordered frozen classifier logits",
    "dtype": "float64",
    "softmax": "subtract row maximum, exponentiate, divide by row sum",
    "MSP": "1 - max(softmax(logits))",
    "entropy": "-sum(softmax(logits) * log_softmax(logits)); natural logarithm",
    "direction": "larger means greater frozen-classifier error risk"}


def _fit_context(root):
    from .data import load_logits

    root, campaign, roles, bindings = _context(root)
    arrays = load_logits(root, "base_train")
    _validate_role_arrays(arrays, roles, "base_train", 100)
    positives = int(np.count_nonzero(arrays["y"] == 1))
    negatives = int(np.count_nonzero(arrays["y"] == 0))
    if positives + negatives != 14400 or not positives or not negatives:
        raise ValueError("The single linear probe needs both outcomes in all 14,400 base rows")
    counts = {"records": 14400, "source_photographs": 1600,
              "positive": positives, "negative": negatives}
    recipe = {"standardize_on": "base_train", "penalty": "l2", "C": 1.0,
              "solver": "lbfgs", "max_iter": 1000, "tol": 1e-6,
              "class_weight": {"0": 1.0, "1": negatives / positives}, "random_state": 7}
    inputs = {**bindings, "training_role": "base_train", "training_counts": counts,
              "training_input_arrays": _array_hashes(arrays),
              "feature_order": FEATURES, "array_hash_policy": "dtype + shape + contiguous C bytes"}
    return root, arrays, inputs, recipe


def validate(root):
    root, arrays, inputs, recipe = _fit_context(root)
    directory = root / "linear"
    complete = protocol.read(directory / "complete.json")
    _match(complete, {"scope_id": SCOPE, "name": NAME, "family": "L", "fit_count": 1,
                       "training_role": "base_train", "inputs": inputs, "recipe": recipe,
                       "dev_eval_used_for_fit": False, "original_test_access": False},
           "Linear completion")
    _verify_files(directory, complete, {"model.json"})
    visible = {path.name for path in directory.glob("*.json") if not path.name.startswith(".")}
    if visible != {"model.json", "complete.json"}:
        raise RuntimeError("The linear baseline must contain exactly one model and its completion")
    if protocol.sha256(directory / ".fit_started.json") != complete["fit_attempt_sha256"]:
        raise RuntimeError("The one linear fitting attempt changed")
    _match(protocol.read(directory / ".fit_started.json"),
           {"inputs": inputs, "recipe": recipe, "fit_count": 1}, "Linear fitting attempt")
    model = protocol.read(directory / "model.json")
    _match(model, {"scope_id": SCOPE, "name": NAME, "family": "L",
                   "training_role": "base_train", "rows": 14400, "source_photos": 1600,
                   "features": FEATURES, "inputs": inputs, "fit_count": 1}, "Linear model")
    _validate_solver(model, COLUMNS, 14400, recipe)
    head_scores(model, arrays["logits"])
    return model, complete


def fit(root):
    protocol.require_slurm()
    root = Path(root).resolve()
    directory = root / "linear"
    with _lock(directory, ".fit.lock"):
        if (directory / "complete.json").exists():
            return validate(root)[1]
        _before_new_readouts(root)
        protocol.check_cutoff(root, "base")
        if any((directory / name).exists() for name in ("model.json", ".fit_started.json")):
            raise RuntimeError("A prior linear fitting attempt is incomplete; no automatic refit or retuning")
        root, arrays, inputs, recipe = _fit_context(root)
        _write_json(directory / ".fit_started.json",
                    {"inputs": inputs, "recipe": recipe, "fit_count": 1,
                     "started_unix": time.time(), "job_id": os.environ["SLURM_JOB_ID"]})
        model = _fit_serialized(
            arrays["logits"], arrays["y"], COLUMNS, recipe,
            {"scope_id": SCOPE, "name": NAME, "family": "L", "training_role": "base_train",
             "rows": 14400, "source_photos": 1600, "features": FEATURES,
             "inputs": inputs, "fit_count": 1})
        _write_json(directory / "model.json", model)
        _validate_solver(model, COLUMNS, 14400, recipe)
        protocol.check_cutoff(root, "base")
        _before_new_readouts(root)
        complete = {"schema_version": 1, "scope_id": SCOPE, "complete": True,
                    "name": NAME, "family": "L", "fit_count": 1, "inputs": inputs, **inputs,
                    "training_role": "base_train", "recipe": recipe,
                    "files": {"model.json": protocol.sha256(directory / "model.json")},
                    "fit_attempt_sha256": protocol.sha256(directory / ".fit_started.json"),
                    "model_sha256": protocol.sha256(directory / "model.json"),
                    "dev_eval_used_for_fit": False, "original_test_access": False,
                    "selection": "none: fixed convex learner, no CV or checkpoint selection",
                    "neural_epoch_history": "not applicable",
                    "seed_sd": None, "seed_sd_reason": "not applicable: one deterministic fit",
                    "completed_unix": time.time(), "job_id": os.environ["SLURM_JOB_ID"]}
        _write_json(directory / "complete.json", complete)
        return validate(root)[1]


def analytic_scores(logits):
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 100 or not np.isfinite(values).all():
        raise ValueError("Static confidence scores require exactly 100 finite logits")
    shifted = values - np.max(values, axis=1, keepdims=True)
    log_probability = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    probability = np.exp(log_probability)
    result = {"MSP": 1.0 - np.max(probability, axis=1),
              "entropy": -np.sum(probability * log_probability, axis=1)}
    if (not np.allclose(probability.sum(axis=1), 1.0, atol=1e-12, rtol=1e-12)
            or any(not np.isfinite(value).all() for value in result.values())):
        raise ValueError("Invalid frozen-logit probability normalization")
    return result


def _gate_inputs(root):
    gate = protocol.require_evaluation_gate(root)
    if gate.get("complete") is not True or gate.get("scope_id") != SCOPE:
        raise RuntimeError("The parent global evaluation freeze is incomplete")
    return {"heads_freeze_sha256": protocol.sha256(Path(root) / "heads_freeze.json"),
            "evaluation_gate_sha256": protocol.sha256(Path(root) / "evaluation_gate.json")}


def _static_context(root):
    protocol.require_slurm()
    root = Path(root).resolve()
    gate = _gate_inputs(root)
    root, campaign, roles, bindings = _context(root, with_base=True)
    model, complete = validate(root)
    inputs = {**bindings, **gate,
              "linear_model_sha256": protocol.sha256(root / "linear/model.json"),
              "linear_complete_sha256": protocol.sha256(root / "linear/complete.json")}
    return root, roles, model, inputs


def read_static(root):
    from .data import load_logits

    root, roles, model, inputs = _static_context(root)
    path = root / "predictions/dev_eval/static.npz"
    sidecar = protocol.read(path.with_suffix(".json"))
    _match(sidecar, {"scope_id": SCOPE, "complete": True, "role": "dev_eval", "rows": 7200,
                     "methods": list(STATIC_METHODS), "normalization": NORMALIZATION,
                     "fit_counts": {"L": 1, "MSP": 0, "entropy": 0},
                     "inputs": inputs, "npz_sha256": protocol.sha256(path)},
           "Static predictions")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    if set(arrays) != set(METADATA) | set(STATIC_METHODS):
        raise RuntimeError("Static export must contain three scores, not per-seed copies or O")
    raw = load_logits(root, "dev_eval")
    _validate_role_arrays(raw, roles, "dev_eval", 100)
    _align_metadata(raw, arrays)
    if sidecar["cache_input_arrays"] != _array_hashes(raw):
        raise RuntimeError("Static score inputs no longer match the immutable classifier logits")
    reconstructed = {"L": head_scores(model, raw["logits"]), **analytic_scores(raw["logits"])}
    for name, wanted in reconstructed.items():
        values = arrays[name]
        if (values.shape != (7200,) or not np.isfinite(values).all()
                or not np.allclose(values, wanted, atol=1e-12, rtol=1e-12)):
            raise RuntimeError("Static error-score reconstruction changed: " + name)
    return arrays, sidecar


def predict_static(root):
    from .data import load_logits

    root, roles, model, inputs = _static_context(root)
    path = root / "predictions/dev_eval/static.npz"
    with _lock(path.parent, ".static.lock"):
        if path.exists() or path.with_suffix(".json").exists():
            if not path.exists() or not path.with_suffix(".json").exists():
                raise RuntimeError("An incomplete static export is preserved, not overwritten")
            return read_static(root)[1]
        protocol.check_cutoff(root, "predictions")
        began = time.monotonic()
        raw = load_logits(root, "dev_eval")
        _validate_role_arrays(raw, roles, "dev_eval", 100)
        arrays = {name: raw[name] for name in METADATA}
        arrays.update(L=head_scores(model, raw["logits"]))
        arrays.update(analytic_scores(raw["logits"]))
        _write_npz(path, arrays)
        protocol.check_cutoff(root, "predictions")
        sidecar = {"schema_version": 1, "scope_id": SCOPE, "complete": True,
                   "role": "dev_eval", "rows": 7200, "source_photographs": 800,
                   "methods": list(STATIC_METHODS), "fit_counts": {"L": 1, "MSP": 0, "entropy": 0},
                   "inputs": inputs, **inputs, "normalization": NORMALIZATION,
                   "cache_input_arrays": _array_hashes(raw), "npz_sha256": protocol.sha256(path),
                   "elapsed_seconds": time.monotonic() - began,
                   "completed_unix": time.time(), "job_id": os.environ["SLURM_JOB_ID"]}
        _write_json(path.with_suffix(".json"), sidecar)
        return read_static(root)[1]


def main():
    protocol.require_slurm()
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("fit", "export"):
        child = commands.add_parser(name)
        child.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = fit(args.root) if args.command == "fit" else predict_static(args.root)
    print(json.dumps({"complete": result["complete"], "method": NAME,
                      "fit_count": 1, "stage": args.command}), flush=True)


if __name__ == "__main__":
    main()
