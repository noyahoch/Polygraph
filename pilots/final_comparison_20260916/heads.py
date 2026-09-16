"""Two signed, meta-only logistic heads per family/seed; immutable legacy imports."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import time
import warnings

import numpy as np

from . import protocol

SCOPE = "final_comparison_20260916_fixed20"
SEEDS = (7, 17, 27)  # must equal protocol.SEEDS (amended 2026-09-16)
ARMS = ("block2", "block5", "block8", "block11")
FAMILIES = ("G", "H", "S")
METADATA = ("record_id", "image_id", "source_id", "severity", "split_id",
            "y", "label", "pred")
ROLE_COUNTS = {"base_train": 1600, "checkpoint": 400, "meta": 400, "dev_eval": 800}
VIEW_KEYS = ((0, 0),) + tuple((source, severity) for source in range(1, 5)
                             for severity in (3, 5))
RECIPE = {"standardize_on": "meta", "penalty": "l2", "C": 1.0,
          "solver": "lbfgs", "max_iter": 1000, "tol": 1e-6,
          "class_weight": None, "random_state": 7}
LEGACY_RECIPE = {key: value for key, value in RECIPE.items() if key != "random_state"}
HEAD_COLUMNS = {"stack": [0, 1, 2, 3], "last_only": [3]}
LEGACY_SCOPES = {7: "layer_ensemble_20260914_core_seed7",
                 17: "layer_ensemble_replication_seed17_27",
                 27: "layer_ensemble_replication_seed17_27"}


def _context(root, *, with_base=False):
    protocol.require_slurm()
    root = Path(root).resolve()
    campaign = protocol.read_campaign(root)
    identity = protocol.source_identity()
    if campaign["source_identity"] != identity or Path(campaign["root"]).resolve() != root:
        raise RuntimeError("Campaign source/root identity changed")
    if (protocol.SCOPE != SCOPE or tuple(protocol.SEEDS) != SEEDS
            or tuple(protocol.ARMS) != ARMS):
        raise RuntimeError("The fixed analysis matrix changed")
    roles_path = root / "role_map.json"
    if protocol.sha256(roles_path) != campaign["roles_sha256"]:
        raise RuntimeError("The original role-map bytes changed")
    bindings = {"campaign_sha256": protocol.campaign_sha(root),
                "roles_sha256": campaign["roles_sha256"],
                "source_identity": identity}
    for key in ("cache_manifest_sha256", "cache_cohort_sha256", "cache_protocol_sha256"):
        bindings[key] = campaign[key]
    if with_base:
        frozen = protocol.read(root / "base_freeze.json")
        if frozen.get("complete") is not True or frozen.get("scope_id") != SCOPE:
            raise RuntimeError("A complete final-comparison base freeze is required")
        bindings["base_freeze_sha256"] = protocol.sha256(root / "base_freeze.json")
    return root, campaign, protocol.read(roles_path), bindings


def _match(value, expected, description):
    for key, wanted in expected.items():
        if key not in value or value[key] != wanted:
            raise RuntimeError(f"{description} binding changed: {key}")


def _inside(root, relative):
    root = Path(root).resolve()
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError("Artifact inventories require relative, contained paths")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise RuntimeError("Artifact path escapes its frozen directory")
    return path


def _verify_files(directory, receipt, expected):
    if receipt.get("complete") is not True or set(receipt.get("files", {})) != set(expected):
        raise RuntimeError("Incomplete or unexpected artifact inventory")
    for name in expected:
        if protocol.sha256(_inside(directory, name)) != receipt["files"][name]:
            raise RuntimeError("Frozen artifact changed: " + name)


@contextmanager
def _lock(directory, name):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / name).open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def _write_bytes(path, content):
    """Publish beside the destination without replacing even a partial prior artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + f".partial.{os.getpid()}")
    try:
        with staging.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(staging, path)
    finally:
        if staging.exists():
            staging.unlink()
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json(path, value):
    _write_bytes(path, (json.dumps(value, ensure_ascii=False, sort_keys=True,
                                  indent=2, allow_nan=False) + "\n").encode("utf-8"))


def _write_npz(path, arrays):
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    _write_bytes(path, stream.getvalue())


def _array_hashes(arrays):
    result = {}
    for name, value in arrays.items():
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise ValueError("Object arrays cannot identify scientific inputs")
        digest = hashlib.sha256()
        digest.update(json.dumps({"dtype": array.dtype.str, "shape": list(array.shape)},
                                 sort_keys=True, separators=(",", ":")).encode())
        digest.update(np.ascontiguousarray(array).tobytes())
        result[name] = digest.hexdigest()
    return result


def _versions():
    return {"python": platform.python_version(),
            **{name: importlib.metadata.version(name)
               for name in ("numpy", "scipy", "scikit-learn")}}


def _before_new_readouts(root):
    root = Path(root)
    if any((root / name).exists() for name in
           ("heads_freeze.json", "evaluation_gate.json", "evaluation/report.json")):
        raise RuntimeError("No new fitting or diagnostic freeze after the global evaluation gate")
    dev = root / "predictions/dev_eval"
    if dev.exists() and any(path.is_file() and not path.name.startswith(".")
                            for path in dev.rglob("*")):
        raise RuntimeError("No new fitting or diagnostic freeze after development exports")


def _validate_role_arrays(arrays, roles, role, columns):
    if role not in ROLE_COUNTS or set(roles["roles"]) != set(ROLE_COUNTS):
        raise ValueError("Only the four fixed development roles are permitted")
    if roles.get("original_test_access") is not False:
        raise ValueError("Original-test access must remain closed")
    seen_photos, seen_records = set(), set()
    for name, count in ROLE_COUNTS.items():
        expected = roles["roles"][name]
        photos, records = expected["photo_ids"], expected["record_ids"]
        if (len(photos) != count or len(set(photos)) != count or len(records) != count * 9
                or len(set(records)) != count * 9 or seen_photos.intersection(photos)
                or seen_records.intersection(records)
                or expected["original_split"] != ("val" if name == "dev_eval" else "train")):
            raise ValueError("Role membership, nine-view exposure, or role disjointness changed")
        seen_photos.update(photos)
        seen_records.update(records)
    size = ROLE_COUNTS[role] * 9
    if set(arrays) != set(METADATA) | {"logits"}:
        raise ValueError("Expected only the eight identity arrays and raw logits")
    for name in METADATA:
        value = np.asarray(arrays[name])
        if value.shape != (size,) or not np.issubdtype(value.dtype, np.integer):
            raise ValueError("Unaligned/non-integral prediction metadata: " + name)
    logits = np.asarray(arrays["logits"])
    if logits.shape != (size, columns) or not np.isfinite(logits).all():
        raise ValueError("Invalid raw-logit dimensions or non-finite values")
    expected = roles["roles"][role]
    if not np.array_equal(arrays["record_id"], expected["record_ids"]):
        raise ValueError("Prediction order differs from the unchanged role map")
    photos, membership, counts = np.unique(arrays["image_id"], return_inverse=True,
                                           return_counts=True)
    if set(photos.tolist()) != set(expected["photo_ids"]) or not np.all(counts == 9):
        raise ValueError("Every assigned image_id must have exactly nine views")
    if not np.all(arrays["split_id"] == (1 if role == "dev_eval" else 0)):
        raise ValueError("A role contains a foreign original split")
    if (not np.isin(arrays["y"], [0, 1]).all()
            or not np.array_equal(arrays["y"], arrays["pred"] != arrays["label"])):
        raise ValueError("The target is the frozen classifier error, not a corruption label")
    if np.unique(arrays["y"]).size != 2:
        raise ValueError("Both correct and incorrect classifier outcomes are required")
    if any(((arrays[name] < 0) | (arrays[name] >= 100)).any() for name in ("label", "pred")):
        raise ValueError("Expected the unchanged 100-class classifier")
    for index in range(len(photos)):
        mask = membership == index
        if (set(zip(arrays["source_id"][mask].tolist(), arrays["severity"][mask].tolist()))
                != set(VIEW_KEYS) or np.unique(arrays["label"][mask]).size != 1):
            raise ValueError("A photograph lost its fixed nine conditions or true label")
    if columns == 100 and not np.array_equal(np.argmax(logits, axis=1), arrays["pred"]):
        raise ValueError("Frozen classifier logits and predictions disagree")


def _align_metadata(reference, candidate):
    for name in METADATA:
        if (name not in reference or name not in candidate
                or not np.array_equal(reference[name], candidate[name])):
            raise ValueError("Methods do not share ordered metadata: " + name)


def head_scores(head, logits):
    values = np.asarray(logits, dtype=np.float64)
    columns = head["columns"]
    if (values.ndim != 2 or not columns or len(set(columns)) != len(columns)
            or any(type(column) is not int or column < 0 or column >= values.shape[1]
                   for column in columns) or not np.isfinite(values).all()):
        raise ValueError("Invalid ordered input columns for a serialized logistic model")
    mean, scale, variance, coefficient = (np.asarray(value, dtype=np.float64) for value in
        (head["scaler"]["mean"], head["scaler"]["scale"],
         head["scaler"]["var"], head["model"]["coef"]))
    if any(value.shape != (len(columns),) or not np.isfinite(value).all()
           for value in (mean, scale, variance, coefficient)):
        raise ValueError("Invalid serialized logistic model dimensions/parameters")
    intercept = head["model"]["intercept"]
    if ((scale <= 0).any() or (variance < 0).any() or not np.isfinite(intercept)
            or head["model"]["classes"] != [0, 1]):
        raise ValueError("Invalid scaling or positive-class direction")
    scores = ((values[:, columns] - mean) / scale) @ coefficient + float(intercept)
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite reconstructed decision scores")
    return scores


def _fit_serialized(logits, labels, columns, recipe, header):
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    values, labels = np.asarray(logits, dtype=np.float64), np.asarray(labels)
    if (values.ndim != 2 or labels.shape != (len(values),)
            or not np.isfinite(values).all() or set(np.unique(labels).tolist()) != {0, 1}):
        raise ValueError("A logistic fit requires aligned finite inputs and both outcomes")
    scaler = StandardScaler()
    standardized = scaler.fit_transform(values[:, columns])
    options = {key: recipe[key] for key in
               ("penalty", "C", "solver", "max_iter", "tol", "class_weight", "random_state")}
    if isinstance(options["class_weight"], dict):
        options["class_weight"] = {int(label): weight
                                   for label, weight in options["class_weight"].items()}
    model = LogisticRegression(**options)
    began = time.monotonic()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        model.fit(standardized, labels)
    warning_rows = [{"category": item.category.__name__, "message": str(item.message)}
                    for item in captured]
    converged = (not any(issubclass(item.category, ConvergenceWarning) for item in captured)
                 and int(np.max(model.n_iter_)) < recipe["max_iter"])
    result = {**header, "schema_version": 1, "columns": list(columns), "recipe": recipe,
              "converged": converged, "warnings": warning_rows, "versions": _versions(),
              "fit_seconds": time.monotonic() - began,
              "scaler": {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
                         "var": scaler.var_.tolist(), "n_samples_seen": int(scaler.n_samples_seen_)},
              "model": {"coef": model.coef_[0].tolist(), "intercept": float(model.intercept_[0]),
                        "classes": model.classes_.tolist(), "n_iter": model.n_iter_.tolist()}}
    restored = head_scores(json.loads(json.dumps(result, allow_nan=False)), values)
    reference = model.decision_function(standardized)
    result["serialization_audit"] = {
        "atol": 1e-12, "rtol": 1e-12,
        "max_abs_difference": float(np.max(np.abs(restored - reference))),
        "passed": bool(np.allclose(restored, reference, atol=1e-12, rtol=1e-12))}
    return result


def _validate_solver(model, columns, rows, recipe):
    if (model.get("columns") != columns or model.get("recipe") != recipe
            or model.get("converged") is not True
            or model["scaler"].get("n_samples_seen") != rows):
        raise RuntimeError("The fixed solver, input columns, convergence, or fitting exposure changed")
    audit = model.get("serialization_audit", {})
    if (audit.get("passed") is not True or audit.get("atol") != 1e-12
            or audit.get("rtol") != 1e-12
            or not np.isfinite(audit.get("max_abs_difference", np.nan))
            or audit["max_abs_difference"] < 0):
        raise RuntimeError("Missing the fixed-tolerance JSON decision-score audit")
    iterations = model["model"].get("n_iter")
    if (not isinstance(iterations, list) or len(iterations) != 1
            or type(iterations[0]) is not int or not 0 < iterations[0] < 1000
            or not isinstance(model.get("warnings"), list)
            or any(item["category"] == "ConvergenceWarning" for item in model["warnings"])):
        raise RuntimeError("An unconverged fit is not an eligible frozen model")
    head_scores(model, np.zeros((1, max(columns) + 1), dtype=np.float64))


def _group_context(root, family, seed):
    from .predict import read_predictions

    if family not in FAMILIES or type(seed) is not int or seed not in SEEDS:
        raise ValueError("Exactly G/H/S and the fixed neural seeds have meta heads")
    root, campaign, roles, bindings = _context(root, with_base=True)
    arrays, sidecar = read_predictions(root, family, seed, "meta")
    _validate_role_arrays(arrays, roles, "meta", 4)
    _match(sidecar, {"role": "meta", "family": family, "seed": seed,
                     "base_freeze_sha256": bindings["base_freeze_sha256"]},
           "Meta prediction")
    path = root / "predictions/meta" / family / f"seed{seed}.npz"
    inputs = {**bindings, "meta_prediction_npz_sha256": protocol.sha256(path),
              "meta_prediction_sidecar_sha256": protocol.sha256(path.with_suffix(".json"))}
    return root, campaign, arrays, inputs


def _legacy_inputs(campaign, seed, meta_sha):
    group = campaign["reuse_groups"][str(seed)]
    expected_status = "complete_late_diagnostic" if seed == 7 else "complete"
    if group["status"] != expected_status:
        raise RuntimeError("Imported G provenance/status may not be relabeled")
    source = Path(group["root"]).resolve()
    required = {"heads/stack.json", "heads/last_only.json", "heads_freeze.json",
                "role_map.json", "base_freeze.json", "execution.json",
                "predictions/meta.npz", "predictions/meta.json"}
    if not required.issubset(group["files"]):
        raise RuntimeError("Legacy import registry lacks head/meta provenance")
    hashes = {}
    for name in sorted(required):
        path = _inside(source, name)
        hashes[name] = protocol.sha256(path)
        if hashes[name] != group["files"][name]:
            raise RuntimeError("Declared legacy source artifact changed: " + name)
    original = protocol.read(source / "heads_freeze.json")
    execution = protocol.read(source / "execution.json")
    if (original.get("complete") is not True or original.get("seed") != seed
            or original.get("scope_id") != LEGACY_SCOPES[seed]
            or execution.get("scope_id") != LEGACY_SCOPES[seed]
            or execution.get("seed") != seed or original.get("recipe") != LEGACY_RECIPE
            or set(original.get("files", {})) != {"heads/stack.json", "heads/last_only.json"}):
        raise RuntimeError("Imported heads are not the identical eligible historical pair")
    _match(original, {"roles_sha256": campaign["roles_sha256"],
                      "execution_sha256": hashes["execution.json"],
                      "base_freeze_sha256": hashes["base_freeze.json"],
                      "meta_prediction_npz_sha256": hashes["predictions/meta.npz"],
                      "meta_prediction_sidecar_sha256": hashes["predictions/meta.json"]},
           "Original head freeze")
    if hashes["role_map.json"] != campaign["roles_sha256"] or meta_sha != hashes["predictions/meta.npz"]:
        raise RuntimeError("Imported heads require byte-identical roles and raw meta predictions")
    for name in ("stack", "last_only"):
        relative = f"heads/{name}.json"
        if original["files"][relative] != hashes[relative]:
            raise RuntimeError("Original head freeze no longer identifies its exact coefficients")
    return {"root": str(source), "original_scope_id": execution["scope_id"],
            "original_status": group["status"], "files": hashes,
            "original_frozen_utc": original["frozen_utc"],
            "original_job_id": original["job_id"]}


def validate_heads(root, family, seed):
    root, campaign, arrays, inputs = _group_context(root, family, seed)
    directory = root / "heads" / family / f"seed{seed}"
    frozen = protocol.read(directory / "freeze.json")
    imported = family == "G" and seed in LEGACY_SCOPES
    expected = {"scope_id": SCOPE, "family": family, "seed": seed, "arms": list(ARMS),
                "recipe": RECIPE, "inputs": inputs, "imported": imported,
                "head_count": 2, "dev_eval_used_for_fit": False, "original_test_access": False}
    _match(frozen, expected, "Head freeze")
    _verify_files(directory, frozen, {"stack.json", "last_only.json"})
    visible = {path.name for path in directory.glob("*.json") if not path.name.startswith(".")}
    if visible != {"stack.json", "last_only.json", "freeze.json"}:
        raise RuntimeError("A family/seed group must contain exactly the two registered heads")
    if imported:
        original = _legacy_inputs(campaign, seed, inputs["meta_prediction_npz_sha256"])
        if frozen.get("import") != original:
            raise RuntimeError("Historical head provenance changed")
    else:
        if protocol.sha256(directory / ".fit_started.json") != frozen["fit_attempt_sha256"]:
            raise RuntimeError("The single meta-fitting attempt changed")
        _match(protocol.read(directory / ".fit_started.json"),
               {"inputs": inputs, "family": family, "seed": seed, "recipe": RECIPE},
               "Meta fit attempt")
    models = {}
    for name, columns in HEAD_COLUMNS.items():
        head = protocol.read(directory / f"{name}.json")
        _match(head, {"name": name, "seed": seed, "training_role": "meta",
                      "rows": 3600, "source_photos": 400, "features": [ARMS[c] for c in columns],
                      "scope_id": LEGACY_SCOPES[seed] if imported else SCOPE}, "Head")
        if imported:
            if frozen["files"][f"{name}.json"] != original["files"][f"heads/{name}.json"]:
                raise RuntimeError("Imported coefficient files must retain their original bytes")
        else:
            _match(head, {"family": family, "inputs": inputs}, "New head")
        _validate_solver(head, columns, 3600, LEGACY_RECIPE if imported else RECIPE)
        head_scores(head, arrays["logits"])
        models[name] = head
    return models, frozen


def fit_heads(root, family, seed):
    protocol.require_slurm()
    if family not in FAMILIES or type(seed) is not int or seed not in SEEDS:
        raise ValueError("Unsupported meta-head family/seed")
    root = Path(root).resolve()
    directory = root / "heads" / family / f"seed{seed}"
    with _lock(directory, ".fit.lock"):
        if (directory / "freeze.json").exists():
            return validate_heads(root, family, seed)[1]
        _before_new_readouts(root)
        protocol.check_cutoff(root, "predictions")
        if any((directory / name).exists()
               for name in ("stack.json", "last_only.json", ".fit_started.json")):
            raise RuntimeError("An incomplete head attempt is preserved; automatic refitting is forbidden")
        root, campaign, arrays, inputs = _group_context(root, family, seed)
        imported = family == "G" and seed in LEGACY_SCOPES
        frozen = {"schema_version": 1, "scope_id": SCOPE, "complete": True,
                  "family": family, "seed": seed, "arms": list(ARMS), "head_count": 2,
                  "recipe": RECIPE, "inputs": inputs, **inputs, "imported": imported,
                  "dev_eval_used_for_fit": False, "original_test_access": False}
        if imported:
            original = _legacy_inputs(campaign, seed, inputs["meta_prediction_npz_sha256"])
            frozen["import"] = original
            for name, columns in HEAD_COLUMNS.items():
                source = Path(original["root"]) / "heads" / f"{name}.json"
                head = protocol.read(source)
                _validate_solver(head, columns, 3600, LEGACY_RECIPE)
                head_scores(head, arrays["logits"])
                _write_bytes(directory / f"{name}.json", source.read_bytes())
        else:
            attempt = {"inputs": inputs, "family": family, "seed": seed, "recipe": RECIPE,
                       "heads": list(HEAD_COLUMNS), "started_unix": time.time(),
                       "job_id": os.environ["SLURM_JOB_ID"]}
            _write_json(directory / ".fit_started.json", attempt)
            frozen["fit_attempt_sha256"] = protocol.sha256(directory / ".fit_started.json")
            for name, columns in HEAD_COLUMNS.items():
                head = _fit_serialized(
                    arrays["logits"], arrays["y"], columns, RECIPE,
                    {"name": name, "scope_id": SCOPE, "family": family, "seed": seed,
                     "training_role": "meta", "rows": 3600, "source_photos": 400,
                     "features": [ARMS[column] for column in columns], "inputs": inputs})
                _write_json(directory / f"{name}.json", head)
                _validate_solver(head, columns, 3600, RECIPE)
        protocol.check_cutoff(root, "predictions")
        _before_new_readouts(root)
        frozen.update(files={f"{name}.json": protocol.sha256(directory / f"{name}.json")
                             for name in HEAD_COLUMNS},
                      frozen_unix=time.time(), job_id=os.environ["SLURM_JOB_ID"])
        _write_json(directory / "freeze.json", frozen)
        return validate_heads(root, family, seed)[1]


def main():
    protocol.require_slurm()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    args = parser.parse_args()
    result = fit_heads(args.root, args.family, args.seed)
    print(json.dumps({"complete": result["complete"], "family": args.family,
                      "seed": args.seed, "imported": result["imported"]}), flush=True)


if __name__ == "__main__":
    main()
