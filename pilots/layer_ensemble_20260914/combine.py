"""Fit exactly two frozen CPU heads using only the reserved meta photographs."""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import warnings

SCOPE = "layer_ensemble_20260914_core_seed7"
ARMS = ("block2", "block5", "block8", "block11")
COUNTS = {"base_train": 1600, "checkpoint": 400, "meta": 400, "dev_eval": 800}
RECIPE = {"standardize_on": "meta", "penalty": "l2", "C": 1.0,
          "solver": "lbfgs", "max_iter": 1000, "tol": 1e-6, "class_weight": None}
DEADLINES = {"base_complete_before": "2026-09-15T03:30:00+03:00",
             "predictions_complete_before": "2026-09-15T07:00:00+03:00"}


def require_slurm():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("All fitting, verification and statistics require an allocated Slurm job")


def read(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp." + str(os.getpid()))
    with temporary.open("w") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def inside(root, path):
    root, path = Path(root).resolve(), Path(path).resolve()
    if not path.is_relative_to(root):
        raise RuntimeError("Artifact path escapes this experiment")
    return path


def code_identity(execution=None):
    directory = Path(__file__).resolve().parent
    names = ["combine.py", "evaluate.py"]
    if execution is not None and execution["scope_id"] != SCOPE:
        names += ["protocol.py", "replication.py"]
    return {name: sha256(directory / name) for name in names}


def context(root):
    """Validate the complete role split and actual frozen four-run artifacts."""
    require_slurm()
    root = Path(root).resolve()
    execution, roles, frozen = (read(root / name) for name in
                                ("execution.json", "role_map.json", "base_freeze.json"))
    if execution.get("scope_id") == SCOPE:
        if execution.get("seed") != 7 or execution.get("matrix") != [[arm, 7] for arm in ARMS]:
            raise RuntimeError("Only the four predeclared single-layer fits at seed7 are allowed")
        deadlines = DEADLINES
    else:
        from .replication import SCOPE as REPLICATION_SCOPE, validate_seed_execution
        if execution.get("scope_id") != REPLICATION_SCOPE:
            raise RuntimeError("Unknown experiment scope")
        validate_seed_execution(root / "execution.json", root / "role_map.json")
        deadlines = execution["deadlines"]
    seed, scope = execution["seed"], execution["scope_id"]
    if roles.get("scope_id") != SCOPE or frozen.get("scope_id") != scope:
        raise RuntimeError("Unexpected experiment scope")
    if execution["training"].get("epochs") != 20 or execution["training"].get("minimum_epochs") != 20:
        raise RuntimeError("All four base models require exactly20 epochs")
    if execution.get("head_recipe") != RECIPE:
        raise RuntimeError("The fixed meta-learning recipe changed")
    if execution.get("deadlines") != deadlines:
        raise RuntimeError("The frozen base/prediction cutoffs changed")
    if execution.get("test_evaluated") is not False or roles.get("original_test_access") is not False:
        raise RuntimeError("Original held-out test must remain closed")
    if set(roles["roles"]) != set(COUNTS):
        raise RuntimeError("Role map must contain exactly the four disjoint roles")
    seen_photos, seen_records = set(), set()
    for name, count in COUNTS.items():
        role = roles["roles"][name]
        photos, records = role["photo_ids"], role["record_ids"]
        if len(photos) != count or len(set(photos)) != count or len(records) != count * 9 or len(set(records)) != count * 9:
            raise RuntimeError("Invalid role membership: " + name)
        if seen_photos.intersection(photos) or seen_records.intersection(records):
            raise RuntimeError("Photograph/record leakage between roles")
        if role.get("original_split") != ("val" if name == "dev_eval" else "train"):
            raise RuntimeError("Role map changed the original development split")
        seen_photos.update(photos)
        seen_records.update(records)
    bindings = {"execution_sha256": sha256(root / "execution.json"),
                "roles_sha256": sha256(root / "role_map.json"),
                "base_freeze_sha256": sha256(root / "base_freeze.json"),
                "cache_manifest_sha256": execution["cache_manifest_sha256"]}
    if execution["roles_sha256"] != bindings["roles_sha256"]:
        raise RuntimeError("Execution is not bound to the current role map")
    if frozen.get("complete") is not True or frozen.get("seed") != seed or frozen.get("arms") != list(ARMS):
        raise RuntimeError("All four base models must be frozen before fitting either head")
    for name in ("execution_sha256", "roles_sha256", "cache_manifest_sha256"):
        if frozen.get(name) != bindings[name]:
            raise RuntimeError("Base freeze identity mismatch: " + name)
    if set(frozen.get("runs", {})) != set(ARMS):
        raise RuntimeError("Missing base model; subset ensembles are forbidden")
    for arm in ARMS:
        run = frozen["runs"][arm]
        path = Path(run["path"])
        directory = inside(root, path if path.is_absolute() else root / path)
        if run.get("completed_epochs") != 20 or run.get("selection_role") != "checkpoint":
            raise RuntimeError("Base fit has the wrong training horizon/selection role")
        for key, name in (("config_sha256", "config.json"), ("best_sha256", "best.safetensors"),
                          ("complete_sha256", "complete.json")):
            if run[key] != sha256(directory / name):
                raise RuntimeError("Frozen base artifact changed: " + arm + "/" + name)
        complete, config = read(directory / "complete.json"), read(directory / "config.json")
        if complete.get("complete") is not True or complete.get("completed_epochs") != 20:
            raise RuntimeError("Base fit did not complete all20 epochs")
        if not 0 < complete.get("completed_unix", 0) < dt.datetime.fromisoformat(deadlines["base_complete_before"]).timestamp():
            raise RuntimeError("Base fit did not complete before the frozen cutoff")
        if complete.get("arm") != arm or complete.get("seed") != seed or config.get("arm") != arm or config.get("seed") != seed:
            raise RuntimeError("Base run identity differs from frozen feature order")
        if complete.get("scope_id") != scope or config.get("scope_id") != scope:
            raise RuntimeError("Base run belongs to a different protocol")
        for name in ("execution_sha256", "roles_sha256", "cache_manifest_sha256"):
            if config.get(name) != bindings[name]:
                raise RuntimeError("Base configuration provenance changed: " + name)
    return execution, roles, bindings


def load_predictions(root, path, role):
    """No fallback role, outcome-based selection, or subset of the four logits."""
    import numpy as np
    if role not in ("meta", "dev_eval"):
        raise RuntimeError("Only reserved meta/dev_eval predictions are supported")
    execution, roles, bindings = context(root)
    path = inside(root, path)
    sidecar = read(path.with_suffix(".json"))
    if sidecar.get("schema_version") != 1 or sidecar.get("role") != role or sidecar.get("arms") != list(ARMS) or sidecar.get("seed") != execution["seed"]:
        raise RuntimeError("Prediction role/feature-order identity mismatch")
    if sidecar.get("rows") != COUNTS[role] * 9 or sidecar.get("npz_sha256") != sha256(path):
        raise RuntimeError("Prediction size/checksum mismatch")
    if execution["scope_id"] != SCOPE and (
            sidecar.get("scope_id") != execution["scope_id"]
            or sidecar.get("replication_manifest_sha256") != execution["replication_manifest_sha256"]):
        raise RuntimeError("Prediction belongs to a different replication protocol")
    deadlines = DEADLINES if execution["scope_id"] == SCOPE else execution["deadlines"]
    if not 0 < sidecar.get("completed_unix", 0) < dt.datetime.fromisoformat(deadlines["predictions_complete_before"]).timestamp():
        raise RuntimeError("Predictions did not complete before the frozen cutoff")
    if role == "dev_eval" and execution["scope_id"] != SCOPE:
        from .replication import validate_joint_heads_freeze
        joint = validate_joint_heads_freeze(execution["replication_root"])
        joint_path = Path(execution["replication_root"]) / "joint_heads_freeze.json"
        if (sidecar.get("joint_heads_freeze_sha256") != sha256(joint_path)
                or sidecar["completed_unix"] <= joint["frozen_unix"]):
            raise RuntimeError("Development predictions require the prior joint seed freeze")
    for name, value in bindings.items():
        if sidecar.get(name) != value:
            raise RuntimeError("Prediction provenance mismatch: " + name)
    with np.load(path, allow_pickle=False) as archive:
        data = {name: archive[name].copy() for name in archive.files}
    required = ("record_id", "image_id", "source_id", "severity", "split_id", "y", "label", "pred")
    if "logits" not in data or any(name not in data for name in required):
        raise RuntimeError("Prediction matrix lacks required scores or identities")
    n = COUNTS[role] * 9
    if data["logits"].shape != (n, 4) or not np.isfinite(data["logits"]).all():
        raise RuntimeError("Expected finite four-column raw error logits")
    if any(data[name].shape != (n,) for name in required):
        raise RuntimeError("Prediction metadata is not aligned")
    expected = roles["roles"][role]
    if not np.array_equal(data["record_id"], np.asarray(expected["record_ids"])):
        raise RuntimeError("Predictions do not match the exact ordered role records")
    photos, counts = np.unique(data["image_id"], return_counts=True)
    if set(photos.tolist()) != set(expected["photo_ids"]) or not np.all(counts == 9):
        raise RuntimeError("Each assigned source photograph must contribute all nine views")
    if not np.isin(data["y"], [0, 1]).all() or not np.array_equal(data["y"], data["pred"] != data["label"]):
        raise RuntimeError("Target must be the frozen ViT classification error")
    if len(np.unique(data["y"])) != 2:
        raise RuntimeError("Both correct and incorrect predictions are required")
    return data, sidecar, bindings


def head_scores(head, logits):
    import numpy as np
    columns = head["columns"]
    values = np.asarray(logits, dtype=np.float64)[:, columns]
    mean, scale = np.asarray(head["scaler"]["mean"]), np.asarray(head["scaler"]["scale"])
    coef = np.asarray(head["model"]["coef"])
    if mean.shape != (len(columns),) or scale.shape != mean.shape or coef.shape != mean.shape:
        raise RuntimeError("Invalid serialized head dimensions")
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or not np.isfinite(coef).all() or (scale <= 0).any():
        raise RuntimeError("Invalid serialized head parameters")
    scores = ((values - mean) / scale) @ coef + float(head["model"]["intercept"])
    if not np.isfinite(scores).all():
        raise RuntimeError("Non-finite head scores")
    return scores


def validate_heads(root, heads=None):
    root = Path(root).resolve()
    execution, roles, bindings = context(root)
    frozen = read(root / "heads_freeze.json")
    if frozen.get("complete") is not True or frozen.get("scope_id") != execution["scope_id"] or frozen.get("seed") != execution["seed"] or frozen.get("arms") != list(ARMS):
        raise RuntimeError("Both heads must be frozen before final development evaluation")
    if frozen.get("recipe") != RECIPE or frozen.get("implementation") != code_identity(execution):
        raise RuntimeError("Head recipe/evaluation implementation changed after freeze")
    for name, value in bindings.items():
        if frozen.get(name) != value:
            raise RuntimeError("Frozen head provenance mismatch: " + name)
    expected_files = {"heads/stack.json", "heads/last_only.json"}
    if set(frozen.get("files", {})) != expected_files:
        raise RuntimeError("Exactly the learned stack and matched last-only control are required")
    if heads is not None and inside(root, heads) != root / "heads":
        raise RuntimeError("Unexpected head directory")
    result = {}
    for name, columns in (("stack", [0, 1, 2, 3]), ("last_only", [3])):
        relative = "heads/" + name + ".json"
        path = inside(root, root / relative)
        if sha256(path) != frozen["files"][relative]:
            raise RuntimeError("Frozen head artifact changed")
        head = read(path)
        if head.get("name") != name or head.get("columns") != columns or head.get("training_role") != "meta" or head.get("rows") != 3600 or head.get("source_photos") != 400:
            raise RuntimeError("Head feature order/training role changed")
        if head.get("converged") is not True or head["model"].get("classes") != [0, 1]:
            raise RuntimeError("A meta head did not converge or has the wrong class direction")
        if head.get("scope_id") != execution["scope_id"] or head.get("seed") != execution["seed"]:
            raise RuntimeError("A head belongs to a different training seed or protocol")
        if head.get("recipe") != RECIPE or head.get("serialization_audit", {}).get("passed") is not True:
            raise RuntimeError("A head lacks the frozen recipe or passing serialization audit")
        result[name] = head
    return result, frozen, bindings


def fit_heads(root, predictions, out):
    import numpy as np
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    require_slurm()
    root = Path(root).resolve()
    out = inside(root, out)
    if out != root / "heads":
        raise RuntimeError("Head files must live in the canonical experiment heads directory")
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".fit.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        data, sidecar, bindings = load_predictions(root, predictions, "meta")
        execution = read(root / "execution.json")
        if (root / "heads_freeze.json").exists():
            _, frozen, _ = validate_heads(root, out)
            if frozen.get("meta_prediction_npz_sha256") != sha256(predictions) or frozen.get("meta_prediction_sidecar_sha256") != sha256(Path(predictions).with_suffix(".json")):
                raise RuntimeError("Frozen heads cannot be refit using different meta predictions")
            return frozen
        if (root / "predictions/dev_eval.npz").exists() or (root / "evaluation/report.json").exists():
            raise RuntimeError("Refusing to fit heads after final development outcomes exist")
        if execution["scope_id"] != SCOPE:
            from .replication import SEEDS, check_deadline, validate_manifest
            replica_root = Path(execution["replication_root"])
            check_deadline(validate_manifest(replica_root), "predictions_complete_before")
            if (replica_root / "joint_heads_freeze.json").exists() or any(
                    (replica_root / f"seed{s}" / "predictions/dev_eval.npz").exists() for s in SEEDS):
                raise RuntimeError("No head fitting after either replication's dev evaluation")
        for name, columns in (("stack", [0, 1, 2, 3]), ("last_only", [3])):
            values = data["logits"][:, columns].astype(np.float64)
            scaler = StandardScaler()
            standardized = scaler.fit_transform(values)
            model = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=1000,
                                       tol=1e-6, class_weight=None, random_state=7)
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                model.fit(standardized, data["y"])
            warning_rows = [{"category": item.category.__name__, "message": str(item.message)} for item in captured]
            converged = not any(issubclass(item.category, ConvergenceWarning) for item in captured)
            if int(np.max(model.n_iter_)) >= 1000:
                converged = False
            head = {"schema_version": 1, "name": name, "scope_id": execution["scope_id"], "seed": execution["seed"],
                    "training_role": "meta", "rows": 3600, "source_photos": 400,
                    "columns": columns, "features": [ARMS[column] for column in columns],
                    "recipe": RECIPE, "converged": converged, "warnings": warning_rows,
                    "scaler": {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
                               "var": scaler.var_.tolist(), "n_samples_seen": int(scaler.n_samples_seen_)},
                    "model": {"coef": model.coef_[0].tolist(), "intercept": float(model.intercept_[0]),
                              "classes": model.classes_.tolist(), "n_iter": model.n_iter_.tolist()}}
            restored = head_scores(head, data["logits"])
            reference = model.decision_function(standardized)
            head["serialization_audit"] = {"atol": 1e-12, "rtol": 1e-12,
                                           "max_abs_difference": float(np.max(np.abs(restored - reference))),
                                           "passed": bool(np.allclose(restored, reference, atol=1e-12, rtol=1e-12))}
            atomic_json(out / (name + ".json"), head)
            if not converged or not head["serialization_audit"]["passed"]:
                raise RuntimeError("Meta fit/serialization failed; no changed recipe or tolerance retry: " + name)
        if execution["scope_id"] != SCOPE:
            check_deadline(validate_manifest(replica_root), "predictions_complete_before")
        frozen = {"schema_version": 1, "scope_id": execution["scope_id"], "complete": True, "seed": execution["seed"],
                  "arms": list(ARMS), "recipe": RECIPE, **bindings,
                  "files": {str(path.relative_to(root)): sha256(path) for path in (out / "stack.json", out / "last_only.json")},
                  "meta_prediction_npz_sha256": sha256(predictions),
                  "meta_prediction_sidecar_sha256": sha256(Path(predictions).with_suffix(".json")),
                  "implementation": code_identity(execution), "versions": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn")},
                  "frozen_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                  "job_id": os.environ["SLURM_JOB_ID"], "dev_eval_used_for_fit": False, "original_test_access": False}
        atomic_json(root / "heads_freeze.json", frozen)
        validate_heads(root, out)
        return frozen


def main():
    require_slurm()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = fit_heads(args.root, args.predictions, args.out)
    print(json.dumps({"complete": result["complete"], "heads_freeze_sha256": sha256(args.root / "heads_freeze.json")}), flush=True)


if __name__ == "__main__":
    main()
