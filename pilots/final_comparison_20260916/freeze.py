"""Global, write-once gates for complete bases, meta heads and development access."""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import fcntl
import json
import os
from pathlib import Path

from .protocol import (
    ARMS, COUNTS, SEEDS, SCOPE, campaign_sha, check_cutoff, model_key, neural_matrix,
    read, read_campaign, read_execution, require_slurm, resolve_run, sha256,
    source_identity, verify_hash, write_frozen, _timestamp,
)


def _header(root):
    return {"schema_version": 1, "scope_id": SCOPE, "complete": True,
            "campaign_sha256": campaign_sha(root), "source_identity": source_identity()}


def _validate_header(root, value, phase):
    if any(value.get(key) != expected for key, expected in _header(root).items()):
        raise RuntimeError("Foreign or incomplete global freeze")
    execution = read_execution(root)
    when = _timestamp(value["frozen_utc"])
    if not _timestamp(execution["authorized_at"]) <= when < _timestamp(execution["deadlines"][phase]):
        raise RuntimeError("Global freeze missed the prospective execution cutoff")


def _base_records(root):
    from .train import verify_complete
    campaign = read_campaign(root)
    records = {}
    for row in neural_matrix():
        family, arm, seed = row["family"], row["arm"], row["seed"]
        complete, config = verify_complete(root, family, arm, seed)
        key = model_key(family, arm, seed)
        directory = resolve_run(root, family, arm, seed)
        if complete.get("complete") is not True or complete.get("completed_epochs") != 20:
            raise RuntimeError("Every neural fit must complete twenty epochs")
        if config.get("seed") != seed or config.get("arm") != arm:
            raise RuntimeError("Base fit identity differs from its matrix slot")
        imported = key in campaign["reuse"]
        if not imported and config.get("family") != family:
            raise RuntimeError("Fresh fit has the wrong model family")
        records[key] = {
            "path": str(directory), "family": family, "arm": arm, "seed": seed,
            "imported": imported, "original_scope_id": config["scope_id"],
            "original_status": campaign["reuse"][key]["status"] if imported else "complete",
            "config_sha256": sha256(directory / "config.json"),
            "complete_sha256": sha256(directory / "complete.json"),
            "best_sha256": sha256(directory / "best.safetensors"),
            "history_sha256": sha256(directory / "history.json"),
        }
    return records


def freeze_base(root):
    require_slurm()
    root = Path(root).resolve()
    with (root / ".base-freeze.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "base_freeze.json").exists():
            return validate_base_freeze(root)
        check_cutoff(root, "base")
        records = _base_records(root)
        check_cutoff(root, "base")
        value = {**_header(root), "neural_fits": COUNTS["neural_fits"], "models": records,
                 "frozen_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                 "job_id": os.environ["SLURM_JOB_ID"]}
        write_frozen(root / "base_freeze.json", value)
        return value


def validate_base_freeze(root):
    require_slurm()
    root = Path(root).resolve()
    value = read(root / "base_freeze.json")
    _validate_header(root, value, "base")
    expected = {model_key(row["family"], row["arm"], row["seed"]) for row in neural_matrix()}
    if value.get("neural_fits") != COUNTS["neural_fits"] or set(value.get("models", {})) != expected:
        raise RuntimeError("All registered neural fits are required, including every bound G import")
    if value["models"] != _base_records(root):
        raise RuntimeError("Selected base models changed after freeze")
    return value


def _head_records(root):
    from .heads import validate_heads
    records = {}
    for family in ("G", "H", "S"):
        for seed in SEEDS:
            models, frozen = validate_heads(root, family, seed)
            if set(models) != {"stack", "last_only"} or frozen.get("complete") is not True:
                raise RuntimeError("Both meta heads must be complete for every family and seed")
            directory = Path(root) / "heads" / family / f"seed{seed}"
            records[f"{family}/seed{seed}"] = {
                "family": family, "seed": seed,
                "files": {name: sha256(directory / name) for name in
                          ("stack.json", "last_only.json", "freeze.json")},
            }
    return records


def freeze_heads(root):
    require_slurm()
    root = Path(root).resolve()
    with (root / ".heads-freeze.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "heads_freeze.json").exists():
            return validate_heads_freeze(root)
        check_cutoff(root, "predictions")
        validate_base_freeze(root)
        records = _head_records(root)
        check_cutoff(root, "predictions")
        value = {**_header(root), "meta_heads": COUNTS["meta_heads"], "groups": records,
                 "base_freeze_sha256": sha256(root / "base_freeze.json"),
                 "frozen_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                 "job_id": os.environ["SLURM_JOB_ID"]}
        write_frozen(root / "heads_freeze.json", value)
        return value


def validate_heads_freeze(root):
    require_slurm()
    root = Path(root).resolve()
    value = read(root / "heads_freeze.json")
    _validate_header(root, value, "predictions")
    validate_base_freeze(root)
    verify_hash(root / "base_freeze.json", value["base_freeze_sha256"])
    if value.get("meta_heads") != COUNTS["meta_heads"] or value.get("groups") != _head_records(root):
        raise RuntimeError("The G/H/S family/seed meta-head pairs changed")
    return value


def _evaluation_bindings(root):
    from .linear import validate as validate_linear
    from .diagnostics import validate as validate_diagnostics
    validate_base_freeze(root)
    validate_heads_freeze(root)
    validate_linear(root)
    validate_diagnostics(root)
    names = (
        "role_map.json", "base_freeze.json", "heads_freeze.json",
        "linear/model.json", "linear/complete.json",
        "diagnostics/training_diagnostics.json", "diagnostics/training_diagnostics.csv",
        "diagnostics/complete.json",
    )
    return {name: sha256(Path(root) / name) for name in names}


def freeze_evaluation(root):
    require_slurm()
    root = Path(root).resolve()
    with (root / ".evaluation-gate.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = root / "evaluation_gate.json"
        if path.exists():
            return validate_evaluation_gate(root)
        check_cutoff(root, "predictions")
        predictions = root / "predictions/dev_eval"
        if predictions.exists() and any(predictions.rglob("*.npz")):
            raise RuntimeError("Cannot create the global freeze after new development exports")
        if (root / "evaluation/report.json").exists():
            raise RuntimeError("Cannot freeze after evaluation results")
        bindings = _evaluation_bindings(root)
        check_cutoff(root, "predictions")
        value = {**_header(root), "neural_fits": COUNTS["neural_fits"], "meta_heads": COUNTS["meta_heads"], "linear_fits": 1,
                 "files": bindings, "original_test_access": False,
                 "diagnostics_policy": "report only; warnings never extend training",
                 "frozen_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                 "job_id": os.environ["SLURM_JOB_ID"]}
        write_frozen(path, value)
        return value


# Per-process memo of successful full gate validations, keyed by (root, gate-file sha256).
# The gate sha is re-hashed on every call, so any change to evaluation_gate.json forces a
# full re-validation; failures raise before insertion and are never cached.
_EVALUATION_GATE_CACHE = {}


def validate_evaluation_gate(root):
    require_slurm()
    root = Path(root).resolve()
    key = (str(root), sha256(root / "evaluation_gate.json"))
    if key in _EVALUATION_GATE_CACHE:
        return copy.deepcopy(_EVALUATION_GATE_CACHE[key])
    value = _validate_evaluation_gate_full(root)
    if sha256(root / "evaluation_gate.json") == key[1]:
        _EVALUATION_GATE_CACHE[key] = copy.deepcopy(value)
    return value


def _validate_evaluation_gate_full(root):
    value = read(root / "evaluation_gate.json")
    _validate_header(root, value, "predictions")
    expected = {"neural_fits": COUNTS["neural_fits"], "meta_heads": COUNTS["meta_heads"], "linear_fits": 1,
                "original_test_access": False}
    if any(value.get(key) != item for key, item in expected.items()):
        raise RuntimeError("The complete final comparison must be frozen before dev_eval")
    if value.get("files") != _evaluation_bindings(root):
        raise RuntimeError("Frozen models, heads or fixed20 diagnostics changed")
    return value


def main():
    require_slurm()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("base", "heads", "evaluation"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    value = {"base": freeze_base, "heads": freeze_heads, "evaluation": freeze_evaluation}[args.stage](args.root)
    print(json.dumps({"scope_id": SCOPE, "stage": args.stage, "complete": value["complete"]}), flush=True)


if __name__ == "__main__":
    main()
