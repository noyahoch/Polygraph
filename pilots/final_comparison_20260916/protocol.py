"""Immutable scientific contract, legacy-run registry and execution gates."""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import re
import time

from pilots.layer_ensemble_20260914.combine import atomic_json, sha256
from pilots.layer_ensemble_20260914.protocol import make_roles
from pilots.layer_screen_20260913.protocol import (
    cohort as cache_cohort, protocol as cache_protocol, digest, require_slurm,
)

SCOPE = "final_comparison_20260916_fixed20"
# Amended 2026-09-16 before any benchmark result: three seeds, exactly those with compatible G fits.
SEEDS = (7, 17, 27)
ARMS = ("block2", "block5", "block8", "block11")
LAYERS = (3, 6, 9, 12)
FAMILIES = ("G", "H", "S", "O")
ROLES = ("base_train", "checkpoint", "meta", "dev_eval")
ROLE_PHOTOS = dict(zip(ROLES, (1600, 400, 400, 800)))
IMPORTED_SEEDS = (7, 17, 27)
HEAD_FAMILIES = ("G", "H", "S")
HEAD_NAMES = ("stack", "last_only")
COUNTS = {
    "neural_fits": len(SEEDS) * (len(HEAD_FAMILIES) * len(ARMS) + 1),
    "imported_neural_fits": len(IMPORTED_SEEDS) * len(ARMS),
    "new_neural_fits": len(SEEDS) * (len(HEAD_FAMILIES) * len(ARMS) + 1) - len(IMPORTED_SEEDS) * len(ARMS),
    "linear_fits": 1,
    "meta_heads": len(HEAD_FAMILIES) * len(SEEDS) * len(HEAD_NAMES),
    "imported_meta_heads": len(IMPORTED_SEEDS) * len(HEAD_NAMES),
    "new_meta_heads": (len(HEAD_FAMILIES) * len(SEEDS) - len(IMPORTED_SEEDS)) * len(HEAD_NAMES),
    "reported_methods": 25,
}
CONTRACT_COUNTS = {key: COUNTS[key] for key in (
    "neural_fits", "imported_neural_fits", "new_neural_fits", "linear_fits", "meta_heads", "reported_methods")}
TRAINING = {
    "epochs": 20, "minimum_epochs": 20, "lr": 0.002, "weight_decay": 0.0001,
    "dropout": 0.15, "batch_size": 24,
    "checkpoint_rule": "strictly greatest checkpoint-role AUROC; earliest tie",
    "loss": "BCEWithLogitsLoss", "class_weight": "base_train negative/positive",
}
MODEL_SPECS = {
    "G": {"architecture": "edge_gated_mean", "width": 64, "gnn_layers": 2,
          "node_dim": 784, "edge_dim": 12, "parameters": 130434},
    "H": {"architecture": "hidden_token_set", "width": 96,
          "node_dim": 772, "edge_dim": 0, "parameters": 129986},
    "S": {"architecture": "node_edge_set", "width": 84,
          "node_dim": 784, "edge_dim": 12, "parameters": 131126},
    "O": {"architecture": "logit_mlp", "dimensions": [100, 64, 32, 1],
          "parameters": 8577},
}
HEAD_RECIPE = {"standardize_on": "meta", "penalty": "l2", "solver": "lbfgs",
               "C": 1.0, "max_iter": 1000, "tol": 1e-6, "class_weight": None, "random_state": 7}
LINEAR_RECIPE = {"features": "all_100_raw_logits", "standardize_on": "base_train",
                 "penalty": "l2", "solver": "lbfgs", "C": 1.0, "max_iter": 1000,
                 "tol": 1e-6, "random_state": 7, "class_weight": "negative=1; positive=base_negative/base_positive",
                 "fits": 1, "primary": False}
PRIMARY_CONTRASTS = (
    ("G_stack", "S_stack"), ("G_mean", "S_mean"),
    ("G_stack", "H_stack"), ("G_mean", "H_mean"),
    ("G_stack", "O"), ("G_mean", "O"),
)
METHODS = tuple(f"{family}_{name}" for family in ("G", "H", "S")
                for name in (*ARMS, "mean", "stack", "last_only")) + ("O", "L", "MSP", "entropy")
ANALYSIS = {
    "primary_contrasts": [list(pair) for pair in PRIMARY_CONTRASTS],
    "training_seeds": list(SEEDS), "bootstrap_draws": 10000, "bootstrap_seed": 20260914,
    "bootstrap_group": "image_id", "shared_draws": True, "percentile_method": "linear",
    "family_alpha": 0.05, "family_size": 6, "correction": "Bonferroni percentile intervals",
    "seed_standard_deviation": "sample ddof=1; static L/MSP/entropy not applicable",
    "estimand": "mean of within-seed paired AUROC differences",
    "undefined_draws": "no redraw; withhold affected interval", "original_test_access": False,
}
DIAGNOSTICS = {
    "required_epochs": list(range(1, 21)), "checkpoint_endpoints": [15, 20],
    "checkpoint_change": "actual checkpoint_auroc[20] minus actual checkpoint_auroc[15]",
    "loss_decrease": "loss[15] minus loss[20]; percentage relative to loss[15]",
    "zero_loss15": "relative decrease undefined, with explicit reason",
    "loss_slope_epochs": [16, 17, 18, 19, 20], "boundary_epoch": 20,
    "warning": "selected20 and checkpoint_change>0 and loss_decrease>0",
    "action": "report only; never extend training or replace models",
}
PRODUCTION_MODULES = ("protocol", "freeze", "data", "models", "extract", "train", "predict",
                      "preflight", "heads", "linear", "diagnostics", "evaluate", "slurm")
_SHA = re.compile(r"[0-9a-f]{64}")


def read(path):
    return json.loads(Path(path).read_text())


def write_frozen(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise RuntimeError("Refusing to replace frozen artifact: " + str(path))
    else:
        atomic_json(path, value)


def verify_hash(path, expected):
    if not isinstance(expected, str) or not _SHA.fullmatch(expected) or sha256(path) != expected:
        raise RuntimeError("Artifact checksum changed: " + str(path))


def neural_matrix():
    return [{"family": family, "arm": arm, "seed": seed}
            for family in FAMILIES for seed in SEEDS
            for arm in (("logits",) if family == "O" else ARMS)]


def missing_matrix():
    return [row for row in neural_matrix()
            if not (row["family"] == "G" and row["seed"] in IMPORTED_SEEDS)]


def model_key(family, arm, seed):
    if (family not in FAMILIES or type(seed) is not int or seed not in SEEDS
            or arm not in (("logits",) if family == "O" else ARMS)):
        raise ValueError("Unknown fixed family/layer/seed identity")
    return f"{family}/{arm}/seed{seed}"


def run_path(root, family, arm, seed):
    return Path(root).resolve() / "runs" / model_key(family, arm, seed)


def resolve_run(root, family, arm, seed):
    campaign = read_campaign(root)
    key = model_key(family, arm, seed)
    return Path(campaign["reuse"][key]["path"]) if key in campaign["reuse"] else run_path(root, family, arm, seed)


def campaign_sha(root):
    return sha256(Path(root) / "campaign.json")


def source_identity():
    repository = Path(__file__).resolve().parents[2]
    files = ["pilots/final_comparison_20260916/" + name + ".py" for name in PRODUCTION_MODULES]
    files += [
        "polygraph/training/models.py", "polygraph/training/train.py", "polygraph/data/graphs.py",
        "pilots/layer_screen_20260913/models.py", "pilots/layer_screen_20260913/data.py",
        "pilots/layer_screen_20260913/train.py", "pilots/layer_screen_20260913/protocol.py",
        "pilots/layer_screen_20260913/loader_runtime.py",
        "pilots/topology_20260910/protocol.py", "pilots/topology_20260910/data.py",
        "pilots/topology_20260910/extract.py", "pilots/topology_20260910/slurm/imports.py",
        "pilots/layer_ensemble_20260914/protocol.py", "pilots/layer_ensemble_20260914/combine.py",
        "pilots/layer_ensemble_20260914/evaluate.py", "pilots/layer_ensemble_20260914/slurm/runtime.py",
    ]
    return {name: sha256(repository / name) for name in files}


def _validate_contract(campaign):
    expected = {
        "schema_version": 1, "scope_id": SCOPE, "seeds": list(SEEDS), "arms": list(ARMS),
        "families": list(FAMILIES), "training": TRAINING, "model_specs": MODEL_SPECS,
        "neural_matrix": neural_matrix(), "missing_matrix": missing_matrix(),
        "head_recipe": HEAD_RECIPE, "linear_recipe": LINEAR_RECIPE,
        "analysis": ANALYSIS, "diagnostics": DIAGNOSTICS,
        "method_names": list(METHODS), "original_test_access": False,
        **CONTRACT_COUNTS,
    }
    if any(campaign.get(key) != value for key, value in expected.items()):
        raise RuntimeError("The approved final comparison contract changed")
    if not isinstance(campaign.get("run_id"), str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", campaign["run_id"]):
        raise RuntimeError("An explicit safe run identity is required")
    wanted = {model_key("G", arm, seed) for seed in IMPORTED_SEEDS for arm in ARMS}
    if set(campaign.get("reuse", {})) != wanted or set(campaign.get("reuse_groups", {})) != set(map(str, IMPORTED_SEEDS)):
        raise RuntimeError("Exactly the compatible historical G fits for IMPORTED_SEEDS must be registered")


def read_campaign(root):
    require_slurm()
    root = Path(root).resolve()
    value = read(root / "campaign.json")
    _validate_contract(value)
    if value["root"] != str(root) or value["hidden_root"] != str(root / "hidden"):
        raise RuntimeError("Campaign namespace changed")
    if value["source_identity"] != source_identity():
        raise RuntimeError("Campaign source changed after its freeze")
    verify_hash(root / "role_map.json", value["roles_sha256"])
    cache = Path(value["cache"])
    if cache.resolve() != cache:
        raise RuntimeError("Cache namespace was redirected")
    verify_hash(cache / "manifest.json", value["cache_manifest_sha256"])
    manifest = read(cache / "manifest.json")
    if (manifest.get("complete") is not True or manifest.get("diagnostic_only") is not False
            or manifest.get("records") != 28800
            or manifest["cohort_sha256"] != value["cache_cohort_sha256"]
            or manifest["protocol_sha256"] != value["cache_protocol_sha256"]):
        raise RuntimeError("The immutable development cache identity changed")
    return value


def role_spec(root, role):
    if role not in ROLES:
        raise ValueError("Only the four frozen development roles are permitted; test is closed")
    campaign = read_campaign(root)
    roles = read(Path(root) / "role_map.json")
    if roles.get("original_test_access") is not False:
        raise RuntimeError("Original test access is forbidden")
    spec = roles["roles"][role]
    if len(spec["photo_ids"]) != ROLE_PHOTOS[role] or len(spec["record_ids"]) != 9 * ROLE_PHOTOS[role]:
        raise RuntimeError("Role membership count changed")
    if sha256(Path(root) / "role_map.json") != campaign["roles_sha256"]:
        raise RuntimeError("Role map changed while loading")
    return spec


def _timestamp(value):
    parsed = dt.datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("Execution timestamps must be timezone qualified")
    return parsed.timestamp()


def read_execution(root):
    require_slurm()
    campaign = read_campaign(root)
    path = Path(root) / "execution.json"
    if not path.exists():
        raise RuntimeError("Scientific execution requires the separately approved resource/deadline record")
    value = read(path)
    if (value.get("approved") is not True or value.get("scope_id") != SCOPE
            or value.get("campaign_sha256") != campaign_sha(root)
            or value.get("run_id") != campaign["run_id"]
            or value.get("original_test_access") is not False):
        raise RuntimeError("Scientific execution approval does not match this campaign")
    deadlines = value.get("deadlines", {})
    if set(deadlines) != {"base", "predictions", "evaluation"}:
        raise RuntimeError("Explicit base/prediction/evaluation cutoffs are required")
    issued = _timestamp(value["authorized_at"])
    times = [_timestamp(deadlines[name]) for name in ("base", "predictions", "evaluation")]
    if not issued < times[0] < times[1] < times[2]:
        raise RuntimeError("Cutoffs must be ordered and prospective at authorization")
    if time.time() < issued:
        raise RuntimeError("Execution authorization is not active yet")
    return value


def check_cutoff(root, stage):
    if stage not in ("base", "predictions", "evaluation"):
        raise ValueError("Unknown execution phase")
    execution = read_execution(root)
    if time.time() >= _timestamp(execution["deadlines"][stage]):
        raise TimeoutError("Fixed execution cutoff reached: " + stage)


def require_evaluation_gate(root):
    from .freeze import validate_evaluation_gate
    return validate_evaluation_gate(root)


def _group_registry(source, seed, roles_sha, cache_sha):
    required = ["role_map.json", "execution.json", "base_freeze.json", "heads_freeze.json",
                "heads/stack.json", "heads/last_only.json",
                "predictions/meta.npz", "predictions/meta.json",
                "predictions/dev_eval.npz", "predictions/dev_eval.json",
                "evaluation/report.json", "evaluation/complete.json"]
    if seed == 7:
        required.append("evaluation/late_diagnostic.json")
    verify_hash(source / "role_map.json", roles_sha)
    execution = read(source / "execution.json")
    report = read(source / "evaluation/report.json")
    evaluation_complete = read(source / "evaluation/complete.json")
    base = read(source / "base_freeze.json")
    heads = read(source / "heads_freeze.json")
    if (execution["seed"] != seed or execution["cache_manifest_sha256"] != cache_sha
            or execution["roles_sha256"] != roles_sha or execution["test_evaluated"] is not False
            or report["seed"] != seed or report["test_evaluated"] is not False or report["complete"] is not True
            or base["seed"] != seed or base["complete"] is not True or base["arms"] != list(ARMS)
            or heads["seed"] != seed or heads["complete"] is not True or heads["arms"] != list(ARMS)):
        raise RuntimeError("Incompatible historical layer-study group")
    if evaluation_complete.get("complete") is not True or evaluation_complete.get("inputs") != report.get("inputs"):
        raise RuntimeError("Historical evaluation completion is not bound to its report")
    verify_hash(source / "evaluation/report.json", evaluation_complete["files"]["report.json"])
    bindings = {
        "execution_sha256": sha256(source / "execution.json"), "roles_sha256": roles_sha,
        "cache_manifest_sha256": cache_sha, "base_freeze_sha256": sha256(source / "base_freeze.json"),
        "heads_freeze_sha256": sha256(source / "heads_freeze.json"),
        "dev_prediction_npz_sha256": sha256(source / "predictions/dev_eval.npz"),
        "dev_prediction_sidecar_sha256": sha256(source / "predictions/dev_eval.json"),
    }
    if any(report["inputs"].get(key) != value for key, value in bindings.items()):
        raise RuntimeError("Historical evaluation inputs changed")
    for role, rows in (("meta", 3600), ("dev_eval", 7200)):
        sidecar = read(source / "predictions" / (role + ".json"))
        if sidecar["seed"] != seed or sidecar["role"] != role or sidecar["rows"] != rows or sidecar["arms"] != list(ARMS):
            raise RuntimeError("Historical prediction group identity changed")
        for key in ("execution_sha256", "roles_sha256", "cache_manifest_sha256", "base_freeze_sha256"):
            if sidecar.get(key) != bindings[key]:
                raise RuntimeError("Historical prediction inputs changed: " + key)
        verify_hash(source / "predictions" / (role + ".npz"), sidecar["npz_sha256"])
    if set(heads["files"]) != {"heads/stack.json", "heads/last_only.json"}:
        raise RuntimeError("Both historical meta heads must be frozen")
    for key in ("execution_sha256", "roles_sha256", "cache_manifest_sha256", "base_freeze_sha256"):
        if heads.get(key) != bindings[key]:
            raise RuntimeError("Historical meta heads use different source inputs")
    verify_hash(source / "predictions/meta.npz", heads["meta_prediction_npz_sha256"])
    verify_hash(source / "predictions/meta.json", heads["meta_prediction_sidecar_sha256"])
    for name, expected in heads["files"].items():
        if name not in ("heads/stack.json", "heads/last_only.json"):
            raise RuntimeError("Unexpected historical meta head")
        verify_hash(source / name, expected)
    status = "complete"
    if seed == 7:
        marker = read(source / "evaluation/late_diagnostic.json")
        if (marker.get("status") != "complete_late_diagnostic"
                or marker.get("complete") is not True
                or not marker.get("original_protocol_status", "").startswith("incomplete")):
            raise RuntimeError("Historical seed7 late/incomplete status must remain explicit")
        status = marker["status"]
    return {"root": str(source), "seed": seed, "scope_id": execution["scope_id"],
            "status": status, "files": {name: sha256(source / name) for name in required}}


def prepare(root, cache, pilot_root, replication_root, run_id):
    require_slurm()
    root, cache, pilot_root, replication_root = map(lambda p: Path(p).resolve(),
                                                   (root, cache, pilot_root, replication_root))
    for original in (cache, pilot_root, replication_root):
        if root == original or root.is_relative_to(original) or original.is_relative_to(root):
            raise RuntimeError("Final campaign must not overlap a historical experiment/cache")
    if (root / "campaign.json").exists():
        existing = read_campaign(root)
        if (existing["run_id"] != run_id or existing["cache"] != str(cache)
                or existing["pilot_root"] != str(pilot_root)
                or existing["replication_root"] != str(replication_root)):
            raise RuntimeError("Existing campaign cannot change identity")
        return existing
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "campaign.json").exists():
            raise RuntimeError("Campaign appeared during preparation; reconcile its identity")
        manifest = read(cache / "manifest.json")
        group = read(cache / "cohort.json")
        roles = read(pilot_root / "role_map.json")
        if (manifest.get("complete") is not True or manifest.get("diagnostic_only") is not False
                or manifest.get("records") != 28800
                or digest(group) != digest(cache_cohort())
                or digest(read(cache / "protocol.json")) != digest(cache_protocol())
                or manifest["cohort_sha256"] != digest(group)
                or manifest["protocol_sha256"] != digest(cache_protocol()) or roles != make_roles(group)):
            raise RuntimeError("The original feature/role contract does not match the approved benchmark")
        role_sha, cache_sha = sha256(pilot_root / "role_map.json"), sha256(cache / "manifest.json")
        reuse, groups = {}, {}
        for seed in IMPORTED_SEEDS:
            source = pilot_root if seed == 7 else replication_root / f"seed{seed}"
            groups[str(seed)] = _group_registry(source, seed, role_sha, cache_sha)
            frozen = read(source / "base_freeze.json")
            for arm in ARMS:
                directory = source / "runs" / arm / f"seed{seed}"
                config, complete = read(directory / "config.json"), read(directory / "complete.json")
                expected = frozen["runs"][arm]
                for name, key in (("config.json", "config_sha256"), ("best.safetensors", "best_sha256"),
                                  ("complete.json", "complete_sha256")):
                    verify_hash(directory / name, expected[key])
                if (config["seed"] != seed or config["arm"] != arm
                        or config["training_role"] != "base_train" or config["selection_role"] != "checkpoint"
                        or config["roles_sha256"] != role_sha or config["cache_manifest_sha256"] != cache_sha
                        or config["fresh_initialization"] is not True
                        or config.get("dtype") != "float32" or config.get("batch_size") != 24
                        or config.get("training_record_count") != 14400
                        or config.get("preprocessing") != {"kind": "none"}
                        or config["training"].get("width") != 64 or config["training"].get("gnn_layers") != 2
                        or any(config["training"].get(k) != v for k, v in TRAINING.items())
                        or complete["complete"] is not True or complete["completed_epochs"] != 20
                        or complete["seed"] != seed or complete["arm"] != arm
                        or config["scope_id"] != read(source / "execution.json")["scope_id"]
                        or complete["scope_id"] != config["scope_id"]):
                    raise RuntimeError("Historical fit cannot be reused under this fixed20 contract")
                for name, expected_sha in complete["artifacts"].items():
                    if name not in ("config.json", "best.safetensors", "latest.pt", "history.json",
                                    "checkpoint.npz", "checkpoint.json"):
                        raise RuntimeError("Unexpected reusable artifact")
                    verify_hash(directory / name, expected_sha)
                reuse[model_key("G", arm, seed)] = {
                    "path": str(directory), "source_root": str(source), "scope_id": config["scope_id"],
                    "status": groups[str(seed)]["status"], "family": "G", "arm": arm, "seed": seed,
                    "config_sha256": sha256(directory / "config.json"),
                    "complete_sha256": sha256(directory / "complete.json"),
                    "best_sha256": sha256(directory / "best.safetensors"), "artifacts": complete["artifacts"],
                }
        role_target = root / "role_map.json"
        if role_target.exists():
            verify_hash(role_target, role_sha)
        else:
            temporary = role_target.with_name("role_map.json.tmp." + str(os.getpid()))
            with temporary.open("wb") as stream:
                stream.write((pilot_root / "role_map.json").read_bytes())
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(role_target)
        campaign = {
            "schema_version": 1, "scope_id": SCOPE, "run_id": run_id, "root": str(root),
            "cache": str(cache), "hidden_root": str(root / "hidden"),
            "pilot_root": str(pilot_root), "replication_root": str(replication_root),
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "seeds": list(SEEDS), "arms": list(ARMS), "families": list(FAMILIES),
            "training": TRAINING, "model_specs": MODEL_SPECS, "neural_matrix": neural_matrix(),
            "missing_matrix": missing_matrix(), "head_recipe": HEAD_RECIPE, "linear_recipe": LINEAR_RECIPE,
            "analysis": ANALYSIS, "diagnostics": DIAGNOSTICS, "method_names": list(METHODS),
            **CONTRACT_COUNTS,
            "roles_sha256": role_sha, "cache_manifest_sha256": cache_sha,
            "cache_cohort_sha256": manifest["cohort_sha256"], "cache_protocol_sha256": manifest["protocol_sha256"],
            "source_identity": source_identity(), "reuse": reuse, "reuse_groups": groups,
            "original_test_access": False, "evaluation_status": "previously studied development benchmark",
            "legacy_status_policy": "preserve original scopes, timestamps and seed7 late diagnostic",
        }
        _validate_contract(campaign)
        write_frozen(root / "campaign.json", campaign)
        return read_campaign(root)


def main():
    require_slurm()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "cache", "pilot-root", "replication-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    campaign = prepare(args.root, args.cache, args.pilot_root, args.replication_root, args.run_id)
    print(json.dumps({"scope_id": SCOPE, "root": str(args.root),
                      "campaign_sha256": campaign_sha(args.root), "new_neural_fits": COUNTS["new_neural_fits"],
                      "linear_fits": 1, "scientific_execution_authorized": False}), flush=True)


if __name__ == "__main__":
    main()
