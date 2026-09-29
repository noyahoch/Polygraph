"""Frozen role exports; historical G matrices are copied byte-for-byte, never rescored."""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch_geometric.data import Batch

from pilots.layer_screen_20260913.train import _atomic_npz, _finite
from pilots.topology_20260910.data import verify_cached_file

from .data import METADATA, RoleDataset, role_metadata, validate_metadata
from .models import HIDDEN_INDEX
from .protocol import (
    ARMS, COUNTS, FAMILIES, IMPORTED_SEEDS, SEEDS, SCOPE, atomic_json, campaign_sha,
    check_cutoff, model_key, neural_matrix, read, read_campaign, read_execution,
    require_evaluation_gate, require_slurm, resolve_run, sha256, source_identity, write_frozen,
)
from .train import (
    check_device, forward, initialize, load_run, make_loader, record_failure, runtime,
    stage_guard, verify_complete, verify_reuse_group,
)


def _role_guard(root, role):
    if role not in ("meta", "dev_eval"):
        raise ValueError("Only meta and gated dev_eval prediction exports are allowed")
    if role == "dev_eval":
        require_evaluation_gate(root)


def prediction_path(root, family, seed, role):
    if family not in FAMILIES or type(seed) is not int or seed not in SEEDS:
        raise ValueError("Unknown prediction family/seed")
    if role not in ("meta", "dev_eval"):
        raise ValueError("Unknown prediction role")
    path = Path(root).resolve() / "predictions" / role / family / f"seed{seed}.npz"
    if path.resolve() != path:
        raise RuntimeError("Prediction outputs may not redirect into another namespace")
    return path


def _base_freeze(root):
    value = read(Path(root) / "base_freeze.json")
    expected = {model_key(row["family"], row["arm"], row["seed"]) for row in neural_matrix()}
    if (value.get("scope_id") != SCOPE or value.get("complete") is not True
            or value.get("campaign_sha256") != campaign_sha(root)
            or value.get("source_identity") != source_identity()
            or value.get("neural_fits") != COUNTS["neural_fits"] or set(value.get("models", {})) != expected):
        raise RuntimeError("All registered bases must be globally frozen before any meta/dev export")
    execution = read_execution(root)
    frozen = dt.datetime.fromisoformat(value["frozen_utc"]).timestamp()
    start = dt.datetime.fromisoformat(execution["authorized_at"]).timestamp()
    cutoff = dt.datetime.fromisoformat(execution["deadlines"]["base"]).timestamp()
    if not start <= frozen < cutoff:
        raise RuntimeError("The base freeze was not created in its authorized window")
    return value


def _bindings(root, family, seed, role, campaign):
    arms = ("logits",) if family == "O" else ARMS
    frozen = _base_freeze(root)
    runs = {}
    for arm in arms:
        key = model_key(family, arm, seed)
        directory = resolve_run(root, family, arm, seed)
        record = frozen["models"][key]
        if (record.get("path") != str(directory) or record.get("family") != family
                or record.get("arm") != arm or record.get("seed") != seed):
            raise RuntimeError("Base freeze binds a different model to this prediction slot")
        for name, field in (("config.json", "config_sha256"), ("best.safetensors", "best_sha256"),
                            ("complete.json", "complete_sha256"), ("history.json", "history_sha256")):
            verify_cached_file(directory / name, record[field])
        runs[arm] = {name: record[name] for name in
                     ("path", "config_sha256", "best_sha256", "complete_sha256", "history_sha256")}
    values = {
        "schema_version": 1, "scope_id": SCOPE, "campaign_sha256": campaign_sha(root),
        "role": role, "family": family, "seed": seed, "arms": list(arms),
        "rows": 3600 if role == "meta" else 7200,
        "roles_sha256": campaign["roles_sha256"],
        "cache_manifest_sha256": campaign["cache_manifest_sha256"],
        "base_freeze_sha256": sha256(Path(root) / "base_freeze.json"),
        "source_identity": campaign["source_identity"], "runs": runs,
        "score": "native_error_logit", "dtype": "float32", "original_test_access": False,
    }
    if role == "dev_eval":
        values["evaluation_gate_sha256"] = sha256(Path(root) / "evaluation_gate.json")
    return values


def _validate_arrays(root, role, values, arms, expected_metadata=None):
    if set(values) != set(METADATA) | {"logits"}:
        raise RuntimeError("Prediction exports require exactly eight metadata columns and raw logits")
    validate_metadata(root, role, values, expected=expected_metadata)
    n = 3600 if role == "meta" else 7200
    if (values["logits"].dtype != np.float32 or values["logits"].shape != (n, len(arms))
            or not np.isfinite(values["logits"]).all()):
        raise RuntimeError("Prediction matrix must contain finite aligned FP32 logits in fixed arm order")


def read_predictions(root, family, seed, role):
    require_slurm()
    _role_guard(root, role)
    campaign = read_campaign(root)
    path = prediction_path(root, family, seed, role)
    sidecar = read(path.with_suffix(".json"))
    expected = _bindings(root, family, seed, role, campaign)
    if sidecar.get("complete") is not True or any(sidecar.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Prediction receipt does not match the frozen campaign/role/models")
    verify_cached_file(path, sidecar["npz_sha256"])
    with np.load(path, allow_pickle=False) as archive:
        values = {name: archive[name].copy() for name in archive.files}
    _validate_arrays(root, role, values, expected["arms"])
    completed = sidecar.get("completed_unix")
    execution = read_execution(root)
    start = dt.datetime.fromisoformat(execution["authorized_at"]).timestamp()
    cutoff = dt.datetime.fromisoformat(execution["deadlines"]["predictions"]).timestamp()
    if (not isinstance(completed, (float, int)) or not np.isfinite(completed)
            or not start < completed < cutoff):
        raise RuntimeError("Prediction export did not complete in its authorized window")
    imported = family == "G" and seed in IMPORTED_SEEDS
    if sidecar.get("imported") is not imported:
        raise RuntimeError("Prediction receipt import flag differs from the registered import set")
    if imported:
        group = verify_reuse_group(root, seed, campaign)
        original = sidecar.get("imported_from", {})
        source = Path(group["root"])
        old_path = source / "predictions" / f"{role}.npz"
        old_json = old_path.with_suffix(".json")
        if (sidecar.get("status") != "imported_unchanged"
                or original.get("status") != group["status"] or original.get("scope_id") != group["scope_id"]
                or original.get("root") != str(source)
                or original.get("npz_sha256") != sidecar["npz_sha256"]
                or original.get("npz_sha256") != group["files"][f"predictions/{role}.npz"]
                or original.get("sidecar_sha256") != group["files"][f"predictions/{role}.json"]
                or original.get("completed_unix") != read(old_json)["completed_unix"]):
            raise RuntimeError("Historical scores/status must be imported unchanged, never relabeled")
        verify_cached_file(old_path, sidecar["npz_sha256"])
    elif sidecar.get("status") != "complete_new":
        raise RuntimeError("Fresh predictions must not masquerade as historical imports")
    verify_cached_file(path, sidecar["npz_sha256"])
    return values, sidecar


def _import_group(root, seed, role, path, campaign, expected, guard):
    group = verify_reuse_group(root, seed, campaign)
    for arm in ARMS:
        verify_complete(root, "G", arm, seed)
    source = Path(group["root"])
    old_path = source / "predictions" / f"{role}.npz"
    old_sidecar = read(old_path.with_suffix(".json"))
    if (old_sidecar.get("seed") != seed or old_sidecar.get("role") != role
            or old_sidecar.get("arms") != list(ARMS) or old_sidecar.get("rows") != expected["rows"]
            or old_sidecar.get("roles_sha256") != campaign["roles_sha256"]
            or old_sidecar.get("cache_manifest_sha256") != campaign["cache_manifest_sha256"]
            or old_sidecar.get("npz_sha256") != group["files"][f"predictions/{role}.npz"]):
        raise RuntimeError("Historical prediction identity/order changed")
    with np.load(old_path, allow_pickle=False) as archive:
        values = {name: archive[name].copy() for name in archive.files}
    _validate_arrays(root, role, values, ARMS)
    guard()
    pending = path.with_name(path.name + f".pending.{time.time_ns()}.{os.getpid()}")
    with old_path.open("rb") as incoming, pending.open("xb") as outgoing:
        shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
        outgoing.flush()
        os.fsync(outgoing.fileno())
    if sha256(pending) != group["files"][f"predictions/{role}.npz"]:
        raise RuntimeError("Historical prediction bytes changed during import")
    guard()
    os.link(pending, path)
    pending.unlink()
    return {
        "status": "imported_unchanged",
        "imported_from": {
            "root": str(source), "scope_id": group["scope_id"], "status": group["status"],
            "npz_sha256": group["files"][f"predictions/{role}.npz"],
            "sidecar_sha256": group["files"][f"predictions/{role}.json"],
            "completed_unix": old_sidecar["completed_unix"],
            "original_protocol_status": ("incomplete; explicitly late diagnostic" if seed == 7
                                         else "completed separate seed17/27 replication"),
            "refitted": False, "rescored": False, "relabeled": False,
        },
    }


@torch.inference_mode()
def _new_scores(root, family, seed, role, device, guard):
    arms = ("logits",) if family == "O" else ARMS
    models, configs, datasets = {}, {}, {}
    for arm in arms:
        models[arm], configs[arm], _ = load_run(root, family, arm, seed, device)
        datasets[arm] = RoleDataset(root, role, family, arm)
    primary = datasets[arms[0]]
    expected_metadata = primary.metadata()
    for dataset in datasets.values():
        if dataset.entries != primary.entries:
            raise RuntimeError("Every arm must predict exactly the same ordered records")
    if family == "O":
        scores = []
        for batch in make_loader(primary, configs["logits"]):
            guard()
            scores.append(forward(models["logits"], batch, "O", device)[0].cpu().numpy())
        result = np.concatenate(scores).astype(np.float32, copy=False)[:, None]
        return {**expected_metadata, "logits": result}
    matrices = []
    for first in range(0, len(primary), 24):
        guard()
        indices = list(range(first, min(first + 24, len(primary))))
        rows = [primary.entries[index] for index in indices]
        if family == "H":
            hidden_payloads = {
                name: primary._load_hidden(name)
                for name in dict.fromkeys(primary.hidden_entries[row["record_id"]]["hidden_shard"] for row in rows)
            }
            final = datasets["block11"]
            cache_payloads = {name: final._load(name) for name in dict.fromkeys(row["shard"] for row in rows)}
        else:
            hidden_payloads = {}
            cache_payloads = {name: primary._load(name) for name in dict.fromkeys(row["shard"] for row in rows)}
        columns = []
        for arm in arms:
            guard()
            dataset = datasets[arm]
            graphs = []
            for index, row in zip(indices, rows):
                hidden = None
                payload = cache_payloads[row["shard"]]
                if family == "H" and HIDDEN_INDEX[arm] != 12:
                    hidden = hidden_payloads[dataset.hidden_entries[row["record_id"]]["hidden_shard"]]
                graphs.append(dataset.graph_at(index, payload=payload, hidden_payload=hidden))
            batch = Batch.from_data_list(graphs)
            columns.append(forward(models[arm], batch, family, device)[0].cpu().numpy())
        matrices.append(np.stack(columns, axis=1).astype(np.float32, copy=False))
        if first % 480 == 0:
            check_cutoff(root, "predictions")
            print(json.dumps({"event": "prediction_progress", "family": family, "seed": seed,
                              "role": role, "records": first + len(rows), "total": len(primary)}), flush=True)
    return {**expected_metadata, "logits": np.concatenate(matrices, axis=0)}


def predict_group(root, family, seed, role, device="cuda"):
    require_slurm()
    _role_guard(root, role)
    campaign = read_campaign(root)
    root = Path(root).resolve()
    path = prediction_path(root, family, seed, role)
    expected = _bindings(root, family, seed, role, campaign)
    guard = stage_guard(root, "predictions")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if path.with_suffix(".json").exists():
            return read_predictions(root, family, seed, role)
        if path.exists():
            raise RuntimeError("Uncommitted prediction matrix exists; do not overwrite or silently adopt it")
        started = time.monotonic()
        if family == "G" and seed in IMPORTED_SEEDS:
            details = _import_group(root, seed, role, path, campaign, expected, guard)
            details.update(imported=True,
                           elapsed_scope="file import/copy only; historical inference not re-timed")
        else:
            selected_device = check_device(family, "cpu" if family == "O" else device)
            initialize(seed)
            if selected_device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(selected_device)
            values = _new_scores(root, family, seed, role, selected_device, guard)
            _finite(values["logits"], "exported native logits")
            _validate_arrays(root, role, values, expected["arms"])
            guard()
            _atomic_npz(path, values)
            details = {
                "status": "complete_new", "imported": False, "numerical_runtime": runtime(),
                "device": str(selected_device),
                "peak_gpu_bytes": (torch.cuda.max_memory_allocated(selected_device)
                                   if selected_device.type == "cuda" else 0),
            }
        check_cutoff(root, "predictions")
        sidecar = {
            **expected, **details, "complete": True, "npz_sha256": sha256(path),
            "elapsed_seconds": time.monotonic() - started, "completed_unix": time.time(),
            "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "job_id": os.environ["SLURM_JOB_ID"],
        }
        guard()
        write_frozen(path.with_suffix(".json"), sidecar)
        return read_predictions(root, family, seed, role)


def main():
    require_slurm()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--role", choices=("meta", "dev_eval"), required=True)
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    try:
        _, receipt = predict_group(args.root, args.family, args.seed, args.role, args.device)
        print(json.dumps({"event": "predictions_complete", "family": args.family, "seed": args.seed,
                          "role": args.role, "npz_sha256": receipt["npz_sha256"]}), flush=True)
    except BaseException as error:
        record_failure(args.root, "predict", {"family": args.family, "seed": args.seed, "role": args.role}, error)
        raise


if __name__ == "__main__":
    main()
