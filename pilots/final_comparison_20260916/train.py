"""Fixed20 training with source-bound epoch-boundary resume and strict imports."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import time
import traceback

import numpy as np
import torch
from safetensors.torch import load_file
from torch_geometric.loader import DataLoader as GraphLoader

from pilots.layer_screen_20260913.loader_runtime import settings as loader_settings
from pilots.layer_screen_20260913.protocol import digest
from pilots.layer_screen_20260913.train import (
    BlockShuffleSampler, _atomic_npz, _atomic_safetensors, _check_labels,
    _cpu_state, _finite, _numerical_runtime, _restore_rng, _rng_state, _run_lock,
    _seed, auroc, compare_scores,
)
from pilots.topology_20260910.data import verify_cached_file

from .data import METADATA, RoleDataset, atomic_torch, validate_metadata
from .models import EXPECTED_PARAMETERS, build_model, validate_arm
from .protocol import (
    ARMS, COUNTS, FAMILIES, IMPORTED_SEEDS, MODEL_SPECS, SEEDS, SCOPE, TRAINING,
    atomic_json, campaign_sha, check_cutoff, model_key, read, read_campaign, read_execution,
    require_slurm, resolve_run, run_path, sha256, source_identity, write_frozen,
)

ARTIFACTS = ("config.json", "best.safetensors", "latest.pt", "history.json",
             "checkpoint.npz", "checkpoint.json")
STOP = False


def _unix(value):
    parsed = dt.datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise RuntimeError("A timezone-qualified completion/cutoff is required")
    return parsed.timestamp()


def stage_guard(root, stage):
    """Validate control once; cheaply check its immutable identity on each batch."""
    require_slurm()
    root = Path(root)
    check_cutoff(root, stage)
    execution = read_execution(root)
    cutoff = _unix(execution["deadlines"][stage])
    bindings = {name: sha256(root / name) for name in ("campaign.json", "execution.json")}

    def check():
        if time.time() >= cutoff:
            raise TimeoutError("Fixed execution cutoff reached: " + stage)
        for name, expected in bindings.items():
            verify_cached_file(root / name, expected)

    return check


def initialize(seed):
    require_slurm()
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError("Only the fixed training seeds are allowed")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG=:4096:8 must be set before Python")
    if torch.are_deterministic_algorithms_enabled():
        raise RuntimeError("Ordinary FP32 CUDA is fixed; deterministic mode is audit-only")
    _seed(seed)
    torch.set_float32_matmul_precision("highest")


def runtime():
    require_slurm()
    return {
        **_numerical_runtime(), "matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }


def versions():
    return {name: importlib.metadata.version(name)
            for name in ("torch", "numpy", "torch-geometric", "safetensors")}


def check_device(family, device):
    require_slurm()
    device = torch.device(device)
    expected = "cpu" if family == "O" else "cuda"
    if device.type != expected or (expected == "cuda" and not torch.cuda.is_available()):
        raise RuntimeError(f"{family} requires an allocated {expected} device; no fallback")
    return device


def array_sha256(value):
    require_slurm()
    values = np.ascontiguousarray(value)
    header = json.dumps({"dtype": values.dtype.str, "shape": list(values.shape)}, sort_keys=True)
    result = hashlib.sha256(header.encode("ascii"))
    result.update(values.tobytes())
    return result.hexdigest()


def fit_normalizer(dataset):
    require_slurm()
    if dataset.role != "base_train" or dataset.diagnostic or dataset.family != "O":
        raise RuntimeError("The O normalizer may be fitted on scientific base_train logits only")
    if len(dataset) != 14400:
        raise RuntimeError("O preprocessing requires all 14,400 base-training rows")
    values = dataset.logits()
    _finite(values, "base-training logits")
    mean, std = values.mean(0), values.std(0, unbiased=False)
    return {
        "kind": "base_train_population_standardization", "mean": mean.tolist(),
        "scale": (std + 1e-6).tolist(), "epsilon": 1e-6,
        "fit_role": "base_train", "fit_count": len(dataset),
        "fit_record_ids_sha256": digest(dataset.metadata()["record_id"].tolist()),
        "fit_logits_sha256": array_sha256(values.numpy()),
        "feature_order": list(range(100)),
    }


def _validate_normalizer(value, config):
    if (value.get("kind") != "base_train_population_standardization"
            or value.get("fit_role") != "base_train" or value.get("fit_count") != 14400
            or value.get("epsilon") != 1e-6 or value.get("feature_order") != list(range(100))
            or value.get("fit_record_ids_sha256") != config["training_record_ids_sha256"]):
        raise RuntimeError("O preprocessing must be bound exclusively to base_train")
    mean, scale = np.asarray(value["mean"]), np.asarray(value["scale"])
    if (mean.shape != (100,) or scale.shape != (100,) or not np.isfinite(mean).all()
            or not np.isfinite(scale).all() or (scale <= 0).any()
            or len(value.get("fit_logits_sha256", "")) != 64):
        raise RuntimeError("Invalid all-100-logit training normalizer")


class NormalizedLogits(torch.utils.data.Dataset):
    def __init__(self, dataset, preprocessing):
        require_slurm()
        values = dataset.logits()
        mean = torch.tensor(preprocessing["mean"], dtype=torch.float32)
        scale = torch.tensor(preprocessing["scale"], dtype=torch.float32)
        if values.shape != (len(dataset), 100) or mean.shape != (100,) or scale.shape != (100,):
            raise RuntimeError("Expected exactly 100 ordered logit features")
        if not torch.isfinite(scale).all() or not (scale > 0).all():
            raise RuntimeError("Invalid frozen logit scale")
        self.x = (values - mean) / scale
        self.y = torch.as_tensor(dataset.labels(), dtype=torch.float32)
        _finite(self.x, "normalized logits")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


def make_loader(dataset, config, *, sampler=None, generator=None, indices=None, batch_size=None):
    require_slurm()
    selected = (NormalizedLogits(dataset, config["preprocessing"])
                if config["family"] == "O" else dataset)
    if indices is not None:
        selected = torch.utils.data.Subset(selected, indices)
    if generator is None:
        generator = torch.Generator().manual_seed(0)
    loader = torch.utils.data.DataLoader if config["family"] == "O" else GraphLoader
    return loader(selected, batch_size=batch_size or config["batch_size"], sampler=sampler,
                  shuffle=False, num_workers=0, generator=generator)


def forward(model, batch, family, device):
    require_slurm()
    if family == "O":
        x, y = (value.to(device) for value in batch)
        _finite(x, "input logits")
        if x.dtype != torch.float32 or x.ndim != 2 or x.shape[1] != 100:
            raise RuntimeError("O requires an FP32 matrix of 100 standardized logits")
        scores = model(x).reshape(-1)
    else:
        batch = batch.to(device)
        _finite(batch.x, "node features")
        if batch.x.dtype != torch.float32:
            raise RuntimeError("Graph/hidden/set inputs must be FP32")
        if family == "H":
            if batch.x.shape[1] != 772 or any(key in batch for key in ("edge_attr", "edge_index", "output_logits")):
                raise RuntimeError("Hidden-only models may not receive attention/incidence/logits")
        elif family in ("G", "S"):
            if batch.x.shape[1] != 784 or batch.edge_attr.shape[1] != 12:
                raise RuntimeError("G and S must have exactly matched feature exposure")
            _finite(batch.edge_attr, "edge features")
        else:
            raise ValueError("Unknown neural family")
        scores, _ = model(batch)
        scores, y = scores.reshape(-1), batch.y.reshape(-1)
    _finite(scores, "native error logits")
    _finite(y, "error labels")
    if scores.shape != y.shape:
        raise RuntimeError("Exactly one error logit per record is required")
    return scores, y


@torch.no_grad()
def collect_checkpoint(model, dataset, config, device, *, guard, indices=None, batch_size=None):
    require_slurm()
    if dataset.role != "checkpoint" or dataset.diagnostic:
        raise RuntimeError("Scientific checkpoint selection requires the complete checkpoint role")
    model.eval()
    result = []
    for batch in make_loader(dataset, config, indices=indices, batch_size=batch_size):
        guard()
        result.append(forward(model, batch, config["family"], device)[0].cpu().numpy())
    scores = np.concatenate(result).astype(np.float32, copy=False)
    _finite(scores, "checkpoint scores")
    return scores


def make_config(root, dataset, family, arm, seed, device):
    require_slurm()
    campaign = read_campaign(root)
    key = model_key(family, arm, seed)
    if (key in campaign["reuse"] or dataset.role != "base_train" or dataset.diagnostic
            or dataset.family != family or dataset.arm != arm or len(dataset) != 14400):
        raise RuntimeError("Only complete scientific base_train data may initialize a new fit")
    if {"family": family, "arm": arm, "seed": seed} not in campaign["missing_matrix"]:
        raise RuntimeError(f"This fit is not one of the {COUNTS['new_neural_fits']} approved new neural fits")
    labels = dataset.labels()
    positive, negative = int((labels == 1).sum()), int((labels == 0).sum())
    if not positive or not negative or positive + negative != len(dataset):
        raise RuntimeError("Base BCE weighting requires both binary error outcomes")
    identity = source_identity()
    return {
        "schema_version": 1, "scope_id": SCOPE, "family": family, "arm": arm, "seed": seed,
        "campaign_sha256": campaign_sha(root), "execution_sha256": sha256(Path(root) / "execution.json"),
        "training": TRAINING, "model_spec": MODEL_SPECS[family],
        "parameter_count": EXPECTED_PARAMETERS[family], "batch_size": 24,
        "pos_weight": negative / positive, "training_correct_count": negative,
        "training_error_count": positive, "training_record_count": len(dataset),
        "training_labels_sha256": digest(labels.astype(int).tolist()),
        "training_record_ids_sha256": digest(dataset.metadata()["record_id"].tolist()),
        "sampler_blocks_sha256": digest(dataset.shard_blocks()),
        "preprocessing": fit_normalizer(dataset) if family == "O" else {"kind": "none"},
        "feature_binding": dataset.feature_binding,
        "training_role": "base_train", "selection_role": "checkpoint", "fresh_initialization": True,
        "roles_sha256": campaign["roles_sha256"],
        "cache_manifest_sha256": campaign["cache_manifest_sha256"],
        "cache_protocol_sha256": campaign["cache_protocol_sha256"],
        "cache_cohort_sha256": campaign["cache_cohort_sha256"],
        "source_identity": identity, "implementation": identity,
        "implementation_sha256": digest(identity),
        "loader": loader_settings(0),
        "sampler": "shuffle shard blocks then rows; independent Python Random(seed+epoch); every row once",
        "numerical_runtime": runtime(), "dtype": "float32", "device_type": device.type,
        "versions": versions(), "torch_cuda_version": torch.version.cuda,
    }


def history_selection(history, *, completed_epochs=20):
    require_slurm()
    if type(completed_epochs) is not int or not 1 <= completed_epochs <= 20:
        raise RuntimeError("Invalid fixed20 history horizon")
    if [row.get("epoch") for row in history] != list(range(1, completed_epochs + 1)):
        raise RuntimeError("History must retain each completed epoch in order, including 15 and 20")
    best_auc, best_epoch = -math.inf, 0
    for row in history:
        for name in ("training_loss", "checkpoint_auroc", "training_checkpoint_seconds"):
            value = row.get(name)
            if not isinstance(value, (float, int)) or not math.isfinite(value):
                raise RuntimeError("Missing/nonfinite immutable epoch value: " + name)
        if (row["training_loss"] < 0 or row["training_checkpoint_seconds"] < 0
                or not 0 <= row["checkpoint_auroc"] <= 1):
            raise RuntimeError("Invalid immutable training curve")
        if row["checkpoint_auroc"] > best_auc:
            best_auc, best_epoch = row["checkpoint_auroc"], row["epoch"]
        if row.get("best_epoch") != best_epoch or row.get("best_checkpoint_auroc") != best_auc:
            raise RuntimeError("Checkpoint selection must retain the earliest strict maximum")
    return best_epoch, best_auc


def verify_reuse_group(root, seed, campaign=None):
    require_slurm()
    campaign = read_campaign(root) if campaign is None else campaign
    if type(seed) is not int or seed not in IMPORTED_SEEDS:
        raise RuntimeError("Only the imported G seeds have compatible imported groups")
    group = campaign["reuse_groups"][str(seed)]
    source = Path(group["root"])
    if source.resolve() != source or group.get("seed") != seed:
        raise RuntimeError("Historical group path/seed changed")
    expected_status = "complete_late_diagnostic" if seed == 7 else "complete"
    if group.get("status") != expected_status:
        raise RuntimeError("Historical seed7 lateness and replication status must remain unchanged")
    for name, expected in group["files"].items():
        path = source / name
        if Path(name).is_absolute() or not path.resolve().is_relative_to(source):
            raise RuntimeError("Historical group inventory escapes its registered root")
        verify_cached_file(path, expected)
    required = {"execution.json", "role_map.json", "base_freeze.json", "heads_freeze.json",
                "predictions/meta.npz", "predictions/meta.json",
                "predictions/dev_eval.npz", "predictions/dev_eval.json"}
    if not required.issubset(group["files"]):
        raise RuntimeError("Incomplete original group provenance")
    execution = read(source / "execution.json")
    if (execution["seed"] != seed or execution["scope_id"] != group["scope_id"]
            or execution["roles_sha256"] != campaign["roles_sha256"]
            or execution["cache_manifest_sha256"] != campaign["cache_manifest_sha256"]
            or execution["test_evaluated"] is not False):
        raise RuntimeError("Imported group does not use the frozen base/checkpoint roles")
    if seed == 7:
        marker = read(source / "evaluation/late_diagnostic.json")
        if (marker.get("status") != expected_status
                or not marker.get("original_protocol_status", "").startswith("incomplete")):
            raise RuntimeError("Seed7 must remain a formally incomplete, late diagnostic")
    return group


def _verify_state(state, config, history, best, scores, dataset, config_sha):
    if (state.get("scope_id") != config["scope_id"] or state.get("config_sha256") != config_sha
            or state.get("completed_epochs") != len(history) or state.get("history") != history):
        raise RuntimeError("Resume state scope/config/history binding changed")
    selected, best_auc = history_selection(history, completed_epochs=len(history))
    if state.get("best_epoch") != selected or state.get("best_auc") != best_auc:
        raise RuntimeError("Resume state selected a different checkpoint")
    sampler = BlockShuffleSampler(dataset, config["seed"])
    sampler.load_state_dict(state["sampler"])
    if sampler.epoch != len(history):
        raise RuntimeError("Resume sampler cursor is not at the completed epoch boundary")
    if set(state["best_state"]) != set(best):
        raise RuntimeError("Selected safetensors/resume-state tensor inventories disagree")
    for name in best:
        if best[name].dtype != torch.float32 or not torch.equal(best[name], state["best_state"][name]):
            raise RuntimeError("Selected native tensors differ from the committed resume state")
        _finite(best[name], "selected tensor " + name)
    if not np.allclose(state["best_scores"], scores, atol=1e-5, rtol=1e-5):
        raise RuntimeError("Selected checkpoint scores differ from the resume state")
    for name in ("python", "numpy", "torch", "cuda"):
        if name not in state["rng"]:
            raise RuntimeError("Missing resumable RNG stream: " + name)
    if state["loader_rng"].dtype != torch.uint8 or not state["loader_rng"].numel():
        raise RuntimeError("Missing dedicated loader-generator state")
    optimizer = state["optimizer"]
    if not optimizer.get("state") or not optimizer.get("param_groups"):
        raise RuntimeError("Missing optimizer state")
    for group in optimizer["param_groups"]:
        if group["lr"] != .002 or group["weight_decay"] != .0001:
            raise RuntimeError("Resume optimizer changed the fixed recipe")
    for entry in optimizer["state"].values():
        for value in entry.values():
            if torch.is_tensor(value):
                _finite(value, "optimizer state")
    if (state["audit"]["indices"] != list(range(48))
            or state["audit"].get("role") != "checkpoint"):
        raise RuntimeError("Resume restoration audit must use the first 48 checkpoint rows")
    _finite(state["audit"]["scores"], "stored restoration audit scores")


def verify_complete(root, family, arm, seed):
    require_slurm()
    campaign = read_campaign(root)
    key = model_key(family, arm, seed)
    imported = key in campaign["reuse"]
    directory = resolve_run(root, family, arm, seed)
    if directory.resolve() != directory:
        raise RuntimeError("A completed run path was redirected")
    complete, config = read(directory / "complete.json"), read(directory / "config.json")
    if (complete.get("complete") is not True or complete.get("completed_epochs") != 20
            or complete.get("selection_role") != "checkpoint"
            or complete.get("test_evaluated") is not False or complete.get("dev_eval_accessed") is not False
            or config.get("training_role") != "base_train" or config.get("selection_role") != "checkpoint"
            or config.get("fresh_initialization") is not True or config.get("dtype") != "float32"
            or config.get("batch_size") != 24 or config.get("training_record_count") != 14400):
        raise RuntimeError("Only complete, checkpoint-selected fixed20 base fits are eligible")
    for value in (complete, config):
        if value.get("arm") != arm or value.get("seed") != seed:
            raise RuntimeError("Run does not belong to this exact matrix slot")
    if any(config["training"].get(name) != value for name, value in TRAINING.items()):
        raise RuntimeError("Fixed20 optimizer/exposure/selection recipe changed")
    for name in ("roles_sha256", "cache_manifest_sha256", "cache_protocol_sha256", "cache_cohort_sha256"):
        if config.get(name) != campaign[name]:
            raise RuntimeError("Run data identity changed: " + name)
    if config.get("implementation_sha256") != digest(config["implementation"]):
        raise RuntimeError("Original executed-source identity is not self-consistent")
    if set(complete.get("artifacts", {})) != set(ARTIFACTS):
        raise RuntimeError("Every native checkpoint, resume state and history artifact is required")
    for name in ARTIFACTS:
        verify_cached_file(directory / name, complete["artifacts"][name])
    if imported:
        registry = campaign["reuse"][key]
        group = verify_reuse_group(root, seed, campaign)
        source = Path(group["root"])
        if (family != "G" or directory != source / "runs" / arm / f"seed{seed}"
                or registry["source_root"] != str(source) or registry["status"] != group["status"]
                or registry["scope_id"] != group["scope_id"]
                or config["scope_id"] != registry["scope_id"] or complete["scope_id"] != registry["scope_id"]
                or config["training"].get("width") != 64 or config["training"].get("gnn_layers") != 2
                or config["preprocessing"] != {"kind": "none"}
                or config.get("device_type") != "cuda" or registry["artifacts"] != complete["artifacts"]):
            raise RuntimeError("Historical fit is not an unchanged compatible G import")
        for name, field in (("config.json", "config_sha256"), ("complete.json", "complete_sha256"),
                            ("best.safetensors", "best_sha256")):
            verify_cached_file(directory / name, registry[field])
        verify_cached_file(source / "execution.json", config["execution_sha256"])
        execution = read(source / "execution.json")
        cutoff = _unix(execution["deadlines"]["base_complete_before"])
        issued = 0.0
    else:
        execution = read_execution(root)
        cutoff, issued = _unix(execution["deadlines"]["base"]), _unix(execution["authorized_at"])
        if (directory != run_path(root, family, arm, seed) or config.get("scope_id") != SCOPE
                or complete.get("scope_id") != SCOPE or config.get("family") != family
                or complete.get("family") != family or config["training"] != TRAINING
                or config.get("model_spec") != MODEL_SPECS[family]
                or config.get("parameter_count") != EXPECTED_PARAMETERS[family]
                or config.get("source_identity") != campaign["source_identity"]
                or config["implementation"] != campaign["source_identity"]
                or config.get("campaign_sha256") != campaign_sha(root)
                or complete.get("campaign_sha256") != campaign_sha(root)
                or config.get("device_type") != ("cpu" if family == "O" else "cuda")):
            raise RuntimeError("New fit source, architecture or execution identity changed")
        verify_cached_file(Path(root) / "execution.json", config["execution_sha256"])
        if family == "O":
            _validate_normalizer(config["preprocessing"], config)
        elif config["preprocessing"] != {"kind": "none"}:
            raise RuntimeError("No graph/hidden/set preprocessing may be fitted")
    when = complete.get("completed_unix")
    if not isinstance(when, (float, int)) or not math.isfinite(when) or not issued < when < cutoff:
        raise RuntimeError("Base fit did not complete inside its original authorized window")
    dataset = RoleDataset(root, "base_train", family, arm)
    _check_labels(dataset)
    labels = dataset.labels()
    positive, negative = int((labels == 1).sum()), int((labels == 0).sum())
    if (not positive or not negative or config.get("pos_weight") != negative / positive
            or config["training_labels_sha256"] != digest(labels.astype(int).tolist())):
        raise RuntimeError("Class weighting or training targets differ from base_train")
    if not imported:
        if (config["feature_binding"] != dataset.feature_binding
                or config["training_record_ids_sha256"] != digest(dataset.metadata()["record_id"].tolist())
                or config["sampler_blocks_sha256"] != digest(dataset.shard_blocks())):
            raise RuntimeError("Training features, ordered exposure or sampler blocks changed")
    history = read(directory / "history.json")
    selected, best_auc = history_selection(history)
    checkpoint = read(directory / "checkpoint.json")
    if (complete.get("best_epoch") != selected or checkpoint.get("best_epoch") != selected
            or checkpoint.get("selection_role") != "checkpoint"
            or checkpoint.get("dev_eval_accessed") is not False
            or checkpoint.get("threshold_fitted") is not False):
        raise RuntimeError("The selected checkpoint must be the earliest checkpoint-role maximum")
    checkpoint_ds = RoleDataset(root, "checkpoint", family, arm)
    with np.load(directory / "checkpoint.npz", allow_pickle=False) as archive:
        values = {name: archive[name].copy() for name in (*METADATA, "score", "logit")}
    validate_metadata(root, "checkpoint", values, expected=checkpoint_ds.metadata())
    scores = values["score"]
    if (scores.dtype != np.float32 or scores.shape != (3600,)
            or not np.isfinite(scores).all() or not np.array_equal(scores, values["logit"])
            or abs(auroc(values["y"], scores) - checkpoint["checkpoint_auroc"]) > 1e-12):
        raise RuntimeError("Selected checkpoint scores/labels/AUROC changed")
    state = torch.load(directory / "latest.pt", map_location="cpu", weights_only=False)
    best = load_file(str(directory / "best.safetensors"), device="cpu")
    _verify_state(state, config, history, best, scores, dataset, sha256(directory / "config.json"))
    if sum(value.numel() for value in best.values()) != EXPECTED_PARAMETERS[family]:
        raise RuntimeError("Selected native parameter count differs from the approved architecture")
    for name in ARTIFACTS:
        verify_cached_file(directory / name, complete["artifacts"][name])
    return complete, config


def load_run(root, family, arm, seed, device):
    require_slurm()
    device = check_device(family, device)
    complete, config = verify_complete(root, family, arm, seed)
    directory = resolve_run(root, family, arm, seed)
    if config["scope_id"] != SCOPE:
        current = source_identity()
        forward_sources = ("polygraph/training/models.py", "polygraph/training/train.py",
                           "polygraph/data/graphs.py", "pilots/layer_screen_20260913/models.py",
                           "pilots/layer_screen_20260913/data.py")
        if any(config["implementation"].get(name) != current.get(name) for name in forward_sources):
            raise RuntimeError("Historical forward source differs; use its archived executed source, never relabel it")
    model = build_model(family, arm).to(device)
    model.load_state_dict(load_file(str(directory / "best.safetensors"), device="cpu"), strict=True)
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise RuntimeError("Restored detector parameters must remain FP32")
    model.eval().requires_grad_(False)
    return model, config, complete


def _stop_handler(_signum, _frame):
    global STOP
    STOP = True


def record_failure(root, stage, identity, error, *, status="failed"):
    root = Path(root).resolve()
    campaign_file = root / "campaign.json"
    if not campaign_file.exists() or read(campaign_file).get("scope_id") != SCOPE:
        return
    name = f"{stage}_{time.time_ns()}_{os.getpid()}.json"
    atomic_json(root / "failures" / name, {
        "scope_id": SCOPE, "complete": False, "status": status, "stage": stage,
        **identity, "error": repr(error), "traceback": traceback.format_exc(),
        "failed_unix": time.time(), "job_id": os.environ.get("SLURM_JOB_ID"),
    })


def fit(root, family, arm, seed, device):
    require_slurm()
    root = Path(root).resolve()
    key = model_key(family, arm, seed)
    campaign = read_campaign(root)
    if key in campaign["reuse"]:
        raise RuntimeError("Refusing to retrain an imported fit: " + key)
    device = check_device(family, device)
    guard = stage_guard(root, "base")
    initialize(seed)
    train_ds = RoleDataset(root, "base_train", family, arm)
    checkpoint_ds = RoleDataset(root, "checkpoint", family, arm)
    _check_labels(train_ds)
    _check_labels(checkpoint_ds)
    config = make_config(root, train_ds, family, arm, seed, device)
    directory = run_path(root, family, arm, seed)
    if directory.resolve() != directory:
        raise RuntimeError("Fresh fits must write exclusively into the new run namespace")
    directory.mkdir(parents=True, exist_ok=True)
    with _run_lock(directory):
        path, latest = directory / "config.json", directory / "latest.pt"
        if path.exists():
            if read(path) != config:
                raise RuntimeError("Resume config, source, inputs, execution or numerical runtime changed")
        else:
            if any((directory / name).exists() for name in (*ARTIFACTS[1:], "complete.json")):
                raise RuntimeError("Unrecognized artifacts cannot initialize a fresh run")
            write_frozen(path, config)
        if (directory / "complete.json").exists():
            verify_complete(root, family, arm, seed)
            return True
        initialize(seed)
        model = build_model(family, arm).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.0001)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(config["pos_weight"], device=device))
        sampler = BlockShuffleSampler(train_ds, seed)
        loader_rng = torch.Generator().manual_seed(seed + 1000003)
        history, best_state, best_scores = [], None, None
        best_auc, best_epoch, start_epoch = -math.inf, 0, 0
        audit_ids = list(range(48))
        attempts_path = directory / "attempts.json"
        attempts = read(attempts_path) if attempts_path.exists() else []
        attempt = {"started_unix": time.time(), "job_id": os.environ["SLURM_JOB_ID"],
                   "resume": latest.exists(), "status": "running", "device": str(device)}
        attempts.append(attempt)
        atomic_json(attempts_path, attempts)
        try:
            if latest.exists():
                state = torch.load(latest, map_location="cpu", weights_only=False)
                if state.get("scope_id") != SCOPE or state.get("config_sha256") != sha256(path):
                    raise RuntimeError("Foreign or unbound resume container")
                history, start_epoch = state["history"], state["completed_epochs"]
                _verify_state(state, config, history, state["best_state"], state["best_scores"],
                              train_ds, sha256(path))
                model.load_state_dict(state["model"], strict=True)
                optimizer.load_state_dict(state["optimizer"])
                sampler.load_state_dict(state["sampler"])
                loader_rng.set_state(state["loader_rng"])
                best_state, best_scores = state["best_state"], state["best_scores"]
                best_auc, best_epoch = state["best_auc"], state["best_epoch"]
                restored = collect_checkpoint(model, checkpoint_ds, config, device, guard=guard, indices=audit_ids)
                audit = compare_scores(state["audit"]["scores"], restored)
                audit["alternative_partition"] = compare_scores(
                    restored, collect_checkpoint(model, checkpoint_ds, config, device,
                                                 guard=guard, indices=audit_ids, batch_size=12))
                atomic_json(directory / "resume_audit.json", audit)
                _restore_rng(state["rng"])
                _atomic_safetensors(directory / "best.safetensors", best_state)
                atomic_json(directory / "history.json", history)
            loader = make_loader(train_ds, config, sampler=sampler, generator=loader_rng)
            for epoch in range(start_epoch, 20):
                check_cutoff(root, "base")
                epoch_started = time.monotonic()
                model.train()
                total_loss, count = 0.0, 0
                for batch in loader:
                    guard()
                    optimizer.zero_grad(set_to_none=True)
                    scores, labels = forward(model, batch, family, device)
                    loss = criterion(scores, labels)
                    _finite(loss, "training loss")
                    loss.backward()
                    for name, parameter in model.named_parameters():
                        if parameter.grad is not None:
                            _finite(parameter.grad, "gradient " + name)
                    optimizer.step()
                    total_loss += float(loss.detach()) * len(labels)
                    count += len(labels)
                if count != 14400 or count != len(train_ds):
                    raise RuntimeError("Each epoch must expose every base-training record exactly once")
                scores = collect_checkpoint(model, checkpoint_ds, config, device, guard=guard)
                checkpoint_auc = auroc(checkpoint_ds.labels(), scores)
                if checkpoint_auc > best_auc:
                    best_auc, best_epoch = checkpoint_auc, epoch + 1
                    best_state, best_scores = _cpu_state(model), scores.copy()
                    _atomic_safetensors(directory / "best.safetensors", best_state)
                history.append({
                    "epoch": epoch + 1, "training_loss": total_loss / count,
                    "checkpoint_auroc": checkpoint_auc, "best_epoch": best_epoch,
                    "best_checkpoint_auroc": best_auc,
                    "training_checkpoint_seconds": time.monotonic() - epoch_started,
                })
                audit_scores = collect_checkpoint(model, checkpoint_ds, config, device,
                                                  guard=guard, indices=audit_ids)
                state = {
                    "scope_id": SCOPE, "config_sha256": sha256(path),
                    "model": _cpu_state(model), "optimizer": optimizer.state_dict(),
                    "completed_epochs": epoch + 1, "history": history,
                    "sampler": sampler.state_dict(), "loader_rng": loader_rng.get_state(), "rng": _rng_state(),
                    "best_state": best_state, "best_scores": best_scores,
                    "best_auc": best_auc, "best_epoch": best_epoch,
                    "training_checkpoint_seconds": sum(row["training_checkpoint_seconds"] for row in history),
                    "audit": {"indices": audit_ids, "scores": audit_scores, "role": "checkpoint",
                              "device_type": device.type, "batch_size": 24, "dtype": "float32"},
                }
                check_cutoff(root, "base")
                atomic_torch(latest, state)
                atomic_json(directory / "history.json", history)
                print(json.dumps({"event": "epoch_complete", "family": family, "arm": arm,
                                  "seed": seed, **history[-1]}), flush=True)
                if STOP and epoch + 1 < 20:
                    attempt.update(status="interrupted_at_epoch_boundary", completed_epochs=epoch + 1,
                                   ended_unix=time.time(), complete=False)
                    atomic_json(attempts_path, attempts)
                    record_failure(root, "train", {"family": family, "arm": arm, "seed": seed},
                                   RuntimeError("Stopped after a committed epoch; no eligible complete fit"),
                                   status="interrupted_at_epoch_boundary")
                    return False
            history_selection(history)
            _atomic_safetensors(directory / "best.safetensors", best_state)
            model.load_state_dict(load_file(str(directory / "best.safetensors"), device="cpu"), strict=True)
            restored = collect_checkpoint(model, checkpoint_ds, config, device, guard=guard)
            if not np.allclose(best_scores, restored, atol=1e-5, rtol=1e-5):
                raise RuntimeError("Full checkpoint-role restoration exceeded fixed atol=rtol=1e-5")
            reference = restored[audit_ids]
            audit = compare_scores(reference, collect_checkpoint(
                model, checkpoint_ds, config, device, guard=guard, indices=audit_ids))
            audit["alternative_partition"] = compare_scores(reference, collect_checkpoint(
                model, checkpoint_ds, config, device, guard=guard, indices=audit_ids, batch_size=12))
            _atomic_npz(directory / "checkpoint.npz",
                        {**checkpoint_ds.metadata(), "score": restored, "logit": restored})
            atomic_json(directory / "checkpoint.json", {
                "selection_role": "checkpoint", "best_epoch": best_epoch,
                "checkpoint_auroc": auroc(checkpoint_ds.labels(), restored),
                "restoration_audit": audit, "threshold_fitted": False, "dev_eval_accessed": False,
            })
            check_cutoff(root, "base")
            complete = {
                "schema_version": 1, "scope_id": SCOPE, "complete": True,
                "family": family, "arm": arm, "seed": seed, "completed_epochs": 20,
                "best_epoch": best_epoch, "selection_role": "checkpoint",
                "campaign_sha256": campaign_sha(root),
                "artifacts": {name: sha256(directory / name) for name in ARTIFACTS},
                "test_evaluated": False, "dev_eval_accessed": False,
                "training_checkpoint_seconds": sum(row["training_checkpoint_seconds"] for row in history),
                "completed_unix": time.time(), "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "job_id": os.environ["SLURM_JOB_ID"],
            }
            guard()
            write_frozen(directory / "complete.json", complete)
            verify_complete(root, family, arm, seed)
            attempt.update(status="complete", completed_epochs=20, ended_unix=time.time())
            atomic_json(attempts_path, attempts)
            return True
        except BaseException as error:
            attempt.update(status="failed", complete=False, ended_unix=time.time(),
                           error=repr(error), traceback=traceback.format_exc())
            atomic_json(attempts_path, attempts)
            raise


def main():
    require_slurm()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument("--arm", choices=(*ARMS, "logits"), required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), required=True)
    args = parser.parse_args()
    validate_arm(args.family, args.arm)
    previous = {name: signal.signal(name, _stop_handler) for name in (signal.SIGTERM, signal.SIGINT)}
    try:
        done = fit(args.root, args.family, args.arm, args.seed, args.device)
        if not done:
            raise SystemExit(2)
    except BaseException as error:
        record_failure(args.root, "train", {"family": args.family, "arm": args.arm, "seed": args.seed}, error)
        raise
    finally:
        for name, handler in previous.items():
            signal.signal(name, handler)


if __name__ == "__main__":
    main()
