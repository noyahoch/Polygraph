"""Fresh, resumable detector fits and hash-bound inference for the fixed protocol.

Run only inside Slurm. ``latest.pt`` is this trainer's trusted resume container;
``best.safetensors`` contains only the selected detector's native tensor state.
No historical checkpoint, test statistic, or test label participates in training.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import signal
import time
import traceback

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch_geometric.loader import DataLoader as GraphLoader

from .data import CachedDataset, atomic_torch, check_freeze
from .models import build_model
from .protocol import ARMS, SEEDS, atomic_json, digest, file_sha256, protocol, require_slurm
from .rewiring_decision import DECISION_PATH, DECISION_SHA256

SCHEMA_VERSION = 1
ATOL = RTOL = 1e-5
_STOP_REQUESTED = False
_ARTIFACTS = {"config_sha256": "config.json", "best_sha256": "best.safetensors",
              "validation_npz_sha256": "validation.npz", "validation_json_sha256": "validation.json"}


def _value(dataset, name):
    value = getattr(dataset, name)
    return value() if callable(value) else value


def _finite(value, name):
    valid = torch.isfinite(value).all().item() if torch.is_tensor(value) else np.isfinite(value).all()
    if not valid:
        raise FloatingPointError(f"Non-finite {name}")


def _implementation_identity():
    root = Path(__file__).resolve().parents[2]
    files = ["pilots/topology_20260910/" + name + ".py" for name in ("train", "data", "models", "protocol", "rewiring_decision")]
    files += [DECISION_PATH]
    files += ["polygraph/training/train.py", "polygraph/training/models.py", "polygraph/data/sidecars.py"]
    return {name: file_sha256(root / name) for name in files}


def _cache_identity(cache, arm=None):
    cache = Path(cache)
    identity = {"protocol_sha256": digest(json.loads((cache / "protocol.json").read_text())),
            "cohort_sha256": digest(json.loads((cache / "cohort.json").read_text())),
            "cache_manifest_sha256": file_sha256(cache / "manifest.json")}
    if arm == "full_rewired":
        identity["rewire_manifest_sha256"] = file_sha256(cache / "rewire/manifest.json")
        identity["rewire_admission_sha256"] = file_sha256(cache / "rewire/admission.json")
        identity["rewire_decision_sha256"] = DECISION_SHA256
    return identity


def _check_labels(dataset):
    metadata = _value(dataset, "metadata")
    labels = np.asarray(_value(dataset, "labels"))
    expected = np.asarray(metadata["pred"]) != np.asarray(metadata["label"])
    if labels.shape != expected.shape or not np.equal(labels, expected).all():
        raise ValueError("y must be frozen predicted class != true CIFAR class; corruption is not an error label")
    if not np.equal(labels, np.asarray(metadata["y"])).all():
        raise ValueError("Dataset labels and metadata y disagree")


def _artifact_hashes(run_dir):
    return {key: file_sha256(Path(run_dir) / name) for key, name in _ARTIFACTS.items()}


def _atomic_safetensors(path, state):
    path = Path(path)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    save_file({key: value.detach().cpu().contiguous().clone() for key, value in sorted(state.items())}, str(temporary))
    os.replace(temporary, path)


def _atomic_npz(path, values):
    path = Path(path)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **values)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        if len(state["cuda"]) != torch.cuda.device_count():
            raise RuntimeError("Resume CUDA device count differs from checkpoint")
        torch.cuda.set_rng_state_all(state["cuda"])


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


class BlockShuffleSampler(torch.utils.data.Sampler):
    """Visit each row exactly once; shuffle shards and rows, with explicit state."""

    def __init__(self, dataset, seed):
        self.blocks = [list(block) for block in _value(dataset, "shard_blocks")]
        flat = sorted(item for block in self.blocks for item in block)
        if flat != list(range(len(dataset))):
            raise ValueError("Shard blocks must partition every training row exactly once")
        self.seed, self.epoch = int(seed), 0
        self.blocks_sha256 = digest(self.blocks)

    def __len__(self):
        return sum(map(len, self.blocks))

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        order = list(range(len(self.blocks)))
        rng.shuffle(order)
        for index in order:
            block = list(self.blocks[index])
            rng.shuffle(block)
            yield from block

    def state_dict(self):
        return {"seed": self.seed, "epoch": self.epoch, "blocks_sha256": self.blocks_sha256}

    def load_state_dict(self, value):
        if value["seed"] != self.seed or value["blocks_sha256"] != self.blocks_sha256:
            raise RuntimeError("Resume training sampler identity mismatch")
        self.epoch = int(value["epoch"])


def _cached_logits(dataset):
    # The logit arm must not reread multi-GB graph shards for every audit pass.
    if not hasattr(dataset, "_training_logits_cache"):
        dataset._training_logits_cache = _value(dataset, "logits").float()
    return dataset._training_logits_cache


class _Logits(torch.utils.data.Dataset):
    def __init__(self, dataset, scaler):
        logits = _cached_logits(dataset)
        _finite(logits, "classifier logits")
        if logits.shape != (len(dataset), 100):
            raise ValueError("Expected one vector of 100 logits per record")
        mean, scale = torch.tensor(scaler["mean"]), torch.tensor(scaler["scale"])
        self.x = (logits - mean) / scale
        _finite(self.x, "normalized logits")
        self.y = torch.as_tensor(_value(dataset, "labels"), dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


def _loader(dataset, config, *, sampler=None, batch_size=None, generator=None, indices=None):
    selected = _Logits(dataset, config["preprocessing"]) if config["arm"] == "logit" else dataset
    if indices is not None:
        selected = torch.utils.data.Subset(selected, indices)
    loader_type = torch.utils.data.DataLoader if config["arm"] == "logit" else GraphLoader
    # A dedicated generator prevents evaluation's DataLoader setup consuming the
    # training dropout RNG; the persistent training generator is checkpointed.
    if generator is None:
        generator = torch.Generator().manual_seed(0)
    return loader_type(selected, batch_size=batch_size or config["batch_size"],
                       sampler=sampler, shuffle=False, num_workers=0, generator=generator)


def _forward(model, batch, arm, device):
    if arm == "logit":
        x, y = (value.to(device) for value in batch)
        _finite(x, "input logits")
        scores = model(x).reshape(-1)
    else:
        batch = batch.to(device)
        _finite(batch.x, "node features")
        _finite(batch.edge_attr, "edge features")
        scores, _ = model(batch)
        scores, y = scores.reshape(-1), batch.y.reshape(-1)
    _finite(scores, "failure scores")
    _finite(y, "labels")
    if scores.shape != y.shape:
        raise RuntimeError("One failure score per input row is required")
    return scores, y


@torch.no_grad()
def _collect(model, dataset, config, device, *, batch_size=None, indices=None):
    model.eval()
    values = []
    for batch in _loader(dataset, config, batch_size=batch_size, indices=indices):
        score, _ = _forward(model, batch, config["arm"], device)
        values.append(score.detach().cpu().numpy())
    result = np.concatenate(values).astype(np.float32, copy=False)
    _finite(result, "collected failure scores")
    return result


def auroc(labels, scores):
    """Mann–Whitney AUROC with half credit for each tied positive-negative pair."""
    labels, scores = np.asarray(labels), np.asarray(scores)
    _finite(scores, "AUROC scores")
    if labels.shape != scores.shape or not np.isin(labels, [0, 1]).all():
        raise ValueError("AUROC inputs must be aligned binary labels and scores")
    positives, negatives = int((labels == 1).sum()), int((labels == 0).sum())
    if not positives or not negatives:
        raise ValueError("Validation AUROC is undefined without both classes")
    order = np.argsort(scores, kind="stable")
    ordered_scores, ordered_labels = scores[order], labels[order]
    starts = np.r_[0, np.flatnonzero(np.diff(ordered_scores)) + 1]
    ends = np.r_[starts[1:], len(scores)]
    positive_counts = np.add.reduceat(ordered_labels.astype(np.int64), starts)
    negative_counts = ends - starts - positive_counts
    previous_negatives = np.cumsum(negative_counts) - negative_counts
    return float(np.sum(positive_counts * (previous_negatives + 0.5 * negative_counts)) / (positives * negatives))


def validation_threshold(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores)
    _finite(scores, "validation threshold scores")
    correct = np.sort(scores[labels == 0])
    if not len(correct):
        raise ValueError("Cannot freeze a threshold without correct validation predictions")
    rank = math.ceil(0.95 * len(correct))
    threshold = float(correct[rank - 1])
    alarm = scores > threshold
    wrong = labels == 1
    return {"threshold": threshold, "score": "native_failure_logit", "alarm_rule": "score > threshold",
            "selection": "correct-validation ascending rank ceil(0.95*n), one-based; conservative ties",
            "correct_rank": rank, "correct_count": int(len(correct)), "error_count": int(wrong.sum()),
            "false_alarm_count": int(alarm[~wrong].sum()),
            "validation_false_alarm_rate": float(alarm[~wrong].mean()),
            "error_alarm_count": int(alarm[wrong].sum()),
            "validation_error_recall": float(alarm[wrong].mean()) if wrong.any() else None,
            "equal_threshold_count": int((scores == threshold).sum()), "record_count": int(len(scores))}


def compare_scores(reference, restored, threshold=None):
    """Tolerance audit, robust-pair ordering, and conservative threshold audit.

    Called on at most 48 fixed rows, so explicit pair masks stay bounded.
    """
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    restored = np.asarray(restored, dtype=np.float64).reshape(-1)
    if reference.shape != restored.shape:
        raise RuntimeError("Restored score shape mismatch")
    _finite(reference, "reference audit scores")
    _finite(restored, "restored audit scores")
    errors = ATOL + RTOL * np.abs(reference)
    if not np.allclose(reference, restored, atol=ATOL, rtol=RTOL):
        raise RuntimeError(f"Restored failure scores exceed atol/rtol=1e-5: max error {np.max(np.abs(reference-restored))}")
    delta = reference[:, None] - reference[None, :]
    changed = np.sign(delta) != np.sign(restored[:, None] - restored[None, :])
    pairs = np.triu(np.ones(delta.shape, dtype=bool), 1)
    robust = np.abs(delta) > errors[:, None] + errors[None, :]
    if np.any(pairs & robust & changed):
        raise RuntimeError("Restoration changed a numerically separated pair's ordering")
    result = {"atol": ATOL, "rtol": RTOL, "count": len(reference),
              "max_absolute_error": float(np.max(np.abs(reference - restored), initial=0)),
              "robust_pairs": int((pairs & robust).sum()),
              "near_tie_order_changes": int((pairs & ~robust & changed).sum())}
    if threshold is not None:
        decisions_changed = (reference > threshold) != (restored > threshold)
        safe = np.abs(reference - threshold) > errors
        if np.any(safe & decisions_changed):
            raise RuntimeError("Restoration changed a decision safely outside threshold tolerance")
        result["near_threshold_decision_changes"] = int((~safe & decisions_changed).sum())
    return result


def _make_config(cache, dataset, arm, seed, device):
    labels = np.asarray(_value(dataset, "labels"))
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Training labels must indicate classifier errors")
    positive, negative = int((labels == 1).sum()), int((labels == 0).sum())
    if not positive or not negative:
        raise ValueError("Training BCE weighting requires both classes")
    preprocessing = {"kind": "none"}
    if arm == "logit":
        values = _cached_logits(dataset)
        _finite(values, "training logits")
        mean, std = values.mean(dim=0), values.std(dim=0, unbiased=False)
        preprocessing = {"kind": "train_population_standardization", "mean": mean.tolist(),
                         "scale": (std + 1e-6).tolist(), "epsilon": 1e-6,
                         "fit_split": "train", "fit_count": len(dataset)}
    training = protocol()["training"]
    implementation = _implementation_identity()
    versions = {name: importlib.metadata.version(name) for name in ("torch", "numpy", "torch-geometric", "safetensors")}
    config = {"schema_version": SCHEMA_VERSION, "arm": arm, "seed": seed, "arm_spec": ARMS[arm],
              "training": training, "batch_size": training["logit_batch_size" if arm == "logit" else "batch_size"],
              "preprocessing": preprocessing, "pos_weight": negative / positive,
              "training_correct_count": negative, "training_error_count": positive,
              "training_record_count": len(dataset), "training_labels_sha256": digest(labels.astype(int).tolist()),
              "implementation": implementation, "implementation_sha256": digest(implementation),
              "versions": versions, "torch_cuda_version": torch.version.cuda,
              "device_type": device.type, "dtype": "float32", "initialization": "fresh_seeded_build_model",
              "sampler": "shuffle shard blocks then rows, every row once, independent Python Random(seed+epoch)",
              **_cache_identity(cache, arm)}
    if config["protocol_sha256"] != digest(protocol()):
        raise RuntimeError("Training implementation protocol differs from cache protocol")
    return config


def _verify_config(config):
    if config.get("schema_version") != SCHEMA_VERSION or config.get("arm") not in ARMS or config.get("seed") not in SEEDS:
        raise RuntimeError("Unrecognized run; historical checkpoints cannot be reused")
    if config["implementation"] != _implementation_identity() or config["implementation_sha256"] != digest(config["implementation"]):
        raise RuntimeError("Run implementation has changed")
    if config["protocol_sha256"] != digest(protocol()):
        raise RuntimeError("Run protocol has changed")


def _verify_complete(run_dir):
    run_dir = Path(run_dir)
    complete = json.loads((run_dir / "complete.json").read_text())
    if complete.get("complete") is not True or complete.get("selection_split") != "val":
        raise RuntimeError("Run is not a completed validation-selected fit")
    for key, actual in _artifact_hashes(run_dir).items():
        if complete.get(key) != actual:
            raise RuntimeError(f"Completed artifact changed: {key}")
    if complete.get("latest_sha256") != file_sha256(run_dir / "latest.pt"):
        raise RuntimeError("Completed resume state is missing or changed")
    if complete.get("history_sha256") != file_sha256(run_dir / "history.json"):
        raise RuntimeError("Completed training history changed")
    return complete


def load_run(run_dir, device):
    """Return ``(model, config)`` after verifying a completed native checkpoint."""
    require_slurm()
    run_dir, device = Path(run_dir), torch.device(device)
    _verify_complete(run_dir)
    config = json.loads((run_dir / "config.json").read_text())
    _verify_config(config)
    model = build_model(config["arm"]).to(device)
    model.load_state_dict(load_file(str(run_dir / "best.safetensors"), device="cpu"), strict=True)
    model.eval()
    return model, config


def predict_split(run_dir, cache, split, device, freeze=None):
    """Aligned metadata plus ``score``/``logit`` native error scores.

    Test access validates every frozen contestant before constructing its dataset.
    This function never fits or modifies a threshold, checkpoint, or scaler.
    """
    require_slurm()
    run_dir, cache, device = Path(run_dir), Path(cache), torch.device(device)
    config = json.loads((run_dir / "config.json").read_text())
    if split == "test":
        frozen = check_freeze(cache, freeze)
        key = f"{config['arm']}/seed{config['seed']}"
        entry = frozen.get("runs", {}).get(key)
        if not isinstance(entry, dict):
            raise RuntimeError(f"Run is absent from frozen matrix: {key}")
        expected = {**_artifact_hashes(run_dir), "complete_sha256": file_sha256(run_dir / "complete.json")}
        for name, value in expected.items():
            if entry.get(name) != value:
                raise RuntimeError(f"Requested run differs from frozen artifact: {key}/{name}")
    model, config = load_run(run_dir, device)
    for key, value in _cache_identity(cache, config["arm"]).items():
        if config[key] != value:
            raise RuntimeError(f"Prediction cache differs from training: {key}")
    dataset = CachedDataset(cache, split, config["arm"], freeze=freeze)
    result = {key: np.asarray(value) for key, value in _value(dataset, "metadata").items()}
    scores = _collect(model, dataset, config, device)
    if any(len(value) != len(scores) for value in result.values()):
        raise RuntimeError("Prediction metadata alignment failure")
    result.update(score=scores, logit=scores)
    return result


@contextlib.contextmanager
def _run_lock(run_dir):
    with (Path(run_dir) / ".training.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another process owns this arm/seed run") from error
        yield


def _request_stop(signum, frame):
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


def _fit(cache, run_dir, arm, seed, device):
    train_ds, val_ds = CachedDataset(cache, "train", arm), CachedDataset(cache, "val", arm)
    _check_labels(train_ds)
    _check_labels(val_ds)
    config = _make_config(cache, train_ds, arm, seed, device)
    config_path, latest_path = run_dir / "config.json", run_dir / "latest.pt"
    if config_path.exists():
        existing = json.loads(config_path.read_text())
        if digest(existing) != digest(config):
            raise RuntimeError("Resume configuration, data, implementation, or environment changed")
    else:
        if any((run_dir / name).exists() for name in ("best.safetensors", "latest.pt", "complete.json")):
            raise RuntimeError("Refusing unrecognized checkpoint reuse without this trainer's configuration")
        atomic_json(config_path, config)
    if (run_dir / "complete.json").exists():
        _verify_complete(run_dir)
        return True
    _seed(seed)
    model = build_model(arm).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["lr"], weight_decay=config["training"]["weight_decay"])
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(config["pos_weight"], device=device))
    sampler = BlockShuffleSampler(train_ds, seed)
    loader_rng = torch.Generator().manual_seed(seed + 1000003)
    history, best_state, best_scores = [], None, None
    best_auc, best_epoch, stale, start_epoch, prior_seconds = -math.inf, 0, 0, 0, 0.0
    audit_indices = list(range(min(48, len(val_ds))))
    attempts = json.loads((run_dir / "attempts.json").read_text()) if (run_dir / "attempts.json").exists() else []
    attempt = {"started_unix": time.time(), "slurm_job_id": os.environ["SLURM_JOB_ID"],
               "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"), "device": str(device),
               "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
               "resume": latest_path.exists(), "status": "running"}
    attempts.append(attempt)
    atomic_json(run_dir / "attempts.json", attempts)
    if latest_path.exists():
        # This is our own trusted resume format, not a third-party model pickle.
        state = torch.load(latest_path, map_location="cpu", weights_only=False)
        if state.get("schema_version") != SCHEMA_VERSION or state.get("config_sha256") != file_sha256(config_path):
            raise RuntimeError("Resume checkpoint is not bound to this run configuration")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        sampler.load_state_dict(state["sampler"])
        loader_rng.set_state(state["loader_rng"])
        history, best_state, best_scores = state["history"], state["best_state"], state["best_scores"]
        best_auc, best_epoch, stale = state["best_auc"], state["best_epoch"], state["stale_epochs"]
        start_epoch, prior_seconds = state["completed_epochs"], state["training_seconds"]
        if sampler.epoch != start_epoch or len(history) != start_epoch:
            raise RuntimeError("Resume epoch, sampler, and history disagree")
        if state["audit"]["indices"] != audit_indices or state["audit"]["device_type"] != device.type:
            raise RuntimeError("Resume audit layout/device differs")
        restored = _collect(model, val_ds, config, device, indices=audit_indices)
        audit = compare_scores(state["audit"]["scores"], restored)
        alternative = _collect(model, val_ds, config, device, indices=audit_indices, batch_size=max(1, config["batch_size"] // 2))
        audit["alternative_partition"] = compare_scores(restored, alternative)
        atomic_json(run_dir / "resume_audit.json", audit)
        _restore_rng(state["rng"])
        # A preemption between best and latest replacement may leave an orphaned
        # best file. The committed resume state is the sole source of truth.
        _atomic_safetensors(run_dir / "best.safetensors", best_state)
        atomic_json(run_dir / "history.json", history)
    training_loader = _loader(train_ds, config, sampler=sampler, generator=loader_rng)
    started = time.monotonic()
    try:
        if stale < config["training"]["patience"]:
            for epoch in range(start_epoch, config["training"]["epochs"]):
                epoch_started = time.monotonic()
                model.train()
                total_loss, count = 0.0, 0
                for batch in training_loader:
                    optimizer.zero_grad(set_to_none=True)
                    scores, labels = _forward(model, batch, arm, device)
                    loss = criterion(scores, labels)
                    _finite(loss, "training loss")
                    loss.backward()
                    for name, parameter in model.named_parameters():
                        if parameter.grad is not None:
                            _finite(parameter.grad, f"gradient {name}")
                    optimizer.step()
                    total_loss += float(loss.detach()) * len(labels)
                    count += len(labels)
                if count != len(train_ds):
                    raise RuntimeError("An epoch did not visit each training row once")
                val_scores = _collect(model, val_ds, config, device)
                val_auc = auroc(_value(val_ds, "labels"), val_scores)
                improved = val_auc > best_auc
                if improved:
                    best_auc, best_epoch, stale = val_auc, epoch + 1, 0
                    best_state, best_scores = _cpu_state(model), val_scores.copy()
                    _atomic_safetensors(run_dir / "best.safetensors", best_state)
                else:
                    stale += 1
                history.append({"epoch": epoch + 1, "training_loss": total_loss / count,
                                "validation_auroc": val_auc, "best_epoch": best_epoch,
                                "best_validation_auroc": best_auc, "stale_epochs": stale,
                                "selected": improved, "seconds": time.monotonic() - epoch_started})
                audit_scores = _collect(model, val_ds, config, device, indices=audit_indices)
                state = {"schema_version": SCHEMA_VERSION, "config_sha256": file_sha256(config_path),
                         "model": _cpu_state(model), "optimizer": optimizer.state_dict(),
                         "completed_epochs": epoch + 1, "history": history,
                         "best_state": best_state, "best_scores": best_scores,
                         "best_auc": best_auc, "best_epoch": best_epoch, "stale_epochs": stale,
                         "rng": _rng_state(), "sampler": sampler.state_dict(), "loader_rng": loader_rng.get_state(),
                         "training_seconds": prior_seconds + time.monotonic() - started,
                         "audit": {"indices": audit_indices, "scores": audit_scores, "device_type": device.type,
                                   "batch_size": config["batch_size"], "dtype": "float32"}}
                atomic_torch(latest_path, state)
                atomic_json(run_dir / "history.json", history)
                print(json.dumps({"arm": arm, "seed": seed, **history[-1]}), flush=True)
                if stale >= config["training"]["patience"]:
                    break
                if _STOP_REQUESTED:
                    attempt.update(status="interrupted_at_epoch_boundary", completed_epochs=epoch + 1, ended_unix=time.time())
                    atomic_json(run_dir / "attempts.json", attempts)
                    return False
        if best_state is None or not latest_path.exists():
            raise RuntimeError("No validation-selected state and resumable checkpoint available")
        # Validate the serialized selected model, including an alternative batch
        # partition. These rows are validation-only and never test examples.
        model.load_state_dict(best_state, strict=True)
        reference = _collect(model, val_ds, config, device, indices=audit_indices)
        _atomic_safetensors(run_dir / "best.safetensors", best_state)
        restored_model = build_model(arm).to(device)
        restored_model.load_state_dict(load_file(str(run_dir / "best.safetensors"), device="cpu"), strict=True)
        restored = _collect(restored_model, val_ds, config, device, indices=audit_indices)
        selected_scores = _collect(restored_model, val_ds, config, device)
        # Full-vector check is linear; pair ordering is separately audited on the
        # fixed bounded batch to avoid a quadratic 7200-record matrix.
        _finite(selected_scores, "restored complete validation scores")
        if not np.allclose(best_scores, selected_scores, atol=ATOL, rtol=RTOL):
            raise RuntimeError("Selected checkpoint failed complete-validation score restoration")
        threshold = validation_threshold(_value(val_ds, "labels"), selected_scores)
        audit = compare_scores(reference, restored, threshold["threshold"])
        alternative = _collect(restored_model, val_ds, config, device, indices=audit_indices, batch_size=max(1, config["batch_size"] // 2))
        audit["alternative_partition"] = compare_scores(restored, alternative, threshold["threshold"])
        metadata = {key: np.asarray(value) for key, value in _value(val_ds, "metadata").items()}
        if any(len(value) != len(selected_scores) for value in metadata.values()):
            raise RuntimeError("Validation artifact alignment failure")
        _atomic_npz(run_dir / "validation.npz", {**metadata, "score": selected_scores, "logit": selected_scores})
        atomic_json(run_dir / "validation.json", {**threshold, "schema_version": SCHEMA_VERSION,
                    "arm": arm, "seed": seed, "selection_split": "val", "best_epoch": best_epoch,
                    "selected_epoch_validation_auroc": best_auc,
                    "restored_validation_auroc": auroc(_value(val_ds, "labels"), selected_scores),
                    "preprocessing": config["preprocessing"], "restoration_audit": audit,
                    "config_sha256": file_sha256(config_path), "best_sha256": file_sha256(run_dir / "best.safetensors")})
        attempt.update(status="complete", ended_unix=time.time(), completed_epochs=len(history))
        atomic_json(run_dir / "attempts.json", attempts)
        atomic_json(run_dir / "complete.json", {"schema_version": SCHEMA_VERSION, "complete": True,
                    "arm": arm, "seed": seed, "selection_split": "val", "best_epoch": best_epoch,
                    "best_validation_auroc": best_auc, "completed_epochs": len(history),
                    "stop_reason": "patience" if stale >= config["training"]["patience"] else "max_epochs",
                    "training_seconds": prior_seconds + time.monotonic() - started,
                    "config_digest": digest(config), "implementation_sha256": config["implementation_sha256"],
                    **_artifact_hashes(run_dir), "latest_sha256": file_sha256(latest_path),
                    "history_sha256": file_sha256(run_dir / "history.json"),
                    "test_evaluated": False, "completed_unix": time.time()})
        _verify_complete(run_dir)
        return True
    except BaseException as error:
        attempt.update(status="failed", ended_unix=time.time(), error=repr(error), traceback=traceback.format_exc())
        atomic_json(run_dir / "attempts.json", attempts)
        raise


def main():
    require_slurm()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--arm", choices=tuple(ARMS), required=True)
    parser.add_argument("--seed", choices=SEEDS, type=int, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    run_dir = args.run_root / args.arm / f"seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGUSR1, _request_stop)
    with _run_lock(run_dir):
        finished = _fit(args.cache, run_dir, args.arm, args.seed, torch.device(args.device))
    print(json.dumps({"arm": args.arm, "seed": args.seed, "complete": finished, "run_dir": str(run_dir)}), flush=True)
    if not finished:
        raise SystemExit(75)


if __name__ == "__main__":
    main()
